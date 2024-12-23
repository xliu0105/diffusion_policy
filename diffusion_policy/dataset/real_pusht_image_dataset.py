from typing import Dict, List
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
'''
    为什么要区分pusht_image_dataset和real_pusht_image_dataset，是因为在pusht_image_dataset中，self.replay_buffer = ReplayBuffer.copy_from_path这一个指令直接将
    zarr数据集中的数据完全加载到了内存中，并由ReplayBuffer高效管理。这样的做法在数据集较小的时候是没有问题的，但是在数据集较大的时候，会导致内存不足。
'''

class RealPushTImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,  # shape_meta是一个字典，数据形状的元信息，可以参考训练任务的.yaml文件
            dataset_path: str,  # 数据集路径，这个代码用Zarr数据集
            horizon=1,  # 参考论文定义的Horizon
            pad_before=0,  # pad_before参数是在一个episode最一开始pad几个，这个和Horizon以及n_obs_steps有关
            pad_after=0,  # pad_after参数是在一个episode最后面pad几个，这个和Horizon以及n_action_steps有关
            n_obs_steps=None,
            n_latency_steps=0,
            # use_cache是用来控制是否启用缓存机制。数据集中的数据在训练前需要经过预处理，如果每次加载数据都要重新预处理，那么会导致训练时间过长。
            # 因此，可以将预处理后的数据保存到磁盘中，下次加载数据的时候直接从磁盘中加载，这样可以节省大量的时间。
            use_cache=False,
            seed=42,
            val_ratio=0.0,  # 验证数据集的占比
            max_train_episodes=None,  # 限制训练集的最大episodes数量
            delta_action=False,  # 是否将action转化为相邻时间步的差分形式
        ):
        assert os.path.isdir(dataset_path)
        
        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta
            '''
            shape_meta_json和shape_meta_hash是为了生成唯一标识符，用于缓存文件的管理。它们的目标是确保每种shape_meta配置
            都有一个独立且对应的缓存文件，防止因shape_meta配置的不同导致缓存数据冲突。
            '''
            # 将shape_meta转为json格式的字符串表示
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            # 根据shape_meta的json字符串生成一个hash值，这个hash值是唯一的，用于标识shape_meta，相同的shape_meta生成的hash值是相同的；不同的shape_meta生成的hash值是不同的
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            # 在数据集路径下创建一个缓存文件，缓存文件的命名规则是shape_meta的hash值加上.zarr.zip的后缀
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            # 创建与缓存文件对应的锁文件，用于保证多进程读写的安全性
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            # 文件锁，用来保证多进程读写的安全性
            with FileLock(cache_lock_path):
                # 判断如果缓存文件不存在，那么就创建缓存文件；如果缓存文件存在，那么就直接从缓存文件中加载数据
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore()  # zarr.MemoryStore代表在内存中存储，而不是在磁盘中存储，如果数据集较大，可能会导致内存不足
                        )
                        '''
                        store还有其他很多选择，如zarr.TempStore()，将数据存储在临时文件夹(一般在/tmp目录下)中，适用于需要中间存储但不需要长期保存的场景
                        可以通过store = zarr.TempStore(); print(store.path)来查看临时文件夹的路径
                        '''
                        print('Saving cache to disk.')
                        # 将replay_buffer保存到缓存文件中
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        # TODO: 为什么还要完全copy一份数据，为什么不能直接用原始数据集？一定要copy_from_store?
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore()
            )
        
        # 可以将action转化为相邻时间步的差分形式，通过delta_action参数控制
        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # 这个的作用应该是，SequenceSampler在采样的时候，会采一整个Horizon的数据，但对于obs来说，只需要取前n_obs_steps个数据，因此设置这个加速采样
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        # train_mask和val_mask用来标记哪些episode是训练集，哪些是验证集，这里的mask是一个bool数组，train_mask当然就是val_mask的反
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        # 通过downsample_mask函数来降低训练集的数量，这里的max_train_episodes是一个阈值，如果训练集的数量超过这个阈值，就会进行下采样
        # 但不知道为什么要限制训练集的数量，是为了节省内存？
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # 创建序列采样器，应该是支持时间序列的采样
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        # 浅拷贝当前对象，返回一个新对象，但与原对象共享可变子对象数据（如列表、字典、集合、自定义对象实例等）
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        
        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        
        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        # 从sampler中采样一个序列数据
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        # 创建一个从0到n_obs_steps的切片
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            # np.moveaxis是移动轴的位置，这里是将最后一个轴移动到第二个位置
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            del data[key]  # 删除data中的key，释放内存
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]
        
        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data

'''
zarr_resize_index_last_dim函数根据idxs参数对zarr_arr的最后一个维度进行resize操作，idxs是一个列表，列表中的元素是要保留的维度的索引。
比如zarr_arr的形状是(100,5)，idxs=[0,1]，那么resize后的zarr_arr的形状是(100,2)。
'''
def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[...,idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr

def _get_replay_buffer(dataset_path, shape_meta, store):
    # parse shape meta
    # rgb_keys和lowdim_keys是存储shape_meta中的obs字段中的所有键名字，如agent_pos, image等
    rgb_keys = list()
    lowdim_keys = list()
    out_resolutions = dict()  # 输出图像的分辨率
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():  # 遍历 shape_meta['obs'] 中的所有字段
        type = attr.get('type', 'low_dim')  # 获取字段的类型，如果没有指定，默认为low_dim
        shape = tuple(attr.get('shape'))  # 获取字段的形状
        # 如果字段的类型是rgb，那么将字段的键名加入到rgb_keys中，并将字段的分辨率加入到out_resolutions字典中
        if type == 'rgb':
            rgb_keys.append(key)
            c,h,w = shape
            out_resolutions[key] = (w,h)
        # 如果字段的类型是low_dim，那么将字段的键名加入到lowdim_keys中，并将字段的形状加入到lowdim_shapes字典中
        elif type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            # 如果low_dim类型的字段的键名中包含pose，那么将字段的形状限制为(2,)或(6,)，否则报错
            if 'pose' in key:
                assert tuple(shape) in [(2,),(6,)]
    
    # 获取action_shape的形状，并且要求action_shape的形状只能是(2,)或(6,)
    action_shape = tuple(shape_meta['action']['shape'])
    assert action_shape in [(2,),(6,)]

    # load data
    cv2.setNumThreads(1)  # 将 OpenCV的线程数限制为1
    with threadpool_limits(1):  # 限制线程池的线程数量
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_keys + ['action'],
            image_keys=rgb_keys
        )

    # transform lowdim dimensions
    # 可能是由于real world收集来的数据包含XYZ，但是实际上只需要XY，因此需要对数据进行剔除操作
    if action_shape == (2,):
        # 2D action space, only controls X and Y
        zarr_arr = replay_buffer['action']
        zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])
    
    for key, shape in lowdim_shapes.items():
        if 'pose' in key and shape == (2,):
            # only take X and Y
            zarr_arr = replay_buffer[key]
            zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])

    return replay_buffer


def test():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize('../diffusion_policy/config'):
        cfg = hydra.compose('train_robomimic_real_image_workspace')
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'][:])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
    _ = plt.hist(dists, bins=100); plt.title('real action velocity')
