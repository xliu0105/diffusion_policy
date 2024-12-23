from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

# Dataset类，最根上需要继承于torch.utils.data.Dataset类，同时必须要实现__len__和__getitem__方法
class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,  # pad_before参数是在一个episode最一开始pad几个，这个和Horizon以及n_obs_steps有关
            pad_after=0,  # pad_after参数是在一个episode最后面pad几个，这个和Horizon以及n_action_steps有关
            seed=42,
            val_ratio=0.0,  # 验证数据集占比
            max_train_episodes=None
            ):
        
        super().__init__()
        # 调用replayBuffer从zarr数据集中加载数据，加载指定字段，应该是默认加载data group下的提供字段的数据
        # IMPORTANT: ReplayBuffer.copy_from_path是将zarr数据集中的所有指定字段的数据加载到内存中，这个操作是非常耗内存的，因此如果在数据集较大的情况下，需要使用
        #            real_pusht_image_dataset.py中RealPushTImageDataset类。
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        # train_mask和val_mask用来标记哪些episode是训练集，哪些是验证集，这里的mask是一个bool数组，train_mask当然就是val_mask的反
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        # 通过downsample_mask函数来降低训练集的数量，这里的max_train_episodes是一个阈值，如果训练集的数量超过这个阈值，就会进行下采样
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        # 初始化序列采样器，应该是支持时间序列的采样
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    # 返回一个数据归一化工具
    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    # 将__getitem__ sample出来的数据做一下处理，做一下坐标转换，转为np，同时放入字典中返回
    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        # np.moveaxis是移动轴的位置，这里是将最后一个轴移动到第二个位置
        image = np.moveaxis(sample['img'],-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
