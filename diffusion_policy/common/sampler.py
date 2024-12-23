from typing import Optional
import numpy as np
import numba
from diffusion_policy.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray,   # 数据库中的episode_ends，代表每个episode的结束位置
    sequence_length:int,  # 采样的序列长度，Horizon+n_latency_steps
    episode_mask: np.ndarray,
    pad_before: int=0, 
    pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    '''
    函数的目的是生成从ReplayBuffer中采样的索引，用于时间序列任务。同时支持序列长度设置和边界填充(pad_before, pad_after)，
    并根据episode_mask来选择是否采样某个episode。
    '''
    episode_mask.shape == episode_ends.shape
    # 限制pad_before和pad_after的取值范围，确保不会超出sequence_length且不会小于0
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    # 遍历每个episode
    for i in range(len(episode_ends)):
        # 如果episode_mask[i]为False，则跳过该episode
        if not episode_mask[i]:
            # skip episode
            continue
        # 注意，start_idx是在ReplayBuffer中，这个episode起始位置的索引
        start_idx = 0
        # 由于在episode_ends中存储的是每个episode的结束位置，因此需要根据i-1来获取这个episode的开始位置
        if i > 0:
            start_idx = episode_ends[i-1]
        # 获取当前episode的结束位置
        end_idx = episode_ends[i]
        # 计算当前episode的长度
        episode_length = end_idx - start_idx
        
        # 限制采样的最小起始位置和最大起始位置，根据pad_before和pad_after来限制
        # 若pad_before=5，则可以在-5的位置开始采样，因为可以前向pad 5个数据，则min_start=-5
        # 若pad_after=5，则可以在episode_length-5的位置结束采样，因为可以后向pad 5个数据，则max_start=episode_length - Horizon +5
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            # 注意，buffer_start_idx是这个idx对应的在ReplayBuffer中的原始数据索引，要有一个max限制，否则就可能会采样到上一个episode的数据
            buffer_start_idx = max(idx, 0) + start_idx
            # 同样，buffer_end_idx也要有一个min限制，否则就可能会采样到下一个episode的数据
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            # start_offset是采样的起始位置相对于buffer_start_idx的偏移量，主要是针对有pad_before的情况，数据采样需要从ReplayBuffer的buffer_start_idx开始，
            # 而还需要在前面添加start_offset个数据，以保证采样的数据长度为sequence_length，注意，start_offset≥0
            start_offset = buffer_start_idx - (idx+start_idx)
            # 和start_offset类似，end_offset是采样的结束位置相对于buffer_end_idx的偏移量，主要是针对有pad_after的情况，数据采样需要在ReplayBuffer的buffer_end_idx结束，
            # 而还需要在后面添加end_offset个数据，以保证采样的数据长度为sequence_length
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            # 对于这个idx的采样，buffer_start_idx的位置是这个idx的squence_length的哪个位置
            sample_start_idx = 0 + start_offset
            # 对于这个idx的采样，buffer_end_idx的位置事这个idx的squence_length的哪个位置
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            # 将数据添加到indices中
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    # 经过两个循环，这时的indices已经包含了所有可以采样的episode中，所有可以用来采样的数据的索引
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    # 首先创建全0的bool mask，然后根据总的episode数量以及验证集占比val_ratio，来随机选择一部分episode作为验证集，同时保证至少有一个episode作为验证集，至少有一个episode作为训练集
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    # 通过随机选择的方式，对一个bool掩码进行下采样(减少为True的数量)，以限制选中的样本数量不超过max_n
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer,  # 包含所有数据库数据的ReplayBuffer对象
        sequence_length:int,  # 采样的序列长度，Horizon+n_latency_steps
        pad_before:int=0,  # pad_before参数是在一个episode最一开始pad几个
        pad_after:int=0,  # pad_after参数是在一个episode最后面pad几个
        keys=None,  # 如果传入了keys，则只采样这些keys的数据；如果没有传入keys，则默认使用replay_buffer中data组下的所有keys
        key_first_k=dict(),  # SequenceSampler在采样的时候，会采一整个Horizon的数据，但对于obs来说，只需要取前n_obs_steps个数据，因此设置key_first_k加速采样
        episode_mask: Optional[np.ndarray]=None,  # 如果是训练集，就传入train_mask；如果是验证集，就传入val_mask
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
            SequenceSampler在采样的时候，会采一整个Horizon的数据，但对于obs来说，只需要取前n_obs_steps个数据，因此设置key_first_k加速采样
        """

        super().__init__()
        assert(sequence_length >= 1)
        # 如果没有传入keys，则默认使用replay_buffer中data组下的所有keys
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        # 如果没有传入episode_mask，则默认全为True，即所有episode都参与采样
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        # np.any用于检查数组中是否至少存在一个元素为True(或非零)，如果存在，则返回True，否则返回False
        if np.any(episode_mask):
            # 获得所有可用episode中所有可以用来采样的数据的索引
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)  # 形状为(0,4)的空数组，没有实际数据

        # indices中每个元素的结构：(buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        # 按idx从indices中取出对应元素
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            # 使用ReplayBuffer的__getitem__方法，默认从数据库的'data'组下获取数据
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                # 如果input_arr的shape是(100,10,10)，n_data是16，那么(n_data,) + input_arr.shape[1:] = (16,10,10)
                # 创建一个全为nan的数组，第一维是n_data，剩下的维度就是原始数据的维度
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    # 将sample全为nan的数组，前k_data个数据替换为input_arr[buffer_start_idx:buffer_start_idx+k_data]
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            # 下面的嵌套if语句，就是为了处理pad_before和pad_after的情况，根据sample_start_idx和sample_end_idx
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                # 再次强调，sample_start_idx代表的是，buffer_start_idx位置的数据在这个idx的squence_length的那个位置
                # sample_end_idx代表的是，buffer_end_idx位置的数据在这个idx的squence_length的哪个位置
                data[sample_start_idx:sample_end_idx] = sample
            # 将每个key的数据，存储在名为result的字典中
            result[key] = data
        return result
