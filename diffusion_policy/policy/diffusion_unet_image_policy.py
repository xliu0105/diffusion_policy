from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
# 这里只用了diffusers库的DDPMScheduler类，是用来定义 扩散过程 的时间步长、噪声注入方式，以及在生成时逐步去噪的策略
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler  # diffusers是huggingface的扩散模型库

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,  # 对观测图像的编码器
            horizon,  # 预测的时间步长
            n_action_steps,  # 从预测的时间步长中取出的动作步长去执行
            n_obs_steps,  # 输入给模型的观测步长
            num_inference_steps=None,  # 生成时的时间步长，即加噪步长T，也是去噪步长
            obs_as_global_cond=True,  # 是否将观测作为全局condition，如果为false，则将观测的特征放入input中
            diffusion_step_embed_dim=256,  # 对timesteps进行编码的维度
            down_dims=(256,512,1024),  # Unet下采样的维度
            kernel_size=5,  # 卷积核的大小，应该对应1维卷积吧
            n_groups=8,  # 组归一化的组数
            cond_predict_scale=True,  # 是否根据condition计算scale和bias，对模型的输出进行缩放和偏移，如果为False，则直接将condition编码后的结果加到模型输出上
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        # 获取action_shape/action_dim，就是一个动作是几维的，如xyz+轴角就是6维，像pushT任务的动作就是2维，只有x和y
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1  # action_shape应该是一维的
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]  # 根据传入的obs_encoder获取观测要编码成多少维度的

        # create diffusion model
        # 设置input_dim，如果obs_as_global_cond为true，则input_dim只是action_dim，否则input_dim则为action_dim+obs_feature_dim
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps  # 由于有n_obs_steps个历史观测值，则global_cond_dim为obs_feature_dim乘上n_obs_steps

        # 创建1维Unet模型
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(  # 生成mask的类，用于表示在模型的input中，哪些是要被去噪的action，哪些是condition(有的地方condition作为全局condition，就不需要这个mask)
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    # 之前提到过，如果obs_as_global_cond为false，则就要将观测的特征放到input中，因此这里就有condition_data，其中包含了action和观测的特征(condition)，其中观测的特征是真实的，而action只是用来占位的，去采样相同维度的噪音
    # condition_mask就是要标记，在condition_data中哪些是condition(观测的特征)，哪些是要去噪的(action)
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(  # 采样一个和condition_data相同维度的噪音，注意，如果obs_as_global_cond为false，则condition_data中包含了action+condition维度
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)  # DDPM的调度器

        for t in scheduler.timesteps:  # 这个for循环是逆向去噪过程
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]  # 将condition_data中，condition_mask为True的部分赋值给trajectory对应位置

            # 2. predict model output
            # 估计噪音
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            # 使用DDPM的调度器，计算出x_t-1的值应该是什么
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]  # 再次将condition_data中，condition_mask为True的部分赋值给trajectory对应位置   

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)  # 对观测进行归一化，应该是在训练时计算了观测的均值和方差，这里用相同的参数来归一化
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]  # B是batch_size，To是观测的时间步长
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        # 以下这个if else是根据obs_as_global_cond是否为true来处理input和condition
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            # 对nobs字典中所有的值进行处理，但不改变字典的结构。将nobs字典中的值的维度进行变换，即将B, T, ... -> B*T, ...
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))  # *x.shape[2:]是解包操作，将x.shape[2:]的元素作为参数传入reshape
            nobs_features = self.obs_encoder(this_nobs)  # 对观测进行encoder，encoder中会根据字典的obs的key对图片进行编码
            # reshape back to B, Do*To
            global_cond = nobs_features.reshape(B, -1)  # 对观测encoder后的特征进行reshape，前面加一个batch_size维度，此时global_cond维度为：(B, Do*To)
            # empty data for action
            # 如果obs_as_global_cond为true，则condition会作为全局condition，因此这里只需要设置好action的维度即可，这里设置为全0的action和全false的mask
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            # 对nobs字典中所有的值进行处理，但不改变字典的结构。将nobs字典中的值的维度进行变换，即将B, T, ... -> B*T, ...
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)  # 对观测进行encoder
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)  # 由于obs_as_global_cond为false，则给模型的input包含action和观测的特征
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features  # 观测的encoder后的特征放在指定位置
            cond_mask[:,:To,Da:] = True  # 设置mask，指定观测的特征位置为True

        # run sampling
        # 执行采样，得到结果
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]  # 从结果中取出action部分
        # 对得出的action反归一化，应该是在训练中，对数据集中的action进行了归一化，应该需要计算归一化的均值和方差，这里用相同的参数来反归一化
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        # 这一部分是获取n_action_steps部分的action，即从预测的时间步长中取出的动作步长去执行
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        # batch是一个字典，包括了obs和action
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]  # 此时传入的naction的维度应该是：(B, horizon, action_dim)

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T, ...
            this_nobs = dict_apply(nobs, lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        # 在condition_mask中，true表示condition，false对应的是要被去噪的action部分，因此在计算loss的时候，需要将condition_mask取反，true对应要被去噪的action部分
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        # 在config文件中设置prediction_type，可以是epsilon或sample，epsilon表示直接预测噪音，sample表示预测样本，注意这里的样本应该是x0，而不是x_t-1
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')  # reduction='none'表示不对loss进行求和，返回的loss维度和输入的target一样
        loss = loss * loss_mask.type(loss.dtype)  # 只保留loss_mask中为True的部分
        loss = reduce(loss, 'b ... -> b (...)', 'mean')  # 将...维度进行平均，最终得到的loss维度为：(B,)
        loss = loss.mean()  # 对batch_size维度进行平均，得到最终的loss
        return loss
