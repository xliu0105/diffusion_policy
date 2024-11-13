from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

# 这个模型类其实整个是一个resnet网络，用这个网络+上下采样可以实现unet网络
class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,  # condition的维度
            kernel_size=3,  # 卷积核的大小
            n_groups=8,  # 在Conv1dBlock模块中，组归一化的大小
            cond_predict_scale=False):  # 是否根据condition得到scale和bias，然后对输出进行缩放和偏移，这是FiLM的一种实现；如果为False，则直接将condition编码后的信息加到输出上
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels  # 默认情况下，cond_channels为out_channels
        if cond_predict_scale:  # 注意这里，如果cond_predict_scale为True，则cond_channels为out_channels * 2
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),  # 比较复杂的激活函数，可以用来提升模型的非线性表达能力
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),  # 使用einops库的Rearrange函数，将输入的维度从'batch t'转换为'batch t 1'
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ], 这里的in_channels就是一个action的维度, 比如位置action就包括xyz和rpy. 文中对姿态用了其他的表示方法, 如轴角
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)  # 对输入特征x进行初步卷积操作
        embed = self.cond_encoder(cond)  # 处理cond信息，将cond信息编码为embed
        if self.cond_predict_scale:  # 根据cond_predict_scale的值决定如何处理embed
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)  # reshape embed维度，注意在init函数中已经将cond_channels设置为out_channels * 2
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias  # 如果cond_predict_scale为True，则out = scale * out + bias，其中scale和bias是embed的两个部分，embed是由cond编码得到的
        else:  # 如果cond_predict_scale为False，则直接将embed加到out上
            out = out + embed
        out = self.blocks[1](out)  # 对out进行第二次卷积操作
        out = out + self.residual_conv(x)  # 残差连接
        return out


class ConditionalUnet1D(nn.Module):  # 1D的conditional unet网络架构
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,  # 对timesteps进行编码的维度
        down_dims=[256,512,1024],
        kernel_size=3,  # 卷积核的大小，应该是1维卷积吧
        n_groups=8,  # 分组归一化的组数，在Conv1dBlock中的归一化操作
        cond_predict_scale=False 
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim  # 对timesteps进行编码的维度
        diffusion_step_encoder = nn.Sequential(  # 对diffusion的步数step进行正弦位置编码
            SinusoidalPosEmb(dsed),  # 将输入的timesteps进行正弦位置编码，编码出的维度为dsed
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed  # condition的一部分是对timesteps的编码
        if global_cond_dim is not None:  # 如果有全局condition，直接加到cond_dim上
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))  # 将两个列表的对应元素对应起来，形成一个元组列表，如果all_dims=[1,2,3]，则in_out=[(1,2),(2,3)]

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        # 下采样后的中间模块
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        # 下采样模块
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        # 上采样模块
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(  # 补充：为什么上采样这里第一个要是dim_out*2：因为在上采样层中，输入的特征来自下采样路径的特征和上采样路径的上一层特征拼起来的结果
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)  # 对timesteps进行sinusoidal位置编码

        if global_cond is not None:  # 如果有global condition，将全局condition加到global_feature后面
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        # 如果有local condition，则将local condition编码后储存在h_local中
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):  # Unet的下采样模块
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:  # 如果是第一个down模块，并且有local condition，则将local condition加到x上
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:  # Unet的中间模块
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):  # Unet的上采样模块
            x = torch.cat((x, h.pop()), dim=1)  # 这里就是Unet上采样的时候，会将下采样路径的同一层的特征和上采样路径的上一层特征拼接起来
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:  # 如果是最后一个up模块，并且有local condition，则将local condition加到x上
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)  # 最后的卷积操作

        x = einops.rearrange(x, 'b t h -> b h t')  # 将输出的维度从'b t h'转换为'b h t'
        return x

