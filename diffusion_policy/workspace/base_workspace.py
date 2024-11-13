from typing import Optional
import os
import pathlib
import hydra
import copy
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import dill
import torch
import threading


class BaseWorkspace:
    include_keys = tuple()
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir: Optional[str]=None):
        self.cfg = cfg  # 将传给workspace的配置文件保存在类对象内部的cfg中
        self._output_dir = output_dir
        self._saving_thread = None

    # @property装饰器的作用是将一个方法变成一个属性调用，可以像访问属性一样调用这个方法，允许在访问属性值时执行一些操作
    @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def run(self):
        """
        Create any resource shouldn't be serialized as local variables
        """
        pass

    def save_checkpoint(self, path=None, tag='latest',  # tag指定检查点标签，默认为latest
            exclude_keys=None,  # 用于指定不需要保存的键
            include_keys=None,  # 用于指定需要保存的键
            use_thread=True):  # 用于指定是否使用线程异步保存
        if path is None:
            # joinpath()方法用于在后面添加路径，且不需要考虑路径分隔符
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')  # path最终为.ckpt模型文件
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:  # 如果没有指定不需要保存的键，则使用默认的exclude_keys
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        # path.parent是获取path的父目录，mkdir创建一个文件夹，parents=True表示如果父目录不存在则创建父目录，exist_ok=True表示如果文件夹已经存在则不会报错
        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()  # 所有希望保存的，带不是state_dict的对象的，会保存在pickles中
        } 

        # self.__dict__是一个类的属性，代表实例的属性字典，以字典的形式存储了实例的所有属性及其对应的值
        for key, value in self.__dict__.items():
            # hasattr()函数用于判断对象是否包含对应的属性，属性用字符串来表示
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):  # 如果value有state_dict和load_state_dict方法，则保存在state_dicts中
                # modules, optimizers and samplers etc
                if key not in exclude_keys:  # 对于有state_dict和load_state_dict方法的对象，如果不在exclude_keys中，则全都保存在state_dicts中
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())  # 如果要用多线程保存，则需要将state_dict拷贝到cpu上，避免多线程保存时出现问题
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:  # 所有希望保存的，带不是state_dict的对象的，会保存在pickles中
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:  # 如果使用线程异步保存，则创建一个线程来保存
            self._saving_thread = threading.Thread(
                # open('wb')代表以二进制写的方式打开文件
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))  # pickle_module=dill表示使用dill模块来序列化
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    def get_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys:
                self.__dict__[key].load_state_dict(value, **kwargs)
        for key in include_keys:
            if key in payload['pickles']:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
    
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, **kwargs)  # open('rb')代表以二进制读的方式打开文件
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    # @classmethod装饰器的作用是将一个方法定义为类方法，类方法的第一个参数cls，表示类本身，而不是实例，执行这个类方法时，不需要创建实例
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)


def _copy_to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().to('cpu')
    elif isinstance(x, dict):
        result = dict()
        for k, v in x.items():
            result[k] = _copy_to_cpu(v)
        return result
    elif isinstance(x, list):
        return [_copy_to_cpu(k) for k in x]
    else:
        return copy.deepcopy(x)
