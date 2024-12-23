"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# use line-buffering for both stdout and stderr
# 这两行代码是修改了标准输出和标准错误的缓冲模式，使得它们变为行缓冲模式（line-buffering），这意味着每输出一行，都会立即刷新
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra  # hydra库用于管理复杂的配置，允许通过配置文件和命令行动态指定参数
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace  # 这是主要的功能实现代码部分

# allows arbitrary python code execution in configs using the ${eval:''} resolver
# 这行代码是注册一个解释器，名为"eval"，关联到python的eval函数，这样在配置文件中就可以使用${eval:''}方式去执行python代码执行运算
OmegaConf.register_new_resolver("eval", eval, replace=True)

"""
@hydra.main代表装饰器指定了一个函数main,获取了参数后会传给main函数
在程序执行时, hydra会自动从指定路径加载配置文件, 并将配置文件转化为cfg对象, 传给指定的main函数
针对@hydra的额外说明: Hydra 支持在命令行中通过参数覆盖配置文件中的内容, 如使用python my_app.py db.user=root db.pass=1234修改配置文件中的db.user和db.pass的值
@hydra装饰器支持多个参数:常见的有--config-path、--config-name等。config_path用于指定配置文件所在的目录, config_name用于指定要加载的配置文件名, 否则会加载目录下的config.yaml文件
@hydra装饰器后面需要直接跟一个函数, 就像所有装饰器一样
@hydra装饰器最终传给main函数的是一个OmegaConf对象, 该对象是一个字典, 可以通过字典的方式访问配置文件中的内容
"""
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))  # 指定配置文件的路径
)

def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)  # 解析配置文件，在配置文件中可能会使用${}语法来引用其他配置项或环境变量，如created_at: ${now:%Y-%m-%d}来获取当前时间，这里就是解析这些语法

    cls = hydra.utils.get_class(cfg._target_)  # 将cfg._target_指定的类名字的字符串转换为可以调用的类名字，并赋值给cls，之后可以直接用cls创建对象
    # 创建workspace对象
    workspace: BaseWorkspace = cls(cfg)  # 根据上一行cfg._target_指定的类名创建一个对象，传入参数cfg
    workspace.run()  # 调用对象的run方法，开始运行

if __name__ == "__main__":
    main()
