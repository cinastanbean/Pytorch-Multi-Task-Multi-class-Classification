

# 1. Pytorch-Multi-Task-Multi-class-Classification

**MTMC-Pytorch:**
MTMC-Pytorch = Multi-Task Multi-Class Classification Project using Pytorch.

**目的：**
旨在搭建一个分类问题在Pytorch框架下的通解，批量解决单任务多分类问题、多任务多分类问题。

**使用：**
需要做的准备工作是将样本整理成如下格式按文件夹存放；
MTMC自动解析任务获取类别标签、自适应样本均衡、模型训练、模型评估等过程。

```
MLDataloader load MTMC dataset as following directory tree.
Make sur train-val directory tree keeps consistency.

data_root_path
├── task_A
│   ├── train
│   │   ├── class_1
│   │   ├── class_2
│   │   ├── class_3
│   │   └── class_4
│   └── val
│       ├── class_1
│       ├── class_2
│       ├── class_3
│       └── class_4
└── task_B
    ├── train
    │   ├── class_1
    │   ├── class_2
    │   └── class_3
    └── val
        ├── class_1
        ├── class_2
        └── class_3
```

**备注：**
1. 通用的，而不是对任意问题都是最优的；
2. 尚有诸多Tricks未集成；
3. 项目不再更新，止步于AutoML；

# 2. Pytorch Version Info

```
$  conda list | grep torch
pytorch                   0.4.1           py36_cuda0.0_cudnn0.0_1    pytorch
torchvision               0.2.1                    py36_1    pytorch
```

# 3. Train-Val Logs

日志文件保存为*.txt文件，使用Excel打开展示结果如下：

![](https://ws4.sinaimg.cn/large/006tNc79ly1fz1bikc8edj30ql0p8wid.jpg)



