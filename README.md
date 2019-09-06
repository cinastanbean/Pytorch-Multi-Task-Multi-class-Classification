

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
2. 目的是集成分类问题诸多训练Tricks；
3. 项目不再更新，止步于Gluon CV；（https://github.com/dmlc/gluon-cv，尽管他不是很完整。Ref: Bag of Tricks for Image Classification with Convolutional Neural Networks https://arxiv.org/abs/1812.01187v2）

# 2. Pytorch Version Info

```
$  conda list | grep torch
pytorch                   0.4.1           py36_cuda0.0_cudnn0.0_1    pytorch
torchvision               0.2.1                    py36_1    pytorch
```

# 3. Train-Val Logs


## 3.1 类目分类问题（Trainval_log_2018-07-09）

**Argmax运算前，网络输出值与准确率关系如下：**

![](https://ws4.sinaimg.cn/large/006tKfTcgy1ft6v36bcjtj30k40dfgoc.jpg)

**Softmax运算后，网络输出值与准确率关系如下：**

![](https://ws3.sinaimg.cn/large/006tKfTcgy1ft6xjm5m4yj30k40df0u5.jpg)

**类目判别在检测之后，另在Softmax后分布显示准确度收敛较好，所以类目部分直接输出Argmax对应的标签做为判定结果。**

## 3.2 印花分类问题

多类别值置信度值预测

**Argmax运算前，网络输出值与准确率关系如下：**

![](https://ws2.sinaimg.cn/large/006tKfTcgy1ft6x7qvvv8j30k40df76z.jpg)

**Softmax运算后，网络输出值与准确率关系如下：**

![](https://ws2.sinaimg.cn/large/006tKfTcgy1ft6x7zo20bj30k40df76y.jpg)

**鉴于印花问题存在以下问题：标签定义域关系相重叠，存在未定义印花种类。
搭建服务时，在服务内部进行置信度值映射：**

```
class_idx_dict = {
        "0" : "五角星",
        "1" : "人物",
        "2" : "几何",
        "3" : "动物鸟虫",
        "4" : "千鸟",
        "5" : "卡通",
        "6" : "复古",
        "7" : "大花",
        "8" : "字母数字汉字",
        "9" : "手绘",
        "10" : "斑马纹",
        "11" : "条纹",
        "12" : "格子",
        "13" : "植物风景",
        "14" : "波点",
        "15" : "渐变色",
        "16" : "爱心",
        "17" : "牛仔",
        "18" : "碎花",
        "19" : "纯色",
        "20" : "色块拼色",
        "21" : "豹纹蛇纹",
        "22" : "迷彩",
        "23" : "食物水果"}

pvals = [0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98]

ptmat_dict = {
    "五角星" : [1.30, 1.80, 2.30, 2.70, 3.00, 3.30, 3.50, 3.80, 4.00, 4.20, 4.40, 4.50, 4.70, 4.90, 5.00, 5.20, 5.30, 5.50, 5.60, 5.80, 5.90, 6.00, 6.20, 6.30, 6.50, 6.60, 6.70, 6.90, 7.00, 7.20, 7.30, 7.50, 7.60, 7.80, 7.90, 8.10, 8.30, 8.50, 8.70, 8.90, 9.10, 9.40, 9.70, 10.10, 11.20],
    ...
    ...
    "食物水果" : [2.40, 2.90, 3.30, 3.60, 3.90, 4.20, 4.50, 4.70, 4.90, 5.10, 5.30, 5.50, 5.70, 5.80, 6.00, 6.10, 6.30, 6.50, 6.60, 6.70, 6.90, 7.00, 7.20, 7.30, 7.50, 7.60, 7.70, 7.90, 8.00, 8.10, 8.30, 8.40, 8.60, 8.80, 8.90, 9.10, 9.30, 9.50, 9.70, 10.00, 10.30, 10.60, 11.00, 11.60, 12.70]
}
```

**不使用Softmax之后的值，而使用网络的输出值，可以得到两个置信度都比较高的分类判别信息。**
**例如下图人物卡通图案，会在“人物”&“卡通”两个类别下都有较高的置信度值：**

![](https://ws1.sinaimg.cn/large/006tNc79gy1ft4we0p3puj30l30g27c9.jpg)


**另，输出结果如果要用作半结构化信息时，采用归一化过的置信度值是欠妥的。**



## 3.3 日志文件保存为*.txt文件，使用Excel打开展示结果如下：

![](https://ws4.sinaimg.cn/large/006tNc79ly1fz1bikc8edj30ql0p8wid.jpg)



