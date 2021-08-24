## 1 结果展示

### 1.1 生成图片
| Epoch0                                                       | Epoch20                                                      | Epoch100                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ![](https://ai-studio-static-online.cdn.bcebos.com/967972ee668b402b8ed7d2bc9dbcf3de53a58721e63541f697786448c4c6a71d)|  ![](https://ai-studio-static-online.cdn.bcebos.com/e264f6fd61cc41f2acaa1ca5c33de5208dc9884055374537b14d7505cf33e263)| ![](https://ai-studio-static-online.cdn.bcebos.com/66ce25e920a543bf84d44f597bceb5243a74b6979c4e45f3a4480cd86bfd9ff0)|


### 1.2 loss
![](https://ai-studio-static-online.cdn.bcebos.com/6b025dc85477444893d0a52224a27992858e9ab3d0614889844b5ba3445e7db4)

图中，D_loss 为判别器的loss

G_loss 为编码器的loss

recon_loss 为图片的loss，此处选二元交叉熵函数（BCEloss）

recon loss = BCE(X_{sample}, X_{decoder})

如图所示，D_loss 与G_loss在epoch为50轮左右达到稳定。而recon_loss则逐渐减小，在20-30轮稳定。

### 1.3 Likelihood

#### 1.3.1 Parzen 窗长σ在验证集上交叉验证曲线

原文中提到对于窗长σ的选取需要通过交叉验证实现。本文使用1000个数据样本的验证集进行选择，从而得到一个符合最大似然准则下的窗长。

![](https://ai-studio-static-online.cdn.bcebos.com/fc1cc33731c64ee2a5a385001f85d9e52bae016c769d4d72aa53f90919985216)

根据少量样本验证集的结果，选择使得negative log likelihood最大的σ值作为窗长

#### 1.3.2 复现结果
|      | MNIST (10K)                                                  |
| ---- | ------------------------------------------------------------ |
| 原文 | 340 ± 2                                                   |
| 复现 | 345 ± 1.9432                                              |
|      | ![](https://ai-studio-static-online.cdn.bcebos.com/297396923f824dedb8a9bf7cd78532dd267bc8fa4f364d6893c1e61138344c0c) |

## 2、训练与测试方式

- 数据生成

```
python data_maker.py
```

- 训练

```
python train.py
```

- 测试log_likelihood

```
python eval.py
```

## 3、实现细节说明

- 遵循原文原则，Encoder采用两层全连接层，Decoder采用两层全连接层，Discriminator采用两层全连接层。由于原文并未公布代码及相关详细参数，因此笔者在此处加入了Dropout层，以及按照原文附录中所述加入了re-parametrization trick ，即将生成的隐变量z重新参数化为高斯分布。
- 高斯先验分布为8维，方差为5，神经元数为1200. 网络结构与原文一致

## 4、快速评估
由于github大小限制，模型保存在aistudio项目中的`./model/model199.pkl`，为第199轮输出结果。可以快速对模型进行评估，但需先运行`datamaker.py`产生分割后的数据集。
[链接](https://aistudio.baidu.com/studio/project/partial/verify/2301660/d1b6c79020f64b5b81dcd38ca23ebe60)

log文件见`./logs/2021-8-24-12-40/*`，重新运行会在`./logs/`内产生新的输出
