# 论文与实现审计

审计日期：2026-11-08。论文 DOI：[10.1007/s11704-023-2662-3](https://doi.org/10.1007/s11704-023-2662-3)。对照源码固定为 [BestXuan/NPLAM@73690d3](https://github.com/BestXuan/NPLAM/tree/73690d3ad9ae8687221775d37c3b4edc478f95a8)。未发现根目录 LICENSE，故仅阅读审计，**没有复制、移植或重新许可其代码**。本项目从公式独立实现；不分发原代码、论文 PDF 或真实数据。

## 精确语义

|项目|论文|参考实现|本项目|
|---|---|---|---|
|特征门|实数 g1，同时进入两条路径|输入逐项乘 g1|相同；无 sigmoid、无截断|
|结构门|实数 g2，只乘非线性输入|`Dense_sd` 上路径乘 g2，下路径乘 1|相同；没有互补门|
|线性系数|w_tilde * g1|实际前向为 w_tilde * g1|正确导出有效系数|
|随机隐藏映射|v,b ~ N(0,1)，冻结|冻结|注册为 buffer，随 checkpoint 保存|
|激活|第3页明确 ReLU|`tf.nn.leaky_relu` 默认负半轴斜率0.2|paper=ReLU；legacy=LeakyReLU(0.2)|
|正则|λ1 sum abs(g1)+λ2 sum abs(g2)+λ3 sum abs(A)|前两项分别除 d，A 除 d*h|paper 求和；legacy 取均值|
|初始化|方法节规定 v,b 分布|g1=1，g2约0.7，w=1，A=0|沿用这些数值设计；框架随机数不等价|
|损失|均方误差|逐样本 MSE 后框架 reduction|显式样本均值|
|L1 在0处|子梯度设0|TF abs 默认|PyTorch abs 在0处梯度0，已测试|
|偏置|全局 β，隐藏 b 冻结|同左|同左|
|最佳模型|验证集最佳|保存 val_mse 最佳，另打印末轮结果|恢复最佳权重后才评估测试集|
|缩放|说明不够完整|先对全部 X/y min-max，再划分|先划分，**仅训练集**拟合 X/y 缩放器|

`Dense_sd` 的 `kernel_down=1` **不是需要改成 1-g2 的错误**：论文式(3)本来就没有互补门。真正不一致在 `model.py:get_g`，它报告的线性权重又乘了 `(1-g2)`，与实际前向不一致。当前导出的 `effective_linear_weights` 严格等于 g1*w。测试用独立标量公式核对，防止把解释输出误当架构。

λ 的等价换算：在相同前向和数据尺度下，paper λ1=legacy λ1/d、paper λ2=legacy λ2/d、paper λ3=legacy λ3/(d*h)。模式切换但不换算 λ 会改变优化问题。消融中的 legacy 是独立候选，不宣称其与 paper 实验只有一个变量不同。

## 重要边界

1. g1=0 时该特征的贡献成为常数，未必为0；冻结 b 与可训练 A 仍产生常数。解释图中心化，但训练前向保留原公式。
2. g2=0 时非线性分支成为常数，线性路径仍工作。非线性识别使用 abs(g1*g2)>阈值，与论文5.3节路径解释一致；同时保存原始 g2。
3. 未给门控加正性、范围、单调或概率约束。门控大小不具有概率解释。
4. 线性路径存在尺度不唯一：g1 缩小、w 放大可能保持相同预测，正则只惩罚 g1。没有通过额外 w 正则悄悄修补论文；因此结构恢复差不等于预测实现错误。
5. legacy 仅复现已审计的前向/正则语义；**未进行 TensorFlow 2.2 与 PyTorch 数值逐步对齐**。优化器 epsilon、随机流、shuffle 和浮点细节不同。
6. 原论文第20式 dense30 的第18个特征是 -exp(-x18)，参考生成器却是 +exp(-x18)。本项目默认使用论文负号，差异进入数据测试和文档。
7. 参考源默认800 epoch、batch100；本轮有界预算采用更短训练、早停、小网格。实际运行清单以 manifests 为准。

## 可核验证据

论文原文已在用户此前下载目录读取，17页；SHA256：`be16ed0c92477be98265573446a47ace29dd163ba3d39a68eae194f554db6a94`。第3页公式(2)-(4)已渲染并人工检查。第6-11页与附录用于核对实验。原文不进入版本库。

源码参考文件：[model.py](https://github.com/BestXuan/NPLAM/blob/73690d3ad9ae8687221775d37c3b4edc478f95a8/model.py)、[layer.py](https://github.com/BestXuan/NPLAM/blob/73690d3ad9ae8687221775d37c3b4edc478f95a8/layer.py)、[createdata.py](https://github.com/BestXuan/NPLAM/blob/73690d3ad9ae8687221775d37c3b4edc478f95a8/createdata.py)、[read_data.py](https://github.com/BestXuan/NPLAM/blob/73690d3ad9ae8687221775d37c3b4edc478f95a8/read_data.py)、[result.py](https://github.com/BestXuan/NPLAM/blob/73690d3ad9ae8687221775d37c3b4edc478f95a8/result.py)。
