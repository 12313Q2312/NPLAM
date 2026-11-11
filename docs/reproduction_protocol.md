# 复现协议

## 实验范围

| 配置 | 任务 | 样本与种子 | 训练预算 |
| --- | --- | --- | --- |
| `synthetic.yaml` | `mixed30`、`linear30`、`nonlinear30`、`dense30` | 每任务 1,600 行，10–19 | 400 epoch 上限，batch 256 |
| `highdim.yaml` | `sparse300` | 2,000 行，10–12 | 250 epoch 上限，batch 256 |
| `ablations.yaml` | `mixed30` | 1,600 行，10–12 | 400 epoch 上限，batch 256 |
| `real.yaml` | California、Superconductivity、Beijing | 每任务 4,096 行，10–12 | 200 epoch 上限，batch 256 |
| `smoke.yaml` | `mixed30` | 200 行，种子 0 | 10 epoch 上限，batch 64 |

四个 30 维任务分别是五个有效变量中线性与非线性混合、五个全部线性、五个全部非线性，以及 30 个全部有效（各 15 个线性与非线性）。300 维任务沿用混合任务的五个有效变量，加入 295 个无关变量。输入独立服从 U(-2.5, 2.5)，目标加标准正态噪声。

代码用 NumPy 的种子独立生成数据与划分索引，比例为 60/20/20。各候选共享同一数据划分。输入和目标的 min-max 参数只从训练集估计，验证和测试集沿用该变换且保留超出训练范围的值。训练使用确定性 CPU 运算和每进程一个 Torch 线程。只按验证集 MSE 选超参数、恢复最佳权重，随后计算测试指标。

## 对照与消融

MLP 为两个 25 单元的 ReLU 隐藏层。Lasso 使用同一缩放输入和验证集选 alpha。NAM 基线对每个变量使用独立的单隐藏层 ReLU 网络并相加，所有隐藏参数可训练。它是简化的 NAM 型对照，完整 NAM 论文还讨论专用单元和集成。[NAM 原论文](https://arxiv.org/abs/2004.13912)

各候选的搜索网格见 YAML。模型参数量及搜索耗时写入结果，模型之间的参数量、候选数和实际训练步数存在差异。

消融使用一组固定的基准超参数。分别移除门控 L1（`no_l1`、`no_l2`），固定某个门为 1 并移除对应惩罚（`no_fs`、`no_sd`），移除全部 L1（`no_reg`），或解冻随机隐藏映射（`train_hidden`）。`legacy` 同时使用旧实现激活、正则归约与指定的正则系数，作为实现对照单独解读。

## 数据来源与许可

| 数据 | 来源和处理 |
| --- | --- |
| California Housing | [StatLib 来源](https://lib.stat.cmu.edu/datasets/houses.zip) · [scikit-learn 固定归档说明](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_california_housing.py)。读取原始 8 个预测字段，保留房价原始美元单位与房间等原始计数。研究使用的原始归档仅本地缓存。 |
| Superconductivity | Hamidieh (2018)，[UCI 数据](https://doi.org/10.24432/C53P47)，CC BY 4.0。读取 `train.csv` 的 81 个预测字段，预测临界温度。 |
| Beijing Multi-Site Air Quality | Chen (2017)，[UCI 数据](https://doi.org/10.24432/C5RK5G)，CC BY 4.0。只读取官方 PRSA 站点表，按时间和站点排序，去除缺失行，将风向映射到 16 个方向编号，均匀抽取 4,096 行后按时间顺序切分。目标是同一时刻的 PM2.5 浓度，输入包括同一时刻 PM10 等观测。 |
| Boston Housing | 未纳入本项目的运行范围。需要单独确认数据来源、使用条件及涉及的历史变量。 |

真实数据归档及原始行不随仓库分发。缓存中记录下载地址、许可来源和 SHA-256。California 与 Superconductivity 每个种子无放回抽样，Beijing 保持同一时间划分并改变训练种子。

## 运行命令

先按照 [README](../README.md) 安装。以下命令在仓库目录运行，`python` 指向刚安装依赖的环境。

```text
python -m nplam.cli run --config configs/synthetic.yaml --output runs/synthetic-replay
python -m nplam.cli run --config configs/highdim.yaml --output runs/highdim-replay
python -m nplam.cli run --config configs/ablations.yaml --output runs/ablations-replay
python -m nplam.cli fetch california
python -m nplam.cli fetch superconductivity
python -m nplam.cli fetch beijing
python -m nplam.cli run --config configs/real.yaml --output runs/real-replay
```

也可以使用 `python scripts/benchmark.py runs/full-replay --workers 4`。它把任务分开运行，保存各任务的标准输出、调用命令与退出码。只需要离线合成实验时，运行前三条即可。`extended_300d.yaml` 提供更大计算预算的后续配置，已执行的高维结果来自 `highdim.yaml`。

中断后可用新的输出目录和 `--reuse-root`（批量）或 `--reuse-from`（单任务）承接已有结果。复用检查配置、核心方法源文件、依赖版本、数据指纹及预测文件的 MSE；不完整的运行重新计算。详细来源写入新清单，旧目录保留。单任务可用 `python -m nplam.cli summarize <运行目录>` 重新生成汇总表。

批量实验可用 `python scripts/benchmark.py runs/full-replay --summarize-only` 重新汇总。完成九个任务后，用 `python scripts/export_evidence.py runs/full-replay evidence/replay` 验证各任务覆盖了配置中的种子和模型，并从保存的预测重新计算 MSE，再导出紧凑的结果表、代表图与清单。

## 查看结果

每个目录包含 `manifest.json`、`results.csv`、`summary.csv` 和 `REPORT.md`。每个任务与种子有划分索引、预处理参数和数据指纹，每个模型有验证候选、测试预测、学习历史与 checkpoint。首个种子生成预测、门控和贡献图。

MSE、RMSE、MAE 使用原目标尺度，同时记录缩放后的 MSE 和 R²。特征选择报告 TP、FP、TN、FN、F1。CF 为正确判为线性或非线性的有效特征比例；UF 和 OF 分别为非线性误判线性、线性误判非线性的比例，三者与遗漏率的分母均为真实有效特征数。MLP、NAM 这里仅报告预测结果。Lasso 可识别入选变量，全部入选变量按线性计。真实数据没有已知的变量结构真值。均值和样本标准差按种子统计。
