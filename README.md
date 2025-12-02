<div align="center">
  <img src="szdx.jpg" width="120" alt="Suzhou University Logo"/>
  <h1>2024中国高校计算机大赛——大数据挑战赛<br>全国二等奖（亚军）方案</h1>
  <h3>Team: 元胞自动机 (Cellular Automaton)</h3>
  
  <p>
    <a href="https://www.python.org/">
      <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
    </a>
    <a href="https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg" alt="PyTorch">
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/Rank-2%2F2626-gold.svg" alt="Rank">
    </a>
    <a href="#">
      <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
    </a>
  </p>
</div>

## 📖 简介 (Introduction)

本项目为 **2024中国高校计算机大赛——大数据挑战赛（Big Data Challenge 2024）** 的复现代码库。我们的团队 **“元胞自动机”** 在 1777 支参赛队伍中脱颖而出，最终获得 **全国总决赛亚军（第2名）**。

本赛题的核心挑战在于**位置信息脱敏**条件下的**全球到区域（Global-to-Local）** 域泛化预测。我们需要利用全球 3850 个站点的历史气象数据（训练集），对中国区域站点（测试集）未来 72 小时的气温和风速进行精准预测。

> **核心突破点**：提升模型的时空域外泛化能力（OOD Generalization）。我们通过隐式气候聚类构建验证集，并使用物理感知的特征工程与鲁棒损失函数（Huber/MAE）替代传统的 MSE，大幅提升了模型预测振幅的稳定性。

## 🏆 成绩回顾 (Leaderboard)

我们的方案在比赛各阶段均保持了稳健的性能提升：

| 阶段 | 榜单 | 排名 | 备注 |
| :--- | :---: | :---: | :--- |
| **初赛** | A榜 | 26 | 初步探索 |
| **初赛** | B榜 | 14 | 优化特征 |
| **复赛** | A榜 | 11 | 引入 iTransformer |
| **复赛** | B榜 | 4 | 聚类验证策略生效 |
| **决赛** | **Final** | **2** | **全国一等奖 (Runner-up)** |

## 💡 核心方法 (Methodology)

我们的解决方案涵盖了数据挖掘、验证策略、模型架构与损失函数优化四个维度：

### 1. 隐式气候聚类验证 (Implicit Climate Clustering Validation)
针对训练集（全球）与测试集（中国）分布不一致的问题，传统的随机 K-Fold 验证失效。
- 我们提取站点的**统计指纹**（均值、方差、波动率等）。
- 使用 **K-Means** 将全球站点划分为 13 个隐式气候域。
- 采用 **“留一簇交叉验证 (Leave-One-Cluster-Out)”**，模拟跨域迁移场景，确保线下分数与线上表现高度一致 。

### 2. 模型架构: iTransformer + LSTM
- **Encoder**: 采用 **iTransformer** (Inverted Transformer) ，将整条时间序列视为 Token，显式建模风速、温度、气压等多变量间的**动力学耦合**。
- **Decoder**: 引入单层 **LSTM** ，利用其递归归纳偏置（Inductive Bias），解决 Transformer 生成长序列时的局部震荡问题，保证预测曲线的时间连续性。

### 3. 物理感知特征工程 (Physics-Aware Feature Engineering)
- **缺失值处理**：采用前向填充（Forward Fill）保护日变化周期 。
- **物理特征**：引入热力学与动力学方程，构建**热通量 (Heat Flux)** 和 **风冷指数 (WCI)** 等特征 ，弥补地理位置缺失带来的信息损失。

### 4. 鲁棒损失函数 (Robust Loss Function)
- 将传统的 MSE 替换为 **Huber Loss** 和 **MAE** ，增强模型对极端天气（大风、骤温）的敏感度，解决预测趋于平滑的问题。
- 引入**线性递增的时间步加权**，抑制 72 小时长时预测中的累积误差。

## 🛠️ 环境依赖 (Requirements)

请确保安装了以下依赖库：

```bash
pip install torch numpy pandas scikit-learn
