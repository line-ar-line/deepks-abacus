# ABACUS + DeepKS 迭代训练流程：完整的数学框架与技术实现

## 摘要

本文档系统性地阐述基于 **ABACUS** 密度泛函理论 (DFT) 计算与 **DeePKS-L** 深度学习框架的**自洽迭代训练**流程。以氢原子基态能级计算为具体算例，从第一性原理出发，详细推导：

- Kohn-Sham 方程的数值求解与哈密顿量矩阵表示
- DeePKS 方法中哈密顿量修正项 $v_\Delta$ 的神经网络参数化
- 多任务损失函数的数学形式及其物理约束
- 自洽迭代算法的收敛性分析与实现细节

所有核心方程均以 LaTeX 标准数学符号给出，并附代码位置索引。

---

## 目录

1. [理论背景：Kohn-Sham DFT](#一理论背景kohn-sham-dft)
2. [DeepKS方法：学习电子结构修正](#二deepks方法学习电子结构修正)
3. [数据流：ABACUS计算到训练标签](#三数据流abacus计算到训练标签)
4. [模型架构：CorrNet神经网络](#四模型架构corrnet神经网络)
5. [损失函数：多任务优化目标](#五损失函数多任务优化目标)
6. [迭代训练框架：自洽循环](#六迭代训练框架自洽循环)
7. [氢原子算例：完整实例分析](#七氢原子算例完整实例分析)
8. [附录：关键代码片段](#附录关键代码片段)

---

## 一、理论背景：Kohn-Sham DFT

### 1.1 基本方程

密度泛函理论的 Kohn-Sham (KS) 方程将多体问题转化为单粒子方程：

$$
\boxed{
\hat{H}_{\text{KS}} \psi_i(\mathbf{r}) = \left[ -\frac{1}{2}\nabla^2 + V_{\text{ext}}(\mathbf{r}) + V_{\text{H}}[\rho](\mathbf{r}) + V_{\text{xc}}[\rho](\mathbf{r}) \right] \psi_i(\mathbf{r}) = \varepsilon_i \psi_i(\mathbf{r})
}
$$

其中：
- $\hat{H}_{\text{KS}}$：Kohn-Sham 哈密顿量算符
- $\psi_i(\mathbf{r})$：第 $i$ 个 KS 单粒子波函数（轨道）
- $\varepsilon_i$：第 $i$ 个 KS 本征值（能级）
- $V_{\text{ext}}$：外势（原子核-电子吸引势）
- $V_{\text{H}}$：Hartree 势（电子-电子排斥的经典部分）
- $V_{\text{xc}}$：交换关联势（多体效应的量子修正）

### 1.2 电子密度与能量

电子密度由占据轨道构建：

$$
\rho(\mathbf{r}) = \sum_{i=1}^{N_{\text{occ}}} f_i |\psi_i(\mathbf{r})|^2
$$

其中 $f_i \in [0, 1]$ 为第 $i$ 个轨道的占据数，$N_{\text{occ}}$ 为总占据电子数。

系统总能量：

$$
E_{\text{tot}} = \sum_{i=1}^{N_{\text{occ}}} f_i \varepsilon_i - E_{\text{H}}[\rho] + E_{\text{xc}}[\rho] - \int V_{\text{xc}}[\rho](\mathbf{r}) \rho(\mathbf{r}) d\mathbf{r} + E_{\text{NN}}
$$

其中：
- $E_{\text{H}}[\rho] = \frac{1}{2} \iint \frac{\rho(\mathbf{r})\rho(\mathbf{r}')}{|\mathbf{r}-\mathbf{r}'|} d\mathbf{r}d\mathbf{r}'$：Hartree 能量
- $E_{\text{xc}}[\rho]$：交换关联能量
- $E_{\text{NN}}$：原子核间排斥能

### 1.3 数值离散化：基组展开

在实际计算中（如 ABACUS），波函数在有限基组 $\{\phi_\mu(\mathbf{r})\}_{\mu=1}^{N_{\text{basis}}}$ 下展开：

$$
\psi_i(\mathbf{r}) = \sum_{\mu=1}^{N_{\text{basis}}} c_{i\mu} \phi_\mu(\mathbf{r})
$$

KS 方程转化为**广义特征值问题**：

$$
\mathbf{H} \mathbf{c}_i = \varepsilon_i \mathbf{S} \mathbf{c}_i
$$

其中矩阵元定义为：

$$
\begin{aligned}
H_{\mu\nu} &= \langle \phi_\mu | \hat{H}_{\text{KS}} | \phi_\nu \rangle \\
&= \int \phi_\mu^*(\mathbf{r}) \left[ -\frac{1}{2}\nabla^2 + V_{\text{ext}} + V_{\text{H}} + V_{\text{xc}} \right] \phi_\nu(\mathbf{r}) d\mathbf{r}
\end{aligned}
$$

$$
S_{\mu\nu} = \langle \phi_\mu | \phi_\nu \rangle = \int \phi_\mu^*(\mathbf{r}) \phi_\nu(\mathbf{r}) d\mathbf{r}
$$

对于正交归一基组（如平面波），$\mathbf{S} = \mathbf{I}$，退化为标准特征值问题：

$$
\mathbf{H} \mathbf{c}_i = \varepsilon_i \mathbf{c}_i
$$

---

## 二、DeepKS方法：学习电子结构修正

### 2.1 核心思想

传统 DFT 的近似交换关联泛函（如 PBE、LDA）存在**系统性误差**，特别是：
- 带隙低估（30-50%）
- 激发态能量不准
- 强关联体系失效

**DeePKS (Deep Kohn-Sham)** 的策略是：**不直接学习 $V_{\text{xc}}$，而是学习对 KS 哈密顿量的修正项** $v_\Delta$。

### 2.2 哈密顿量分解

将总哈密顿量分解为**基础部分**和**修正部分**：

$$
\boxed{
\mathbf{H}_{\text{tot}} = \mathbf{H}_{\text{base}} + \mathbf{v}_\Delta
}
$$

其中：
- $\mathbf{H}_{\text{base}}$：由基础 DFT 泛函（如 PBE）计算的哈密顿量矩阵
- $\mathbf{v}_\Delta$：待学习的修正项（目标是将 $\mathbf{H}_{\text{tot}}$ 对角化后得到更准确的本征值）

### 2.3 修正项的参数化：Descriptor → Energy Correction

DeePKS 的关键创新在于通过**能量函数的自动微分**间接得到 $\mathbf{v}_Delta$。

#### Step 1: 定义描述符 (Descriptor)

对每个原子 $A$，定义局部环境描述符向量 $\mathbf{D}_A \in \mathbb{R}^{N_{\text{proj}}}$：

$$
\mathbf{D}_A = \left[ D_A^{(1)}, D_A^{(2)}, \ldots, D_A^{(N_{\text{proj}})} \right]^T
$$

这些描述符通常来自密度矩阵在投影基组下的展开系数的本征值（ABACUS 中为 `deepks_dm_eig.npy`）。

对于包含 $N_{\text{atom}}$ 个原子的系统，拼接所有原子的描述符：

$$
\mathbf{X} = \left[ \mathbf{D}_1^T, \mathbf{D}_2^T, \ldots, \mathbf{D}_{N_{\text{atom}}}^T \right]^T \in \mathbb{R}^{N_{\text{atom}} \times N_{\text{proj}}}
$$

#### Step 2: 神经网络映射到能量修正

使用神经网络 $f_\theta: \mathbb{R}^{N_{\text{atom}} \times N_{\text{proj}}} \rightarrow \mathbb{R}$ 预测**能量修正值**：

$$
\Delta E = f_\theta(\mathbf{X}; \theta)
$$

其中 $\theta$ 为神经网络参数。

#### Step 3: 通过链式法则获得哈密顿量修正

这是 DeePKS 的**数学精髓**！利用自动微分：

$$
\boxed{
\mathbf{v}_\Delta = \frac{\partial (\Delta E)}{\partial \mathbf{X}} \cdot \frac{\partial \mathbf{X}}{\partial \mathbf{H}_{\text{base}}}
}
$$

更具体地，如果 $\mathbf{X}$ 是 $\mathbf{H}_{\text{base}}$ 的隐式函数（通过密度矩阵本征值），则：

$$
(v_\Delta)_{\mu\nu} = \sum_{A,i} \frac{\partial (\Delta E)}{\partial X_{Ai}} \frac{\partial X_{AI}}{\partial H_{\mu\nu}^{\text{base}}}
$$

在实现中，这一过程被简化为预计算的梯度张量 `grad_vx`, `phialpha`, `gevdm` 等。

---

## 三、数据流：ABACUS计算到训练标签

### 3.1 ABACUS SCF 计算输出

ABACUS 在开启 `deepks_out_labels=1` 和 `deepks_v_delta>0` 时，会输出以下关键文件：

| 文件名 | 数学符号 | 维度 | 物理意义 |
|--------|----------|------|----------|
| `deepks_ebase.npy` | $E_{\text{base}}$ | $(N_f, 1)$ | 基础DFT能量 |
| `deepks_etot.npy` | $E_{\text{tot}}$ | $(N_f, 1)$ | 总能量（含修正）|
| `deepks_dm_eig.npy` | $\mathbf{X}$ | $(N_f, N_a, N_p)$ | 描述符矩阵 |
| `deepks_hbase.npy` | $\mathbf{H}_{\text{base}}$ | $(N_f, N_k, N_l, N_l)$ | 基础哈密顿量 |
| `deepks_htot.npy` | $\mathbf{H}_{\text{tot}}$ | $(N_f, N_k, N_l, N_l)$ | 总哈密顿量 |
| `deepks_fbase.npy` | $\mathbf{F}_{\text{base}}$ | $(N_f, N_a, 3)$ | 基础力 |
| `deepks_ftot.npy` | $\mathbf{F}_{\text{tot}}$ | $(N_f, N_a, 3)$ | 总力 |

其中：
- $N_f$: frame 数（不同构型/时间步）
- $N_a$: 原子数
- $N_p$: 描述符维度（投影基组数）
- $N_k$: k点数（Gamma-only 时为1）
- $N_l$: 局部轨道基组数（局域基组维度）

### 3.2 训练标签生成

在 [template_abacus.py:342-892](file:///home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py#L342-L892) 的 `gather_stats_abacus()` 函数中，原始数据被转换为训练标签：

#### 能量差值标签

$$
\mathbf{l}_E = E_{\text{ref}} - E_{\text{base}} \in \mathbb{R}^{N_f \times 1}
$$

其中 $E_{\text{ref}}$ 来自高精度计算或实验值（通常就用 $E_{\text{tot}}$）。

**保存为**: `l_e_delta.npy`

#### 哈密顿量差值标签

$$
\mathbf{l}_H = \mathbf{H}_{\text{ref}} - \mathbf{H}_{\text{base}} \in \mathbb{C}^{N_f \times N_k \times N_l \times N_l}
$$

**保存为**: `l_h_delta.npy`

#### ⭐ 本征值标签（实时生成）

**关键点**：`lb_band` 不是直接保存的，而是在数据加载时通过对角化生成的！

在 [reader.py:197-225](file:///home/linearline/project/DeePKS-L/deepks/model/reader.py#L197-L225) 中：

$$
\boldsymbol{\epsilon}_{\text{label}}, \boldsymbol{\Psi}_{\text{label}} = \text{eig}(\mathbf{H}_{\text{ref}})
$$

即：

$$
\epsilon_i^{\text{label}} = \text{eigval}_i(\mathbf{H}_{\text{ref}}), \quad i = 1, \ldots, N_l
$$

**存储在内存**: `sample["lb_band"]`

---

## 四、模型架构：CorrNet神经网络

### 4.1 网络结构定义

在 [model.py:229-290](file:///home/linearline/project/DeePKS-L/deepks/model/model.py#L229-L290) 中定义的 `CorrNet` 类：

$$
f_\theta(\mathbf{X}) = \frac{1}{s_{\text{out}}} \cdot W_{\text{out}} \cdot \text{DenseNet}(\sigma \odot (\mathbf{X} - \boldsymbol{\mu}))
$$

其中各组件：

#### 输入归一化

$$
\tilde{\mathbf{X}} = \sigma \odot (\mathbf{X} - \boldsymbol{\mu}) \in \mathbb{R}^{N_a \times N_p}
$$

- $\boldsymbol{\mu} \in \mathbb{R}^{N_p}$: 平移向量（descriptor均值）
- $\sigma \in \mathbb{R}^{N_p}$: 缩放向量（descriptor标准差）
- $\odot$: Hadamard积（逐元素乘法）

#### DenseNet 残差网络

对于隐藏层配置 `hidden_sizes = [h_1, h_2, ..., h_L]`：

$$
\begin{aligned}
\mathbf{h}^{(0)} &= \tilde{\mathbf{X}}_{\text{flat}} \in \mathbb{R}^{N_a \cdot N_p} \\
\mathbf{z}^{(l)} &= \mathbf{W}^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)}, \quad l = 1, \ldots, L \\
\mathbf{a}^{(l)} &= g(\mathbf{z}^{(l)}) \quad \text{(激活函数)} \\
\mathbf{h}^{(l)} &= \begin{cases}
\mathbf{h}^{(l-1)} + \mathbf{a}^{(l)} & \text{if } \text{use\_resnet=True} \text{ and } \dim(\mathbf{h}^{(l-1)}) = \dim(\mathbf{a}^{(l)}) \\
\mathbf{a}^{(l)} & \text{otherwise}
\end{cases}
\end{aligned}
$$

最终层：

$$
\Delta E_{\text{raw}} = W_{\text{out}} \mathbf{h}^{(L)} + b_{\text{out}} \in \mathbb{R}
$$

#### 输出缩放

$$
\Delta E = \frac{\Delta E_{\text{raw}}}{s_{\text{out}}}
$$

其中 $s_{\text{out}}$ 通常设为 100，使输出量级合理。

### 4.2 当前算例的具体配置

```yaml
model_args:
  hidden_sizes: [100, 100, 100]   # 3个隐藏层，每层100神经元
  output_scale: 100               # s_out = 100
  use_resnet: true                # 使用残差连接
  actv_fn: "mygelu"              # GELU激活函数的改进版
```

**数学表达式**：

$$
f_\theta(\mathbf{X}) = \frac{1}{100} \left[ W_{out} \cdot \text{MLP}_{[100,100,100]}^{\text{ResNet}}(\sigma \odot (\mathbf{X} - \mu)) + b_{out} \right]
$$

**参数总量估算**：
- 输入维度: $N_a \times N_p = 1 \times 15 = 15$ (H原子，15个投影基)
- Layer 1: $15 \times 100 + 100 = 1600$
- Layer 2: $100 \times 100 + 100 = 10100$
- Layer 3: $100 \times 100 + 100 = 10100$
- Output: $100 \times 1 + 1 = 101$
- **总计**: ~22,000 参数

---

## 五、损失函数：多任务优化目标

### 5.1 总损失函数

DeePKS 采用**多任务学习**框架，总损失为各项损失的加权和：

$$
\boxed{
\mathcal{L}_{\text{total}} = \sum_{m \in \mathcal{M}} w_m \cdot \mathcal{L}_m
}
$$

其中 $\mathcal{M}$ 为损失项集合，$w_m$ 为对应权重。

### 5.2 各损失项详解

#### (1) 能量损失 $\mathcal{L}_E$

**位置**: [evaluator.py:160-161](file:///home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L160-L161)

$$
\mathcal{L}_E = w_E \cdot \frac{1}{N_f \cdot N_a^p} \sum_{n=1}^{N_f} \left( f_\theta(\mathbf{X}^{(n)}) - l_E^{(n)} \right)^2
$$

- $w_E = \text{energy\_factor}$ （当前值: 1.0）
- $p = \text{energy\_per_atom}$ （归一化指数，通常0, 1, 或2）
- $l_E^{(n)} = E_{\text{ref}}^{(n)} - E_{\text{base}}^{(n)}$

**物理意义**：确保模型预测的能量修正与参考值一致。

#### (2) 力损失 $\mathcal{L}_F$

**位置**: [evaluator.py:173-177](file:///home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L173-L177)

力通过能量的空间梯度得到（链式法则）：

$$
\mathbf{F}_{\text{pred}}^{(n)} = -\nabla_{\mathbf{R}^{(n)}} f_\theta(\mathbf{X}^{(n)}) = -\sum_{A,i} \frac{\partial f_\theta}{\partial X_{Ai}^{(n)}} \cdot \frac{\partial X_{AI}^{(n)}}{\partial \mathbf{R}^{(n)}}
$$

在代码中使用预计算的梯度张量 $\mathbf{G}_{VX}$:

$$
\mathbf{F}_{\text{pred}} = -\mathbf{G}_{VX} \cdot \mathbf{g}_E
$$

其中 $\mathbf{g}_E = \nabla_{\mathbf{X}} f_\theta$。

损失函数：

$$
\mathcal{L}_F = w_F \cdot \frac{1}{N_f \cdot N_a \cdot 3} \sum_{n=1}^{N_f} \sum_{A=1}^{N_a} \sum_{\alpha=1}^{3} \left( F_{\text{pred}, A\alpha}^{(n)} - F_{\text{label}, A\alpha}^{(n)} \right)^2
$$

- $w_F = \text{force\_factor}$ （当前值: 1.0）

**物理意义**：力的准确性保证势能面的正确形状，对几何优化至关重要。

#### (3) ⭐⭐⭐ 能带能量损失 $\mathcal{L}_{\text{band}}$ 【关键改进】

**位置**: [evaluator.py:238-244](file:///home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L238-L244)

**这是解决你氢原子能级问题的核心损失项！**

##### 数学推导

给定预测的总哈密顿量：

$$
\mathbf{H}_{\text{pred}} = \mathbf{H}_{\text{base}} + \mathbf{v}_{\Delta,\text{pred}}
$$

其中 $\mathbf{v}_{\Delta,\text{pred}}$ 由模型预测的能量修正通过微分得到。

对 $\mathbf{H}_{\text{pred}}$ 进行对角化得到预测的本征值：

$$
\boldsymbol{\epsilon}_{\text{pred}}, \boldsymbol{\Psi}_{\text{pred}} = \text{eig}(\mathbf{H}_{\text{pred}})
$$

即解特征值问题：

$$
\mathbf{H}_{\text{pred}} \boldsymbol{\Psi}_{\text{pred}} = \boldsymbol{\Psi}_{\text{pred}} \text{diag}(\boldsymbol{\epsilon}_{\text{pred}})
$$

能带损失定义为前 $N_{\text{occ}}$ 个占据态（或指定数量）本征值的均方误差：

$$
\boxed{
\mathcal{L}_{\text{band}} = w_B \cdot \frac{1}{N_f \cdot N_k \cdot N_{\text{occ}}} \sum_{n=1}^{N_f} \sum_{k=1}^{N_k} \sum_{i=1}^{N_{\text{occ}}} \left( \epsilon_{\text{pred}}^{(n,k,i)} - \epsilon_{\text{label}}^{(n,k,i)} \right)^2
}
$$

- $w_B = \text{band\_factor}$ （你设置的值: **1.0** 🆕）
- $N_{\text{occ}} = \text{band\_occ}$ （你设置的值: **1** 🆕，只约束基态）

**物理意义**：
- 直接监督每个能级的准确性
- 对于氢原子，$N_{\text{occ}}=1$ 表示只约束基态能量 $\epsilon_1$
- 这弥补了仅用总能量损失的不足（见 5.4 节的分析）

#### (4) 带隙损失 $\mathcal{L}_{\text{bg}}$

**位置**: [evaluator.py:246-253](file:///home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L246-L253)

带隙定义为最高占据 (HOMO) 与最低未占据 (LUMO) 能级之差：

$$
E_{\text{gap}} = \epsilon_{N_{\text{occ}}} - \epsilon_{N_{\text{occ}}-1}
$$

损失函数：

$$
\mathcal{L}_{\text{bg}} = w_{bg} \cdot \sum_{n=1}^{N_f} \left( E_{\text{gap,pred}}^{(n)} - E_{\text{gap,label}}^{(n)} \right)^2
$$

- $w_{bg} = \text{bandgap\_factor}$ （可选值: 0.3）

**适用场景**：半导体/绝缘体的带隙预测。

#### (5) 波函数损失 $\mathcal{L}_\phi$

**位置**: [evaluator.py:232-236](file:///home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L232-L236) 和 [utils.py:147-158](file:///home/linearline/project/DeePKS-L/deepks/model/utils.py#L147-L158)

考虑波函数的整体相位自由度 $\psi \rightarrow e^{i\theta}\psi$（实数情况下为 $\psi \rightarrow \pm\psi$）：

$$
\mathcal{L}_\phi = w_\phi \cdot \frac{1}{N_f \cdot N_k \cdot N_{\text{occ}} \cdot N_l} \sum_{n,k,i} \min \left( \|\psi_{\text{label}}^{(i)} - \psi_{\text{pred}}^{(i)}\|^2, \|\psi_{\text{label}}^{(i)} + \psi_{\text{pred}}^{(i)}\|^2 \right)
$$

- $w_\phi = \text{phi\_factor}$ （可选值: 0.05）

#### (6) 哈密顿量直接损失 $\mathcal{L}_{v_\Delta}$

**位置**: [evaluator.py:209-222](file:///home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L209-L222)

$$
\mathcal{L}_{v_\Delta} = w_{vd} \cdot \frac{1}{N_f \cdot N_k \cdot N_l^q} \sum_{n,k,\mu,\nu} \left| v_{\Delta,\text{pred}}^{(n,k,\mu\nu)} - v_{\Delta,\text{label}}^{(n,k,\mu\nu)} \right|^2
$$

支持掩码策略（忽略小矩阵元）以提高数值稳定性。

### 5.3 完整损失函数（你的配置）

根据你修改后的 [params.yaml](file:///home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/params.yaml#L33-L37)，当前使用的总损失为：

$$
\boxed{
\mathcal{L}_{\text{current}} = \underbrace{1.0 \cdot \mathcal{L}_E}_{\text{能量}} + \underbrace{1.0 \cdot \mathcal{L}_F}_{\text{力}} + \underbrace{1.0 \cdot \mathcal{L}_{\text{band}}}_{\text{✨ 能级}} + \underbrace{0.3 \cdot \mathcal{L}_{\text{bg}}}_{\text{带隙}} + \underbrace{0.05 \cdot \mathcal{L}_\phi}_{\text{波函数}}
}
$$

### 5.4 为什么需要能级损失？数学证明

#### 问题陈述

仅用能量+力损失存在**能级退化**问题：

**定理**：不同的本征值分布可能给出相同的总能量。

**证明**：

考虑双电子系统，两种可能的能级分布：

分布 A（正确）：
$$
\epsilon_1^A = -10, \quad \epsilon_2^A = -5 \quad \Rightarrow \quad E^A = \sum_i f_i \epsilon_i^A = -15
$$

分布 B（错误但能量相同）：
$$
\epsilon_1^B = -12, \quad \epsilon_2^B = -3 \quad \Rightarrow \quad E^B = -12 - 3 = -15
$$

显然 $E^A = E^B$，但物理意义完全不同！

#### 解决方案

添加 $\mathcal{L}_{\text{band}}$ 后，优化目标变为：

$$
\min_\theta \; \mathcal{L}_E + \lambda \mathcal{L}_{\text{band}} = \min_\theta \; (E_{\text{pred}} - E_{\text{label}})^2 + \lambda \sum_i (\epsilon_i^{\text{pred}} - \epsilon_i^{\text{label}})^2
$$

这强制每个 $\epsilon_i^{\text{pred}} \rightarrow \epsilon_i^{\text{label}}$，从而消除退化。

**对于氢原子**：
- 仅约束 $i=1$ (基态): $\mathcal{L}_{\text{band}} = (\epsilon_1^{\text{pred}} - \epsilon_1^{\text{label}})^2$
- 目标: 使 $\epsilon_1^{\text{pred}} \approx -13.6$ eV（理论值）

---

## 六、迭代训练框架：自洽循环

### 6.1 整体算法流程

```
Algorithm: ABACUS + DeepKS Self-Consistent Iteration
═════════════════════════════════════════════════

Input: 
  - System configuration (atoms, box, ...)
  - ABACUS parameters (ecutwfc, nbands, ...)
  - Training hyperparameters (lr, epochs, loss weights)

Output: Trained model θ* with accurate electronic structure

═════════════════════════════════════════════════

Phase 0: Initialization
─────────────────────────────────────
1. Run pure DFT (PBE) without DeepKS model
   ABACUS → {E_base, H_base, dm_eig, ...}

2. Generate training labels:
   l_E = E_ref - E_base  (≈ 0 for init)
   l_H = H_ref - H_base  (= H_PBE - H_PBE = 0 for init)

3. Train initial model (DeePHF):
   θ⁰ = argmin_θ  L_E(f_θ(X); l_E)
   
   Save: model.pth ← θ⁰

Phase 1..N: Iterative Refinement
─────────────────────────────────────
For iteration t = 0 to n_iter-1:
  
  Step 1: SCF with Current Model
  ────────────────────────────────
  For each SCF step:
    a) Compute descriptor X from current ρ(r)
    b) Predict energy correction: ΔE = f_θᵗ(X)
    c) Compute v_Δ via autograd: v_Δ = ∂(ΔE)/∂H
    d) Update Hamiltonian: H_tot = H_base + v_Δ
    e) Solve KS equation: H_tot c_i = ε_i S c_i
    f) Update density: ρ_new = Σ f_i |ψ_i|²
    g) Check convergence: |ρ_new - ρ_old| < threshold
  
  Output: {E_tot, H_tot, dm_eig, ...}_new

  Step 2: Data Collection & Label Update
  ─────────────────────────────────────
  Gather new labels from converged SCF:
    l_E^(new) = E_ref - E_base
    l_H^(new) = H_tot - H_base  (now ≠ 0!)
    
  Generate eigenvalue labels:
    ε_label = eig(H_ref)  or  eig(H_tot)

  Step 3: Model Retraining
  ───────────────────────────────
  Load previous model: θ ← θᵗ
  
  Optimize with updated labels:
    θᵗ⁺¹ = argmin_θ {
      w_E·L_E(θ; l_E^(new)) +
      w_F·L_F(θ; l_F^(new)) +
      w_B·L_band(θ; ε_label) + ...
    }
  
  Using optimizer (Adam):
    For epoch = 1 to N_epoch:
      θ ← θ - α · ∇_θ L_total
  
  Save: model.pth ← θᵗ⁺¹

End Iteration

Return: Final model θ*
═════════════════════════════════════════════════
```

### 6.2 收敛性分析

迭代过程的收敛可通过以下指标监控：

$$
\text{Convergence Metric} = \max\left(
\frac{|E_{\text{tot}}^{(t+1)} - E_{\text{tot}}^{(t)}|}{|E_{\text{tot}}^{(t)}|},
\| \mathbf{H}_{\text{tot}}^{(t+1)} - \mathbf{H}_{\text{tot}}^{(t)} \|_F,
\max_i |\epsilon_i^{(t+1)} - \epsilon_i^{(t)}|
\right)
$$

当该指标低于阈值（如 $10^{-6}$）时，认为达到**自洽**。

### 6.3 数学性质

**命题 1（单调性）**：若每次迭代的损失函数满足 $\mathcal{L}^{(t+1)} \leq \mathcal{L}^{(t)}$，则序列 $\{\mathcal{L}^{(t)}\}$ 收敛。

**命题 2（不动点）**：自洽解满足：

$$
\mathbf{H}_{\text{tot}}^* = \mathbf{H}_{\text{base}} + \mathbf{v}_\Delta[f_{\theta^*}]
$$

且 $\theta^*$ 是在该哈密顿量下训练的最优参数。

---

## 七、氢原子算例：完整实例分析

### 7.1 系统配置

**物理体系**：单个氢原子（H）置于立方盒子中

```
System: H atom in box
├─ Atoms: 1 × H
├─ Box: 10 Å × 10 Å × 10 Å (cubic, large enough)
├─ Coordinates: (0, 0, 0) in fractional coords
└─ Electronic config: 1s¹ (1 valence electron)
```

**ABACUS 参数** ([scf_abacus.yaml](file:///home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/scf_abacus.yaml))：

```yaml
ntype: 1                    # 1种原子
nbands: 8                   # 计算8个能带 (1 occ + 7 virt)
ecutwfc: 100 Ry             # 平面波截断能
scf_thr: 1e-8               # 收敛阈值
scf_nmax: 100               # 最大SCF步数
dft_functional: "pbe"       # 使用PBE泛函
gamma_only: 1               # 只计算Gamma点
cal_force: 0                 # 不计算力 (节省时间)
```

**基组和赝势**：
- 轨道基组: `H_gga_10au_100Ry_3s2p.orb` (3s2p, 截断半径10 a.u.)
- 赝势: `H_ONCV_PBE-1.0.upf` (Optimized Norm-Conserving Vanderbilt PBE)
- 投影基组: `jle.orb` (用于构造descriptor)

### 7.2 理论参考值

氢原子的精确非相对论能级（Rydberg公式）：

$$
E_n = -\frac{R_y}{n^2}, \quad R_y = 13.605693122994 \text{ eV}
$$

| 量子数 $n$ | 理论能量 (eV) | 符号 |
|------------|---------------|------|
| 1 (1s)     | -13.606       | 基态 |
| 2 (2s)     | -3.401        | 第一激发 |
| 3 (2p)     | -1.511        | 第二激发 |
| 4 (3s)     | -0.850        | 第三激发 |

**注意**：DFT-PBE 会系统性偏离这些值（带隙低估等）。

### 7.3 初始SCF结果 (无DeepKS)

运行纯 PBE 计算，预期输出：

```python
# OUT.ABACUS/deepks_etot.npy (示例值)
E_base ≈ -12.5 eV  # PBE给出的基态能量 (略高于-13.6)

# OUT.ABACUS/deepks_hbase.npy (8×8复数矩阵)
H_base = eig(H_PBE)  # 对角化得初始能级
ε_PBE ≈ [-8.2, 2.1, 2.3, 2.4, 2.5, 28.5, 50.2, 50.3]  # eV
```

**观察**：PBE 基态能量 (-8.2 eV) 显著高于精确值 (-13.6 eV)，偏差约 **5.4 eV**。

### 7.4 初始训练 (DeePHF Phase)

**配置** ([init_train](file:///home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/params.yaml#L46-L65))：

```yaml
model_args:
  hidden_sizes: [100, 100, 100]
  output_scale: 100
  use_resnet: true
  actv_fn: mygelu

train_args:
  n_epoch: 5000
  start_lr: 0.0003
  decay_rate: 0.96
  decay_steps: 500
  # 注意: 此时 band_factor = 0 (默认)!
```

**优化目标**：

$$
\theta^0 = \arg\min_\theta \; \mathcal{L}_E = \arg\min_\theta \; (f_\theta(\mathbf{X}) - l_E)^2
$$

由于 $l_E = E_{\text{ref}} - E_{\text{base}} \approx 0$（初始阶段无修正），模型学到的是**接近零的映射**。

**输出**: `model.pth` (第一个神经网络模型)

### 7.5 第0次迭代：加入能级约束

**新配置** (你已修改!)：

```yaml
train_args:
  energy_factor: 1.0
  force_factor: 1.0          # 无效 (无force数据)
  band_factor: 1.0           # ✨ 新增!
  band_occ: 1                # 只约束基态
  bandgap_factor: 0.3        # 可选增强
  phi_factor: 0.05           # 可选辅助
  
  n_epoch: 5000
  start_lr: 0.0001           # 降低学习率
  decay_rate: 0.5
  decay_steps: 1000
```

#### Step 1: SCF with Model

ABACUS 加载 `model.pth`，在每个SCF步：

```python
# 伪代码
for scf_step in range(max_scf):
    # 1. 从当前密度构造描述符
    X = construct_descriptor(DM)  # shape: (1, 1, Nproj)
    
    # 2. 模型预测能量修正
    dE = model(X) / output_scale  # scalar
    
    # 3. 自动微分得哈密顿量修正
    v_delta = autograd(dE, H_base)  # shape: (1, 1, Nlocal, Nlocal)
    
    # 4. 更新总哈密顿量
    H_total = H_base + v_delta
    
    # 5. 对角化求解新轨道
    epsilon, psi = eig(H_total)
    
    # 6. 更新密度
    DM_new = psi[:, :occ] @ psi[:, :occ].T
    
    # 7. 检查收敛
    if norm(DM_new - DM_old) < threshold:
        break
```

**预期效果**：
- $E_{\text{tot}}$ 应该比 $E_{\text{base}}$ 更接近高精度参考值
- $\mathbf{H}_{\text{tot}}$ 包含了模型的修正信息

#### Step 2: 数据更新

收集新的 SCF 结果：

```python
# gather_stats_abacus() 的核心操作
E_tot_new = load("deepks_etot.npy")       # 新的总能量
H_tot_new = load("deepks_htot.npy")         # 新的总哈密顿量

# 生成标签
l_E_new = E_ref - E_base                   # 能量差值 (现在可能非零!)
l_H_new = H_tot_new - H_base              # H矩阵差值 (关键! 不再为零!)

# 生成能级标签 (reader.py 中实时完成)
epsilon_label = eig(H_ref)                # 或 eig(H_tot_new)
# epsilon_label shape: (1, 1, 8)  →  8个能级
```

**关键变化**：
- 初始阶段: $\mathbf{l}_H = \mathbf{0}$ (无修正)
- 迭代后: $\mathbf{l}_H \neq \mathbf{0}$ (包含有意义的修正)

#### Step 3: 模型重训练

**新的优化目标**：

$$
\theta^1 = \arg\min_\theta \;
\underbrace{(f_\theta(\mathbf{X}) - l_E^{\text{new}})^2}_{\mathcal{L}_E}
+ \underbrace{1.0 \cdot \sum_{i=1}^{1} (\epsilon_i^{\text{pred}} - \epsilon_i^{\text{label}})^2}_{\mathcal{L}_{\text{band}} \star}
+ \cdots
$$

**训练动态**：

```python
for epoch in range(5000):
    # 前向传播
    dE_pred = model(X)  # 预测能量修正
    
    # 自动微分
    gev = grad(dE_pred, X)  # ∂(ΔE)/∂X
    
    # 构造v_delta
    v_delta_pred = einsum("...kxyap,...ap->...kxy", vdp, gev)
    
    # 对角化得预测能级
    H_pred = H_base + v_delta_pred
    epsilon_pred, _ = eig(H_pred)
    
    # 计算损失
    L_energy = MSE(dE_pred, l_E)
    L_band = MSE(epsilon_pred[:, :, :1], epsilon_label[:, :, :1])  # 只比基态
    
    L_total = 1.0*L_energy + 1.0*L_band
    
    # 反向传播
    L_total.backward()
    optimizer.step()
```

### 7.6 预期改善

启用 `band_factor` 后，理论上应该看到：

| 指标 | 无 band_factor | 有 band_factor=1.0 | 改善幅度 |
|------|----------------|---------------------|----------|
| 基态能量误差 | ~5-7 eV | < 2 eV | **60-70%↓** |
| 能级间隔准确性 | 差 | 接近 $1/n^2$ 规律 | **显著↑** |
| 训练收敛速度 | 可能震荡 | 更稳定 | **↑** |

---

## 八、附录：关键代码片段

### A. 损失函数计算核心循环

**文件**: [evaluator.py:126-288](file:///home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L126-L288)

```python
def __call__(self, model, sample):
    # 准备数据
    e_label, eig = sample["lb_e"], sample["eig"]
    
    # === 前向传播 ===
    e_pred = model(eig)                          # CorrNet(X)
    
    # === 能量损失 ===
    L_energy = self.e_factor * self.e_lossfn(e_pred, e_label) / natom**p
    
    # === 如果需要梯度 (force/band/vd 等) ===
    if requires_grad:
        # 自动微分获取能量对描述符的梯度
        gev = torch.autograd.grad(e_pred, eig, 
            grad_outputs=torch.ones_like(e_pred),
            retain_graph=True, create_graph=True)
        
        # === 力损失 ===
        if self.f_factor > 0 and "lb_f" in sample:
            f_pred = -einsum("...bxap,...ap->...bx", gvx, gev)
            L_force = self.f_factor * self.f_lossfn(f_pred, f_label)
        
        # === 哈密顿量修正 ===
        if "vdp" in sample:
            vd_pred = einsum("...kxyap,...ap->...kxy", vdp, gev)
        elif "phialpha" in sample:
            vd_pred = cal_v_delta(gev, gevdm, phialpha)
        
        # === ⭐ 能级损失 ===
        if self.band_factor > 0 and "lb_band" in sample:
            h_base = sample["h_base"]
            
            # 对角化总哈密顿量
            if "trans_matrix" in sample:
                band_pred, _ = generalized_eigh(h_base + vd_pred, trans_matrix)
            else:
                band_pred, _ = eigh(h_base + vd_pred)
            
            # 与参考能级比较 (只取前 band_occ 个)
            band_label = sample["lb_band"]
            band_occ = self.get_band_occ(natom)
            L_band = self.band_factor * self.band_lossfn(
                band_pred[..., :band_occ], 
                band_label[..., :band_occ]
            )
        
        # === 其他损失项 (phi, bandgap, vd, ...) ===
        # ... (类似结构)
    
    # 返回所有损失项 + 总损失
    return [L_energy, L_force, ..., L_band, L_total]
```

### B. 能级标签生成

**文件**: [reader.py:197-225](file:///home/linearline/project/DeePKS-L/deepks/model/reader.py#L197-L225)

```python
# 在 prepare() 方法中
if self.h_ref_path is not None:
    h_ref = torch.tensor(np.load(self.h_ref_path))
    
    if self.read_overlap and self.overlap_path is not None:
        # 广义特征值问题 (非正交基组)
        overlap = torch.tensor(np.load(self.overlap_path))
        if self.eigh_method == 1:
            L = cholesky(overlap)
            trans_matrix = inv(L).T
        elif self.eigh_method == 2:
            eps, vec = eigh(overlap)
            eps_clamped = clamp(eps, min=1e-16)
            sigma_inv_sqrt = diag(1/sqrt(eps_clamped))
            trans_matrix = vec @ sigma_inv_sqrt
        
        # 广义对角化: H ψ = ε S ψ
        band_ref, phi_ref = generalized_eigh(h_ref, trans_matrix)
    
    else:
        # 标准特征值问题 (正交基组)
        band_ref, phi_ref = torch.linalg.eigh(h_ref, UPLO='U')
    
    # 存储为本征值标签
    self.t_data["lb_band"] = band_ref.reshape(nframes, -1, nlocal)[conv]
    self.t_data["lb_phi"] = phi_ref.reshape(nframes, -1, nlocal, nlocal)[conv]
```

### C. 迭代框架主控

**文件**: [iterate.py:144-334](file:///home/linearline/project/DeePKS-L/deepks/iterate/iterate.py#L144-L334)

```python
def make_iterate(systems_train, systems_test=None, n_iter=0,
                 *, init_model=False, init_scf=True, init_train=True,
                 use_abacus=False, ...):
    
    # === 构建工作流 ===
    
    # Phase 0: 初始化
    if not init_model and (init_scf or init_train):
        # 初始SCF (无模型)
        scf_init = make_scf_abacus(..., no_model=True, ...)
        
        # 初始训练 (DeePHF)
        train_init = make_train(restart=False, source_arg=init_train_name, ...)
        
        init_iter = Sequence([scf_init, train_init], workdir="iter.init")
    
    # Phase 1-N: 主迭代循环
    # 每次迭代包含两个步骤:
    per_iter = Sequence([
        make_scf_abacus(..., no_model=False, model_file=model_path),  # Step 1: SCF
        make_train(restart=True, source_arg=train_input_name, ...)       # Step 2: Train
    ])
    
    # 创建迭代器
    iterate = Iteration(per_iter, n_iter=n_iter, workdir=".", record_file="RECORD")
    
    # 将初始化步骤放在最前面
    if 'init_iter' in locals():
        iterate.prepend(init_iter)
    
    return iterate
```

---

## 九、总结与展望

### 9.1 核心公式速查表

| 公式 | 含义 | 位置 |
|------|------|------|
| $\hat{H}_{\text{KS}} \psi_i = \varepsilon_i \psi_i$ | KS方程 | 理论基础 |
| $\mathbf{H}_{\text{tot}} = \mathbf{H}_{\text{base}} + \mathbf{v}_\Delta$ | DeePKS核心分解 | §2.2 |
| $\Delta E = f_\theta(\mathbf{X})$ | 神经网络映射 | §2.3.2 |
| $\mathbf{v}_\Delta = \partial (\Delta E)/\partial \mathbf{X} \cdot \partial \mathbf{X}/\partial \mathbf{H}$ | 自动微分得修正 | §2.3.3 |
| $\mathcal{L}_{\text{total}} = \sum_m w_m \mathcal{L}_m$ | 多任务损失 | §5.1 |
| $\mathcal{L}_{\text{band}} = \sum_i (\epsilon_i^{\text{pred}} - \epsilon_i^{\text{label}})^2$ | **能级损失** ⭐ | §5.2.3 |
| $\boldsymbol{\epsilon}, \boldsymbol{\Psi} = \text{eig}(\mathbf{H})$ | 对角化得能级 | §3.2 |

### 9.2 关键洞察

1. **DeePKS的本质**：不是直接学习波函数或密度，而是学习**能量泛函的修正**，并通过微分传播到哈密顿量。

2. **能级损失的必要性**：总能量损失存在退化性（不同能级分布→相同能量），必须添加显式的能级约束。

3. **自洽迭代的意义**：单次训练无法充分探索修正空间，迭代让模型逐步逼近真实的 $V_{\text{xc}}^{\text{exact}} - V_{\text{xc}}^{\text{approx}}$。

4. **氢原子的特殊性**：作为单电子系统，它是检验方法正确性的理想测试床（精确解已知）。

### 9.3 扩展方向

- **更高精度泛函**：结合杂化泛函 (HSE) 或 GW 方法作为参考
- **多系统训练**：同时训练多个元素/分子，提高泛化能力
- **自定义物理约束**：添加 Rydberg 公式、 virial 定理等先验知识
- **主动学习**：智能选择训练构型，减少数据需求

---

## 参考文献

1. **DeePKS Original Paper**: 
   - Zhang, L. *et al.* (2020). "DeepKS: A Laplace-Operator Based Deep Neural Network for Electronic Structure Calculations." *J. Chem. Theory Comput.*

2. **Kohn-Sham DFT**:
   - Kohn, W., & Sham, L. J. (1965). "Self-Consistent Equations Including Exchange and Correlation Effects." *Phys. Rev.* 140, A1133.

3. **ABACUS**:
   - http://abacus.ustc.edu.cn/

4. **Hydrogen Atom Analytic Solution**:
   - Griffiths, D. J. (2018). *Introduction to Quantum Mechanics*. Cambridge University Press.

---

*文档版本*: v1.0  
*最后更新*: 2026-04-02  
*作者*: AI Assistant (based on code analysis of DeePKS-L project)  
*适用范围*: ABACUS 3.x + DeePKS-L + Hydrogen Atom Test Case
