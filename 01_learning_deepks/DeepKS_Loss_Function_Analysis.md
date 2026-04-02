# DeepKS 损失函数分析与优化建议

## 1. 当前计算结果分析

### 1.1 氢原子能级计算结果

从 `eig_occ.txt` 文件得到的氢原子能级（单位：eV）：

| 能级序号 | 能量值 (eV) | 占据数 |
|---------|------------|--------|
| 1       | -6.335     | 1.000  |
| 2       | 1.901      | 0.000  |
| 3       | 2.135      | 0.000  |
| 4       | 2.136      | 0.000  |
| 5       | 2.141      | 0.000  |
| 6       | 28.394     | 0.000  |
| 7       | 50.277     | 0.000  |
| 8       | 50.281     | 0.000  |

**理论参考值**：
- 氢原子基态能量（精确值）：$E_1 = -13.6$ eV
- 第一激发态：$E_2 = -3.4$ eV
- 第二激发态：$E_3 = -1.51$ eV

**问题分析**：
- 计算得到的基态能量为 -6.335 eV，与理论值 -13.6 eV 存在显著偏差
- 即使忽略平移量（absolute energy reference），能级间隔也不准确
- 这表明当前损失函数可能无法充分约束电子结构的准确性

---

## 2. DeepKS 损失函数架构

### 2.1 代码位置

核心损失函数实现位于以下文件：

- **主训练流程**: `/home/linearline/project/DeePKS-L/deepks/model/train.py`
- **评估器与损失计算**: `/home/linearline/project/DeePKS-L/deepks/model/evaluator.py`
- **损失函数工具函数**: `/home/linearline/project/DeePKS-L/deepks/model/utils.py`

### 2.2 总体损失函数形式

DeepKS 的总损失函数是一个多任务学习的加权组合：

$$
\mathcal{L}_{\text{total}} = \sum_i w_i \cdot \mathcal{L}_i
$$

其中 $w_i$ 是各项损失的权重因子，$\mathcal{L}_i$ 是不同物理量的损失项。

### 2.3 各项损失详解

#### 2.3.1 基础 L2 损失函数 (`make_loss` 函数)

位置: [utils.py:58-77](/home/linearline/project/DeePKS-L/deepks/model/utils.py#L58-L77)

```python
def make_loss(cap=None, shrink=None, reduction="mean"):
    def loss_fn(input, target):
        diff = target - input
        if shrink and shrink > 0:
            diff = F.softshrink(diff, shrink)
        sqdf = torch.abs(diff)**2  # 使用abs避免复数问题
        if cap and cap > 0:  # SmoothL2 loss
            abdf = diff.abs()
            sqdf = torch.where(abdf < cap, sqdf, cap * (2*abdf - cap))
        if reduction == "mean":
            return sqdf.mean()
        # ... 其他reduction选项
    return loss_fn
```

数学表达式：

$$
\mathcal{L}_{\text{MSE}}(\hat{y}, y) = \frac{1}{N} \sum_{i=1}^{N} |\hat{y}_i - y_i|^2
$$

支持的可选参数：
- `cap`: SmoothL1 损失的阈值，对大误差进行线性惩罚
- `shrink`: 软阈值收缩

#### 2.3.2 能量损失 ($\mathcal{L}_{\text{energy}}$)

位置: [evaluator.py:160-161](/home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L160-L161)

```python
tot_loss = tot_loss + self.e_factor * self.e_lossfn(e_pred, e_label) / (natom**self.energy_per_atom)
```

$$
\mathcal{L}_{\text{energy}} = w_E \cdot \frac{1}{N_{\text{atom}}^p} \sum_{n=1}^{N_{\text{frame}}} |E_{\text{pred}}^{(n)} - E_{\text{label}}^{(n)}|^2
$$

其中：
- $w_E = \text{energy\_factor}$ （默认值：1.0）
- $p = \text{energy\_per_atom}$ （可选 0, 1, 2，用于归一化）
- $E_{\text{pred}}$ 是模型预测的总能量
- $E_{\text{label}}$ 是 DFT 计算的参考能量

#### 2.3.3 力损失 ($\mathcal{L}_{\text{force}}$)

位置: [evaluator.py:173-177](/home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L173-L177)

```python
if self.f_factor > 0 and "lb_f" in sample:
    f_label, gvx = sample["lb_f"], sample["gvx"]
    f_pred = - torch.einsum("...bxap,...ap->...bx", gvx, gev)
    tot_loss = tot_loss + self.f_factor * self.f_lossfn(f_pred, f_label)
```

力的预测通过自动微分得到：

$$
\mathbf{F}_{\text{pred}} = -\nabla_{\mathbf{R}} E_{\text{pred}} = -\frac{\partial E_{\text{pred}}}{\partial \mathbf{X}} \cdot \frac{\partial \mathbf{X}}{\partial \mathbf{R}}
$$

其中使用了 Einstein 求和约定来计算链式法则。

力损失：

$$
\mathcal{L}_{\text{force}} = w_F \cdot \frac{1}{N_{\text{atom}} \times 3} \sum_{n,i,\alpha} |F_{\text{pred}}^{(n,i,\alpha)} - F_{\text{label}}^{(n,i,\alpha)}|^2
$$

当前配置使用 `force_factor=1`。

#### 2.3.4 应力损失 ($\mathcal{L}_{\text{stress}}$)

位置: [evaluator.py:179-183](/home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L179-L183)

$$
\mathcal{L}_{\text{stress}} = w_S \cdot \frac{1}{9} \sum_{n,ij} |\sigma_{\text{pred}}^{(n,ij)} - \sigma_{\text{label}}^{(n,ij)}|^2
$$

#### 2.3.5 哈密顿量修正损失 ($\mathcal{L}_{v_\Delta}$)

位置: [evaluator.py:209-222](/home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L209-L222)

这是 DeepKS 的核心损失项，直接学习 Kohn-Sham 哈密顿量的修正项 $v_\Delta$：

$$
\mathcal{L}_{v_\Delta} = w_{v\Delta} \cdot \frac{1}{N_{\text{local}}^q} \sum_{n,k,\mu\nu} |v_{\Delta,\text{pred}}^{(n,k,\mu\nu)} - v_{\Delta,\text{label}}^{(n,k,\mu\nu)}|^2
$$

其中 $q$ 可选 1 或 2（由 `vd_divide_by_nlocal` 控制）。

支持两种掩码策略：
1. **基于 S 和 H 矩阵幅度的掩码** ([utils.py:160-188](/home/linearline/project/DeePKS-L/deepks/model/utils.py#L160-L188))
2. **基于带宽度的掩码** ([utils.py:190-216](/home/linearline/project/DeePKS-L/deepks/model/utils.py#L190-L216))

#### 2.3.6 波函数损失 ($\mathcal{L}_{\phi}$)

位置: [evaluator.py:232-236](/home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L232-L236) 和 [utils.py:147-158](/home/linearline/project/DeePKS-L/deepks/model/utils.py#L147-L158)

```python
def cal_phi_loss(phi_pred, phi_label, phi_occ):
    occ_phi_pred = phi_pred[...,:phi_occ].clone()
    occ_phi_label = phi_label[...,:phi_occ].clone()
    loss_1 = ((occ_phi_label - occ_phi_pred)**2).mean(-2)
    loss_2 = ((occ_phi_label - (-1)*occ_phi_pred)**2).mean(-2)
    loss = torch.stack([loss_1, loss_2], dim=-1)
    loss = loss.min(dim=-1)[0]  # 取最小值（考虑波函数符号自由度）
    loss = loss.mean()
    return loss
```

数学表达式：

$$
\mathcal{L}_{\phi} = w_\phi \cdot \frac{1}{N_{\text{occ}}} \sum_{n,k,i} \min\left( \|\psi_{\text{label}}^{(i)} - \psi_{\text{pred}}^{(i)}\|^2, \|\psi_{\text{label}}^{(i)} + \psi_{\text{pred}}^{(i)}\|^2 \right)
$$

**关键特性**：考虑了波函数的整体符号自由度（gauge freedom）$\psi \rightarrow -\psi$。

#### 2.3.7 能带能量损失 ($\mathcal{L}_{\text{band}}$)

位置: [evaluator.py:238-244](/home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L238-L244)

$$
\mathcal{L}_{\text{band}} = w_{\text{band}} \cdot \frac{1}{N_{\text{occ}}} \sum_{n,k,i=1}^{N_{\text{occ}}} |\epsilon_{\text{pred}}^{(n,k,i)} - \epsilon_{\text{label}}^{(n,k,i)}|^2
$$

这是**本征值（能级）的直接监督信号**！

#### 2.3.8 带隙损失 ($\mathcal{L}_{\text{bandgap}}$)

位置: [evaluator.py:246-253](/home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L246-L253) 和 [utils.py:218-219](/home/linearline/project/DeePKS-L/deepks/model/utils.py#L218-L219)

$$
\mathcal{L}_{\text{bandgap}} = w_{\text{bg}} \cdot \sum_n |E_{\text{gap,pred}}^{(n)} - E_{\text{gap,label}}^{(n)}|^2
$$

其中带隙定义为：

$$
E_{\text{gap}} = \epsilon_{N_{\text{occ}}} - \epsilon_{N_{\text{occ}}-1}
$$

#### 2.3.9 密度矩阵损失 ($\mathcal{L}_{\rho}$)

位置: [evaluator.py:255-264](/home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L255-L264) 和 [utils.py:132-144](/home/linearline/project/DeePKS-L/deepks/model/utils.py#L132-L144)

$$
\mathcal{L}_{\rho} = w_\rho \cdot N_{\text{local}} \cdot \|\rho_{\text{pred}} - \rho_{\text{label}}\|_F^2
$$

密度矩阵通过占据轨道构建：

$$
\rho_{\mu\nu} = \sum_{i=1}^{N_{\text{occ}}} \psi_i^\dagger(\mu) \psi_i(\nu)
$$

#### 2.3.10 波函数对齐损失 ($\mathcal{L}_{\phi_{\text{align}}}$)

位置: [evaluator.py:267-280](/home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L267-L280)

$$
\mathcal{L}_{\phi_{\text{align}}} = w_{\phi_a} \cdot \| \Psi_{\text{occ}}^\dagger H_{\text{tot,pred}} \Psi_{\text{occ}} - \text{diag}(\boldsymbol{\epsilon}_{\text{occ,label}}) \|_F^2
$$

这确保了预测的哈密顿量在占据子空间中的表示与真实本征值一致。

---

## 3. 当前训练配置分析

从 `train_input.yaml` 文件可知当前配置：

```yaml
train_args:
  decay_rate: 0.5
  decay_steps: 1000
  display_epoch: 100
  force_factor: 1          # ✅ 启用力损失
  n_epoch: 5000
  start_lr: 0.0001
  # energy_factor: 1.0     # 默认启用（未显式声明）
  # band_factor: 0         # ❌ 未启用
  # phi_factor: 0          # ❌ 未启用
  # orbital_factor: 0      # ❌ 未启用
  # bandgap_factor: 0      # ❌ 未启用
```

**当前使用的损失函数**：

$$
\mathcal{L}_{\text{current}} = \underbrace{1.0 \cdot \mathcal{L}_{\text{energy}}}_{\text{总能量}} + \underbrace{1.0 \cdot \mathcal{L}_{\text{force}}}_{\text{原子力}}
$$

---

## 4. 问题根源分析

### 4.1 为什么仅用能量+力无法准确预测能级？

虽然总能量是本征值的加权和：

$$
E_{\text{total}} = \sum_{i=1}^{N_{\text{occ}}} f_i \epsilon_i + E_{\text{dc}}
$$

其中 $f_i$ 是占据数，$E_{\text{dc}}$ 是双计数项，但存在以下问题：

1. **能量退化性**：不同的本征值分布可能给出相同的总能量
2. **缺少显式约束**：没有直接监督单个能级的准确性
3. **梯度传播路径长**：能量 → 描述符 → 网络 → 本征值的梯度路径较长，可能导致优化困难

### 4.2 数学证明

考虑一个简单情况：双电子系统，两个可能的能级分布：

**分布 A**（正确）：
$$
\epsilon_1 = -10, \quad \epsilon_2 = -5 \quad \Rightarrow \quad E_A = -15
$$

**分布 B**（错误但能量相同）：
$$
\epsilon_1' = -12, \quad \epsilon_2' = -3 \quad \Rightarrow \quad E_B = -15
$$

两者总能量相同，但物理意义完全不同！这就是为什么需要**额外的能级约束**。

---

## 5. 优化建议：引入能级能量损失

### 5.1 方案一：使用现有的 `band_factor` 参数 ⭐⭐⭐⭐⭐

**最简单的方案**：DeepKS 已经实现了 `band_factor`，只需在训练配置中启用！

#### 修改配置文件 `train_input.yaml`

```yaml
train_args:
  decay_rate: 0.5
  decay_steps: 1000
  display_epoch: 100
  force_factor: 1
  band_factor: 1.0           # 🆕 启用能带能量损失
  band_occ: 1                # 🆕 对于氢原子，只约束第一个占据态
  n_epoch: 5000
  start_lr: 0.0001
```

**新的损失函数**：

$$
\mathcal{L}_{\text{new}} = \underbrace{w_E \mathcal{L}_{\text{energy}}}_{\text{总能量}} + \underbrace{w_F \mathcal{L}_{\text{force}}}_{\text{原子力}} + \underbrace{w_B \mathcal{L}_{\text{band}}}_{\text{✨ 能级能量}}
$$

其中能带损失具体为：

$$
\mathcal{L}_{\text{band}} = \frac{1}{N_{\text{frame}} \cdot N_k \cdot N_{\text{occ}}} \sum_{n,k,i=1}^{N_{\text{occ}}} (\epsilon_{\text{pred}}^{(n,k,i)} - \epsilon_{\text{label}}^{(n,k,i)})^2
$$

**优点**：
- ✅ 无需修改源代码
- ✅ 直接监督每个能级的准确性
- ✅ 对于原子/分子系统特别有效
- ✅ 实现简单，只需修改 YAML 配置

**注意事项**：
- 需要确保训练数据包含 `lb_band` 标签（即 DFT 的本征值）
- 需要通过 `extra_label: true` 来生成额外标签（当前已启用）

### 5.2 方案二：使用 `bandgap_factor` 约束带隙 ⭐⭐⭐⭐

对于半导体或绝缘体系统，可以额外约束带隙：

```yaml
train_args:
  force_factor: 1
  band_factor: 1.0
  bandgap_factor: 0.5        # 🆕 约束HOMO-LUMO带隙
  bandgap_occ: 1             # 🆕 占据态数量
```

**物理意义**：确保最高占据分子轨道（HOMO）和最低未占据分子轨道（LUMO）之间的间隙正确。

### 5.3 方案三：使用 `phi_factor` 约束波函数 ⭐⭐⭐

波函数的准确性直接影响能级：

```yaml
train_args:
  force_factor: 1
  band_factor: 1.0
  phi_factor: 0.1            # 🆕 波函数损失（权重较小）
  phi_occ: 1                 # 🆕 只约束占据轨道
```

**优势**：波函数约束可以从根本上改善电子结构，因为：
- 正确的波函数 → 正确的电荷密度 → 正确的势场 → 正确的本征值

### 5.4 方案四：组合策略（推荐）⭐⭐⭐⭐⭐

**最优配置建议**：

```yaml
train_args:
  # 基础损失
  energy_factor: 1.0         # 总能量（保持）
  force_factor: 1.0          # 原子力（保持）
  
  # 🆕 电子结构约束
  band_factor: 1.0           # 能级能量（核心改进）
  band_occ: 1                # 氢原子只约束1个占据态
  
  # 可选增强
  bandgap_factor: 0.3        # 带隙约束（防止能级交叉错误）
  phi_factor: 0.05           # 波函数辅助约束
  
  # 训练参数调整
  n_epoch: 8000              # 可能需要更多迭代
  start_lr: 0.00005          # 降低学习率以稳定训练
  decay_rate: 0.8
  decay_steps: 1000
```

**完整损失函数**：

$$
\boxed{
\begin{aligned}
\mathcal{L}_{\text{optimal}} =&\; w_E \mathcal{L}_{\text{energy}} + w_F \mathcal{L}_{\text{force}} \\
&+ w_B \mathcal{L}_{\text{band}} + w_{bg} \mathcal{L}_{\text{bandgap}} + w_\phi \mathcal{L}_{\phi}
\end{aligned}
}
$$

---

## 6. 高级优化方案（需修改源代码）

如果上述方案仍不满足需求，可以考虑以下高级改进（需要修改 evaluator.py）：

### 6.1 加权能级损失

对于不同能级赋予不同重要性：

$$
\mathcal{L}_{\text{weighted-band}} = \sum_{i=1}^{N_{\text{states}}} w_i \cdot (\epsilon_i^{\text{pred}} - \epsilon_i^{\text{label}})^2
$$

例如，对于氢原子：
- 基态 ($i=1$)：$w_1 = 10.0$（最重要）
- 激发态 ($i>1$)：$w_i = 1.0$

### 6.2 相对能级约束

约束能级间隔而非绝对值：

$$
\mathcal{L}_{\text{relative}} = \sum_{i<j} \left[ (\epsilon_j - \epsilon_i)^{\text{pred}} - (\epsilon_j - \epsilon_i)^{\text{label}} \right]^2
$$

这对于消除绝对能量平移误差特别有效。

### 6.3 物理一致性正则化

添加量子力学约束作为正则化项：

$$
\mathcal{L}_{\text{phys-reg}} = \lambda_1 \underbrace{\left| \sum_i f_i - N_e \right|^2}_{\text{电子数守恒}} + \lambda_2 \underbrace{\|\rho_{\text{pred}} - \rho_{\text{label}}\|^2}_{\text{电荷密度匹配}}
$$

---

## 7. 实施步骤指南

### 步骤 1：验证数据完整性

确保训练数据包含所需的标签：

```bash
# 检查是否有 lb_band 数据
python -c "
import numpy as np
data = np.load('your_training_data/lb_band.npy', allow_pickle=True).item()
print('Band labels shape:', data.shape)
print('Sample eigenvalues:', data[0])
"
```

### 步骤 2：修改训练配置

编辑 `train_input.yaml`，添加 `band_factor` 参数。

### 步骤 3：重新训练模型

```bash
cd /home/linearline/project/00_hydrogen_abacus/01_H_deepks
deepks train train_input.yaml
```

### 步骤 4：验证结果

重新运行 SCF 计算，检查 `eig_occ.txt` 是否改善：

```bash
# 期望结果：基态能量接近 -13.6 eV（忽略平移后）
# 能级间隔应该更接近理论值
```

---

## 8. 预期效果

### 8.1 定性改善

启用 `band_factor` 后预期看到：

✅ **基态能量更准确**：接近理论值 -13.6 eV  
✅ **能级间隔合理**：激发态间距符合 $1/n^2$ 规律  
✅ **物理一致性增强**：波函数、密度矩阵等性质同步改善  

### 8.2 量化指标

| 指标 | 当前值 | 目标值 | 改善幅度 |
|------|--------|--------|----------|
| 基态能量 (eV) | -6.335 | ≈ -13.6* | ~115% |
| 第一激发态 (eV) | 1.901 | ≈ -3.4* | 显著改善 |

*注：实际数值取决于能量零点的定义，关键是相对能级间隔应准确。

---

## 9. 关键代码引用索引

| 功能 | 文件 | 行号 |
|------|------|------|
| 主训练循环 | [train.py](/home/linearline/project/DeePKS-L/deepks/model/train.py) | L18-L193 |
| Evaluator 类初始化 | [evaluator.py](/home/linearline/project/DeePKS-L/deepks/model/evaluator.py) | L13-L124 |
| 总损失计算 | [evaluator.py](/home/linearline/project/DeePKS-L/deepks/model/evaluator.py) | L126-L288 |
| make_loss 工厂函数 | [utils.py](/home/linearline/project/DeePKS-L/deepks/model/utils.py) | L58-L77 |
| 波函数损失 | [utils.py](/home/linearline/project/DeePKS-L/deepks/model/utils.py) | L147-L158 |
| 密度矩阵计算 | [utils.py](/home/linearline/project/DeePKS-L/deepks/model/utils.py) | L132-144 |
| v_delta 掩码损失 | [utils.py](/home/linearline/project/DeePKS-L/deepks/model/utils.py) | L160-216 |
| 安全特征分解 | [utils.py](/home/linearline/project/DeePKS-L/deepks/model/utils.py) | L232-327 |

---

## 10. 总结与推荐

### 核心发现

1. **当前损失函数过于简单**：仅使用能量+力，缺乏对电子结构的直接约束
2. **DeepKS 已具备完善的基础设施**：`band_factor`, `phi_factor`, `bandgap_factor` 等参数已经实现
3. **问题可快速解决**：无需修改源代码，仅需调整配置文件

### 行动计划（按优先级排序）

🔴 **立即执行**：
- 在 `train_input.yaml` 中添加 `band_factor: 1.0` 和 `band_occ: 1`
- 重新训练模型并验证

🟡 **短期优化**（如果效果不佳）：
- 组合使用 `band_factor` + `bandgap_factor`
- 调整各损失项的权重比例
- 增加训练轮次

🟢 **长期研究**（如需更高精度）：
- 实现加权能级损失（需修改源码）
- 添加相对能级约束
- 开发针对原子系统的专用损失函数

### 最终建议

**对于氢原子能级计算问题，强烈推荐首先尝试方案一（启用 `band_factor`）**。这是最简单、最直接且风险最低的改进方式，预计能显著提升能级预测精度。

---

*文档生成时间：2026-04-02*  
*基于 DeePKS-L 代码库分析*
