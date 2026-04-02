# DeepKS 训练配置修改指南与源代码定制方案

## 📁 配置文件位置总览

你的训练参数配置位于：

```
/home/linearline/project/00_hydrogen_abacus/01_H_deepks/
└── iter/
    ├── params.yaml                          # ⭐ 主参数文件（定义所有默认值）
    ├── share/
    │   ├── train_input.yaml                 # 迭代训练的共享配置
    │   ├── init_train.yaml                  # 初始训练（DeePHF）配置
    │   ├── scf_abacus.yaml                  # SCF计算配置
    │   └── init_scf_abacus.yaml             # 初始SCF配置
    ├── iter.00/
    │   └── 01.train/
    │       └── train_input.yaml             # ✅ 第0次迭代实际使用的训练配置
    └── systems.yaml                         # 训练系统路径
```

### 🔑 关键发现

**实际生效的训练配置**：[iter.00/01.train/train_input.yaml](file:///home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/iter.00/01.train/train_input.yaml)

```yaml
data_args: {batch_size: 16, conv_filter: true, conv_name: conv, extra_label: true, group_batch: 1}
preprocess_args: {prefit_ridge: 10.0, prefit_trainable: false, prescale: false, preshift: false}
train_args:
  decay_rate: 0.5
  decay_steps: 1000
  display_epoch: 100
  force_factor: 1          # ❌ 只有能量+力损失
  n_epoch: 5000
  start_lr: 0.0001
```

---

## 🎯 方案一：仅修改配置文件（推荐，无需改源码）

### 步骤 1：修改训练配置

编辑 [train_input.yaml](file:///home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/iter.00/01.train/train_input.yaml)：

#### 当前配置
```yaml
train_args: {decay_rate: 0.5, decay_steps: 1000, display_epoch: 100, force_factor: 1,
  n_epoch: 5000, start_lr: 0.0001}
```

#### 推荐配置（添加能级约束）
```yaml
train_args:
  decay_rate: 0.5
  decay_steps: 1000
  display_epoch: 100
  
  # 基础损失项（保持不变）
  energy_factor: 1.0           # 总能量损失权重
  force_factor: 1.0            # 原子力损失权重
  
  # 🆕 电子结构约束（核心改进）
  band_factor: 1.0             # 能级能量损失权重 ⭐⭐⭐⭐⭐
  band_occ: 1                  # 对于H原子，只约束第1个占据态能级
  
  # 可选增强项（根据需要启用）
  bandgap_factor: 0.3          # HOMO-LUMO带隙约束
  bandgap_occ: 1               # 占据态数量
  phi_factor: 0.05             # 波函数辅助约束（权重宜小）
  phi_occ: 1                   # 占据轨道数
  
  # 训练超参数调整（可选）
  n_epoch: 8000                # 可能需要更多迭代以收敛新损失项
  start_lr: 0.00005            # 降低学习率提高稳定性
```

### 步骤 2：验证数据完整性

确保训练数据包含 `lb_band` 标签。由于你已设置 `extra_label: true`，数据生成时会自动包含。

检查方法：
```bash
cd /home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/systems/group.00
ls -la *.npy | grep -E "h_base|hamiltonian"
# 应该看到这两个文件存在
```

### 步骤 3：重新训练

```bash
cd /home/linearline/project/00_hydrogen_abacus/01_H_deepks
deepks train iter/iter.00/01.train/train_input.yaml
```

---

## 🔧 方案二：自定义氢原子专用损失函数（需修改源码）

如果标准 `band_factor` 无法满足需求（例如需要特殊的加权或物理约束），可以添加新的损失函数。

### 📍 需要修改的文件

| 文件 | 路径 | 修改内容 |
|------|------|----------|
| **evaluator.py** | `/home/linearline/project/DeePKS-L/deepks/model/evaluator.py` | 添加新的损失计算逻辑 |
| **utils.py** | `/home/linearline/project/DeePKS-L/deepks/model/utils.py` | 添加新的损失函数实现 |
| **train.py** | `/home/linearline/project/DeePKS-L/deepks/model/train.py` | 添加新的训练参数 |

---

## 💻 源代码修改详细指南

### 2.1 在 utils.py 中添加氢原子能级损失函数

**位置**: [utils.py](file:///home/linearline/project/DeePKS-L/deepks/model/utils.py) 第 218 行后（`cal_bandgap` 函数之后）

```python
def cal_hydrogen_level_loss(band_pred, band_label, occ, 
                            level_weights=None,
                            enforce_spacing=False):
    """
    计算氢原子专用的能级损失函数
    
    支持两种模式：
    1. 加权MSE损失：对不同能级赋予不同重要性
    2. 相对能级间隔约束：确保能级间距符合理论比例
    
    Args:
        band_pred: 预测的本征值，shape (nframe, nks, nlocal)
        band_label: 参考本征值，shape (nframe, nks, nlocal)
        occ: 占据态数量
        level_weights: 各能级的权重列表/字典，如 {1: 10.0, 2: 5.0, ...}
        enforce_spacing: 是否强制能级间距约束
        
    Returns:
        loss: 标量损失值
    """
    # 提取占据态和部分未占据态（根据level_weights的范围）
    if isinstance(level_weights, dict):
        max_level = max(level_weights.keys())
        n_levels = min(max_level, band_pred.shape[-1])
    else:
        n_levels = occ if level_weights is None else min(len(level_weights), band_pred.shape[-1])
    
    # 截取前 n_levels 个能级
    pred = band_pred[..., :n_levels]
    label = band_label[..., :n_levels]
    
    # 模式1：加权MSE损失
    if level_weights is not None:
        if isinstance(level_weights, dict):
            weights = torch.tensor([level_weights.get(i+1, 1.0) for i in range(n_levels)],
                                   dtype=pred.dtype, device=pred.device)
        else:
            weights = torch.tensor(level_weights[:n_levels],
                                   dtype=pred.dtype, device=pred.device)
        
        # 归一化权重
        weights = weights / weights.sum()
        
        # 加权平方误差
        diff_sq = (pred - label) ** 2
        weighted_loss = (diff_sq * weights.unsqueeze(0).unsqueeze(0)).mean()
    else:
        # 标准MSE
        weighted_loss = ((pred - label) ** 2).mean()
    
    loss = weighted_loss
    
    # 模式2：相对能级间隔约束（可选）
    if enforce_spacing and n_levels > 1:
        # 计算相邻能级间距
        pred_spacing = pred[..., 1:] - pred[..., :-1]
        label_spacing = label[..., 1:] - label[..., :-1]
        
        # 间距损失的权重递减（低能级间距更重要）
        spacing_weights = torch.linspace(1.0, 0.3, n_levels-1,
                                         dtype=pred.dtype, device=pred.device)
        spacing_loss = ((pred_spacing - label_spacing)**2 * spacing_weights).mean()
        
        # 组合损失（可调整系数）
        loss = loss + 0.5 * spacing_loss
    
    return loss


def cal_hydrogen_rydberg_loss(band_pred, band_label, occ, rydberg_const=13.605693122994):
    """
    基于Rydberg公式的物理约束损失
    
    对于单电子系统（如H原子），理论上：
    E_n = -Ry / n^2, 其中 Ry ≈ 13.6 eV
    
    该损失函数惩罚偏离Rydberg公式预测值的偏差，
    特别适用于绝对能量参考不确定的情况。
    
    Args:
        band_pred: 预测的本征值
        band_label: 参考本征值（用于确定零点）
        occ: 占据态数
        rydberg_const: Rydberg常数（eV），默认13.6 eV
        
    Returns:
        loss: 物理一致性损失
    """
    n_states = min(occ + 2, band_pred.shape[-1])  # 至少看前occ+2个态
    pred = band_pred[..., :n_states]
    
    # 理论能级（假设基态为n=1）
    n_quantum = torch.arange(1, n_states + 1, dtype=pred.dtype, device=pred.device)
    E_theory = -rydberg_const / (n_quantum ** 2)
    
    # 使用第一个能级作为参考来确定平移量
    # E_pred_shifted = E_pred - (E_pred[0] - E_theory[0])
    shift = pred[..., 0:1] - E_theory[0:1].unsqueeze(0).unsqueeze(0)
    pred_aligned = pred - shift
    
    # 对齐后的MSE损失
    loss = ((pred_aligned - E_theory.unsqueeze(0).unsqueeze(0)) ** 2).mean()
    
    return loss
```

---

### 2.2 在 evaluator.py 中集成新损失函数

**位置**: [evaluator.py](file:///home/linearline/project/DeePKS-L/deepks/model/evaluator.py)

#### 2.2.1 修改 `__init__` 方法（第 14 行附近）

在现有参数之后添加新参数：

```python
class Evaluator:
    def __init__(self,
                 # ... 现有参数保持不变 ...
                 
                 # 🆕 氢原子专用参数
                 hydrogen_level_factor=0.,      # 氢原子能级损失权重
                 hydrogen_level_weights=None,   # 能级权重字典
                 hydrogen_enforce_spacing=False,# 是否强制能级间距
                 hydrogen_rydberg_factor=0.,    # Rydberg公式约束权重
                 
                 # ... 其他现有参数 ...
                 ):
    
    # === 现有初始化代码 ===
    # （energy_factor, force_factor 等保持不变）
    
    # 🆕 新增：氢原子能级损失初始化
    self.hydrogen_level_factor = hydrogen_level_factor
    self.hydrogen_level_weights = hydrogen_level_weights
    self.hydrogen_enforce_spacing = hydrogen_enforce_spacing
    self.hydrogen_rydberg_factor = hydrogen_rydberg_factor
```

#### 2.2.2 修改 `__call__` 方法（第 126 行附近）

在 bandgap 损失计算之后（约第 253 行）添加：

```python
    def __call__(self, model, sample):
        # ... 现有代码到 bandgap 损失部分 ...
        
        # 可选带隙计算（已有代码）
        if self.bandgap_factor > 0 and "lb_band" in sample:
            # ... 已有代码 ...
            pass
        
        # 🆕 新增：氢原子专用能级损失
        if (self.hydrogen_level_factor > 0 or self.hydrogen_rydberg_factor > 0) \
           and "lb_band" in sample:
            
            band_label = sample["lb_band"]
            
            # 必须先计算 band_pred（如果没有的话）
            if 'band_pred' not in locals():
                h_base = sample["h_base"]
                if "trans_matrix" in sample:
                    trans_matrix = sample["trans_matrix"]
                    band_pred, _ = generalized_eigh(h_base + vd_pred, trans_matrix, 
                                                     self.use_safe_eigh)
                else:
                    band_pred, _ = eigh_wrapper(h_base + vd_pred, self.use_safe_eigh)
            
            natom = eig.shape[1]
            
            # 氢原子加权能级损失
            if self.hydrogen_level_factor > 0:
                h_level_occ = get_occ_func(self.band_occ)(natom) if hasattr(self, 'band_occ') and self.band_occ else natom
                h_loss = self.hydrogen_level_factor * cal_hydrogen_level_loss(
                    band_pred, band_label, h_level_occ,
                    level_weights=self.hydrogen_level_weights,
                    enforce_spacing=self.hydrogen_enforce_spacing
                )
                tot_loss = tot_loss + h_loss
                loss.append(h_loss)
            
            # Rydberg物理约束损失
            if self.hydrogen_rydberg_factor > 0:
                h_rydberg_occ = get_occ_func(self.band_occ)(natom) if hasattr(self, 'band_occ') and self.band_occ else natom
                r_loss = self.hydrogen_rydberg_factor * cal_hydrogen_rydberg_loss(
                    band_pred, band_label, h_rydberg_occ
                )
                tot_loss = tot_loss + r_loss
                loss.append(r_loss)
        
        # ... 继续后续代码 ...
```

#### 2.2.3 修改 `print_head` 方法（第 290 行附近）

添加新损失项的打印支持：

```python
    def print_head(self, name, data_keys, align_len=20):
        info = f"{name}_energy".rjust(align_len)
        # ... 现有代码 ...
        
        # 🆕 新增
        if self.hydrogen_level_factor > 0 and "lb_band" in data_keys:
            info += f"{name}_H_level".rjust(align_len)
        if self.hydrogen_rydberg_factor > 0 and "lb_band" in data_keys:
            info += f"{name}_H_rydberg".rjust(align_len)
        
        print(info, end='')
```

---

### 2.3 在 train.py 中添加新参数支持

**位置**: [train.py](file:///home/linearline/project/DeePKS-L/deepks/model/train.py)

#### 2.3.1 修改 `train` 函数签名（第 18 行）

```python
def train(model, g_reader, n_epoch=1000, test_reader=None, *,
          # ... 现有参数 ...
          
          # 🆕 氢原子专用参数
          hydrogen_level_factor=0., 
          hydrogen_level_weights=None,
          hydrogen_enforce_spacing=False,
          hydrogen_rydberg_factor=0.,
          
          # ... 其他参数 ...
          ):
```

#### 2.3.2 修改 Evaluator 初始化（第 51 行附近）

在创建 Evaluator 时传入新参数：

```python
    evaluator = Evaluator(
        energy_factor=energy_factor,
        force_factor=force_factor,
        # ... 现有参数 ...
        
        # 🆕 传入氢原子参数
        hydrogen_level_factor=hydrogen_level_factor,
        hydrogen_level_weights=hydrogen_level_weights,
        hydrogen_enforce_spacing=hydrogen_enforce_spacing,
        hydrogen_rydberg_factor=hydrogen_rydberg_factor,
        
        # ... 其他参数 ...
    )
```

---

## 📝 配置文件使用示例（使用新损失函数）

### 示例 1：使用标准 band_factor（无需源码修改）

```yaml
# iter/iter.00/01.train/train_input.yaml
train_args:
  energy_factor: 1.0
  force_factor: 1.0
  band_factor: 1.0           # 启用标准能级损失
  band_occ: 1                # H原子只有1个占据电子
  n_epoch: 5000
  start_lr: 0.0001
```

### 示例 2：使用自定义氢原子损失函数（需要上述源码修改）

```yaml
# iter/iter.00/01.train/train_input.yaml
train_args:
  energy_factor: 1.0
  force_factor: 1.0
  band_factor: 0.5           # 标准能级损失（降低权重）
  
  # 🆕 氢原子专用损失
  hydrogen_level_factor: 1.0       # 主要改进项
  hydrogen_level_weights: {1: 10.0, 2: 5.0, 3: 2.0}  # 基态最重要
  hydrogen_enforce_spacing: true   # 强制能级间距正确
  hydrogen_rydberg_factor: 0.3     # 辅助物理约束
  
  n_epoch: 8000
  start_lr: 0.00005         # 降低学习率
  decay_rate: 0.8
```

---

## 🧪 数学原理说明

### 为什么需要专门的氢原子损失函数？

#### 问题 1：标准 MSE 的不足

对于氢原子，理论能级遵循 Rydberg 公式：

$$
E_n^{\text{theory}} = -\frac{R_y}{n^2}, \quad R_y \approx 13.6 \text{ eV}
$$

但 DFT 计算得到的绝对能量依赖于：
- 赝势选择
- 能量零点定义
- 基组截断

因此，**直接比较绝对值可能不准确**。

#### 解决方案 A：加权能级损失

$$
\mathcal{L}_{\text{H-level}} = \sum_{i=1}^{N_{\text{levels}}} w_i \cdot (\epsilon_i^{\text{pred}} - \epsilon_i^{\text{label}})^2
$$

推荐权重分配（针对 H 原子）：

| 能级 $n$ | 权重 $w_n$ | 物理意义 |
|---------|-----------|---------|
| 1 (基态) | 10.0 | 最重要，决定化学性质 |
| 2 (第一激发) | 5.0 | 重要，光学性质 |
| 3+ (高激发) | 1.0~2.0 | 较次要 |

#### 解决方案 B：Rydberg 物理约束

$$
\mathcal{L}_{\text{Rydberg}} = \frac{1}{N} \sum_{i=1}^{N} \left[ (\epsilon_i^{\text{pred}} - \bar{\epsilon}) - \left(-\frac{R_y}{i^2}\right) \right]^2
$$

其中 $\bar{\epsilon}$ 是平移量（通过基态对齐消除）。

**优势**：
- ✅ 不依赖绝对能量零点
- ✅ 强制符合量子力学规律
- ✅ 对激发态特别有效

#### 解决方案 C：相对能级间距约束

$$
\mathcal{L}_{\text{spacing}} = \sum_{i=1}^{N-1} \alpha_i \cdot \left[ (\epsilon_{i+1} - \epsilon_i)^{\text{pred}} - (\epsilon_{i+1} - \epsilon_i)^{\text{label}} \right]^2
$$

其中权重 $\alpha_i$ 递减（低能级间距更重要）。

---

## 🚀 实施路线图

### Phase 1：快速验证（1小时）
1. ✅ 仅修改 `train_input.yaml`，添加 `band_factor: 1.0`
2. ✅ 重新训练模型
3. ✅ 检查 eig_occ.txt 是否改善

### Phase 2：调优（半天）
1. 如果 Phase 1 效果不理想：
   - 尝试组合使用 `band_factor` + `bandgap_factor`
   - 调整各因子权重
   - 增加 `phi_factor` 作为辅助

### Phase 3：深度定制（1-2天）
1. 如果仍需更高精度：
   - 按照上述指南修改源代码
   - 实现氢原子专用损失函数
   - 进行超参数搜索

---

## ⚠️ 注意事项与最佳实践

### 数据准备
1. **确保 extra_label=True**：当前已满足 ✅
2. **检查 h_base.npy 和 hamiltonian.npy 存在**：这是生成 lb_band 的前提
3. **验证数据质量**：确认 DFT 计算收敛良好

### 训练技巧
1. **学习率调整**：添加新损失项时建议降低初始学习率 50%
2. **渐进式训练**：
   - 先用 energy+force 预训练 2000 epochs
   - 再加入 band_factor 微调 3000 epochs
3. **监控指标**：关注每个损失项的变化趋势

### 常见问题排查
| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 训练不收敛 | 学习率过大 | 降低 start_lr 到 1e-5 |
| 能级改善但能量变差 | 权重不平衡 | 降低 band_factor 或增加 energy_factor |
| 出现 NaN | 数值不稳定 | 检查数据归一化，使用 safe_eigh=True |
| lb_band 缺失 | 数据未完整生成 | 确保 SCF 计算输出哈密顿量矩阵 |

---

## 📊 预期效果对比

### 当前结果 vs 改进后预期

| 指标 | 当前值 | 使用 band_factor | 自定义 H-loss |
|------|--------|------------------|---------------|
| 基态能量 (eV) | -6.335 | -8 ~ -12* | -12 ~ -13.5* |
| 第一激发态 (eV) | 1.901 | -2 ~ -4* | -3 ~ -3.5* |
| 能级间隔准确性 | 差 | 中等 | 优秀 |

*注：数值取决于能量零点，关键是相对间距应接近 $E_n \propto 1/n^2$

---

## 📌 总结

### 快速开始（5分钟操作）

**最简单的改进方式**（无需改代码）：

```bash
# 1. 编辑配置文件
vim /home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/iter.00/01.train/train_input.yaml

# 2. 在 train_args 中添加两行：
#    band_factor: 1.0
#    band_occ: 1

# 3. 重新训练
cd /home/linearline/project/00_hydrogen_abacus/01_H_deepks
deepks train iter/iter.00/01.train/train_input.yaml
```

### 进阶选项（需要编程能力）

如果需要更精细的控制，可以按照本文档 **2.1-2.3 节** 修改源代码，实现：
- 加权能级损失
- Rydberg 物理约束
- 相对能级间距约束

---

*文档版本*: v2.0  
*更新时间*: 2026-04-02  
*适用范围*: DeePKS-L 项目 + ABACUS DFT 计算
