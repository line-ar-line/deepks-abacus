# DeepKS 激发态能量计算优化指南

## 🎯 核心问题诊断

### 当前能级结果（严重问题！）

| 能级 | 当前值 (eV) | 理论值 (eV) | 误差 |
|------|------------|-------------|------|
| 基态 1s | **-6.33** | **-13.61** | +7.3 eV ❌ |
| 第一激发 2s/2p | **+1.90** | **-3.40** | **+5.3 eV, 符号完全错误!** ❌❌ |
| 第二激发 | **+2.14** | **-1.51** | **+3.65 eV, 错误!** ❌ |

### 三大致命问题

1. **训练数据极度单一**: 只有1个frame, 1个构型
2. **缺少关键数据**: 没有h_base.npy → band_factor无法生效!
3. **Projector截断不足**: R_cut=6 a.u., 无法捕捉激发态波函数

---

## 📊 一、训练集优化策略（优先级: ⭐⭐⭐⭐⭐）

### 问题根源
DeePKS学习映射 $f_\theta: \mathbf{X} \rightarrow \Delta E$。只有1个样本时：
- 无法学习平移不变性
- 无法捕捉有限尺寸效应  
- 对激发态泛化能力为零

### 方案A: 增加构型多样性（最低要求: ≥5个!）

#### 1. 改变原子位置（验证平移不变性）
```python
import numpy as np
def generate_translated_configs(n=10, box_size=10.0):
    configs = []
    for i in range(n):
        coord = np.random.uniform(box_size*0.2, box_size*0.8, size=(1,3))
        cell = np.eye(3) * box_size
        atom = np.array([[1] + list(coord[0])])
        configs.append({'coord': coord, 'cell': cell, 'atom': atom})
    return configs
```

#### 2. 改变盒子大小（消除有限尺寸效应）
```python
def generate_box_sizes(box_sizes=[8.0, 10.0, 12.0, 15.0, 20.0]):
    # 外推公式: E_inf(L) ≈ E(L) + A/L³
    ...
```

#### 3. 微扰构型（训练力，可选）
```python
def generate_perturbed(base_coord, displacements=[0.05, 0.1, 0.15]):
    ...
```

### 方案B: 提高参考数据质量

当前用PBE, 但PBE带隙低估严重。

升级路径:
- PBE (当前): 低准确度 ★☆☆
- **PBE0**: 中等 ★★☆ ✅推荐!
- **HSE06**: 高 ★★★ 金标准

修改 `scf_abacus.yaml`:
```yaml
dft_functional: "pbe0"  # 或 "hse"
alpha: 0.25
nbands: 12  # 增加能带数
ecutwfc: 120  # 提高截断能
```

### 方案C: 用精确解作为标签（氢原子特有优势!）

氢原子有解析解:
$$E_n = -\frac{R_y}{n^2}, \quad R_y = 13.606 \text{ eV}$$

精确能级:
- n=1: -13.606 eV (1s)
- n=2: -3.401 eV (2s/2p)
- n=3: -1.511 eV (3s/3p/3d)
- n=4: -0.850 eV (4s)

---

## 🎯 二、损失函数优化（优先级: ⭐⭐⭐⭐⭐）

### 当前配置的问题
```yaml
band_occ: 1   # 只约束基态! 完全忽略激发态!
```

### 方案A: 扩展能级监督范围 ⭐⭐⭐⭐⭐

```yaml
train_args:
  band_factor: 2.0      # 提高权重
  band_occ: 4           # 约束前4个能级(1s,2s,2p,3s)
```

数学形式:
$$\mathcal{L}_{\text{band}} = \sum_{i=1}^{4} w_i (\epsilon_i^{\text{pred}} - \epsilon_i^{\text{label}})^2$$

推荐权重分配:

| 能级 | 权重 $w_i$ | 理由 |
|------|-----------|------|
| 1 (1s) | **10.0** | 最重要 |
| 2 (2s/2p) | **5.0** | 光学性质关键 |
| 3 (3s/...) | **3.0** | 光谱特征 |
| 4+ | **1.0~2.0** | 较次要 |

### 方案B: Rydberg物理约束损失 ⭐⭐⭐⭐⭐ (强烈推荐!)

对于单电子系统, 强制符合量子力学规律!

$$\mathcal{L}_{\text{Rydberg}} = \sum_{i=1}^{N_s} \left[(\tilde{\epsilon}_i^{\text{pred}})^2 - \left(\frac{R_y}{i^2}\right)^2\right]^2$$

其中 $\tilde{\epsilon}_i$ 是对齐后的预测值(消除零点偏移).

实现代码 (添加到 utils.py):
```python
def cal_rydberg_loss(band_pred, n_states=4, rydberg_const=13.606):
    pred = band_pred[..., :n_states].clone()
    shift = pred[..., 0:1] - (-rydberg_const)
    pred_aligned = pred - shift
    
    n = torch.arange(1, n_states+1, dtype=pred.dtype, device=pred.device)
    E_theory = -rydberg_const / (n**2)
    
    loss = ((pred_aligned - E_theory)**2).mean()
    return loss
```

在 evaluator.py 中使用:
```python
if self.rydberg_factor > 0 and "lb_band" in sample:
    L_ryd = self.rydberg_factor * cal_rydberg_loss(band_pred, n_states=6)
    tot_loss += L_ryd
```

YAML配置:
```yaml
rydberg_factor: 1.0
rydberg_nstates: 6
```

### 方案C: 能级间距约束

$$\mathcal{L}_{\text{spacing}} = \sum_{i<j} \alpha_{ij}\left[\Delta\epsilon_{ij}^{\text{pred}} - \Delta\epsilon_{ij}^{\text{label}}\right]^2$$

确保能级间隔符合 $1/n^2$ 规律.

### 完整推荐配置

```yaml
train_args:
  energy_factor: 0.5       # 降低(避免主导)
  force_factor: 0.5         # 降低
  
  band_factor: 2.0          # 提高!
  band_occ: 6               # 约束更多能级
  
  rydberg_factor: 1.0       # 新增!
  rydberg_nstates: 6
  bandgap_factor: 0.5
  phi_factor: 0.1
  
  n_epoch: 10000            # 增加
  start_lr: 0.00005         # 降低
```

总损失函数:
$$\boxed{
\begin{aligned}
\mathcal{L}_{\text{optimal}} =&\; 0.5\mathcal{L}_E + 0.5\mathcal{L}_F \\
&+ 2.0\sum_{i=1}^{6}w_i(\epsilon_i^{\text{pred}}-\epsilon_i^{\text{label}})^2 \\
&+ 1.0\sum_{i=1}^{6}\left[(\tilde{\epsilon}_i+\frac{R_y}{i^2})^2\right] \\
&+ 0.5(E_{\text{gap}}^{\text{pred}}-E_{\text{gap}}^{\text{label}})^2 \\
&+ 0.1\mathcal{L}_\phi
\end{aligned}
}$$

---

## 🔬 三、描述子/Projector优化（优先级: ⭐⭐⭐⭐）

### 当前配置分析
```
jle.orb 参数:
├─ Energy Cutoff: 50 Ry     ← 可能偏低
├─ Radius Cutoff: 6 a.u.    ← ⚠️ 不够捕捉激发态!
├─ Lmax: 2 (s,p,d)          ← 缺少f轨道
└─ Orbitals: 78个 (13s+26p+39d)
```

### 关键问题: 截断半径不足!

激发态波函数平均半径:
$$\langle r\rangle_{nl} = \frac{a_0}{2}[3n^2-l(l+1)]$$

| 态 | $\langle r\rangle$ (a.u.) | 所需最小R_cut |
|----|---------------------|---------------|
| 1s | 1.5 | 5 a.u. ✓ |
| 2s | **6.0** | 12 a.u. ⚠️ |
| 2p | 5.0 | 10 a.u. ⚠️ |
| 3s | **13.5** | 27 a.u. ❌❌ |
| 3p | **12.05** | 24 a.u. ❌❌ |

**结论**: 6 a.u.无法捕捉n≥2的激发态!

### 优化方案A: 增大Projector参数

生成新的 `jle_large.orb`:
```
═══════════════════════════
🆕 优化后配置
═══════════════════════════
Energy Cutoff(Ry):     100  (从50↑)
Radius Cutoff(a.u.):   12   (从6↑)
Lmax:                  3    (从2↑, 添加f轨道)

Orbitals breakdown:
  s: 13个
  p: 26个
  d: 39个
  f: 56个  🆕
  ────────
  TOTAL: 134个 (+56)
═══════════════════════════
Mesh: 2000 points (从605↑)
```

如何生成: 使用ABACUS工具或手动指定更大的数值轨道.

### 优化方案B: 多尺度集成

同时使用short-range和long-range描述符:
```python
class MultiScaleDescriptor(nn.Module):
    def __init__(self):
        self.proj_short = load_proj("jle.orb")      # R_cut=6
        self.proj_long = load_proj("jle_large.orb")  # R_cut=12
    
    def forward(self, dm):
        X_short = project(dm, self.proj_short)  # 局域信息
        X_long = project(dm, self.proj_long)    # 渐近行为
        return torch.cat([X_short, X_long], dim=-1)
```
**预期效果**: 激发态误差减少40-60%!

### 优化方案C: 派生特征工程

除了密度矩阵本征值, 还可引入:
1. **径向分布函数(RDF)**: $\rho(r) \cdot 4\pi r^2$
2. **多极矩**: $Q_l = \int r^l Y_l^m\rho d\mathbf{r}$ (l=0,1,2)
3. **动能估算**: 从KS本征值计算

---

## 📋 四、实施路线图

### Phase 1: 快速修复 (1-2天) ⭐⭐⭐⭐⭐

**目标**: 激发态能量符号正确

✅ **Step 1**: 确认h_base.npy存在 (否则band_factor无效!)
```bash
find systems -name "h_base.npy"
# 如果不存在, 在scf_abacus.yaml中设置 deepks_v_delta: 2
```

✅ **Step 2**: 增加到≥5个训练构型
- 不同位置平移
- 不同盒子大小(8,10,12,15Å)
- 微扰位移(±0.05, ±0.1Å)

✅ **Step 3**: 启用完整能级损失
```yaml
band_factor: 2.0
band_occ: 4
n_epoch: 8000
```

**预期效果**: 激发态变为负值!

### Phase 2: 显著改进 (3-7天) ⭐⭐⭐⭐

**目标**: 定量准确 (<0.5 eV误差)

✅ **Step 4**: 升级参考数据 → PBE0泛函
✅ **Step 5**: 添加物理约束 → Rydberg损失
✅ **Step 6**: 增加迭代次数 → n_iter=3

### Phase 3: 精细调优 (1-2周) ⭐⭐⭐

**目标**: 近化学精度 (<0.1 eV)

✅ **Step 7**: 更换大Projector (R_cut=12 a.u.)
✅ **Step 8**: 实现多尺度描述符(可选)
✅ **Step 9**: 添加派生特征(可选)

---

## 🧪 五、验证指标

| 指标 | 当前 | Phase1目标 | Phase2目标 | Phase3目标 |
|------|------|-----------|-----------|-----------|
| **基态误差** | **+7.3 eV** | <3 eV | <1 eV | <0.3 eV |
| **第一激发误差** | **+5.3 eV(❌)** | <2eV(✅符号对) | <0.5eV | <0.1eV |
| **能级间距比** | 错误 | ~0.25 | ±0.02 | ±0.01 |
| **Rydberg拟合度** | N/A | >0.8 | >0.95 | >0.99 |

诊断脚本:
```bash
echo "=== DeepKS Excited States Diagnostic ==="
echo "1. 训练数据统计:"
echo "   Frames: $(find systems -name dm_eig.npy | wc -l)"
echo "   Has h_base? $(find . -name h_base.npy | wc -l)"

echo "2. 能级误差分析:"
python -c "
import numpy as np
eps_calc = np.loadtxt('OUT.H/eig_occ.txt', skiprows=5)[:8, 1]
Ry = 13.605693122994
eps_th = -Ry / np.arange(1, 9)**2
print(f'   Mean Abs Error: {np.abs(eps_calc-eps_th).mean():.3f} eV')
print(f'   Sign correct? {(eps_calc[1:]<0).all()}')
"
```

---

## 💡 六、最佳实践清单

### ✅ 必做项 (立即执行!)

- [ ] **确认h_base.npy存在**
- [ ] **增加训练样本≥5个**
- [ ] **设置band_occ ≥ 3** (不能只看基态!)
- [ ] **考虑使用PBE0替代PBE**

### 🎯 强烈推荐

- [ ] **添加Rydberg约束损失** (对单电子体系极其有效!)
- [ ] **增大Projector R_cut到10-12 a.u.**
- [ ] **将n_iter增加到3-5**
- [ ] **监控能级间距比** (应符合1/n²规律)

### 🔬 进阶选项

- [ ] 多尺度Projector集成
- [ ] RDF/多极矩派生特征
- [ ] 使用GW作为参考标签
- [ ] 加权能级损失实现

### 📝 终极配置模板

```yaml
# params.yaml (激发态优化终极版)
n_iter: 3

scf_abacus:
  dft_functional: "pbe0"
  alpha: 0.25
  nbands: 12
  ecutwfc: 120
  deepks_v_delta: 2
  cal_force: 1

train_input:
  data_args:
    extra_label: true
    
  train_args:
    energy_factor: 0.5
    force_factor: 0.5
    band_factor: 2.0
    band_occ: 6
    rydberg_factor: 1.5
    rydberg_nstates: 6
    bandgap_factor: 0.5
    phi_factor: 0.1
    n_epoch: 10000
    start_lr: 0.00003
    decay_rate: 0.92

# Projector (jle_large.orb)
Energy_Cutoff: 100 Ry
Radius_Cutoff: 12 a.u.
Lmax: 3  # s+p+d+f
```

---

## 附录

### A. 氢原子精确能级表

| n | l | 符号 | $E_n$(eV) | $\langle r\rangle$(a.u.) |
|---|---|------|----------|---------------------|
| 1 | 0 | 1s | -13.606 | 1.5 |
| 2 | 0 | 2s | -3.401 | 6.0 |
| 2 | 1 | 2p | -3.401 | 5.0 |
| 3 | 0 | 3s | -1.511 | 13.5 |
| 3 | 1 | 3p | -1.511 | 12.05 |
| ∞ | ∞ | 连续谱 | 0.000 | ∞ |

### B. 重要文件路径

| 操作 | 路径 |
|------|------|
| 训练参数 | `01_H_deepks/iter/share/train_input.yaml` |
| SCF参数 | `01_H_deepks/iter/scf_abacus.yaml` |
| Projector | `01_H_deepks/iter/jle.orb` |
| 训练数据 | `01_H_deepks/systems/group.00/*.npy` |
| 能级结果 | `00_H_scf/00_deepks_model/OUT.H/eig_occ.txt` |
| 损失函数代码 | `DeePKS-L/deepks/model/utils.py`, `evaluator.py` |

### C. 常用命令

```bash
# 运行训练
cd 01_H_deepks && deepks iterate params.yaml

# 检查数据完整性
python -c "
from deepks.model.reader import Reader
r = Reader('.', extra_label=True)
d = r.sample_all()
print('Has lb_band?', 'lb_band' in d)
print('Band values:', d['lb_band'][0,0,:6])
"

# 验证能级
python -c "
import numpy as np
eps = np.loadtxt('OUT.H/eig_occ.txt', skiprows=5)[:8,1]
Ry = 13.605693122994
print('Errors(eV):', eps.flatten() - (-Ry/np.arange(1,9)**2))
"
```

---

*版本: v1.0 | 更新: 2026-04-02 | 适用: DeePKS-L + ABACUS + H原子激发态*
