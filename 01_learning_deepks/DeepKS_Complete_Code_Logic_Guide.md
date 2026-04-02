# DeepKS 完整代码逻辑图解与参数指南

## 📖 文档目标

系统解析 `01_H_deepks` 算例的**完整实现逻辑**，帮助理解：
- ✅ 从 DFT 计算到模型训练的完整数据流
- ✅ 所有可调参数及其物理意义
- ✅ 损失函数代码位置，为修改提供基础

---

## 🏗️ 一、整体架构概览

### 1.1 项目目录结构

```
00_hydrogen_abacus/
├── 01_H_deepks/                          # ⭐ 主工作目录
│   ├── iter/                              # 迭代训练核心
│   │   ├── params.yaml                    # 🔑 主配置文件（迭代次数、训练参数）
│   │   ├── systems.yaml                   # 训练系统路径
│   │   ├── machines.yaml                  # 计算资源配置
│   │   ├── scf_abacus.yaml                # ABACUS SCF 计算参数
│   │   └── share/                         # 共享配置目录
│   │       ├── train_input.yaml           # 🎯 训练输入参数（实际生效！）
│   │       ├── init_train.yaml            # 初始训练参数（DeePHF阶段）
│   │       └── scf_abacus.yaml            # SCF计算配置
│   └── iter.00/                           # 第0次迭代结果
│       └── 01.train/
│           └── train_input.yaml           # 当前使用的训练配置
│
└── 00_H_scf/00_deepks_model/
    ├── OUT.H/eig_occ.txt                 # 能级结果
    └── *.md                               # 分析文档
```

### 1.2 核心代码位置（DeePKS-L 库）

```
DeePKS-L/deepks/
├── main.py                                # 命令行入口
├── model/
│   ├── train.py                           # ⭐⭐⭐ 训练主流程（损失函数组装）
│   ├── evaluator.py                       # ⭐⭐⭐ Evaluator类（所有损失项实现）
│   ├── utils.py                           # ⭐⭐⭐ 工具函数（make_loss, cal_phi_loss等）
│   ├── reader.py                          # 数据读取器（生成lb_band等标签）
│   └── model.py                           # CorrNet模型定义
├── iterate/
│   ├── iterate.py                         # ⭐⭐⭐ 迭代训练框架
│   ├── template_abacus.py                 # ⭐⭐ ABACUS SCF模板（数据生成）
│   └── generator_abacus.py                # ABACUS输入文件生成
```

---

## 🔄 二、完整数据流与执行流程

### 2.1 总体流程图

```
启动: deepks iterate params.yaml
         ↓
┌─────────────────────────────────────┐
│ Phase 0: 初始化 (init_scf+init_train) │
│ 条件: init_model=F, init_scf=T, init_train=T │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Step 0.1: 初始SCF (无DeepKS模型)     │
│ 📍 template_abacus.py: make_scf()   │
│ 📝 scf_abacus.yaml (init部分)        │
│ 🎯 纯DFT(PBE)计算H原子基态            │
│ 输出: deepks_ebase.npy, dm_eig.npy.. │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Step 0.2: 数据收集与标签生成          │
│ 📍 template_abacus.py: gather_stats()│
│ 输入: OUT.ABACUS/*.npy              │
│ 输出: data_train/group.00/           │
│   ├─ dm_eig.npy      # 描述符       │
│   ├─ l_e_delta.npy   # 能量差值标签  │
│   ├─ h_base.npy      # 基础哈密顿量  │
│   ├─ hamiltonian.npy # 完整哈密顿量  │
│   └─ ...             # 其他标签      │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Step 0.3: 初始训练 (DeePHF)          │
│ 📍 train.py → main()                │
│ 📝 init_train.yaml                  │
│ 🧠 CorrNet([100,100,100])           │
│ 📊 L_energy + L_force               │
│ 输出: model.pth                     │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│ Phase 1: 迭代训练 (n_iter=1)         │
│                                     │
│ Iteration 0:                        │
│ ├─ Step1: SCF with DeepKS Model     │
│ │   使用model.pth预测v_delta         │
│ │   自洽求解新电子结构               │
│ │                                 │
│ └─ Step2: 数据更新 + 重训练          │
│     收集新SCF结果                   │
│     更新标签                        │
│     从model.pth restart继续训练     │
│     🆕 使用band_factor约束能级!      │
└──────────────┬──────────────────────┘
               ↓
最终输出:
  ├─ iter.00/01.train/model.pth
  └─ 00_H_scf/OUT.H/eig_occ.txt
```

### 2.2 详细数据流追踪

#### **阶段A: ABACUS DFT计算**

**入口**: [template_abacus.py](file:///home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py) `make_scf_abacus()`

**三个子步骤**:

1. **convert_data** [template_abacus.py:92-179](file:///home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py#L92-L179)
   - 读取原子坐标 (`atom.npy`, `coord.npy`)
   - 为每个frame创建ABACUS输入:
     - `INPUT`: DFT参数 (ecutwfc=100Ry, nbands=8, dft_functional=pbe)
     - `STRU`: 原子位置、晶格(10×10×10Å)、赝势(H_ONCV_PBE-1.0.upf)、轨道(H_gga_10au_100Ry_3s2p.orb)
     - `KPT`: Gamma点

2. **run_scf_abacus** [template_abacus.py:218-337](file:///home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py#L218-L337)
   - 执行: `mpirun -n 1 abacus_path`
   - 检测收敛状态

3. **gather_stats_abacus** [template_abacus.py:342-892](file:///home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py#L342-L892) ⭐ **核心!**
   
   ```python
   for f in range(nframes):
       load_f_path = f"{sys_path}/ABACUS/{f}/OUT.ABACUS/"
       
       dm_eig = np.load("deepks_dm_eig.npy")   # 描述符(特征向量)
       e_base = np.load("deepks_ebase.npy")    # 基础能量
       e_tot  = np.load("deepks_etot.npy")      # 总能量
       
       if cal_force:
           f_base = np.load("deepks_fbase.npy") # 基础力
           gvx    = np.load("deepks_gradvx.npy") # 力梯度
       
       if deepks_v_delta > 0:  # ⭐⭐⭐ 关键!
           h_base = np.load("deepks_hbase.npy")  # 基础H矩阵
           h_tot  = np.load("deepks_htot.npy")    # 总H矩阵
   
   # 保存训练标签
   np.save("l_e_delta.npy", energy_ref - e_base)
   np.save("l_h_delta.npy", hamiltonian_ref - h_base)  # H矩阵差值
   np.save("h_base.npy", h_base)                       # 用于对角化得能级
   np.save("hamiltonian.npy", hamiltonian_ref)          # 参考H矩阵
   ```

#### **阶段B: 数据读取与预处理**

**入口**: [reader.py](file:///home/linearline/project/DeePKS-L/deepks/model/reader.py) `Reader` 类

**关键**: `prepare()` 方法 [reader.py:106-240](file:///home/linearline/project/DeePKS-L/deepks/model/reader.py#L106-L240)

```python
def prepare(self):
    self.t_data["lb_e"] = torch.tensor(energy_labels)    # 能量差值标签
    self.t_data["eig"]  = torch.tensor(descriptors)      # 特征向量
    
    if h_matrix_files_exist:
        self.t_data["lb_vd"] = torch.tensor(hamiltonian_delta)
        self.t_data["h_base"] = torch.tensor(h_base)
        
        # ⭐⭐⭐ 生成能级标签! 对角化hamiltonian矩阵
        if hamiltonian_ref_exists:
            h_ref = torch.tensor(hamiltonian_ref)
            
            if overlap_exists:
                band_ref, phi_ref = generalized_eigh(h_ref, trans_matrix)
            else:
                band_ref, phi_ref = torch.linalg.eigh(h_ref)
            
            self.t_data["lb_band"] = band_ref   # ⭐ 本征值(能级)!
            self.t_data["lb_phi"]  = phi_ref     # 波函数
```

**重要发现**: 
- `lb_band` 不是直接从文件读取，而是通过对 `hamiltonian.npy` **实时对角化**生成!
- 这就是为什么需要 `extra_label: true` 和 `h_base.npy` + `hamiltonian.npy`

#### **阶段C: 模型训练**

**入口**: [train.py](file:///home/linearline/project/DeePKS-L/deepks/model/train.py) `train()` 函数

```python
def train(model, g_reader, n_epoch=5000,
          energy_factor=1., force_factor=1.,
          band_factor=0., band_occ=0,  # ⭐ 能级约束
          ...):
    
    evaluator = Evaluator(
        energy_factor=energy_factor,
        force_factor=force_factor,
        band_factor=band_factor,      # ⭐ 能级损失权重
        band_occ=band_occ,            # ⭐ 占据态数量
    )
    
    for epoch in range(n_epoch):
        for batch in g_reader:
            optimizer.zero_grad()
            loss = evaluator(model, batch)  # ⭐⭐⭐ 核心
            loss[-1].backward()
            optimizer.step()
```

#### **阶段D: 损失函数计算**

**入口**: [evaluator.py](file:///home/linearline/project/DeePKS-L/deepks/model/evaluator.py) `__call__()` [L126-288](file:///home/linearline/project/DeePKS-L/deepks/model/evaluator.py#L126-L288)

```python
def __call__(self, model, sample):
    e_label, eig = sample["lb_e"], sample["eig"]
    e_pred = model(eig)                    # CorrNet前向传播
    
    # 1. 能量损失
    loss_energy = energy_factor * MSE(e_pred, e_label)
    
    # 2. 自动微分获取梯度
    gev = ∂e_pred/∂eig
    
    # 3. 力损失
    f_pred = -gvx @ gev
    loss_force = force_factor * MSE(f_pred, f_label)
    
    # 4. 哈密顿量修正
    vd_pred = einsum("...kxyap,...ap->...kxy", vdp, gev)
    
    # 5. ⭐⭐⭐ 能级损失 - 关键改进点!
    if band_factor > 0 and "lb_band" in sample:
        band_pred, _ = eigh(h_base + vd_pred)  # 对角化得预测能级
        loss_band = band_factor * MSE(band_pred[:occ], band_label[:occ])
    
    return [loss_energy, loss_force, ..., loss_band, total_loss]
```

---

## ⚙️ 三、所有可调参数详解

### 3.1 迭代控制 ([params.yaml](file:///home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/params.yaml))

| 参数 | 当前值 | 物理意义 |
|------|--------|----------|
| `n_iter` | 1 | 迭代次数 (0=仅DeePHF, 1+=自洽迭代) |
| `init_model` | false | 是否使用已有模型 |
| `init_scf` | True | 是否初始SCF (无模型) |
| `init_train` | dict | 初始训练参数 |

### 3.2 SCF参数 ([scf_abacus.yaml](file:///home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/scf_abacus.yaml))

| 参数 | 当前值 | 物理意义 | 建议 |
|------|--------|----------|------|
| `nbands` | 8 | 计算能带数 | ≥占据+未占据 |
| `ecutwfc` | 100 Ry | 截断能 | 越大越精确 |
| `scf_thr` | 1e-8 | 收敛阈值 | 1e-7~1e-9 |
| `cal_force` | 0 | 是否计算力 | 需要力标签时=1 |
| `dft_functional` | "pbe" | 交换关联泛函 | pbe/lda/hse |
| `gamma_only` | 1 | 仅Gamma点 | 原子/分子=1 |
| `lattice_vector` | [[10,0,0]...] Å | 盒子大小 | 足够大避免相互作用 |

### 3.3 训练参数 ([train_input.yaml](file:///home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/iter.00/01.train/train_input.yaml))

#### A. 数据处理 (`data_args`)

| 参数 | 当前值 | 影响 |
|------|--------|------|
| `batch_size` | 16 | 显存占用 |
| `extra_label` | **true** | ⭐ 必须true才能用band损失! |
| `conv_filter` | true | 只用收敛frame |

#### B. 预处理 (`preprocess_args`)

| 参数 | 当前值 | 建议 |
|------|--------|------|
| `preshift` | false | true通常更好 |
| `prefit_ridge` | 10.0 | Ridge正则化强度 |

#### C. 模型 (`model_args`, 在init_train中)

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `hidden_sizes` | [100,100,100] | 隐藏层神经元数 |
| `output_scale` | 100 | 输出缩放 |
| `use_resnet` | true | 残差连接 |
| `actv_fn` | "mygelu" | 激活函数 |

**模型架构**:
```
Input(nproj) → Shift&Scale → Linear(→100)→GELU→Res → 
               Linear(100→100)→GELU→Res → 
               Linear(100→100)→GELU→Res → 
               Linear(100→1) → Output/output_scale
```

#### D. **损失函数权重** ⭐⭐⭐ **最关键!**

| 参数 | 当前值 | 默认值 | 数学表达式 | 建议 |
|------|--------|--------|------------|------|
| `energy_factor` | 1.0 | 1.0 | $w_E \cdot\text{MSE}(E_{pred},E_{label})$ | 保持1.0 |
| `force_factor` | **1.0** | 0.0 | $w_F \cdot\text{MSE}(F_{pred},F_{label})$ | 1.0 |
| **`band_factor`** | **1.0🆕** | **0.0** | **$w_B \cdot\text{MSE}(\epsilon_{pred},\epsilon_{label})$** | **1.0~2.0** |
| **`band_occ`** | **1🆕** | **0** | **只约束前N个能级** | **H原子=1** |
| `bandgap_factor` | 0.3🆕 | 0.0 | 带隙损失 | 0.3~0.5 |
| `phi_factor` | 0.05🆕 | 0.0 | 波函数损失 | 0.05~0.1 |

#### E. 优化器

| 参数 | 当前值 | 建议 |
|------|--------|------|
| `n_epoch` | 5000 | 3000~10000 |
| `start_lr` | 0.0001 | 1e-4~1e-3 |
| `decay_rate` | 0.5 | 0.8~0.96 |
| `decay_steps` | 1000 | 500~2000 |

---

## 🔍 四、关键代码定位与修改指南

### 4.1 损失函数代码地图

```
调用链:

train.py:154  →  loss = evaluator(model, sample)
     ↓
evaluator.py:126  →  def __call__(self, model, sample):
     ↓
┌──────────────────────────────────────────────────┐
│ L160: loss_energy = e_factor * MSE(e_pred, e_l)  │
│ L176: loss_force  = f_factor * MSE(f_pred, f_l)  │
│ L217: loss_vd     = vd_factor * MSE(vd_p, vd_l)  │
│ L234: loss_phi    = phi_factor * cal_phi_loss(..)│
│ L241: loss_band   = band_factor * MSE(band_p,b_l)│⭐
│ L251: loss_bg     = bg_factor * MSE(bg_p, bg_l)   │
└──────────────────────────────────────────────────┘
     ↓
utils.py:58-77   →  make_loss()  (基础MSE/L1/Huber)
utils.py:147-158 →  cal_phi_loss()  (波函数损失)
```

### 4.2 如何添加自定义损失

#### 方案A: 用现有band_factor (推荐，无需改码)

编辑 `train_input.yaml`:
```yaml
train_args:
  band_factor: 1.0
  band_occ: 1
```

#### 方案B: 修改源码 (需改3个文件)

**Step 1**: [utils.py](file:///home/linearline/project/DeePKS-L/deepks/model/utils.py) 第218行后添加:
```python
def cal_custom_loss(band_pred, band_label, occ):
    pred, label = band_pred[...,:occ], band_label[...,:occ]
    weights = torch.linspace(occ, 1, occ)
    return ((pred-label)**2 * weights/weights.sum()).mean()
```

**Step 2**: [evaluator.py](file:///home/linearline/project/DeePKS-L/deepks/model/evaluator.py) 
- `__init__` 添加参数: `custom_factor=0.`
- `__call__` 第253行后添加计算逻辑

**Step 3**: [train.py](file:///home/linearline/project/DeePKS-L/deepks/model/train.py)
- 函数签名添加: `custom_factor=0.`
- Evaluator初始化传入该参数

---

## 🎯 五、当前算例执行日志

基于你的配置，实际流程：

### Phase 0: Init

**Step 0.1**: 初始SCF (PBE, 无DeepKS)
- 计算 H 原子在 10Å 盒中的基态
- 输出: ebase, etot, hbase, htot, dmeig...

**Step 0.2**: 数据收集
- 生成: l_e_delta (≈0, 因为无v_delta), h_base, hamiltonian...
- **注意**: 无force数据 (cal_force=0)

**Step 0.3**: DeePHF训练
- 模型: CorrNet([100,100,100])
- 损失: 仅 L_energy (无力标签)
- 输出: model.pth

### Phase 1: Iteration 0

**Step 1.1**: SCF with model
- ABACUS加载model.pth
- 模型预测v_delta并加入H矩阵
- 自洽求解新电子结构

**Step 1.2**: 重训练 (你已修改!)
```python
# 新增的损失项:
L_total = 1.0*L_energy + 1.0*L_force(无效) + 1.0*L_band 🆕

# L_band计算过程:
1. e_pred = model(eig)
2. gev = ∂e_pred/∂eig  (自动微分)
3. vd_pred = vdp @ gev
4. band_pred = eig(h_base + vd_pred)  # 对角化
5. L_band = MSE(band_pred[:,:,:1], lb_band[:,:,:1])  # 只比基态
```

---

## 📚 六、调试技巧

### 6.1 检查数据完整性

```bash
cd systems/group.00
ls *.npy | grep -E "dm_eig|h_base|hamiltonian"

python -c "
import numpy as np
print('dm_eig:', np.load('dm_eig.npy').shape)
print('h_base:', np.load('h_base.npy').shape)
"
```

### 6.2 验证lb_band存在

```python
from deepks.model.reader import Reader
reader = Reader('.', batch_size=1, extra_label=True)
data = reader.sample_all()
print('keys:', list(data.keys()))
print('lb_band' in data and '✅ 存在' or '❌ 缺失')
if 'lb_band' in data:
    print('shape:', data['lb_band'].shape)
    print('values:', data['lb_band'][0,0,:5])
```

### 6.3 监控训练

查看日志输出:
```
epoch  trn_err   tst_err   lr   ...  trn_energy  trn_band
   0   1.2e-1   1.3e-1  1e-4     1.2e-2     5.6e-2  ← 应快速下降
 100   4.5e-3   4.8e-3  5e-5     2.3e-4     1.2e-3
```

---

## 💡 七、总结

### 你已掌握:

✅ **完整架构**: Init → Iter0 → Iter1 → ...  
✅ **数据流**: DFT → 标签生成 → 训练 → SCF验证  
✅ **所有参数**: SCF截断能到学习率的每个旋钮  
✅ **损失函数位置**: 精确到行号的代码地图  
✅ **修改方法**: 配置 vs 源码两种方案  

### 下一步行动:

**立即 (5分钟)**:
1. ✅ 确认params.yaml有 `band_factor: 1.0, band_occ: 1`
2. ✅ 重跑: `cd 01_H_deepks && deepks iterate params.yaml`
3. ✅ 查看日志是否出现 `trn_band` 列

**短期优化**:
- 组合多个损失项 (band+bandgap+phi)
- 增加训练数据多样性
- 调整模型架构

**深度定制**:
- 实现自定义损失 (如Rydberg物理约束)
- 添加能级间距比例约束

---

*版本: v1.0 | 更新: 2026-04-02*
