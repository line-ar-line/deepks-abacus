# DeePKS算例运行完整分析

## 算例基本信息

- **算例路径**: `/home/linearline/project/00_hydrogen_abacus/01_H_deepks`
- **源码路径**: `/home/linearline/project/DeePKS-L`
- **ABACUS路径**: `/home/linearline/project/abacus-develop`
- **系统**: 氢原子(H)系统
- **目标**: 使用DeePKS方法训练神经网络模型来修正DFT泛函

---

## 一、算例目录结构

```
01_H_deepks/
├── iter/                          # 迭代计算主目录
│   ├── run.sh                     # 启动脚本
│   ├── params.yaml                # 参数配置文件
│   ├── systems.yaml               # 系统配置文件
│   ├── machines.yaml              # 机器配置文件
│   ├── scf_abacus.yaml          # ABACUS SCF配置
│   ├── H_gga_10au_100Ry_3s2p.orb  # 轨道文件
│   ├── H_ONCV_PBE-1.0.upf      # 赝势文件
│   ├── jle.orb                  # 描述符投影基组
│   ├── RECORD                   # 运行记录文件
│   ├── iter.init/               # 初始化迭代
│   │   ├── 00.scf/            # SCF计算
│   │   └── 01.train/          # 模型训练
│   └── iter.00/               # 第一次迭代
│       ├── 00.scf/
│       └── 01.train/
└── systems/                     # 系统数据目录
    └── group.00/
        ├── atom.npy            # 原子坐标数据
        ├── energy.npy          # 能量数据
        └── ABACUS/           # ABACUS计算目录
            └── 0/           # 第0帧
                ├── INPUT      # ABACUS输入文件
                ├── STRU       # 结构文件
                ├── KPT        # K点文件
                └── OUT.ABACUS/ # 输出目录
```

---

## 二、启动流程分析

### 2.1 启动命令

```bash
nohup python -u -m deepks iterate machines.yaml params.yaml systems.yaml scf_abacus.yaml >> log.iter 2> err.iter &
```

### 2.2 Python调用链

#### 第一层：主入口
- **文件**: `/home/linearline/project/DeePKS-L/deepks/__main__.py`
- **函数**: `main_cli()`
- **作用**: 解析命令行参数，路由到子命令

#### 第二层：迭代命令
- **文件**: `/home/linearline/project/DeePKS-L/deepks/main.py`
- **函数**: `iter_cli(args=None)`
- **作用**: 处理iterate子命令，加载配置文件

```python
def iter_cli(args=None):
    parser = argparse.ArgumentParser(...)
    parser.add_argument("argfile", nargs="*", default=[])
    parser.add_argument("--systems-train", ...)
    parser.add_argument("--systems-test", ...)
    parser.add_argument("--n-iter", type=int, ...)
    # 解析参数
    argdict = {}
    for fl in args.argfile:
        argdict = deep_update(argdict, load_yaml(fl))
    # 调用主函数
    from deepks.iterate.iterate import main
    main(**argdict)
```

#### 第三层：迭代主函数
- **文件**: `/home/linearline/project/DeePKS-L/deepks/iterate/iterate.py`
- **函数**: `main(*args, **kwargs)`
- **作用**: 创建并执行迭代工作流

```python
def main(*args, **kwargs):
    iterate = make_iterate(*args, **kwargs)
    if os.path.exists(iterate.record_file):
        iterate.restart()
    else:
        iterate.run()
```

#### 第四层：创建迭代工作流
- **文件**: `/home/linearline/project/DeePKS-L/deepks/iterate/iterate.py`
- **函数**: `make_iterate(...)`
- **作用**: 构建完整的迭代工作流

关键步骤：
1. 收集训练和测试系统
2. 检查共享文件夹中的配置文件
3. 创建SCF步骤（使用ABACUS）
4. 创建训练步骤
5. 组合成迭代工作流

---

## 三、初始化迭代 (iter.init)

### 3.1 SCF步骤 (00.scf)

#### 3.1.1 数据转换阶段
- **文件**: `/home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py`
- **函数**: `convert_data(...)`
- **类**: `PythonTask`
- **作用**: 将系统数据转换为ABACUS输入文件

**输入**:
- `systems_train`: 训练系统路径
- `systems_test`: 测试系统路径
- `orb_files`: 轨道文件列表
- `pp_files`: 赝势文件列表
- `proj_file`: 投影基组文件
- `no_model`: True（初始SCF不使用模型）

**输出**:
- 为每个系统创建ABACUS输入文件：
  - `INPUT`: SCF计算参数
  - `STRU`: 晶体结构
  - `KPT`: K点设置（如需要）

**关键代码**:
```python
def convert_data(systems_train, systems_test=None, 
                no_model=True, model_file=None, pp_files=[], 
                dispatcher=None, **pre_args):
    # 读取原子坐标
    atom_data = np.load(f"{sys_paths[i]}/atom.npy")
    # 创建ABACUS目录
    if not os.path.exists(f"{sys_paths[i]}/ABACUS"):
        os.mkdir(f"{sys_paths[i]}/ABACUS")
    # 为每一帧创建子目录
    for f in range(nframes):
        if not os.path.exists(f"{sys_paths[i]}/ABACUS/{f}"):
            os.mkdir(f"{sys_paths[i]}/ABACUS/{f}")
        # 写入STRU文件
        with open(f"{sys_paths[i]}/ABACUS/{f}/STRU", "w") as stru_file:
            stru_file.write(make_abacus_scf_stru(sys_data, pp_files, pre_args_new))
        # 写入INPUT文件
        with open(f"{sys_paths[i]}/ABACUS/{f}/INPUT", "w") as input_file:
            input_file.write(make_abacus_scf_input(pre_args_new))
```

#### 3.1.2 运行ABACUS SCF
- **文件**: `/home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py`
- **函数**: `make_run_scf_abacus(...)`
- **类**: `GroupBatchTask`
- **作用**: 批量运行ABACUS SCF计算

**命令**:
```bash
cd systems/group.00/ABACUS/0/ && \
mpirun -n 1 /home/linearline/project/abacus-develop/build/abacus_4p > log.scf 2>err.log && \
echo 0`grep -i converge ./OUT.ABACUS/running_scf.log` > conv
```

**ABACUS输入文件示例** (INPUT):
```
INPUT_PARAMETERS
calculation scf
ecutwfc 100.000000
scf_thr 1.000000e-08
scf_nmax 100
basis_type lcao
dft_functional pbe
gamma_only 1
mixing_type pulay
mixing_beta 0.400000
symmetry 0
nbands 8
nspin 1
smearing_method gaussian
smearing_sigma 0.020000
cal_force 0
cal_stress 0
deepks_out_labels 1
deepks_scf 0
deepks_bandgap 0
deepks_v_delta 0
out_wfc_lcao 0
```

**ABACUS结构文件示例** (STRU):
```
ATOMIC_SPECIES
H 1.00 /home/linearline/project/.../H_ONCV_PBE-1.0.upf

LATTICE_CONSTANT
1.8897261258369282

LATTICE_VECTORS
10 0 0 
0 10 0 
0 0 10 

ATOMIC_POSITIONS
Direct

H
0.0
1
0.000000000000 0.000000000000 0.000000000000 0 0 0

NUMERICAL_ORBITAL
/home/linearline/project/.../H_gga_10au_100Ry_3s2p.orb

NUMERICAL_DESCRIPTOR
/home/linearline/project/.../jle.orb
```

#### 3.1.3 C++层面的ABACUS执行

**主要C++类和函数**:

1. **Setup_DeePKS类** (`/home/linearline/project/abacus-develop/source/source_lcao/setup_deepks.h`)
   - `before_runner()`: 初始化DeePKS模块
   - `build_overlap()`: 构建重叠积分
   - `delta_e()`: 计算能量修正
   - `write_forces()`: 写入力数据
   - `write_stress()`: 写入应力数据

2. **LCAO_Deepks类** (`/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/LCAO_deepks.h`)
   - `init()`: 初始化DeePKS参数
   - `allocate_V_delta()`: 分配修正哈密顿量内存
   - `dpks_cal_e_delta_band()`: 计算能带能量修正

**关键C++文件**:
- `setup_deepks.cpp`: DeePKS设置和主流程
- `LCAO_deepks.cpp`: DeePKS核心实现
- `deepks_descriptor.cpp`: 描述符计算
- `deepks_pdm.cpp`: 投影密度矩阵计算
- `deepks_force.cpp`: 力计算
- `deepks_orbital.cpp`: 轨道相关计算

**ABACUS执行流程**:
1. 读取INPUT和STRU文件
2. 初始化晶胞和基组
3. 计算重叠积分 `<phi|alpha>`
4. 如果`deepks_scf=0`，执行常规DFT计算
5. 如果`deepks_out_labels=1`，输出标签数据

**ABACUS输出文件**:
- `OUT.ABACUS/running_scf.log`: SCF迭代日志
- `OUT.ABACUS/deepks_desc.dat`: 描述符数据
- `OUT.ABACUS/deepks_projdm.dat`: 投影密度矩阵
- `OUT.ABACUS/deepks_ebase.npy`: 基础能量
- `OUT.ABACUS/deepks_etot.npy`: 总能量
- `OUT.ABACUS/deepks_dm_eig.npy`: 密度矩阵本征值
- `conv`: 收敛标志

#### 3.1.4 数据收集阶段
- **文件**: `/home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py`
- **函数**: `gather_stats_abacus(...)`
- **类**: `PythonTask`
- **作用**: 收集ABACUS输出数据，组织成训练数据

**输入**:
- ABACUS输出目录
- `train_dump`: 训练数据输出目录
- `test_dump`: 测试数据输出目录

**输出**:
- `data_train/group.00/atom.npy`: 原子坐标
- `data_train/group.00/energy.npy`: 能量
- `data_train/group.00/e_base.npy`: 基础能量
- `data_train/group.00/e_tot.npy`: 总能量
- `data_train/group.00/l_e_delta.npy`: 能量修正标签
- `data_train/group.00/dm_eig.npy`: 密度矩阵本征值（描述符）
- `data_train/group.00/conv.npy`: 收敛标志

**关键代码**:
```python
def gather_stats_abacus(systems_train, systems_test, 
                      train_dump, test_dump, **stat_args):
    for i in range(len(systems_train)):
        for f in range(nframes):
            load_f_path = f"{sys_train_paths[i]}/ABACUS/{f}/OUT.ABACUS/"
            # 读取描述符（密度矩阵本征值）
            des = np.load(load_f_path + "deepks_dm_eig.npy")
            dm_eig[f] = des
            # 读取基础能量
            ene = np.load(load_f_path + "deepks_ebase.npy")
            e_base[f] = ene
            # 读取总能量
            ene = np.load(load_f_path + "deepks_etot.npy")
            e_tot[f] = ene
            # 计算能量修正标签
            l_e_delta[f] = e_tot[f] - e_base[f]
```

### 3.2 训练步骤 (01.train)

#### 3.2.1 训练主函数
- **文件**: `/home/linearline/project/DeePKS-L/deepks/model/train.py`
- **函数**: `main(...)`
- **作用**: 加载数据，训练神经网络模型

**输入参数**:
- `train_paths`: 训练数据路径
- `test_paths`: 测试数据路径
- `model_args`: 模型架构参数
- `data_args`: 数据加载参数
- `preprocess_args`: 数据预处理参数
- `train_args`: 训练超参数

**模型架构** (来自params.yaml):
```yaml
model_args:
  hidden_sizes: [100, 100, 100]  # 隐藏层大小
  output_scale: 100                # 输出缩放因子
  use_resnet: true                # 使用残差连接
  actv_fn: mygelu                # 激活函数
```

**训练参数**:
```yaml
train_args:
  n_epoch: 5000
  start_lr: 0.0003
  decay_rate: 0.96
  decay_steps: 500
  display_epoch: 100
  force_factor: 1
```

#### 3.2.2 模型定义
- **文件**: `/home/linearline/project/DeePKS-L/deepks/model/model.py`
- **类**: `CorrNet`
- **作用**: 定义神经网络模型结构

**模型组件**:
1. **Embedding层**: 将描述符投影到特征空间
   - `TraceEmbedding`: 迹嵌入
   - `ThermalEmbedding`: 热力学嵌入

2. **DenseNet**: 全连接神经网络
   - 多层感知机
   - 支持残差连接
   - 可选层归一化

3. **输出层**: 预测能量修正

**关键代码**:
```python
class CorrNet(nn.Module):
    def __init__(self, ndesc, hidden_sizes, output_scale=1.0,
                 use_resnet=True, actv_fn='relu', 
                 embedder_type='trace', **kwargs):
        super().__init__()
        self.ndesc = ndesc
        self.output_scale = output_scale
        # 嵌入层
        self.embedder = make_embedder(embedder_type, shell_sec, **kwargs)
        # 主网络
        sizes = [self.embedder.ndesc] + hidden_sizes + [1]
        self.net = DenseNet(sizes, actv_fn, use_resnet, **kwargs)
    
    def forward(self, desc):
        x = self.embedder(desc)
        x = self.net(x)
        return x / self.output_scale
```

#### 3.2.3 数据加载
- **文件**: `/home/linearline/project/DeePKS-L/deepks/model/reader.py`
- **类**: `GroupReader`
- **作用**: 批量加载训练数据

**数据字段**:
- `eig`: 描述符（密度矩阵本征值）
- `lb_e`: 能量标签（l_e_delta）

**预处理步骤**:
1. **平移**: `preshift: true` - 将描述符平移到零均值
2. **缩放**: `prescale: false` - 不进行方差缩放
3. **Ridge回归**: `prefit_ridge: 1e1` - 使用岭回归进行预拟合

#### 3.2.4 训练循环
- **文件**: `/home/linearline/project/DeePKS-L/deepks/model/train.py`
- **函数**: `train(...)`
- **作用**: 执行模型训练

**训练过程**:
```python
def train(model, g_reader, n_epoch=1000, test_reader=None, ...):
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, decay_steps, decay_rate)
    
    # 训练循环
    for epoch in range(1, n_epoch+1):
        model.train()
        for sample in g_reader:
            optimizer.zero_grad()
            loss = evaluator(model, sample)
            loss[-1].backward()
            optimizer.step()
        scheduler.step()
        
        # 定期评估
        if epoch % display_epoch == 0:
            model.eval()
            # 计算训练和测试损失
            trn_loss = evaluate(model, g_reader)
            tst_loss = evaluate(model, test_reader)
            # 保存模型
            model.save(ckpt_file)
```

**损失函数**:
- 主要损失: 能量MSE损失
- `energy_factor`: 能量损失权重
- `force_factor`: 力损失权重（本例中为0）

**训练日志示例**:
```
# epoch      trn_err   tst_err        lr  trn_time  tst_time     trn_loss_energy
  0         3.89e-04  3.89e-04  3.00e-04      0.00      0.03          1.5143e-07
  100       7.65e-09  5.69e-09  3.00e-04      0.01      0.00          5.8537e-17
  200       1.86e-13  8.86e-14  3.00e-04      0.01      0.00          3.4567e-26
...
  5000      0.00e+00  0.00e+00  1.99e-04      0.00      0.00          0.0000e+00
```

**输出文件**:
- `model.pth`: 训练好的PyTorch模型
- `log.train`: 训练日志
- `log.test`: 测试日志

---

## 四、迭代步骤 (iter.00)

### 4.1 SCF步骤 (00.scf) - 使用模型

与初始化SCF的主要区别：
- `no_model=False`: 使用训练好的模型
- `deepks_scf=1`: 在SCF中使用DeePKS修正

#### 4.1.1 模型追踪和编译
- **文件**: `/home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py`
- **函数**: `convert_data(...)`
- **作用**: 将PyTorch模型转换为TorchScript格式

**关键代码**:
```python
if not no_model:
    if model_file is not None:
        from deepks.model import CorrNet
        model = CorrNet.load(model_file)
        model.compile_save(CMODEL_FILE)
        pre_args.update(deepks_scf=1, 
                      model_file=os.path.abspath(CMODEL_FILE))
```

**输出**:
- `model.ptg`: TorchScript格式的模型文件

#### 4.1.2 ABACUS SCF with DeePKS

**INPUT文件变化**:
```
deepks_scf 1
deepks_model /path/to/model.ptg
```

**C++执行流程**:

1. **加载模型** (`setup_deepks.cpp`):
```cpp
void Setup_DeePKS<TK>::before_runner(...) {
    if (inp.deepks_scf) {
        // 加载TorchScript模型
        DeePKS_domain::load_model(inp.deepks_model, this->ld.model_deepks);
        // 读取投影密度矩阵
        DeePKS_domain::read_pdm(...);
    }
}
```

2. **SCF迭代**:
   - 每次SCF迭代中：
     a. 计算描述符 `D = eig(pdm)`
     b. 将描述符输入神经网络: `E_delta = model(D)`
     c. 计算哈密顿量修正: `V_delta = dE/dD`
     d. 更新总哈密顿量: `H = H_base + V_delta`
     e. 对角化得到新的密度矩阵

3. **能量计算** (`setup_deepks.cpp`):
```cpp
void Setup_DeePKS<TK>::delta_e(...) {
    if (inp.deepks_scf) {
        this->ld.dpks_cal_e_delta_band(dm_vec, kv.get_nks());
        DeePKS_domain::update_dmr(kv.kvec_d, dm_vec, ucell, orb, pv, gd, this->ld.dm_r);
        f_en.edeepks_scf = this->ld.E_delta - this->ld.e_delta_band;
        f_en.edeepks_delta = this->ld.E_delta;
    }
}
```

**关键C++函数**:
- `LCAO_Deepks::dpks_cal_e_delta_band()`: 计算能带能量修正
- `LCAO_Deepks::cal_descriptor()`: 计算描述符
- `LCAO_Deepks::cal_edelta_gedm()`: 计算能量和梯度

### 4.2 训练步骤 (01.train) - 重启训练

与初始化训练的主要区别：
- `restart=True`: 从已有模型继续训练
- `preshift=false`, `prescale=false`: 不重新计算平移和缩放
- 学习率可能根据`decay_rate_iter`调整

**训练参数** (来自params.yaml):
```yaml
train_args:
  n_epoch: 5000
  start_lr: 0.0001  # 比初始化学习率低
  decay_rate: 0.5
  decay_steps: 1000
```

---

## 五、工作流管理

### 5.1 Workflow类
- **文件**: `/home/linearline/project/DeePKS-L/deepks/task/workflow.py`
- **类**: `Workflow`, `Sequence`, `Iteration`
- **作用**: 管理任务执行顺序和依赖关系

**关键方法**:
```python
class Workflow(AbstructStep):
    def run(self, parent_tag=(), restart_tag=None):
        # 从断点恢复或从头开始
        if restart_tag is not None:
            start_idx = restart_tag[0] + 1
        # 依次执行子任务
        for i in range(start_idx, len(self.child_tasks)):
            task = self.child_tasks[i]
            task.run(curr_tag)
            self.write_record(curr_tag)
    
    def restart(self):
        # 读取RECORD文件，恢复执行
        with self.record_file.open() as lf:
            all_tags = [tuple(map(int, l.split())) for l in lf.readlines()]
        restart_tag = all_tags[-1]
        self.run((), restart_tag=restart_tag)
```

### 5.2 Task类
- **文件**: `/home/linearline/project/DeePKS-L/deepks/task/task.py`
- **类**: `PythonTask`, `BatchTask`, `GroupBatchTask`
- **作用**: 执行具体计算任务

**任务类型**:
1. **PythonTask**: 执行Python函数
2. **BatchTask**: 批量执行Shell命令
3. **GroupBatchTask**: 分组批量执行
4. **DPDispatcherTask**: 使用dpdispatcher提交任务

**任务生命周期**:
```python
class AbstructTask(AbstructStep):
    def run(self, *args, **kwargs):
        self.preprocess()   # 准备工作目录和文件
        self.olddir = os.getcwd()
        os.chdir(self.workdir)
        self.execute()      # 执行任务
        os.chdir(self.olddir)
        self.postprocess()  # 后处理
```

---

## 六、数据流总结

### 6.1 输入数据

**系统数据** (`systems/group.00/`):
- `atom.npy`: 原子坐标 (nframes, natoms, 4)
- `energy.npy`: 参考能量
- `gev.npy`: 几何信息

**配置文件**:
- `params.yaml`: 迭代和训练参数
- `systems.yaml`: 训练和测试系统路径
- `machines.yaml`: 计算资源配置
- `scf_abacus.yaml`: ABACUS SCF参数

**物理文件**:
- `H_gga_10au_100Ry_3s2p.orb`: 数值轨道
- `H_ONCV_PBE-1.0.upf`: 赝势
- `jle.orb`: 描述符投影基组

### 6.2 中间数据

**ABACUS输出** (`systems/group.00/ABACUS/0/OUT.ABACUS/`):
- `deepks_desc.dat`: 描述符
- `deepks_projdm.dat`: 投影密度矩阵
- `deepks_ebase.npy`: 基础能量
- `deepks_etot.npy`: 总能量
- `deepks_dm_eig.npy`: 密度矩阵本征值

**训练数据** (`iter/iter.init/00.scf/data_train/group.00/`):
- `atom.npy`: 原子坐标
- `energy.npy`: 能量
- `e_base.npy`: 基础能量
- `e_tot.npy`: 总能量
- `l_e_delta.npy`: 能量修正标签 (e_tot - e_base)
- `dm_eig.npy`: 描述符
- `conv.npy`: 收敛标志

### 6.3 输出数据

**模型文件**:
- `iter/iter.init/01.train/model.pth`: 初始模型
- `iter/iter.00/01.train/model.pth`: 迭代后模型

**日志文件**:
- `iter/log.iter`: 迭代日志
- `iter/iter.init/00.scf/convert.log`: 转换日志
- `iter/iter.init/01.train/log.train`: 训练日志
- `iter/iter.00/00.scf/log.scf`: SCF日志

**记录文件**:
- `iter/RECORD`: 任务执行记录

---

## 七、关键函数和类汇总

### 7.1 Python端

**主流程**:
- `deepks.main.main_cli()`: 主入口
- `deepks.main.iter_cli()`: 迭代命令处理
- `deepks.iterate.iterate.main()`: 迭代主函数
- `deepks.iterate.iterate.make_iterate()`: 创建迭代工作流

**ABACUS集成**:
- `deepks.iterate.template_abacus.convert_data()`: 数据转换
- `deepks.iterate.template_abacus.make_convert_scf_abacus()`: 创建转换任务
- `deepks.iterate.template_abacus.make_run_scf_abacus()`: 创建运行任务
- `deepks.iterate.template_abacus.gather_stats_abacus()`: 数据收集

**模型训练**:
- `deepks.model.train.main()`: 训练主函数
- `deepks.model.train.train()`: 训练循环
- `deepks.model.model.CorrNet`: 神经网络模型
- `deepks.model.reader.GroupReader`: 数据加载器

**工作流管理**:
- `deepks.task.workflow.Workflow`: 工作流基类
- `deepks.task.workflow.Sequence`: 顺序执行
- `deepks.task.workflow.Iteration`: 迭代执行
- `deepks.task.task.PythonTask`: Python任务
- `deepks.task.task.BatchTask`: 批量任务
- `deepks.task.task.GroupBatchTask`: 分组批量任务

### 7.2 C++端

**DeePKS核心**:
- `Setup_DeePKS::before_runner()`: 初始化
- `Setup_DeePKS::build_overlap()`: 构建重叠积分
- `Setup_DeePKS::delta_e()`: 计算能量修正
- `LCAO_Deepks::init()`: 初始化DeePKS
- `LCAO_Deepks::dpks_cal_e_delta_band()`: 计算能带修正
- `LCAO_Deepks::cal_descriptor()`: 计算描述符
- `LCAO_Deepks::cal_edelta_gedm()`: 计算能量和梯度

**辅助模块**:
- `DeePKS_domain::load_model()`: 加载模型
- `DeePKS_domain::read_pdm()`: 读取投影密度矩阵
- `DeePKS_domain::build_phialpha()`: 构建投影重叠积分
- `LCAO_deepks_io::save_matrix2npy()`: 保存numpy文件

---

## 八、文件调用关系图

```
run.sh
  └─> python -m deepks iterate
        └─> deepks.main.iter_cli()
              └─> deepks.iterate.iterate.main()
                    ├─> deepks.iterate.iterate.make_iterate()
                    │     ├─> deepks.iterate.template_abacus.make_scf_abacus()
                    │     │     ├─> make_convert_scf_abacus()
                    │     │     │     └─> convert_data() [PythonTask]
                    │     │     ├─> make_run_scf_abacus()
                    │     │     │     └─> mpirun abacus [GroupBatchTask]
                    │     │     │           └─> Setup_DeePKS::before_runner()
                    │     │     │           └─> LCAO_Deepks::dpks_cal_e_delta_band()
                    │     │     └─> make_stat_scf_abacus()
                    │     │           └─> gather_stats_abacus() [PythonTask]
                    │     └─> deepks.iterate.template.make_train()
                    │           └─> deepks.model.train.main()
                    │                 └─> deepks.model.train.train()
                    │                       └─> deepks.model.model.CorrNet.forward()
                    └─> deepks.task.workflow.Iteration.run()
                          └─> deepks.task.workflow.Sequence.run()
```

---

## 九、配置参数说明

### 9.1 params.yaml

```yaml
n_iter: 1                    # 迭代次数
workdir: "."                 # 工作目录
share_folder: "share"        # 共享文件夹

# 初始化SCF
init_scf: True              # 执行初始化SCF

# 初始化训练
init_train:
  model_args:
    hidden_sizes: [100, 100, 100]
    output_scale: 100
    use_resnet: true
    actv_fn: mygelu
  data_args:
    batch_size: 16
    group_batch: 1
  preprocess_args:
    preshift: true
    prescale: false
    prefit_ridge: 1e1
  train_args:
    n_epoch: 5000
    start_lr: 0.0003
    decay_rate: 0.96
    decay_steps: 500
    display_epoch: 100

# 迭代训练
train_input:
  data_args:
    batch_size: 16
    group_batch: 1
    extra_label: true
    conv_filter: true
  preprocess_args:
    preshift: false
    prescale: false
    prefit_ridge: 1e1
  train_args:
    n_epoch: 5000
    start_lr: 0.0001
    decay_rate: 0.5
    decay_steps: 1000
    display_epoch: 100
```

### 9.2 scf_abacus.yaml

```yaml
scf_abacus:
  ntype: 1
  nbands: 8
  ecutwfc: 100
  scf_thr: 1e-8
  scf_nmax: 100
  cal_force: 0
  dft_functional: "pbe"
  smearing_method: "gaussian"
  gamma_only: 1
  orb_files: ["H_gga_10au_100Ry_3s2p.orb"]
  pp_files: ["H_ONCV_PBE-1.0.upf"]
  proj_file: ["jle.orb"]
  lattice_constant: 1.8897261258369282
  lattice_vector: [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
  coord_type: "Direct"
  run_cmd: "mpirun"
  abacus_path: "/home/linearline/project/abacus-develop/build/abacus_4p"
```

### 9.3 machines.yaml

```yaml
scf_machine:
  group_size: 125
  resources:
    task_per_node: 1
  sub_size: 1
  dispatcher:
    context: local
    batch: shell

train_machine:
  dispatcher:
    context: local
    batch: shell
  python: "python"

use_abacus: true
```

---

## 十、总结

### 10.1 完整流程

1. **启动**: 执行`run.sh`，调用`python -m deepks iterate`
2. **初始化迭代** (iter.init):
   - SCF计算：不使用模型，生成标签数据
   - 模型训练：使用标签数据训练初始模型
3. **迭代步骤** (iter.00):
   - SCF计算：使用模型进行DeePKS修正
   - 模型训练：使用新数据更新模型
4. **重复迭代**：根据`n_iter`参数重复步骤3

### 10.2 关键技术点

1. **描述符**: 使用投影密度矩阵的本征值作为描述符
2. **神经网络**: 全连接网络，支持残差连接
3. **SCF修正**: 在每次SCF迭代中添加神经网络预测的修正项
4. **自洽迭代**: 通过多次迭代提高模型精度

### 10.3 文件路径汇总

**Python源码**:
- `/home/linearline/project/DeePKS-L/deepks/__main__.py`
- `/home/linearline/project/DeePKS-L/deepks/main.py`
- `/home/linearline/project/DeePKS-L/deepks/iterate/iterate.py`
- `/home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py`
- `/home/linearline/project/DeePKS-L/deepks/model/train.py`
- `/home/linearline/project/DeePKS-L/deepks/model/model.py`
- `/home/linearline/project/DeePKS-L/deepks/task/workflow.py`
- `/home/linearline/project/DeePKS-L/deepks/task/task.py`

**C++源码**:
- `/home/linearline/project/abacus-develop/source/source_lcao/setup_deepks.h`
- `/home/linearline/project/abacus-develop/source/source_lcao/setup_deepks.cpp`
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/LCAO_deepks.h`
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/LCAO_deepks.cpp`

**算例文件**:
- `/home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/run.sh`
- `/home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/params.yaml`
- `/home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/systems.yaml`
- `/home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/machines.yaml`
- `/home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/scf_abacus.yaml`
- `/home/linearline/project/00_hydrogen_abacus/01_H_deepks/systems/group.00/atom.npy`

---

## 附录：RECORD文件解读

RECORD文件记录了任务执行的顺序，用于断点续传：

```
0 0 0    # iter.init, 00.scf, 任务0
0 0 1    # iter.init, 00.scf, 任务1
0 0 2    # iter.init, 00.scf, 任务2
0 0       # iter.init, 00.scf完成
0 1 0    # iter.init, 01.train, 任务0
0 1       # iter.init, 01.train完成
0          # iter.init完成
1 0 0    # iter.00, 00.scf, 任务0
1 0 1    # iter.00, 00.scf, 任务1
1 0 2    # iter.00, 00.scf, 任务2
1 0       # iter.00, 00.scf完成
1 1 0    # iter.00, 01.train, 任务0
1 1       # iter.00, 01.train完成
1          # iter.00完成
```

每行表示一个完成的任务，数字序列表示工作流中的路径。
