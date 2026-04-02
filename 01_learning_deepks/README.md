# DeePKS算例分析总结

## 概述

本分析基于氢原子DeePKS算例，详细解析了从Python启动到C++执行的完整流程。

**算例路径**: `/home/linearline/project/00_hydrogen_abacus/01_H_deepks`
**源码路径**: `/home/linearline/project/DeePKS-L`
**ABACUS路径**: `/home/linearline/project/abacus-develop`

---

## 分析文档列表

### 1. DeePKS算例运行完整分析.md

**内容概要**:
- 算例基本信息和目录结构
- 启动流程和Python调用链
- 初始化迭代（iter.init）的详细步骤
- 迭代步骤（iter.00）的详细步骤
- 工作流管理机制
- 数据流总结
- 关键函数和类汇总
- 配置参数说明
- 完整流程总结

**适合人群**: 想要全面了解DeePKS算例运行流程的用户

### 2. DeePKS调用流程详解.md

**内容概要**:
- 启动流程的详细步骤
- Python模块导入链
- 主函数调用链（4层）
- 初始化迭代执行（iter.init）
  - SCF步骤（00.scf）
    - 数据转换任务
    - 运行ABACUS任务
    - 数据收集任务
  - 训练步骤（01.train）
    - 训练任务
    - 模型定义
    - 训练循环
- 迭代步骤执行（iter.00）
- 完整调用链图
- 关键文件路径

**适合人群**: 想要深入了解代码调用关系和执行顺序的开发者

### 3. DeePKS_C++接口详解.md

**内容概要**:
- C++模块概览
- Setup_DeePKS类详解
  - before_runner()方法
  - build_overlap()方法
  - delta_e()方法
  - write_forces()方法
  - write_stress()方法
- LCAO_Deepks类详解
  - init()方法
  - allocate_V_delta()方法
  - dpks_cal_e_delta_band()方法
- DeePKS_domain命名空间
  - load_model()函数
  - cal_pdm()函数
  - cal_descriptor()函数
  - cal_edelta_gedm()函数
- 数据流总结
- 关键算法
  - 投影密度矩阵计算
  - 描述符计算
  - 哈密顿量修正计算

**适合人群**: 想要深入了解C++实现细节和数学原理的开发者

---

## 核心流程总结

### 启动流程

```
run.sh
  └─> python -m deepks iterate machines.yaml params.yaml systems.yaml scf_abacus.yaml
        └─> deepks.__main__.main_cli()
              └─> deepks.main.iter_cli()
                    └─> deepks.iterate.iterate.main()
                          └─> deepks.iterate.iterate.make_iterate()
                                └─> 创建工作流并执行
```

### 初始化迭代（iter.init）

```
iter.init/
  ├─> 00.scf/
  │     ├─> convert_data() [PythonTask]
  │     │     └─> 创建ABACUS输入文件（INPUT, STRU, KPT）
  │     ├─> run_abacus() [GroupBatchTask]
  │     │     └─> mpirun abacus
  │     │           └─> Setup_DeePKS::before_runner()
  │     │           └─> LCAO_Deepks::dpks_cal_e_delta_band()
  │     │           └─> 输出标签数据
  │     └─> gather_stats_abacus() [PythonTask]
  │           └─> 收集训练数据
  └─> 01.train/
        └─> train() [PythonTask]
              └─> 训练神经网络模型
```

### 迭代步骤（iter.00）

```
iter.00/
  ├─> 00.scf/
  │     ├─> convert_data() [PythonTask]
  │     │     └─> 转换PyTorch模型为TorchScript
  │     ├─> run_abacus() [GroupBatchTask]
  │     │     └─> 使用DeePKS修正的SCF
  │     └─> gather_stats_abacus() [PythonTask]
  └─> 01.train/
        └─> train() [PythonTask]
              └─> 继续训练模型
```

---

## 关键技术点

### 1. 描述符

使用投影密度矩阵的本征值作为描述符：

```
pdm_inl = ∑_k ∑_i occ_i <C_i|alpha_inl> <alpha_inl|C_i>
D_inl = eig(pdm_inl)
```

### 2. 神经网络

全连接网络，支持残差连接：

```
描述符 (eig) 
  -> TraceEmbedding (迹嵌入)
    -> DenseNet [100, 100, 100]
      -> 输出 (能量修正) / 100
```

### 3. SCF修正

在每次SCF迭代中添加神经网络预测的修正项：

```
H = H_base + V_delta
V_delta_mu,nu = ∑_inl gedm_inl * <phi_mu|alpha_inl> <alpha_inl|phi_nu>
```

### 4. 能量修正

```
E_delta = NN(D)  // 神经网络预测
e_delta_band = tr(ρ * H_delta)  // 能带部分
edeepks_scf = E_delta - e_delta_band  // SCF能量修正
```

---

## 主要文件路径

### Python源码

- `/home/linearline/project/DeePKS-L/deepks/__main__.py` - 主入口
- `/home/linearline/project/DeePKS-L/deepks/main.py` - 命令行处理
- `/home/linearline/project/DeePKS-L/deepks/iterate/iterate.py` - 迭代主函数
- `/home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py` - ABACUS集成
- `/home/linearline/project/DeePKS-L/deepks/model/train.py` - 模型训练
- `/home/linearline/project/DeePKS-L/deepks/model/model.py` - 模型定义
- `/home/linearline/project/DeePKS-L/deepks/task/workflow.py` - 工作流管理
- `/home/linearline/project/DeePKS-L/deepks/task/task.py` - 任务管理

### C++源码

- `/home/linearline/project/abacus-develop/source/source_lcao/setup_deepks.h` - DeePKS设置
- `/home/linearline/project/abacus-develop/source/source_lcao/setup_deepks.cpp` - DeePKS设置实现
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/LCAO_deepks.h` - DeePKS核心类
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/LCAO_deepks.cpp` - DeePKS核心实现

### 算例文件

- `/home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/run.sh` - 启动脚本
- `/home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/params.yaml` - 参数配置
- `/home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/systems.yaml` - 系统配置
- `/home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/machines.yaml` - 机器配置
- `/home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter/scf_abacus.yaml` - ABACUS配置

---

## 数据流图

### 输入数据

```
systems/group.00/
  ├─> atom.npy (原子坐标)
  ├─> energy.npy (参考能量)
  └─> gev.npy (几何信息)

iter/
  ├─> params.yaml (迭代参数)
  ├─> systems.yaml (系统路径)
  ├─> machines.yaml (资源配置)
  ├─> scf_abacus.yaml (ABACUS参数)
  ├─> H_gga_10au_100Ry_3s2p.orb (轨道文件)
  ├─> H_ONCV_PBE-1.0.upf (赝势文件)
  └─> jle.orb (投影基组)
```

### 中间数据

```
ABACUS输出:
  ├─> deepks_dm_eig.npy (描述符)
  ├─> deepks_ebase.npy (基础能量)
  ├─> deepks_etot.npy (总能量)
  ├─> deepks_projdm.dat (投影密度矩阵)
  └─> conv (收敛标志)

训练数据:
  ├─> atom.npy (原子坐标)
  ├─> dm_eig.npy (描述符)
  ├─> e_base.npy (基础能量)
  ├─> e_tot.npy (总能量)
  └─> l_e_delta.npy (能量修正标签)
```

### 输出数据

```
模型文件:
  ├─> iter.init/01.train/model.pth (初始模型)
  └─> iter.00/01.train/model.pth (迭代后模型)

日志文件:
  ├─> iter/log.iter (迭代日志)
  ├─> iter.init/00.scf/convert.log (转换日志)
  ├─> iter.init/01.train/log.train (训练日志)
  └─> iter.00/00.scf/log.scf (SCF日志)

记录文件:
  └─> iter/RECORD (任务执行记录)
```

---

## 配置参数说明

### params.yaml

```yaml
n_iter: 1                    # 迭代次数
init_scf: True              # 执行初始化SCF
init_train:                 # 初始化训练参数
  ├─> model_args:
  │     ├─> hidden_sizes: [100, 100, 100]
  │     ├─> output_scale: 100
  │     ├─> use_resnet: true
  │     └─> actv_fn: mygelu
  ├─> data_args:
  │     ├─> batch_size: 16
  │     └─> group_batch: 1
  ├─> preprocess_args:
  │     ├─> preshift: true
  │     ├─> prescale: false
  │     └─> prefit_ridge: 1e1
  └─> train_args:
        ├─> n_epoch: 5000
        ├─> start_lr: 0.0003
        ├─> decay_rate: 0.96
        └─> decay_steps: 500
```

### scf_abacus.yaml

```yaml
scf_abacus:
  ├─> ntype: 1
  ├─> nbands: 8
  ├─> ecutwfc: 100
  ├─> scf_thr: 1e-8
  ├─> scf_nmax: 100
  ├─> dft_functional: "pbe"
  ├─> gamma_only: 1
  ├─> orb_files: ["H_gga_10au_100Ry_3s2p.orb"]
  ├─> pp_files: ["H_ONCV_PBE-1.0.upf"]
  ├─> proj_file: ["jle.orb"]
  ├─> lattice_constant: 1.8897261258369282
  ├─> lattice_vector: [[10, 0, 0], [0, 10, 0], [0, 0, 10]]
  ├─> coord_type: "Direct"
  ├─> run_cmd: "mpirun"
  └─> abacus_path: "/path/to/abacus_4p"
```

---

## 关键类和函数

### Python端

**主流程**:
- `deepks.main.main_cli()` - 主入口
- `deepks.main.iter_cli()` - 迭代命令处理
- `deepks.iterate.iterate.main()` - 迭代主函数
- `deepks.iterate.iterate.make_iterate()` - 创建迭代工作流

**ABACUS集成**:
- `deepks.iterate.template_abacus.convert_data()` - 数据转换
- `deepks.iterate.template_abacus.make_convert_scf_abacus()` - 创建转换任务
- `deepks.iterate.template_abacus.make_run_scf_abacus()` - 创建运行任务
- `deepks.iterate.template_abacus.gather_stats_abacus()` - 数据收集

**模型训练**:
- `deepks.model.train.main()` - 训练主函数
- `deepks.model.train.train()` - 训练循环
- `deepks.model.model.CorrNet` - 神经网络模型
- `deepks.model.reader.GroupReader` - 数据加载器

**工作流管理**:
- `deepks.task.workflow.Workflow` - 工作流基类
- `deepks.task.workflow.Sequence` - 顺序执行
- `deepks.task.workflow.Iteration` - 迭代执行
- `deepks.task.task.PythonTask` - Python任务
- `deepks.task.task.BatchTask` - 批量任务
- `deepks.task.task.GroupBatchTask` - 分组批量任务

### C++端

**DeePKS核心**:
- `Setup_DeePKS::before_runner()` - 初始化
- `Setup_DeePKS::build_overlap()` - 构建重叠积分
- `Setup_DeePKS::delta_e()` - 计算能量修正
- `LCAO_Deepks::init()` - 初始化DeePKS
- `LCAO_Deepks::dpks_cal_e_delta_band()` - 计算能带修正
- `LCAO_Deepks::cal_descriptor()` - 计算描述符
- `LCAO_Deepks::cal_edelta_gedm()` - 计算能量和梯度

**辅助模块**:
- `DeePKS_domain::load_model()` - 加载模型
- `DeePKS_domain::read_pdm()` - 读取投影密度矩阵
- `DeePKS_domain::build_phialpha()` - 构建投影重叠积分
- `LCAO_deepks_io::save_matrix2npy()` - 保存numpy文件

---

## 使用建议

### 对于初学者

1. 先阅读 **DeePKS算例运行完整分析.md**，了解整体流程
2. 关注 **核心流程总结** 部分，理解主要步骤
3. 查看 **数据流图**，了解输入输出

### 对于开发者

1. 阅读 **DeePKS调用流程详解.md**，深入了解代码调用关系
2. 关注 **Python调用链** 部分，理解函数调用顺序
3. 查看 **关键文件路径**，快速定位源码

### 对于算法研究者

1. 阅读 **DeePKS_C++接口详解.md**，了解C++实现细节
2. 关注 **关键算法** 部分，理解数学原理
3. 查看 **数据流总结**，了解数据转换过程

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

---

## 总结

本分析文档详细解析了DeePKS算例的完整运行过程，包括：

1. **Python层面的调用链**：从命令行启动到工作流执行
2. **C++层面的实现细节**：从模型加载到SCF修正
3. **数据流的完整路径**：从输入数据到输出结果
4. **关键算法的数学原理**：描述符计算、神经网络、SCF修正

通过这三个文档，读者可以：
- 理解DeePKS的整体工作流程
- 深入了解代码实现细节
- 掌握关键算法的数学原理
- 快速定位和修改相关代码

建议根据自身需求选择合适的文档进行阅读。
