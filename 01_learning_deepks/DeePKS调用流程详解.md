# DeePKS调用流程详解

## 一、启动流程

### 1.1 命令行调用

```bash
cd /home/linearline/project/00_hydrogen_abacus/01_H_deepks/iter
nohup python -u -m deepks iterate machines.yaml params.yaml systems.yaml scf_abacus.yaml >> log.iter 2> err.iter &
```

### 1.2 Python模块导入链

```
deepks.__main__
  └─> import deepks
      └─> from deepks.main import main_cli
```

---

## 二、主函数调用链

### 2.1 第一层：命令行解析

**文件**: `/home/linearline/project/DeePKS-L/deepks/__main__.py`

```python
import os
import sys
try:
    import deepks
except ImportError as e:
    sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../")

from deepks.main import main_cli

if __name__ == "__main__":
    main_cli()
```

**函数**: `main_cli(args=None)`

**输入参数**:
- `args.command`: "iterate"
- `args.args`: ["machines.yaml", "params.yaml", "systems.yaml", "scf_abacus.yaml"]

**输出**: 路由到`iter_cli()`

---

### 2.2 第二层：迭代命令处理

**文件**: `/home/linearline/project/DeePKS-L/deepks/main.py`

**函数**: `iter_cli(args=None)`

**代码流程**:

```python
def iter_cli(args=None):
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(
        prog="deepks iterate",
        description="Run the iteration procedure to train a SCF model.")
    
    # 2. 添加参数
    parser.add_argument("argfile", nargs="*", default=[])
    parser.add_argument("--systems-train", nargs="*")
    parser.add_argument("--systems-test", nargs="*")
    parser.add_argument("--n-iter", type=int)
    parser.add_argument("--workdir")
    parser.add_argument("--share-folder")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--no-strict", action="store_false")
    parser.add_argument("--scf-input")
    parser.add_argument("--scf-machine")
    parser.add_argument("--train-input")
    parser.add_argument("--train-machine")
    parser.add_argument("--init-model")
    parser.add_argument("--init-scf")
    parser.add_argument("--init-train")
    parser.add_argument("--scf-abacus")
    
    # 3. 解析参数
    args = parser.parse_args(args)
    
    # 4. 加载YAML配置文件
    argdict = {}
    for fl in args.argfile:
        argdict = deep_update(argdict, load_yaml(fl))
    del args.argfile
    argdict.update(vars(args))
    
    # 5. 调用迭代主函数
    from deepks.iterate.iterate import main
    main(**argdict)
```

**输入参数** (解析后):
```python
{
    'systems_train': ['../systems/group.00'],
    'systems_test': None,
    'n_iter': 1,
    'workdir': '.',
    'share_folder': 'share',
    'cleanup': False,
    'strict': True,
    'scf_input': False,
    'scf_machine': {
        'group_size': 125,
        'resources': {'task_per_node': 1},
        'sub_size': 1,
        'dispatcher': {'context': 'local', 'batch': 'shell'}
    },
    'train_input': {
        'data_args': {
            'batch_size': 16,
            'group_batch': 1,
            'extra_label': True,
            'conv_filter': True,
            'conv_name': 'conv'
        },
        'preprocess_args': {
            'preshift': False,
            'prescale': False,
            'prefit_ridge': 10.0,
            'prefit_trainable': False
        },
        'train_args': {
            'decay_rate': 0.5,
            'decay_steps': 1000,
            'display_epoch': 100,
            'force_factor': 1,
            'n_epoch': 5000,
            'start_lr': 0.0001
        }
    },
    'init_model': False,
    'init_scf': True,
    'init_train': {
        'model_args': {
            'hidden_sizes': [100, 100, 100],
            'output_scale': 100,
            'use_resnet': True,
            'actv_fn': 'mygelu'
        },
        'data_args': {
            'batch_size': 16,
            'group_batch': 1
        },
        'preprocess_args': {
            'preshift': True,
            'prescale': False,
            'prefit_ridge': 10.0,
            'prefit_trainable': False
        },
        'train_args': {
            'decay_rate': 0.96,
            'decay_steps': 500,
            'display_epoch': 100,
            'n_epoch': 5000,
            'start_lr': 0.0003
        }
    },
    'use_abacus': True,
    'scf_abacus': {
        'ntype': 1,
        'nbands': 8,
        'ecutwfc': 100,
        'scf_thr': 1e-8,
        'scf_nmax': 100,
        'cal_force': 0,
        'dft_functional': 'pbe',
        'smearing_method': 'gaussian',
        'gamma_only': 1,
        'orb_files': ['H_gga_10au_100Ry_3s2p.orb'],
        'pp_files': ['H_ONCV_PBE-1.0.upf'],
        'proj_file': ['jle.orb'],
        'lattice_constant': 1.8897261258369282,
        'lattice_vector': [[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        'coord_type': 'Direct',
        'run_cmd': 'mpirun',
        'abacus_path': '/home/linearline/project/abacus-develop/build/abacus_4p'
    }
}
```

**输出**: 调用`deepks.iterate.iterate.main(**argdict)`

---

### 2.3 第三层：迭代主函数

**文件**: `/home/linearline/project/DeePKS-L/deepks/iterate/iterate.py`

**函数**: `main(*args, **kwargs)`

**代码流程**:

```python
def main(*args, **kwargs):
    # 1. 创建迭代工作流
    iterate = make_iterate(*args, **kwargs)
    
    # 2. 检查是否有记录文件
    if os.path.exists(iterate.record_file):
        # 从断点恢复
        iterate.restart()
    else:
        # 从头开始执行
        iterate.run()
```

**输入**: 同`iter_cli()`的`argdict`

**输出**: `Iteration`对象并执行

---

### 2.4 第四层：创建迭代工作流

**文件**: `/home/linearline/project/DeePKS-L/deepks/iterate/iterate.py`

**函数**: `make_iterate(systems_train=None, systems_test=None, n_iter=0, ...)`

**详细步骤**:

#### 步骤1: 收集训练系统

```python
# 加载训练系统路径
if systems_train is None:
    default_train = os.path.join(share_folder, DEFAULT_TRAIN)
    assert_exist(default_train)
    systems_train = default_train

# 收集系统到共享文件夹
systems_train = collect_systems(systems_train, 
                                os.path.join(share_folder, SYS_TRAIN))
```

**函数**: `collect_systems(systems, folder=None)`

**作用**: 
- 将系统路径转换为绝对路径
- 处理重复的系统名
- 创建符号链接到共享文件夹

**输出**: 
- `share/systems_train/group.00` -> `../systems/group.00`

#### 步骤2: 收集测试系统

```python
if systems_test is None:
    default_test = os.path.join(share_folder, DEFAULT_TEST)
    if os.path.exists(default_test):
        systems_test = default_test
    else:
        # 使用训练集的最后一个系统作为测试集
        systems_test = systems_train[-1]

systems_test = collect_systems(systems_test, 
                                os.path.join(share_folder, SYS_TEST))
```

**输出**: 
- `share/systems_test/group.00` -> `../systems/group.00`

#### 步骤3: 检查配置文件

```python
# 检查SCF配置
scf_args_name = check_share_folder(scf_input, SCF_ARGS_NAME, share_folder)

# 检查训练配置
train_args_name = check_share_folder(train_input, TRN_ARGS_NAME, share_folder)
```

**函数**: `check_share_folder(data, name, share_folder="share")`

**作用**: 
- 如果`data`是`True`，检查文件是否存在
- 如果`data`是字符串，复制文件到共享文件夹
- 如果`data`是字典，保存为YAML文件

**输出**: 返回文件名或None

#### 步骤4: 检查机器参数

```python
scf_machine = check_arg_dict(scf_machine, DEFAULT_SCF_MACHINE, strict)
train_machine = check_arg_dict(train_machine, DEFAULT_TRN_MACHINE, strict)
```

**函数**: `check_arg_dict(data, default, strict=True)`

**作用**: 
- 合并用户参数和默认参数
- 如果`strict=True`，只保留默认参数中的键

**输出**: 完整的参数字典

#### 步骤5: 创建SCF步骤 (使用ABACUS)

```python
if use_abacus:
    # 检查ABACUS配置文件
    scf_abacus_name = check_share_folder(scf_abacus, SCF_ARGS_NAME_ABACUS, share_folder)
    scf_abacus = check_arg_dict(scf_abacus, DEFAULT_SCF_ARGS_ABACUS, strict)
    scf_abacus = dict(scf_abacus, **scf_machine)
    
    # 创建SCF步骤
    scf_step = make_scf_abacus(
        systems_train=systems_train,
        systems_test=systems_test,
        train_dump=DATA_TRAIN,
        test_dump=DATA_TEST,
        no_model=False,
        model_file=MODEL_FILE,
        workdir=SCF_STEP_DIR,
        share_folder=share_folder,
        cleanup=cleanup,
        **scf_abacus
    )
```

**函数**: `make_scf_abacus(...)` (在`template_abacus.py`中)

**作用**: 创建ABACUS SCF工作流

**输出**: `Sequence`对象，包含三个子任务

#### 步骤6: 创建训练步骤

```python
train_step = make_train(
    source_train=DATA_TRAIN,
    source_test=DATA_TEST,
    restart=True,
    source_model=MODEL_FILE,
    save_model=MODEL_FILE,
    source_pbasis=proj_basis,
    source_arg=train_args_name,
    workdir=TRN_STEP_DIR,
    share_folder=share_folder,
    cleanup=cleanup,
    **train_machine
)
```

**函数**: `make_train(...)` (在`template.py`中)

**作用**: 创建模型训练任务

**输出**: `PythonTask`对象

#### 步骤7: 组合迭代步骤

```python
per_iter = Sequence([scf_step, train_step])
iterate = Iteration(per_iter, n_iter, 
                    workdir=".", 
                    record_file=os.path.join(workdir, RECORD))
```

**类**: `Iteration` (在`workflow.py`中)

**作用**: 创建迭代工作流

**输出**: `Iteration`对象

#### 步骤8: 创建初始化步骤

```python
if init_model:
    # 使用已有模型
    init_folder = os.path.join(share_folder, "init")
    check_share_folder(init_model, MODEL_FILE, init_folder)
    iterate.set_init_folder(init_folder)
elif init_scf or init_train:
    # 创建初始化迭代
    init_scf_machine = (check_arg_dict(init_scf_machine, DEFAULT_SCF_MACHINE, strict)
                        if init_scf_machine is not None else scf_machine)
    
    if use_abacus:
        # 创建初始化SCF（不使用模型）
        scf_init = make_scf_abacus(
            systems_train=systems_train,
            systems_test=systems_test,
            train_dump=DATA_TRAIN,
            test_dump=DATA_TEST,
            no_model=True,
            workdir=SCF_STEP_DIR,
            share_folder=share_folder,
            model_file=None,
            cleanup=cleanup,
            **init_scf_abacus
        )
    
    # 创建初始化训练
    init_train_name = check_share_folder(init_train, INIT_TRN_NAME, share_folder)
    init_train_machine = (check_arg_dict(init_train_machine, DEFAULT_TRN_MACHINE, strict)
                        if init_train_machine is not None else train_machine)
    
    train_init = make_train(
        source_train=DATA_TRAIN,
        source_test=DATA_TEST,
        restart=False,
        source_model=MODEL_FILE,
        save_model=MODEL_FILE,
        source_pbasis=proj_basis,
        source_arg=init_train_name,
        workdir=TRN_STEP_DIR,
        share_folder=share_folder,
        cleanup=cleanup,
        **train_machine
    )
    
    # 组合初始化步骤
    init_iter = Sequence([scf_init, train_init], workdir="iter.init")
    iterate.prepend(init_iter)
```

**输出**: 在迭代工作流前添加初始化步骤

**最终工作流结构**:
```
Iteration
  └─> Sequence (iter.init)
        ├─> Sequence (00.scf)
        │     ├─> PythonTask: convert_data
        │     ├─> GroupBatchTask: run_abacus
        │     └─> PythonTask: gather_stats_abacus
        └─> PythonTask: train_model
  └─> Iteration (n_iter=1)
        └─> Sequence (iter.00)
              ├─> Sequence (00.scf)
              │     ├─> PythonTask: convert_data
              │     ├─> GroupBatchTask: run_abacus
              │     └─> PythonTask: gather_stats_abacus
              └─> PythonTask: train_model
```

---

## 三、初始化迭代执行 (iter.init)

### 3.1 工作流执行

**文件**: `/home/linearline/project/DeePKS-L/deepks/task/workflow.py`

**类**: `Iteration`

**方法**: `run(parent_tag=(), restart_tag=None)`

**代码流程**:

```python
def run(self, parent_tag=(), restart_tag=None):
    # 1. 检查是否需要恢复
    start_idx = 0
    if restart_tag is not None:
        last_idx = restart_tag[0]
        rest_tag = restart_tag[1:]
        if rest_tag:
            # 恢复子任务
            self.child_tasks[last_idx].run(last_tag, restart_tag=rest_tag)
        else:
            start_idx = last_idx + 1
    
    # 2. 执行初始化步骤
    for i in range(start_idx, len(self.child_tasks)):
        curr_tag = parent_tag + (i,)
        task = self.child_tasks[i]
        task.run(curr_tag)
        self.write_record(curr_tag)
    
    # 3. 执行迭代步骤
    self.per_iter.run((0,))
```

**输出**: 执行所有任务并记录到RECORD文件

---

### 3.2 SCF步骤 (00.scf)

#### 3.2.1 数据转换任务

**文件**: `/home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py`

**函数**: `make_convert_scf_abacus(...)`

**返回**: `PythonTask`对象

**任务配置**:
```python
PythonTask(
    convert_data,
    call_kwargs=pre_args,
    outlog="convert.log",
    errlog="err",
    workdir='.',
    link_prev_files=link_prev
)
```

**执行函数**: `convert_data(...)`

**详细步骤**:

```python
def convert_data(systems_train, systems_test=None,
                no_model=True, model_file=None, pp_files=[],
                dispatcher=None, **pre_args):
    
    # 1. 如果需要使用模型，进行模型追踪
    if not no_model:
        if model_file is not None:
            from deepks.model import CorrNet
            model = CorrNet.load(model_file)
            model.compile_save(CMODEL_FILE)
            pre_args.update(deepks_scf=1, 
                          model_file=os.path.abspath(CMODEL_FILE))
    
    # 2. 分割系统
    nsys_trn = len(systems_train)
    nsys_tst = len(systems_test)
    train_sets = [systems_train[i::nsys_trn] for i in range(nsys_trn)]
    test_sets = [systems_test[i::nsys_tst] for i in range(nsys_tst)]
    systems = systems_train + systems_test
    sys_paths = [os.path.abspath(s) for s in load_sys_paths(systems)]
    
    # 3. 为每个系统创建ABACUS输入文件
    for i, sset in enumerate(train_sets + test_sets):
        # 读取原子数据
        try:
            atom_data = np.load(f"{sys_paths[i]}/atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(sys_paths[i])
        
        # 检查晶胞数据
        if os.path.isfile(f"{sys_paths[i]}/box.npy"):
            cell_data = np.load(f"{sys_paths[i]}/box.npy")
        
        nframes = atom_data.shape[0]
        
        # 创建ABACUS目录
        if not os.path.exists(f"{sys_paths[i]}/ABACUS"):
            os.mkdir(f"{sys_paths[i]}/ABACUS")
        
        # 更新参数
        pre_args_new = dict(zip(pre_args.keys(), pre_args.values()))
        if os.path.exists(f"{sys_paths[i]}/group_scf_abacus.yaml"):
            stru_abacus = load_yaml(f"{sys_paths[i]}/group_scf_abacus.yaml")
            for k, v in stru_abacus.items():
                pre_args_new[k] = v
        
        # 为每一帧创建输入文件
        for f in range(nframes):
            # 创建帧目录
            if not os.path.exists(f"{sys_paths[i]}/ABACUS/{f}"):
                os.mkdir(f"{sys_paths[i]}/ABACUS/{f}")
            
            # 创建STRU文件
            Path(f"{sys_paths[i]}/ABACUS/{f}/STRU").touch()
            frame_data = atom_data[f]
            atoms = atom_data[f, :, 0]
            nta = Counter(atoms)
            sys_data = {
                'atom_names': [TYPE_NAME[it] for it in nta.keys()],
                'atom_numbs': list(nta.values()),
                'cells': np.array([pre_args_new["lattice_vector"]]),
                'coords': [frame_data[:, 1:]]
            }
            if os.path.isfile(f"{sys_paths[i]}/box.npy"):
                sys_data['cells'] = [cell_data[f]]
            
            with open(f"{sys_paths[i]}/ABACUS/{f}/STRU", "w") as stru_file:
                stru_file.write(make_abacus_scf_stru(sys_data, pp_files, pre_args_new))
            
            # 创建INPUT文件
            with open(f"{sys_paths[i]}/ABACUS/{f}/INPUT", "w") as input_file:
                input_file.write(make_abacus_scf_input(pre_args_new))
            
            # 创建KPT文件
            if pre_args_new["k_points"] is not None or pre_args_new["gamma_only"] == True:
                with open(f"{sys_paths[i]}/ABACUS/{f}/KPT", "w") as kpt_file:
                    kpt_file.write(make_abacus_scf_kpt(pre_args_new))
```

**输入**:
- `systems_train`: `['share/systems_train/group.00']`
- `systems_test`: `['share/systems_test/group.00']`
- `no_model`: True (初始化SCF不使用模型)
- `orb_files`: `['H_gga_10au_100Ry_3s2p.orb']`
- `pp_files`: `['H_ONCV_PBE-1.0.upf']`
- `proj_file`: `['jle.orb']`

**输出**:
- `systems/group.00/ABACUS/0/INPUT`
- `systems/group.00/ABACUS/0/STRU`
- `systems/group.00/ABACUS/0/KPT`

**日志**: `iter.init/00.scf/convert.log`

---

#### 3.2.2 运行ABACUS任务

**文件**: `/home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py`

**函数**: `make_run_scf_abacus(...)`

**返回**: `GroupBatchTask`对象

**任务配置**:
```python
GroupBatchTask(
    batch_tasks=[
        BatchTask(
            cmds="cd systems/group.00/ABACUS/0/ && \
                   mpirun -n 1 /path/to/abacus_4p > log.scf 2>err.log && \
                   echo 0`grep -i converge ./OUT.ABACUS/running_scf.log` > conv && \
                   echo 0`grep -i converge ./OUT.ABACUS/running_scf.log`",
            workdir="systems",
            forward_files=["./group.00/ABACUS/0/"],
            backward_files=["./group.00/ABACUS/0/"]
        )
    ],
    group_size=125,
    workdir="./",
    dispatcher={'context': 'local', 'batch': 'shell'},
    resources={'task_per_node': 1},
    outlog="log.scf",
    share_folder="share",
    forward_files=[...],
    backward_files=[...]
)
```

**执行命令**:
```bash
cd systems/group.00/ABACUS/0/ && \
mpirun -n 1 /home/linearline/project/abacus-develop/build/abacus_4p > log.scf 2>err.log && \
echo 0`grep -i converge ./OUT.ABACUS/running_scf.log` > conv && \
echo 0`grep -i converge ./OUT.ABACUS/running_scf.log`
```

**ABACUS执行流程** (C++):

1. **初始化阶段** (`setup_deepks.cpp`):

```cpp
void Setup_DeePKS<TK>::before_runner(
    const UnitCell &ucell,
    const int nks,
    const LCAO_Orbitals &orb,
    Parallel_Orbitals &pv,
    const Input_para &inp)
{
#ifdef __MLALGO
    // 初始化DeePKS
    LCAO_domain::DeePKS_init(ucell, pv, nks, orb, this->ld, GlobalV::ofs_running);
    
    if (inp.deepks_scf) {
        // 加载模型
        DeePKS_domain::load_model(inp.deepks_model, this->ld.model_deepks);
        
        // 读取投影密度矩阵
        DeePKS_domain::read_pdm(
            (inp.init_chg == "file"),
            inp.deepks_equiv,
            this->ld.init_pdm,
            ucell.nat,
            this->ld.deepks_param,
            *orb.Alpha,
            this->ld.pdm
        );
    }
#endif
}
```

2. **构建重叠积分** (`setup_deepks.cpp`):

```cpp
void Setup_DeePKS<TK>::build_overlap(
    const UnitCell &ucell,
    const LCAO_Orbitals &orb,
    const Parallel_Orbitals &pv,
    const Grid_Driver &gd,
    TwoCenterIntegrator &overlap_orb_alpha,
    const Input_para &inp)
{
#ifdef __MLALGO
    if (PARAM.globalv.deepks_setorb) {
        // 分配 <phi(0)|alpha(R)>
        DeePKS_domain::allocate_phialpha(
            inp.cal_force, ucell, orb, gd, &pv, this->ld.phialpha
        );
        
        // 构建 <phi(0)|alpha(R)>
        DeePKS_domain::build_phialpha(
            inp.cal_force, ucell, orb, gd, &pv, 
            overlap_orb_alpha, this->ld.phialpha
        );
    }
#endif
}
```

3. **SCF迭代** (在ABACUS主循环中):

```cpp
// 每次SCF迭代
for (int iter = 0; iter < scf_nmax; iter++) {
    // 1. 计算投影密度矩阵
    DeePKS_domain::cal_pdm(...);
    
    // 2. 计算描述符
    DeePKS_domain::cal_descriptor(...);
    
    // 3. 如果deepks_scf=1，使用神经网络修正
    if (inp.deepks_scf) {
        // 将描述符输入神经网络
        this->ld.dpks_cal_e_delta_band(dm_vec, kv.get_nks());
        
        // 计算哈密顿量修正
        DeePKS_domain::cal_edelta_gedm(...);
        
        // 更新哈密顿量
        for (int ik = 0; ik < nks; ik++) {
            for (int i = 0; i < nbands; i++) {
                H[ik][i] += this->ld.V_delta[ik][i];
            }
        }
    }
    
    // 4. 对角化哈密顿量
    diagonalize(H, C, eig);
    
    // 5. 更新密度矩阵
    update_dm(C, eig);
    
    // 6. 检查收敛
    if (check_convergence()) break;
}
```

4. **计算能量修正** (`setup_deepks.cpp`):

```cpp
void Setup_DeePKS<TK>::delta_e(
    const UnitCell &ucell,
    const K_Vectors &kv,
    const LCAO_Orbitals &orb,
    const Parallel_Orbitals &pv,
    const Grid_Driver &gd,
    const std::vector<std::vector<TK>>& dm_vec,
    elecstate::fenergy &f_en,
    const Input_para &inp)
{
#ifdef __MLALGO
    if (inp.deepks_scf) {
        this->ld.dpks_cal_e_delta_band(dm_vec, kv.get_nks());
        DeePKS_domain::update_dmr(kv.kvec_d, dm_vec, ucell, orb, pv, gd, this->ld.dm_r);
        f_en.edeepks_scf = this->ld.E_delta - this->ld.e_delta_band;
        f_en.edeepks_delta = this->ld.E_delta;
    }
#endif
}
```

5. **输出标签数据** (在SCF结束后):

```cpp
if (inp.deepks_out_labels) {
    // 输出描述符
    LCAO_deepks_io::save_matrix2npy("deepks_dm_eig.npy", dm_eig, ...);
    
    // 输出基础能量
    LCAO_deepks_io::save_matrix2npy("deepks_ebase.npy", e_base, ...);
    
    // 输出总能量
    LCAO_deepks_io::save_matrix2npy("deepks_etot.npy", e_tot, ...);
    
    // 输出投影密度矩阵
    LCAO_deepks_io::save_matrix2npy("deepks_projdm.dat", pdm, ...);
}
```

**ABACUS输出文件**:
- `OUT.ABACUS/running_scf.log`: SCF迭代日志
- `OUT.ABACUS/deepks_dm_eig.npy`: 描述符
- `OUT.ABACUS/deepks_ebase.npy`: 基础能量
- `OUT.ABACUS/deepks_etot.npy`: 总能量
- `OUT.ABACUS/deepks_projdm.dat`: 投影密度矩阵
- `conv`: 收敛标志

**日志**: `systems/group.00/ABACUS/0/log.scf`

---

#### 3.2.3 数据收集任务

**文件**: `/home/linearline/project/DeePKS-L/deepks/iterate/template_abacus.py`

**函数**: `make_stat_scf_abacus(...)`

**返回**: `PythonTask`对象

**任务配置**:
```python
PythonTask(
    gather_stats_abacus,
    call_kwargs=stat_args,
    outlog="log.data",
    workdir="."
)
```

**执行函数**: `gather_stats_abacus(...)`

**详细步骤**:

```python
def gather_stats_abacus(systems_train, systems_test, 
                      train_dump, test_dump, cal_force=0, cal_stress=0,
                      deepks_bandgap=0, deepks_v_delta=0, **stat_args):
    
    sys_train_paths = [os.path.abspath(s) for s in load_sys_paths(systems_train)]
    sys_test_paths = [os.path.abspath(s) for s in load_sys_paths(systems_test)]
    sys_train_paths = [get_sys_name(s) for s in sys_train_paths]
    sys_test_paths = [get_sys_name(s) for s in sys_test_paths]
    sys_train_names = [os.path.basename(s) for s in sys_train_paths]
    sys_test_names = [os.path.basename(s) for s in sys_test_paths]
    
    # 创建训练数据目录
    if not os.path.exists(train_dump):
        os.mkdir(train_dump)
    
    # 收集训练数据
    for i in range(len(systems_train)):
        load_ref_path = f"{sys_train_paths[i]}/"
        save_path = f"{train_dump}/{sys_train_names[i]}/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        
        # 读取原子数据
        try:
            atom_data = np.load(load_ref_path + "atom.npy")
        except FileNotFoundError:
            atom_data = coord_to_atom(sys_train_paths[i])
        
        nframes = atom_data.shape[0]
        natoms = atom_data.shape[1]
        
        # 初始化属性数组
        conv = np.full((nframes, 1), False)
        dm_eig = None
        e_base = None
        e_tot = None
        
        # 遍历每一帧
        for f in range(nframes):
            load_f_path = f"{sys_train_paths[i]}/ABACUS/{f}/OUT.ABACUS/"
            
            # 读取收敛标志
            with open(f"{sys_train_paths[i]}/ABACUS/{f}/conv", "r") as conv_file:
                ic = conv_file.read().split()
                ic = [item.strip('#') for item in ic]
                if "CONVERGED" in ic and "NOT" not in ic:
                    conv[(int)(ic[0])] = True
            
            # 读取描述符
            des = np.load(load_f_path + "deepks_dm_eig.npy")
            if dm_eig is None:
                dm_eig = np.empty((nframes,) + des.shape, dtype=des.dtype)
            dm_eig[f] = des
            
            # 读取基础能量
            ene = np.load(load_f_path + "deepks_ebase.npy")
            if e_base is None:
                e_base = np.empty((nframes,) + ene.shape, dtype=ene.dtype)
            e_base[f] = ene
            
            # 读取总能量
            ene = np.load(load_f_path + "deepks_etot.npy")
            if e_tot is None:
                e_tot = np.empty((nframes,) + ene.shape, dtype=ene.dtype)
            e_tot[f] = ene
        
        # 保存训练数据
        np.save(save_path + "atom.npy", atom_data)
        np.save(save_path + "conv.npy", conv)
        np.save(save_path + "dm_eig.npy", dm_eig)
        np.save(save_path + "e_base.npy", e_base)
        np.save(save_path + "e_tot.npy", e_tot)
        np.save(save_path + "l_e_delta.npy", e_tot - e_base)
    
    # 类似地收集测试数据...
```

**输入**:
- `systems_train`: `['systems/group.00']`
- `train_dump`: `"data_train"`

**输出**:
- `data_train/group.00/atom.npy`: 原子坐标
- `data_train/group.00/conv.npy`: 收敛标志
- `data_train/group.00/dm_eig.npy`: 描述符
- `data_train/group.00/e_base.npy`: 基础能量
- `data_train/group.00/e_tot.npy`: 总能量
- `data_train/group.00/l_e_delta.npy`: 能量修正标签

**日志**: `iter.init/00.scf/log.data`

---

### 3.3 训练步骤 (01.train)

#### 3.3.1 训练任务

**文件**: `/home/linearline/project/DeePKS-L/deepks/iterate/template.py`

**函数**: `make_train(...)`

**返回**: `PythonTask`对象

**任务配置**:
```python
PythonTask(
    train,
    call_kwargs={
        'train_paths': ['data_train'],
        'test_paths': ['data_test'],
        'restart': False,
        'ckpt_file': 'model.pth',
        'model_args': {...},
        'data_args': {...},
        'preprocess_args': {...},
        'train_args': {...}
    },
    outlog="log.train",
    errlog="err",
    workdir="."
)
```

**执行函数**: `train(...)` (在`model/train.py`中)

**详细步骤**:

```python
def train(train_paths, test_paths=None,
         restart=None, ckpt_file=None,
         model_args=None, data_args=None,
         preprocess_args=None, train_args=None,
         proj_basis=None, fit_elem=False, ...):
    
    # 1. 加载数据
    g_reader = GroupReader(
        train_paths,
        batch_size=data_args['batch_size'],
        group_batch=data_args['group_batch'],
        shuffle=True
    )
    
    if test_paths is not None:
        test_reader = GroupReader(test_paths, ...)
    else:
        test_reader = g_reader
    
    # 2. 创建模型
    model = CorrNet(
        ndesc=g_reader.readers[0].sample_all()['eig'].shape[-1],
        **model_args
    )
    
    # 3. 数据预处理
    if preprocess_args is not None:
        model, g_reader, test_reader = preprocess(
            model, g_reader, test_reader,
            **preprocess_args
        )
    
    # 4. 训练模型
    train(model, g_reader, n_epoch=train_args['n_epoch'],
          test_reader=test_reader, **train_args)
    
    # 5. 保存模型
    if ckpt_file:
        model.save(ckpt_file)
```

**输入**:
- `train_paths`: `['data_train']`
- `test_paths`: `['data_test']`
- `restart`: False
- `ckpt_file`: `'model.pth'`
- `model_args`: `{'hidden_sizes': [100, 100, 100], 'output_scale': 100, ...}`
- `data_args`: `{'batch_size': 16, 'group_batch': 1}`
- `preprocess_args`: `{'preshift': True, 'prescale': False, 'prefit_ridge': 10.0}`
- `train_args`: `{'n_epoch': 5000, 'start_lr': 0.0003, ...}`

**输出**:
- `model.pth`: 训练好的模型
- `log.train`: 训练日志

---

#### 3.3.2 模型定义

**文件**: `/home/linearline/project/DeePKS-L/deepks/model/model.py`

**类**: `CorrNet`

**架构**:

```python
class CorrNet(nn.Module):
    def __init__(self, ndesc, hidden_sizes=[100, 100, 100],
                 output_scale=1.0, use_resnet=True,
                 actv_fn='mygelu', embedder_type='trace',
                 shell_sec=None, **kwargs):
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

**组件**:

1. **TraceEmbedding**:
```python
class TraceEmbedding(nn.Module):
    def __init__(self, shell_sec):
        super().__init__()
        self.shell_sec = shell_sec
        self.ndesc = len(shell_sec)
    
    def forward(self, x):
        x_shells = x.split(self.shell_sec, dim=-1)
        tr_shells = [sx.sum(-1, keepdim=True) for sx in x_shells]
        return torch.cat(tr_shells, dim=-1)
```

2. **DenseNet**:
```python
class DenseNet(nn.Module):
    def __init__(self, sizes, actv_fn=torch.relu,
                 use_resnet=True, with_dt=False, layer_norm=False):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(in_f, out_f) 
            for in_f, out_f in zip(sizes, sizes[1:])
        ])
        self.actv_fn = actv_fn
        self.use_resnet = use_resnet
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            tmp = layer(x)
            if i < len(self.layers) - 1:
                tmp = self.actv_fn(tmp)
            if self.use_resnet and layer.in_features == layer.out_features:
                x = x + tmp
            else:
                x = tmp
        return x
```

**前向传播**:
```
描述符 (eig) 
  -> TraceEmbedding (迹嵌入)
    -> DenseNet (全连接网络)
      -> 输出 (能量修正) / output_scale
```

---

#### 3.3.3 训练循环

**文件**: `/home/linearline/project/DeePKS-L/deepks/model/train.py`

**函数**: `train(model, g_reader, n_epoch=1000, ...)`

**详细步骤**:

```python
def train(model, g_reader, n_epoch=1000, test_reader=None,
          energy_factor=1., force_factor=0., stress_factor=0.,
          orbital_factor=0., v_delta_factor=0., phi_factor=0.,
          band_factor=0., bandgap_factor=0., density_m_factor=0.,
          start_lr=0.001, decay_steps=100, decay_rate=0.96,
          display_epoch=100, ckpt_file="model.pth", device=DEVICE):
    
    # 1. 设置设备
    model = model.to(device)
    model.eval()
    
    # 2. 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, decay_steps, decay_rate)
    
    # 3. 创建评估器
    evaluator = Evaluator(
        energy_factor=energy_factor,
        force_factor=force_factor,
        stress_factor=stress_factor,
        orbital_factor=orbital_factor,
        v_delta_factor=v_delta_factor,
        phi_factor=phi_factor,
        band_factor=band_factor,
        bandgap_factor=bandgap_factor,
        density_m_factor=density_m_factor,
        grad_penalty=grad_penalty
    )
    
    # 4. 初始评估
    trn_loss = evaluate(model, g_reader, evaluator)
    tst_loss = evaluate(model, test_reader, evaluator)
    print(f"  {0:<8d}  {np.sqrt(trn_loss[-1]):>.2e}  {np.sqrt(tst_loss[-1]):>.2e} ...")
    
    # 5. 训练循环
    for epoch in range(1, n_epoch+1):
        model.train()
        
        # 训练一个epoch
        for sample in g_reader:
            optimizer.zero_grad()
            loss = evaluator(model, sample)
            loss[-1].backward()
            optimizer.step()
        
        scheduler.step()
        
        # 定期评估
        if epoch % display_epoch == 0:
            model.eval()
            trn_loss = evaluate(model, g_reader, evaluator)
            tst_loss = evaluate(model, test_reader, evaluator)
            
            # 打印进度
            print(f"  {epoch:<8d}  {np.sqrt(trn_loss[-1]):>.2e}  {np.sqrt(tst_loss[-1]):>.2e} ...")
            
            # 保存模型
            if ckpt_file:
                model.save(ckpt_file)
    
    # 6. 最终保存
    if ckpt_file:
        model.save(ckpt_file)
```

**损失函数**:
```python
def evaluate(model, sample, evaluator):
    # 前向传播
    pred = model(sample['eig'])
    
    # 计算损失
    loss_e = energy_lossfn(pred, sample['lb_e'])
    loss_f = force_lossfn(pred_force, sample['force']) if force_factor > 0 else 0
    loss_s = stress_lossfn(pred_stress, sample['stress']) if stress_factor > 0 else 0
    
    # 加权求和
    total_loss = energy_factor * loss_e + force_factor * loss_f + stress_factor * loss_s
    
    return [loss_e, loss_f, loss_s, total_loss]
```

**训练日志**:
```
# epoch      trn_err   tst_err        lr  trn_time  tst_time     trn_loss_energy
  0         3.89e-04  3.89e-04  3.00e-04      0.00      0.03          1.5143e-07
  100       7.65e-09  5.69e-09  3.00e-04      0.01      0.00          5.8537e-17
  200       1.86e-13  8.86e-14  3.00e-04      0.01      0.00          3.4567e-26
...
  5000      0.00e+00  0.00e+00  1.99e-04      0.00      0.00          0.0000e+00
```

---

## 四、迭代步骤执行 (iter.00)

### 4.1 SCF步骤 (00.scf) - 使用模型

与初始化SCF的主要区别：

1. **`no_model=False`**: 使用训练好的模型
2. **模型追踪**: 将PyTorch模型转换为TorchScript格式
3. **`deepks_scf=1`**: 在SCF中使用DeePKS修正

#### 4.1.1 模型追踪

**代码**:
```python
if not no_model:
    if model_file is not None:
        from deepks.model import CorrNet
        model = CorrNet.load(model_file)
        model.compile_save(CMODEL_FILE)
        pre_args.update(deepks_scf=1, 
                      model_file=os.path.abspath(CMODEL_FILE))
```

**函数**: `CorrNet.compile_save(filename)`

**作用**: 将PyTorch模型转换为TorchScript格式

**输出**: `model.ptg` (TorchScript模型)

#### 4.1.2 ABACUS SCF with DeePKS

**INPUT文件**:
```
deepks_scf 1
deepks_model /path/to/model.ptg
```

**C++执行流程**:

1. **加载TorchScript模型** (`setup_deepks.cpp`):
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
```cpp
for (int iter = 0; iter < scf_nmax; iter++) {
    // 1. 计算投影密度矩阵
    DeePKS_domain::cal_pdm(...);
    
    // 2. 计算描述符
    DeePKS_domain::cal_descriptor(...);
    
    // 3. 将描述符输入神经网络
    this->ld.dpks_cal_e_delta_band(dm_vec, kv.get_nks());
    
    // 4. 计算哈密顿量修正
    DeePKS_domain::cal_edelta_gedm(...);
    
    // 5. 更新哈密顿量
    for (int ik = 0; ik < nks; ik++) {
        for (int i = 0; i < nbands; i++) {
            H[ik][i] += this->ld.V_delta[ik][i];
        }
    }
    
    // 6. 对角化哈密顿量
    diagonalize(H, C, eig);
    
    // 7. 更新密度矩阵
    update_dm(C, eig);
    
    // 8. 检查收敛
    if (check_convergence()) break;
}
```

**关键C++函数**:

- `LCAO_Deepks::dpks_cal_e_delta_band()`: 计算能带能量修正
- `LCAO_Deepks::cal_descriptor()`: 计算描述符
- `LCAO_Deepks::cal_edelta_gedm()`: 计算能量和梯度

### 4.2 训练步骤 (01.train) - 重启训练

与初始化训练的主要区别：

1. **`restart=True`**: 从已有模型继续训练
2. **`preshift=False`**: 不重新计算平移
3. **`prescale=False`**: 不重新计算缩放
4. **学习率调整**: 可能根据`decay_rate_iter`调整

**代码**:
```python
if restart:
    # 加载已有模型
    model = CorrNet.load(ckpt_file)
else:
    # 创建新模型
    model = CorrNet(ndesc, **model_args)
```

---

## 五、总结

### 完整调用链

```
run.sh
  └─> python -m deepks iterate
        └─> deepks.__main__.main_cli()
              └─> deepks.main.iter_cli()
                    └─> deepks.iterate.iterate.main()
                          └─> deepks.iterate.iterate.make_iterate()
                                ├─> collect_systems()
                                ├─> check_share_folder()
                                ├─> check_arg_dict()
                                ├─> deepks.iterate.template_abacus.make_scf_abacus()
                                │     ├─> make_convert_scf_abacus()
                                │     │     └─> PythonTask(convert_data)
                                │     ├─> make_run_scf_abacus()
                                │     │     └─> GroupBatchTask(mpirun abacus)
                                │     │           └─> Setup_DeePKS::before_runner()
                                │     │           └─> LCAO_Deepks::dpks_cal_e_delta_band()
                                │     └─> make_stat_scf_abacus()
                                │           └─> PythonTask(gather_stats_abacus)
                                └─> deepks.iterate.template.make_train()
                                      └─> PythonTask(train)
                                            └─> deepks.model.train.main()
                                                  └─> deepks.model.train.train()
                                                        └─> deepks.model.model.CorrNet.forward()
                          └─> deepks.task.workflow.Iteration.run()
                                └─> deepks.task.workflow.Sequence.run()
                                      └─> deepks.task.task.PythonTask.run()
                                            └─> deepks.task.task.GroupBatchTask.run()
```

### 关键文件路径

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
