# DeePKS C++接口和底层函数调用详解

## 一、C++模块概览

### 1.1 主要头文件

**核心模块**:
- `setup_deepks.h`: DeePKS设置和主流程
- `LCAO_deepks.h`: DeePKS核心实现类
- `deepks_basic.h`: 基础数据结构和工具函数
- `deepks_descriptor.h`: 描述符计算
- `deepks_pdm.h`: 投影密度矩阵
- `deepks_force.h`: 力计算
- `deepks_orbital.h`: 轨道相关计算

**辅助模块**:
- `deepks_check.h`: 检查和验证
- `deepks_fpre.h`: 力预处理
- `deepks_orbpre.h`: 轨道预处理
- `deepks_phialpha.h`: 投影重叠积分
- `deepks_spre.h`: 应力预处理
- `deepks_vdelta.h`: 能量修正计算
- `deepks_vdpre.h`: 能量修正预处理
- `deepks_vdrpre.h`: 能量修正导数预处理

### 1.2 主要实现文件

- `setup_deepks.cpp`: DeePKS设置和主流程实现
- `LCAO_deepks.cpp`: DeePKS核心类实现
- `deepks_basic.cpp`: 基础功能实现
- `deepks_descriptor.cpp`: 描述符计算实现
- `deepks_pdm.cpp`: 投影密度矩阵实现
- `deepks_force.cpp`: 力计算实现
- `deepks_orbital.cpp`: 轨道计算实现

---

## 二、Setup_DeePKS类详解

### 2.1 类定义

**文件**: `/home/linearline/project/abacus-develop/source/source_lcao/setup_deepks.h`

```cpp
template <typename TK>
class Setup_DeePKS
{
public:
    Setup_DeePKS();
    ~Setup_DeePKS();

#ifdef __MLALGO
    LCAO_Deepks<TK> ld;  // DeePKS核心对象
#endif

    std::string dpks_out_type;

    // 主要方法
    void before_runner(
        const UnitCell &ucell,           // 晶胞
        const int nks,                   // k点数
        const LCAO_Orbitals &orb,       // 轨道信息
        Parallel_Orbitals &pv,          // 并行轨道
        const Input_para &inp);          // 输入参数

    void build_overlap(
        const UnitCell &ucell,
        const LCAO_Orbitals &orb,
        const Parallel_Orbitals &pv,
        const Grid_Driver &gd,
        TwoCenterIntegrator &overlap_orb_alpha,
        const Input_para &inp);

    void delta_e(
        const UnitCell& ucell,
        const K_Vectors &kv,
        const LCAO_Orbitals& orb,
        const Parallel_Orbitals &pv,
        const Grid_Driver &gd,
        const std::vector<std::vector<TK>>& dm_vec,
        elecstate::fenergy &f_en,
        const Input_para &inp);

    void write_forces(
        const ModuleBase::matrix &fcs,
        const ModuleBase::matrix &fvnl_dalpha,
        const Input_para &inp);

    void write_stress(
        const ModuleBase::matrix &scs,
        const ModuleBase::matrix &svnl_dalpha,
        const double &omega,
        const Input_para &inp);
};
```

### 2.2 before_runner()方法

**文件**: `/home/linearline/project/abacus-develop/source/source_lcao/setup_deepks.cpp`

**作用**: 在SCF运行前初始化DeePKS模块

**代码**:
```cpp
template <typename TK>
void Setup_DeePKS<TK>::before_runner(
    const UnitCell &ucell,
    const int nks,
    const LCAO_Orbitals &orb,
    Parallel_Orbitals &pv,
    const Input_para &inp)
{
#ifdef __MLALGO
    // 1. 初始化DeePKS
    LCAO_domain::DeePKS_init(ucell, pv, nks, orb, 
                              this->ld, GlobalV::ofs_running);
    
    // 2. 如果使用DeePKS SCF
    if (inp.deepks_scf) {
        // 加载TorchScript模型
        DeePKS_domain::load_model(inp.deepks_model, this->ld.model_deepks);
        
        // 读取投影密度矩阵
        DeePKS_domain::read_pdm(
            (inp.init_chg == "file"),      // 是否从文件读取
            inp.deepks_equiv,              // 是否使用等价原子
            this->ld.init_pdm,            // 初始化pdm标志
            ucell.nat,                   // 原子数
            this->ld.deepks_param,         // DeePKS参数
            *orb.Alpha,                   // 投影基组
            this->ld.pdm                  // 投影密度矩阵
        );
    }
#endif
}
```

**调用流程**:
```
before_runner()
  └─> DeePKS_init()
        ├─> 初始化参数
        ├─> 分配内存
        └─> 设置索引
  
  └─> load_model()
        ├─> torch::jit::load()
        └─> 加载到model_deepks
  
  └─> read_pdm()
        ├─> 读取文件
        └─> 填充pdm数组
```

**输入参数**:
- `ucell`: 晶胞信息（原子位置、晶格矢量等）
- `nks`: k点数量
- `orb`: 轨道信息（基组、轨道数等）
- `pv`: 并行轨道分布
- `inp`: 输入参数（包含deepks_scf、deepks_model等）

**输出**:
- `this->ld`: 初始化的DeePKS对象
- `this->ld.model_deepks`: 加载的神经网络模型
- `this->ld.pdm`: 投影密度矩阵

---

### 2.3 build_overlap()方法

**作用**: 构建投影重叠积分 `<phi(0)|alpha(R)>`

**代码**:
```cpp
template <typename TK>
void Setup_DeePKS<TK>::build_overlap(
    const UnitCell &ucell,
    const LCAO_Orbitals &orb,
    const Parallel_Orbitals &pv,
    const Grid_Driver &gd,
    TwoCenterIntegrator &overlap_orb_alpha,
    const Input_para &inp)
{
#ifdef __MLALGO
    // 检查是否需要构建重叠积分
    if (PARAM.globalv.deepks_setorb) {
        // 1. 分配 <phi(0)|alpha(R)>
        DeePKS_domain::allocate_phialpha(
            inp.cal_force,           // 是否计算力
            ucell,                 // 晶胞
            orb,                   // 轨道
            gd,                    // 网格驱动
            &pv,                   // 并行轨道
            this->ld.phialpha       // 输出：phialpha数组
        );
        
        // 2. 构建 <phi(0)|alpha(R)>
        DeePKS_domain::build_phialpha(
            inp.cal_force,
            ucell,
            orb,
            gd,
            &pv,
            overlap_orb_alpha,       // 双中心积分器
            this->ld.phialpha
        );
        
        // 3. 如果需要，进行单元测试
        if (inp.deepks_out_unittest) {
            DeePKS_domain::check_phialpha(
                inp.cal_force,
                ucell,
                orb,
                gd,
                &pv,
                this->ld.phialpha,
                GlobalV::MY_RANK
            );
        }
    }
#endif
}
```

**调用流程**:
```
build_overlap()
  └─> allocate_phialpha()
        ├─> 分配phialpha内存
        └─> 初始化为零
  
  └─> build_phialpha()
        ├─> 遍历原子对
        ├─> 计算双中心积分
        └─> 填充phialpha数组
  
  └─> check_phialpha() [可选]
        └─> 验证积分正确性
```

**数学定义**:
```
<phi_mu(0)|alpha_nu(R)> = ∫ phi_mu(r) * alpha_nu(r-R) dr
```

其中：
- `phi_mu(r)`: 数值原子轨道
- `alpha_nu(r-R)`: 描述符投影基组
- `R`: 原子位置

**数据结构**:
```cpp
// phialpha是一个向量，包含：
// phialpha[0]: <phi(0)|alpha(R)>  (自身)
// phialpha[1]: d/dx <phi(0)|alpha(R)>  (x导数，计算力时需要)
// phialpha[2]: d/dy <phi(0)|alpha(R)>  (y导数)
// phialpha[3]: d/dz <phi(0)|alpha(R)>  (z导数)
std::vector<hamilt::HContainer<double>*> phialpha;
```

---

### 2.4 delta_e()方法

**作用**: 计算DeePKS能量修正

**代码**:
```cpp
template <typename TK>
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
    // 如果使用DeePKS SCF
    if (inp.deepks_scf) {
        // 1. 计算能带能量修正
        this->ld.dpks_cal_e_delta_band(dm_vec, kv.get_nks());
        
        // 2. 更新实空间密度矩阵
        DeePKS_domain::update_dmr(
            kv.kvec_d,      // k点矢量
            dm_vec,         // k空间密度矩阵
            ucell,          // 晶胞
            orb,            // 轨道
            pv,             // 并行轨道
            gd,             // 网格驱动
            this->ld.dm_r    // 输出：实空间密度矩阵
        );
        
        // 3. 计算能量修正
        f_en.edeepks_scf = this->ld.E_delta - this->ld.e_delta_band;
        f_en.edeepks_delta = this->ld.E_delta;
    }
#endif
}
```

**调用流程**:
```
delta_e()
  └─> dpks_cal_e_delta_band()
        ├─> 计算投影密度矩阵
        ├─> 计算描述符
        ├─> 输入神经网络
        ├─> 计算能量修正
        └─> 计算哈密顿量修正
  
  └─> update_dmr()
        ├─> Fourier变换
        └─> 得到实空间密度矩阵
  
  └─> 计算能量
        ├─> E_delta: 总能量修正
        └─> e_delta_band: 能带部分
```

**能量组成**:
```
E_delta = E_NN(D)  // 神经网络预测的能量
e_delta_band = tr(ρ * H_delta)  // 能带部分

edeepks_scf = E_delta - e_delta_band  // SCF能量修正
edeepks_delta = E_delta  // 总能量修正
```

---

### 2.5 write_forces()方法

**作用**: 写入力数据到文件

**代码**:
```cpp
template <typename TK>
void Setup_DeePKS<TK>::write_forces(
    const ModuleBase::matrix &fcs,           // 总力
    const ModuleBase::matrix &fvnl_dalpha,    // 赝势力
    const Input_para &inp)
{
#ifdef __MLALGO
    // 如果需要输出标签
    if (inp.deepks_out_labels) {
        // 1. 输出总力
        if (inp.deepks_out_base == "none" || 
            (inp.deepks_out_base != "none" && this->dpks_out_type == "tot")) {
            const std::string file_ftot = 
                PARAM.globalv.global_out_dir + 
                (inp.deepks_out_labels == 1 ? "deepks_ftot.npy" : "deepks_force.npy");
            LCAO_deepks_io::save_matrix2npy(file_ftot, fcs, GlobalV::MY_RANK);
            
            // 2. 输出基础力
            if (inp.deepks_out_labels == 1) {
                const std::string file_fbase = 
                    PARAM.globalv.global_out_dir + "deepks_fbase.npy";
                if (inp.deepks_scf) {
                    LCAO_deepks_io::save_matrix2npy(
                        file_fbase, 
                        fcs - fvnl_dalpha,  // F_base = F_tot - F_vnl_dalpha
                        GlobalV::MY_RANK
                    );
                } else {
                    LCAO_deepks_io::save_matrix2npy(
                        file_fbase, 
                        fcs,  // 没有scf时，F_base = F_tot
                        GlobalV::MY_RANK
                    );
                }
            }
        }
    }
#endif
}
```

**力组成**:
```
F_tot = F_base + F_vnl_dalpha  // 总力
F_base = F_tot - F_vnl_dalpha  // 基础力（不包含DeePKS修正）
```

---

### 2.6 write_stress()方法

**作用**: 写入应力数据到文件

**代码**:
```cpp
template <typename TK>
void Setup_DeePKS<TK>::write_stress(
    const ModuleBase::matrix &scs,           // 总应力
    const ModuleBase::matrix &svnl_dalpha,    // 赝势应力
    const double &omega,                      // 晶胞体积
    const Input_para &inp)
{
#ifdef __MLALGO
    if (inp.deepks_out_labels == 1) {
        assert(omega > 0.0);
        
        // 1. 输出总应力
        if (inp.deepks_out_base == "none" || 
            (inp.deepks_out_base != "none" && this->dpks_out_type == "tot")) {
            const std::string file_stot = 
                PARAM.globalv.global_out_dir + "deepks_stot.npy";
            LCAO_deepks_io::save_matrix2npy(
                file_stot, scs, GlobalV::MY_RANK, omega, 'U'  // 转换为Ry单位
            );
            
            // 2. 输出基础应力
            const std::string file_sbase = 
                PARAM.globalv.global_out_dir + "deepks_sbase.npy";
            if (inp.deepks_scf) {
                LCAO_deepks_io::save_matrix2npy(
                    file_sbase, 
                    scs - svnl_dalpha,  // S_base = S_tot - S_vnl_dalpha
                    GlobalV::MY_RANK, 
                    omega, 
                    'U'
                );
            } else {
                LCAO_deepks_io::save_matrix2npy(
                    file_sbase, 
                    scs,  // 没有scf时，S_base = S_tot
                    GlobalV::MY_RANK, 
                    omega, 
                    'U'
                );
            }
        }
    }
#endif
}
```

**应力组成**:
```
S_tot = S_base + S_vnl_dalpha  // 总应力
S_base = S_tot - S_vnl_dalpha  // 基础应力
```

---

## 三、LCAO_Deepks类详解

### 3.1 类定义

**文件**: `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/LCAO_deepks.h`

```cpp
template <typename T>
class LCAO_Deepks
{
public:
    // 公共变量
    double E_delta = 0.0;              // NN预测的能量修正 (Ry)
    double e_delta_band = 0.0;          // 能带部分 (Ry)
    std::vector<std::vector<T>> V_delta;  // 哈密顿量修正

    // DeePKS参数
    DeePKS_Param deepks_param;

    // 初始化标志
    bool init_pdm = false;

    // 神经网络模型
    torch::jit::script::Module model_deepks;

    // 投影重叠积分
    std::vector<hamilt::HContainer<double>*> phialpha;

    // 实空间密度矩阵
    hamilt::HContainer<double>* dm_r = nullptr;

    // 投影密度矩阵
    std::vector<torch::Tensor> pdm;

    // 能量对描述符的导数
    double** gedm = nullptr;

    // 公共方法
    explicit LCAO_Deepks();
    ~LCAO_Deepks();

    void init(const LCAO_Orbitals& orb,
              const int nat,
              const int ntype,
              const int nks,
              const Parallel_Orbitals& pv_in,
              std::vector<int> na,
              std::ofstream& ofs);

    void allocate_V_delta(const int nat, const int nks = 1);

    void init_DMR(const UnitCell& ucell,
                  const LCAO_Orbitals& orb,
                  const Parallel_Orbitals& pv,
                  const Grid_Driver& GridD);

    void dpks_cal_e_delta_band(const std::vector<std::vector<T>>& dm, 
                              const int nks);

private:
    bool hr_cal = true;
};
```

### 3.2 init()方法

**作用**: 初始化DeePKS参数和数据结构

**代码** (在`LCAO_deepks.cpp`中):
```cpp
template <typename T>
void LCAO_Deepks<T>::init(
    const LCAO_Orbitals& orb,
    const int nat,
    const int ntype,
    const int nks,
    const Parallel_Orbitals& pv_in,
    std::vector<int> na,
    std::ofstream& ofs)
{
    // 1. 初始化参数
    this->deepks_param.init(nat, ntype, na, orb, pv_in, ofs);
    
    // 2. 分配投影密度矩阵内存
    this->pdm.resize(this->deepks_param.tot_Inl);
    for (int i = 0; i < this->deepks_param.tot_Inl; i++) {
        int l = this->deepks_param.l_nchi[i];
        int size = (2 * l + 1) * (2 * l + 1);
        this->pdm[i] = torch::zeros({size}, torch::kFloat64);
    }
    
    // 3. 分配gedm内存
    this->gedm = new double*[this->deepks_param.tot_Inl];
    for (int i = 0; i < this->deepks_param.tot_Inl; i++) {
        int l = this->deepks_param.l_nchi[i];
        int size = (2 * l + 1) * (2 * l + 1);
        this->gedm[i] = new double[size];
    }
}
```

**初始化内容**:
- `tot_Inl`: 总投影基函数数
- `l_nchi`: 每个投影基的角动量
- `pdm`: 投影密度矩阵
- `gedm`: 能量对描述符的导数

---

### 3.3 allocate_V_delta()方法

**作用**: 分配哈密顿量修正的内存

**代码**:
```cpp
template <typename T>
void LCAO_Deepks<T>::allocate_V_delta(const int nat, const int nks)
{
    // 1. 分配V_delta内存
    this->V_delta.resize(nks);
    for (int ik = 0; ik < nks; ik++) {
        this->V_delta[ik].resize(nat);
        for (int i = 0; i < nat; i++) {
            this->V_delta[ik][i].create(
                this->deepks_param.nchi[i], 
                this->deepks_param.nchi[i]
            );
        }
    }
}
```

**数据结构**:
```cpp
// V_delta[ik][i][mu][nu]
// ik: k点索引
// i: 原子索引
// mu, nu: 轨道索引
std::vector<std::vector<std::vector<T>>> V_delta;
```

---

### 3.4 dpks_cal_e_delta_band()方法

**作用**: 计算能量修正和哈密顿量修正

**代码** (在`LCAO_deepks.cpp`中):
```cpp
template <typename T>
void LCAO_Deepks<T>::dpks_cal_e_delta_band(
    const std::vector<std::vector<T>>& dm, 
    const int nks)
{
    // 1. 计算投影密度矩阵
    DeePKS_domain::cal_pdm(
        dm, 
        this->phialpha, 
        this->deepks_param, 
        this->pdm
    );
    
    // 2. 计算描述符
    DeePKS_domain::cal_descriptor(
        this->pdm, 
        this->deepks_param, 
        this->dm_r
    );
    
    // 3. 将描述符输入神经网络
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(this->dm_r->toTensor());
    
    // 4. 前向传播
    torch::jit::IValue output = this->model_deepks.forward(inputs);
    torch::Tensor e_delta_tensor = output.toTensor();
    
    // 5. 提取能量修正
    this->E_delta = e_delta_tensor.item<double>();
    
    // 6. 计算梯度
    e_delta_tensor.backward();
    
    // 7. 提取gedm
    for (int i = 0; i < this->deepks_param.tot_Inl; i++) {
        torch::Tensor grad = this->pdm[i].grad();
        std::memcpy(this->gedm[i], 
                   grad.data_ptr<double>(), 
                   grad.numel() * sizeof(double));
    }
    
    // 8. 计算哈密顿量修正
    DeePKS_domain::cal_edelta_gedm(
        this->phialpha, 
        this->gedm, 
        this->deepks_param, 
        this->V_delta
    );
    
    // 9. 计算能带部分
    this->e_delta_band = 0.0;
    for (int ik = 0; ik < nks; ik++) {
        for (int i = 0; i < this->deepks_param.nat; i++) {
            for (int mu = 0; mu < this->deepks_param.nchi[i]; mu++) {
                for (int nu = 0; nu < this->deepks_param.nchi[i]; nu++) {
                    this->e_delta_band += 
                        std::real(dm[ik][i](mu, nu) * this->V_delta[ik][i](mu, nu));
                }
            }
        }
    }
}
```

**计算流程**:
```
1. 计算投影密度矩阵:
   pdm[i][l][m][m'] = ∑_k ∑_i occ_i <C_i|alpha_i,l,m> <alpha_i,l,m'|C_i>

2. 计算描述符:
   D[i][l] = eig(pdm[i][l])  // 对pdm[i][l]对角化

3. 输入神经网络:
   E_delta = NN(D)

4. 计算梯度:
   gedm[i][l][m][m'] = dE_delta/dD[i][l][m][m']

5. 计算哈密顿量修正:
   V_delta[i][mu][nu] = ∑_l,m,m' gedm[i][l][m][m'] * 
                        <phi_mu|alpha_i,l,m> <alpha_i,l,m'|phi_nu>

6. 计算能带部分:
   e_delta_band = ∑_k,i,mu,nu dm[k][i][mu][nu] * V_delta[k][i][mu][nu]
```

**数学表达式**:
```
pdm_inl = ∑_k ∑_i occ_i C_i^* <phi_i|alpha_inl> <alpha_inl|phi_i>

D_inl = eig(pdm_inl)  // 描述符

E_delta = NN(D)  // 神经网络预测

gedm_inl = dE_delta/dD_inl  // 自动微分

V_delta_mu,nu = ∑_inl gedm_inl * <phi_mu|alpha_inl> <alpha_inl|phi_nu>

e_delta_band = ∑_k,i,mu,nu dm_k_i_mu,nu * V_delta_k_i_mu,nu
```

---

## 四、DeePKS_domain命名空间

### 4.1 主要函数

**初始化和设置**:
```cpp
namespace DeePKS_domain {
    void DeePKS_init(...);
    void load_model(...);
    void read_pdm(...);
    void write_pdm(...);
}
```

**投影密度矩阵**:
```cpp
namespace DeePKS_domain {
    void cal_pdm(...);
    void allocate_phialpha(...);
    void build_phialpha(...);
    void check_phialpha(...);
}
```

**描述符**:
```cpp
namespace DeePKS_domain {
    void cal_descriptor(...);
    void update_dmr(...);
}
```

**能量和力**:
```cpp
namespace DeePKS_domain {
    void cal_edelta_gedm(...);
    void cal_force(...);
    void cal_stress(...);
}
```

### 4.2 load_model()函数

**作用**: 加载TorchScript模型

**代码** (推测):
```cpp
namespace DeePKS_domain {
    void load_model(
        const std::string& model_file,
        torch::jit::script::Module& model_deepks)
    {
        // 1. 加载模型
        model_deepks = torch::jit::load(model_file);
        
        // 2. 设置为评估模式
        model_deepks.eval();
        
        // 3. 移动到CPU（如果需要）
        model_deepks.to(torch::kCPU);
    }
}
```

**输入**:
- `model_file`: TorchScript模型文件路径
- `model_deepks`: 输出的模型对象

**输出**: 加载的TorchScript模型

---

### 4.3 cal_pdm()函数

**作用**: 计算投影密度矩阵

**代码** (推测):
```cpp
namespace DeePKS_domain {
    void cal_pdm(
        const std::vector<std::vector<T>>& dm,
        const std::vector<hamilt::HContainer<double>*>& phialpha,
        const DeePKS_Param& deepks_param,
        std::vector<torch::Tensor>& pdm)
    {
        // 遍历所有投影基
        for (int inl = 0; inl < deepks_param.tot_Inl; inl++) {
            int iat = deepks_param.inl2iat[inl];  // 原子索引
            int l = deepks_param.l_nchi[inl];     // 角动量
            int nchi = deepks_param.nchi[iat];   // 轨道数
            
            // 初始化pdm
            pdm[inl].zero_();
            
            // 遍历k点
            for (int ik = 0; ik < dm.size(); ik++) {
                // 遍历占据轨道
                for (int i = 0; i < deepks_param.nbands; i++) {
                    if (deepks_param.occ[ik][i] > 0.0) {
                        // 计算投影系数
                        std::vector<double> proj(nchi);
                        for (int mu = 0; mu < nchi; mu++) {
                            proj[mu] = 0.0;
                            for (int nu = 0; nu < nchi; nu++) {
                                proj[mu] += std::real(dm[ik][iat](mu, nu)) * 
                                            (*phialpha[0])(iat, mu, nu);
                            }
                        }
                        
                        // 累加到pdm
                        for (int m = 0; m < 2*l+1; m++) {
                            for (int mp = 0; mp < 2*l+1; mp++) {
                                pdm[inl][m*(2*l+1)+mp] += 
                                    deepks_param.occ[ik][i] * 
                                    proj[m] * proj[mp];
                            }
                        }
                    }
                }
            }
        }
    }
}
```

**数学表达式**:
```
pdm_inl[m,m'] = ∑_k ∑_i occ_i 
               <C_i|alpha_inl,m> <alpha_inl,m'|C_i>

其中:
< C_i | alpha_inl,m > = ∑_nu C_i,nu * <phi_nu|alpha_inl,m>
```

---

### 4.4 cal_descriptor()函数

**作用**: 计算描述符（对投影密度矩阵对角化）

**代码** (推测):
```cpp
namespace DeePKS_domain {
    void cal_descriptor(
        const std::vector<torch::Tensor>& pdm,
        const DeePKS_Param& deepks_param,
        hamilt::HContainer<double>* dm_r)
    {
        // 遍历所有投影基
        for (int inl = 0; inl < deepks_param.tot_Inl; inl++) {
            int l = deepks_param.l_nchi[inl];
            int size = 2 * l + 1;
            
            // 将pdm转换为矩阵
            torch::Tensor pdm_matrix = pdm[inl].view({size, size});
            
            // 对角化
            auto [eigvals, eigvecs] = torch::linalg::eigh(pdm_matrix);
            
            // 保存特征值（描述符）
            dm_r->set_value(inl, eigvals.data_ptr<double>());
        }
    }
}
```

**数学表达式**:
```
D_inl = eig(pdm_inl)

其中:
pdm_inl = [pdm_inl[m,m']]  // (2l+1)×(2l+1)矩阵
D_inl = [λ_0, λ_1, ..., λ_2l]  // 特征值
```

---

### 4.5 cal_edelta_gedm()函数

**作用**: 计算哈密顿量修正

**代码** (推测):
```cpp
namespace DeePKS_domain {
    void cal_edelta_gedm(
        const std::vector<hamilt::HContainer<double>*>& phialpha,
        double** gedm,
        const DeePKS_Param& deepks_param,
        std::vector<std::vector<T>>& V_delta)
    {
        // 遍历所有原子
        for (int iat = 0; iat < deepks_param.nat; iat++) {
            int nchi = deepks_param.nchi[iat];
            
            // 遍历轨道
            for (int mu = 0; mu < nchi; mu++) {
                for (int nu = 0; nu < nchi; nu++) {
                    double v_delta = 0.0;
                    
                    // 遍历投影基
                    for (int inl = 0; inl < deepks_param.tot_Inl; inl++) {
                        if (deepks_param.inl2iat[inl] == iat) {
                            int l = deepks_param.l_nchi[inl];
                            
                            // 遍历磁量子数
                            for (int m = 0; m < 2*l+1; m++) {
                                for (int mp = 0; mp < 2*l+1; mp++) {
                                    v_delta += 
                                        gedm[inl][m*(2*l+1)+mp] *
                                        (*phialpha[0])(iat, mu, inl, m) *
                                        (*phialpha[0])(iat, nu, inl, mp);
                                }
                            }
                        }
                    }
                    
                    // 设置V_delta
                    for (int ik = 0; ik < V_delta.size(); ik++) {
                        V_delta[ik][iat](mu, nu) = v_delta;
                    }
                }
            }
        }
    }
}
```

**数学表达式**:
```
V_delta_mu,nu = ∑_inl ∑_m,m' gedm_inl[m,m'] * 
                <phi_mu|alpha_inl,m> <alpha_inl,m'|phi_nu>

其中:
gedm_inl[m,m'] = dE_delta/dD_inl[m,m']
```

---

## 五、数据流总结

### 5.1 输入数据

**从ABACUS主程序**:
- `dm_vec`: k空间密度矩阵
- `ucell`: 晶胞信息
- `orb`: 轨道信息
- `kv`: k点信息

**从文件**:
- `model.ptg`: TorchScript模型
- `pdm.dat`: 投影密度矩阵（可选）

### 5.2 中间数据

**投影重叠积分**:
```cpp
phialpha[0]: <phi(0)|alpha(R)>  // 自身
phialpha[1]: d/dx <phi(0)|alpha(R)>  // x导数
phialpha[2]: d/dy <phi(0)|alpha(R)>  // y导数
phialpha[3]: d/dz <phi(0)|alpha(R)>  // z导数
```

**投影密度矩阵**:
```cpp
pdm[inl][m][mp]  // tot_Inl个投影基
```

**描述符**:
```cpp
dm_r[inl]  // 特征值，作为描述符
```

**神经网络输出**:
```cpp
E_delta  // 能量修正
gedm[inl][m][mp]  // 梯度
```

**哈密顿量修正**:
```cpp
V_delta[ik][iat][mu][nu]  // k点、原子、轨道
```

### 5.3 输出数据

**能量**:
- `deepks_ebase.npy`: 基础能量
- `deepks_etot.npy`: 总能量
- `deepks_edelta.npy`: 能量修正

**力**:
- `deepks_ftot.npy`: 总力
- `deepks_fbase.npy`: 基础力

**应力**:
- `deepks_stot.npy`: 总应力
- `deepks_sbase.npy`: 基础应力

**描述符**:
- `deepks_dm_eig.npy`: 密度矩阵本征值（描述符）

**投影密度矩阵**:
- `deepks_projdm.dat`: 投影密度矩阵

---

## 六、关键算法

### 6.1 投影密度矩阵计算

```cpp
// 伪代码
for (int inl = 0; inl < tot_Inl; inl++) {
    int iat = inl2iat[inl];
    int l = l_nchi[inl];
    
    for (int ik = 0; ik < nks; ik++) {
        for (int i = 0; i < nbands; i++) {
            if (occ[ik][i] > 0) {
                // 计算投影系数
                for (int mu = 0; mu < nchi[iat]; mu++) {
                    proj[mu] = ∑_nu dm[ik][iat](mu,nu) * phialpha[0](iat,mu,inl)
                }
                
                // 累加到pdm
                for (int m = 0; m < 2*l+1; m++) {
                    for (int mp = 0; mp < 2*l+1; mp++) {
                        pdm[inl][m,mp] += occ[ik][i] * proj[m] * proj[mp]
                    }
                }
            }
        }
    }
}
```

### 6.2 描述符计算

```cpp
// 伪代码
for (int inl = 0; inl < tot_Inl; inl++) {
    int l = l_nchi[inl];
    int size = 2 * l + 1;
    
    // 将pdm转换为矩阵
    Matrix pdm_matrix(size, size);
    for (int m = 0; m < size; m++) {
        for (int mp = 0; mp < size; mp++) {
            pdm_matrix(m, mp) = pdm[inl][m, mp];
        }
    }
    
    // 对角化
    Vector eigvals(size);
    diagonalize(pdm_matrix, eigvals);
    
    // 保存特征值
    dm_r[inl] = eigvals;
}
```

### 6.3 哈密顿量修正计算

```cpp
// 伪代码
for (int iat = 0; iat < nat; iat++) {
    for (int mu = 0; mu < nchi[iat]; mu++) {
        for (int nu = 0; nu < nchi[iat]; nu++) {
            double v_delta = 0.0;
            
            for (int inl = 0; inl < tot_Inl; inl++) {
                if (inl2iat[inl] == iat) {
                    int l = l_nchi[inl];
                    
                    for (int m = 0; m < 2*l+1; m++) {
                        for (int mp = 0; mp < 2*l+1; mp++) {
                            v_delta += gedm[inl][m,mp] * 
                                       phialpha[0](iat,mu,inl,m) * 
                                       phialpha[0](iat,nu,inl,mp);
                        }
                    }
                }
            }
            
            V_delta[iat](mu, nu) = v_delta;
        }
    }
}
```

---

## 七、文件路径汇总

### C++头文件
- `/home/linearline/project/abacus-develop/source/source_lcao/setup_deepks.h`
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/LCAO_deepks.h`
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/deepks_basic.h`
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/deepks_descriptor.h`
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/deepks_pdm.h`
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/deepks_force.h`

### C++实现文件
- `/home/linearline/project/abacus-develop/source/source_lcao/setup_deepks.cpp`
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/LCAO_deepks.cpp`
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/deepks_basic.cpp`
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/deepks_descriptor.cpp`
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/deepks_pdm.cpp`
- `/home/linearline/project/abacus-develop/source/source_lcao/module_deepks/deepks_force.cpp`

### 输出文件
- `OUT.ABACUS/deepks_dm_eig.npy`: 描述符
- `OUT.ABACUS/deepks_ebase.npy`: 基础能量
- `OUT.ABACUS/deepks_etot.npy`: 总能量
- `OUT.ABACUS/deepks_ftot.npy`: 总力
- `OUT.ABACUS/deepks_fbase.npy`: 基础力
- `OUT.ABACUS/deepks_projdm.dat`: 投影密度矩阵
