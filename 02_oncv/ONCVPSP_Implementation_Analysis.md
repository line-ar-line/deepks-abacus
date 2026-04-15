# ONCVPSP 赝势实现逻辑分析

## 1. 概述

ONCVPSP (Optimized Norm-Conserving Vanderbilt PSeudopotential) 是一个用于生成优化的范德堡（Vanderbilt）或克莱曼-拜兰德（Kleinman-Bylander）规范守恒赝势的程序，基于 D. R. Hamann 的工作（Phys. Rev. B 88, 085117 (2013)）。本文档将详细分析其实现逻辑，并用数学语言描述完整流程。

## 2. 理论基础

### 2.1 规范守恒赝势的基本原理

规范守恒赝势的核心思想是将原子的核心电子和价电子分离，用一个赝势来代替核心电子对系统的影响，同时保持价电子波函数在截断半径外与全电子波函数一致。

对于单电子薛定谔方程：

$$eft[ -rac{1}{2}\nabla^2 + V_{ext}(\mathbf{r}) + V_{xc}(\mathbf{r}) \right] \psi_i(\mathbf{r}) = \varepsilon_i \psi_i(\mathbf{r})$$

其中 $V_{ext}$ 是原子核和其他电子产生的库仑势，$V_{xc}$ 是交换关联势。

规范守恒赝势将其替换为：

$$eft[ -rac{1}{2}\nabla^2 + V_{ps}(\mathbf{r}) \right] \tilde{\psi}_i(\mathbf{r}) = \varepsilon_i \tilde{\psi}_i(\mathbf{r})$$

其中 $V_{ps}$ 是赝势，$\tilde{\psi}_i$ 是赝波函数。

### 2.2 范德堡形式的赝势

范德堡形式的赝势可以表示为：

$$V_{ps}(\mathbf{r}) = V_{loc}(\mathbf{r}) + \sum_{lm}\sum_{l'm'} |\phi_{lm}(\mathbf{r})\rangle D_{ll'mm'}\langle\phi_{l'm'}(\mathbf{r})|$$

其中 $V_{loc}$ 是局域势，$\phi_{lm}$ 是投影函数，$D_{ll'mm'}$ 是非局域势的强度矩阵。

## 3. 实现流程

### 3.1 输入参数处理

ONCVPSP 读取输入文件中的参数，包括：
- 原子符号、原子序数、核心电子数、价电子数
- 交换关联泛函类型
- 截断半径、能量参数、连续性约束等
- 局域势参数
- 投影算子参数
- 模型核心电荷参数
- 输出网格参数
- 测试配置

### 3.2 全电子参考原子计算

首先，程序计算全电子参考原子的波函数和能量：

$$\left[ -rac{1}{2}\nabla^2 - \frac{Z}{r} + V_{H}(r) + V_{xc}(r) \right] \psi_i(r) = \varepsilon_i \psi_i(r)$$

其中 $Z$ 是原子序数，$V_H$ 是哈特利势，$V_{xc}$ 是交换关联势。

这一步通过自洽场方法求解，得到全电子波函数 $\psi_i$ 和能量 $\varepsilon_i$。

### 3.3 赝势波函数优化

对于每个角动量通道 $l$，程序优化赝势波函数 $\tilde{\psi}_l$，使其在截断半径 $r_c$ 外与全电子波函数一致，并满足规范守恒条件。

优化目标是最小化以下泛函：

$$E[\tilde{\psi}_l] = \int_0^{r_c} \left| \frac{d^n}{dr^n} (\psi_l(r) - \tilde{\psi}_l(r)) \right|^2 r^2 dr$$

其中 $n$ 是连续性约束的阶数。

同时，需要满足规范守恒条件：

$$\int_0^{r_c} \tilde{\psi}_l(r) \psi_{l'}(r) r^2 dr = \delta_{ll'}$$

### 3.4 构建投影算子

对于每个角动量通道，程序构建范德堡-克莱曼-拜兰德投影算子：

$$|\phi_{lm}\rangle = \sum_i c_{il} \tilde{\psi}_{il}(r) Y_{lm}(\hat{\mathbf{r}})$$

其中 $\tilde{\psi}_{il}$ 是优化后的赝势波函数，$c_{il}$ 是系数，$Y_{lm}$ 是球谐函数。

### 3.5 计算赝势电荷密度

程序计算赝势电荷密度：

$$\rho_{ps}(r) = \sum_i f_i |\tilde{\psi}_i(r)|^2$$

其中 $f_i$ 是占据数。

### 3.6 构建模型核心电荷

为了改进赝势的性能，程序构建模型核心电荷：

$$\rho_{core}^{mod}(r) = \rho_{core}(r) + f_{cfact} \rho_{ps}(r)$$

其中 $\rho_{core}$ 是核心电荷密度，$f_{cfact}$ 是核心电荷因子。

### 3.7 计算屏蔽势

程序计算屏蔽势：

$$V_{scr}(r) = V_H^{ps}(r) + V_{xc}^{ps}(r)$$

其中 $V_H^{ps}$ 是赝势电荷的哈特利势，$V_{xc}^{ps}$ 是赝势电荷的交换关联势。

### 3.8 构建最终赝势

最终的赝势为：

$$V_{ps}(r) = V_{loc}(r) + \sum_{lm}\sum_{l'm'} |\phi_{lm}(r)\rangle D_{ll'mm'}\langle\phi_{l'm'}(r)| - V_{scr}(r)$$

其中 $V_{loc}$ 是局域势，$D_{ll'mm'}$ 是非局域势的强度矩阵。

## 4. 核心算法

### 4.1 优化算法

ONCVPSP 使用以下步骤优化赝势波函数：

1. 初始化赝势波函数 $\tilde{\psi}_l$。
2. 计算全电子波函数 $\psi_l$。
3. 在截断半径 $r_c$ 处匹配波函数及其导数。
4. 最小化目标泛函，得到优化后的赝势波函数。
5. 检查规范守恒条件是否满足。
6. 重复步骤 3-5，直到收敛。

### 4.2 规范守恒条件的实现

规范守恒条件通过以下方式实现：

$$\int_0^{r_c} \tilde{\psi}_l(r) \psi_{l'}(r) r^2 dr = \delta_{ll'}$$

这确保了赝势波函数与全电子波函数在截断半径内的正交性。

### 4.3 非局域势的构建

非局域势的强度矩阵 $D$ 通过以下方式计算：

$$D = (\langle\phi|h|\phi\rangle)^{-1}$$

其中 $h$ 是单电子哈密顿量。

## 5. 数值实现

### 5.1 径向网格

ONCVPSP 使用指数网格：

$$r_i = r_1 e^{\alpha (i-1)}$$

其中 $r_1$ 是最小半径，$\alpha$ 是网格参数。

### 5.2 薛定谔方程的求解

程序使用数值方法求解薛定谔方程，包括：
- 有限差分法
- 自洽场迭代

### 5.3 优化过程

优化过程使用共轭梯度法或其他优化算法，最小化目标泛函。

## 6. 输入文件格式

ONCVPSP 的输入文件包含以下部分：

1. **原子和参考配置**
   - 原子符号、原子序数、核心电子数、价电子数、交换关联泛函类型、输出文件格式
   - 每个电子的主量子数(n)、角量子数(l)、占据数(f)

2. **赝势和优化参数**
   - lmax（最大角量子数）
   - 每个角动量通道的参数：l, rc（截断半径）, ep（能量参数）, ncon（连续性约束）, nbas（基函数数量）, qcut（截断波矢）

3. **局域势参数**
   - lloc（局域势的角量子数）、lpopt（局域势优化选项）、rc(5)、dvloc0

4. **范德堡-克莱曼-拜兰德投影算子参数**
   - 每个角动量通道的 nproj（投影算子数量）、debl（能量偏移）

5. **模型核心电荷参数**
   - icmod（核心电荷模型类型）、fcfact（核心电荷因子）、rcfact（核心电荷截断半径因子）

6. **对数导数分析参数**
   - epsh1、epsh2、depsh

7. **输出网格参数**
   - rlmax（最大径向距离）、drl（径向步长）

8. **测试配置**
   - ncnf（测试配置数量）
   - 每个测试配置的价电子数和电子组态

## 7. 输出文件

ONCVPSP 生成以下输出文件：
- `<prefix>.out`：主输出文件，包含输入参数、计算结果和诊断信息
- `<prefix>.psp8` 或 `<prefix>.upf`：生成的赝势文件
- 用于 gnuplot 绘图的数据文件和脚本

## 8. 诊断和验证

ONCVPSP 提供以下诊断信息：
- 输入参数的回显
- 全电子能量和波函数信息
- 赝势波函数的优化结果
- 规范守恒条件的满足情况
- 投影算子的信息
- 赝势的能量和电荷密度
- 测试配置的结果

## 9. 代码结构

ONCVPSP 的代码结构如下：
- `oncvpsp.f90`：主程序
- `oncvpsp_r.f90`：相对论版本
- `oncvpsp_nr.f90`：非相对论版本
- `functionals.F90`：交换关联泛函实现
- `linout.f90`：psp8 格式输出
- `upfout.f90`：upf 格式输出

## 10. 总结

ONCVPSP 是一个功能强大的赝势生成程序，通过以下步骤生成高质量的规范守恒赝势：

1. 计算全电子参考原子
2. 优化赝势波函数，满足规范守恒条件
3. 构建范德堡-克莱曼-拜兰德投影算子
4. 计算赝势电荷密度和屏蔽势
5. 输出赝势文件

该程序的核心是优化算法，通过最小化赝势波函数与全电子波函数的差异，同时满足规范守恒条件，生成准确的赝势。

ONCVPSP 生成的赝势被广泛应用于各种从头计算方法中，如密度泛函理论（DFT）计算，为材料科学和凝聚态物理研究提供了重要工具。

## 11. 代码优化建议

1. **并行计算**：对于大原子和复杂系统，可以引入并行计算，提高计算效率。
2. **自动参数优化**：开发自动参数优化算法，减少用户的手动调整。
3. **更多交换关联泛函**：增加对更多交换关联泛函的支持。
4. **用户友好的界面**：开发图形用户界面，简化输入参数的设置。
5. **与其他代码的接口**：提供与其他量子化学和材料模拟代码的接口，方便集成。

## 12. 应用示例

### 12.1 生成硅的赝势

输入文件示例：

```
# ATOM AND REFERENCE CONFIGURATION
# atsym  z   nc   nv     iexc    psfile
  Si     14   2   4      13      psp8

#   n    l    f
  1     0    2.0
  2     0    2.0
  2     1    6.0
  3     0    2.0
  3     1    2.0

# PSEUDOPOTENTIAL AND OPTIMIZATION
# lmax
  2

#   l,   rc,      ep,       ncon, nbas, qcut
  0    2.00    -6.0000      4     10    2.00
  1    2.00    -2.0000      4     10    2.00
  2    2.00    -0.5000      4     10    2.00

# LOCAL POTENTIAL
# lloc, lpopt,  rc(5),   dvloc0
  1     1      2.00      0.0000

# VANDERBILT-KLEINMAN-BYLANDER PROJECTORS
# l, nproj, debl
  0    2      1.000
  1    2      1.000
  2    1      0.000

# MODEL CORE CHARGE
# icmod, fcfact, rcfact
  1     0.00    0.00

# LOG DERIVATIVE ANALYSIS
# epsh1, epsh2, depsh
  10.0   20.0   0.1

# OUTPUT GRID
# rlmax, drl
  10.0   0.02

# TEST CONFIGURATIONS
# ncnf
  2

# nvcnf
  2
#   n    l    f
  3     0    2.0
  3     1    0.0

# nvcnf
  6
#   n    l    f
  3     0    2.0
  3     1    4.0
```

### 12.2 运行命令

```bash
./oncvpsp < Si.dat > Si.out
```

## 13. 参考文献

1. D. R. Hamann, Phys. Rev. B 88, 085117 (2013)
2. D. R. Hamann, Phys. Rev. B 80, 165104 (2009)
3. G. Kresse and D. Joubert, Phys. Rev. B 59, 1758 (1999)
4. G. B. Bachelet, D. R. Hamann, and M. Schlüter, Phys. Rev. B 26, 4199 (1982)
5. D. Vanderbilt, Phys. Rev. B 41, 7892 (1990)
6. L. Kleinman and D. M. Bylander, Phys. Rev. Lett. 48, 1425 (1982)
