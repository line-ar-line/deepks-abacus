# 氢原子 ONCV 赝势完整计算流程分析

> 基于 ONCVPSP v4.0.1 源代码及 H_PBE 输入文件，参照 Hamann, Phys. Rev. B 88, 085117 (2013)

---

## 0. 输入文件解析

氢原子输入文件 `H.dat` 内容如下：

```
# ATOM AND REFERENCE CONFIGURATION
# atsym  z    nc    nv    iexc   psfile
  H  1.00     0     1     4      upf
#   n    l    f        energy (Ha)
    1    0    1.00
# PSEUDOPOTENTIAL AND OPTIMIZATION
# lmax
    0
#   l,   rc,     ep,   ncon, nbas, qcut
    0   1.13748  -0.23860    5    8   9.72141
# LOCAL POTENTIAL
# lloc, lpopt,  rc(5),   dvloc0
    4    5   0.49352      0.00000
# VANDERBILT-KLEINMAN-BYLANDER PROJECTORs
# l, nproj, debl
    0    2   1.27464
# MODEL CORE CHARGE
# icmod, fcfact
    0   0.00000
# LOG DERIVATIVE ANALYSIS
# epsh1, epsh2, depsh
   -5.00    3.00    0.02
# OUTPUT GRID
# rlmax, drl
    6.00    0.01
# TEST CONFIGURATIONS
# ncnf
    0
```

**关键参数解读：**

| 参数 | 值 | 含义 |
|------|-----|------|
| `zz=1.00` | 原子序数 | 氢原子 Z=1 |
| `nc=0` | 核态数 | 氢无芯电子 |
| `nv=1` | 价态数 | 1s 价电子 |
| `iexc=4` | PBE-GGA | 交换关联泛函 |
| `lmax=0` | 最大角动量 | 仅 s 态 |
| `rc=1.13748` | 截断半径 | s 态赝势核心半径 |
| `ncon=5` | 约束数 | 匹配函数值+4阶导数 |
| `nbas=8` | 基函数数 | 球贝塞尔函数基组大小 |
| `qcut=9.72141` | 截断波矢 | 残余动能定义中的 $q_c$ |
| `lloc=4` | 局域势类型 | 多项式外推AE势 |
| `nproj=2` | 投影子数 | Vanderbilt 双投影子方法 |
| `icmod=0` | 无芯修正 | 无非线性芯校正 |

---

## 1. 对数径向网格构建

**源代码位置：** [oncvpsp.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/oncvpsp.f90#L214-L254)

```fortran
amesh=1.006d0
al=dlog(amesh)
rr1=0.0005d0/zz
rr1=dmin1(rr1,0.0005d0/10)
mmax=dlog(45.0d0/rr1)/al

do ii=1,mmax
  rr(ii)=rr1*exp(al*(ii-1))
end do
```

**数学公式：**

对数径向网格定义为：

$$r_i = r_1 \cdot e^{\alpha(i-1)}, \quad i = 1, 2, \ldots, M$$

其中：
- $\alpha = \ln(\text{amesh}) = \ln(1.006) \approx 0.005982$
- $r_1 = \min(0.0005/Z, 0.00005)$，对氢原子 $r_1 = 0.0005$
- $M = \lfloor \ln(45/r_1)/\alpha \rfloor$，约 1470 个网格点

对氢原子：$r_1 = 0.0005$，网格覆盖 $[0.0005, 45]$ Bohr。

---

## 2. 全电子自洽标量相对论原子计算（sratom）

**源代码位置：** [sratom.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/sratom.f90#L19-L210)

**调用入口：**

```fortran
call sratom(na,la,ea,fa,rpk,nc,nc+nv,it,rhoc,rho, &
            rr,vfull,zz,mmax,iexc,etot,ierr,srel)
```

### 2.1 初始势：Thomas-Fermi 势

**源代码位置：** [tfapot.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/tfapot.f90#L20-L56)

```fortran
do ii=1,mmax
  vi(ii)=tfapot(rr(ii),zz)
end do
```

**数学公式：** 广义 Thomas-Fermi 势（Latter 修正）：

$$V_{\text{TF}}(r) = -\frac{\tilde{Z}(r)}{r}$$

其中：

$$\tilde{Z}(r) = \frac{Z}{1 + \sqrt{x}(0.02747 - x(0.1486 - 0.007298x)) + x(1.243 + x(0.2302 + 0.006944x))}$$

$x = r/b$，$b = (0.69395656/Z)^{1/3}$。若 $\tilde{Z}(r) < 1$，则令 $\tilde{Z}(r) = 1$。

### 2.2 初始本征值估计

```fortran
do ii=1,ncv
  sf=sf+fa(ii)
  zion=zz+1.0d0-sf
  ea(ii)=-0.5d0*(zion/na(ii))**2
end do
```

**数学公式：** 类氢估计：

$$\varepsilon_i^{(0)} = -\frac{1}{2}\left(\frac{Z_{\text{ion}}}{n_i}\right)^2$$

对氢原子 1s 态：$\varepsilon^{(0)} = -0.5$ Ha。

### 2.3 自洽迭代循环

**核心流程：**

```
循环 it = 1, 100:
  1. 求解每个束缚态的径向薛定谔方程 (lschfb)
  2. 累积电荷密度 ρ(r) 和本征值
  3. 计算输出势 V_out (Hartree + XC)
  4. Anderson 混合法更新势 V_in
  5. 检查收敛
```

**数学公式：** 自洽条件为 Kohn-Sham 方程：

$$\left[-\frac{1}{2}\nabla^2 + V_{\text{eff}}[\rho](r)\right]\psi_i = \varepsilon_i \psi_i$$

$$V_{\text{eff}}(r) = V_{\text{ext}}(r) + V_H(r) + V_{xc}(r)$$

其中 $V_{\text{ext}}(r) = -Z/r$，Hartree 势：

$$V_H(r) = \int \frac{\rho(r')}{|r-r'|} dr'$$

对氢原子（iexc=4），交换关联势使用 PBE-GGA：

$$V_{xc}(r) = V_{xc}^{\text{PBE}}[\rho(r)]$$

**Anderson 混合法：**

$$V_{\text{new}} = (1-\beta)\left[(1-\theta)V_{\text{in}} + \theta V_{\text{in}}'\right] + \beta\left[(1-\theta)V_{\text{out}} + \theta V_{\text{out}}'\right]$$

其中 $\beta = 0.5$，$\theta$ 由残差最小化确定。

**总能量：**

$$E_{\text{tot}} = \sum_i f_i \varepsilon_i + E_{xc} - \frac{1}{2}E_{e-e}$$

氢原子计算结果：$E_{\text{tot}} = -0.458934602$ Ha，迭代 22 次收敛。

---

## 3. 标量相对论径向薛定谔方程求解（lschfb）

**源代码位置：** [lschfb.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/lschfb.f90#L19-L291)

### 3.1 Pauli 型标量相对论方程

**数学公式：** 径向薛定谔方程（对 $u = rR$）：

$$-\frac{d^2u}{dr^2} + \left[\frac{l(l+1)}{r^2} + 2(V(r) - \varepsilon)\right]u + f_r(r)u + f_r'(r)\frac{du}{dr} = 0$$

其中标量相对论修正项：

$$f_r(r) = \alpha^2 r^2\left[-\frac{\alpha^2}{4}(V-\varepsilon)^2 + \frac{\alpha^2}{2}\frac{dV/dr}{r(1+\frac{\alpha^2}{2}(\varepsilon-V))}\right]$$

$$f_r'(r) = -\alpha r \frac{\alpha^2}{2}\frac{dV/dr}{1+\frac{\alpha^2}{2}(\varepsilon-V)}$$

$\alpha = 1/137.036$ 为精细结构常数。

**对数网格变换：** 令 $t = \ln(r/r_1)$，则 $u(r) \to u(t)$：

$$\alpha_L^2 \frac{d^2u}{dt^2} = \left[\alpha_L^2 l(l+1) + 2\alpha_L^2(V-\varepsilon)r^2 + f_r r^2\right]u + f_r' r \alpha_L \frac{du}{dt}$$

其中 $\alpha_L = \ln(\text{amesh})$。

### 3.2 数值求解策略

```fortran
! 外向积分：从 r=0 到经典转折点
do ii=4,mch-1
  uu(ii+1)=uu(ii)+aeo(up,ii)       ! Adams-Bashforth 预测
  up(ii+1)=up(ii)+aeo(upp,ii)
  do it=1,2
    upp(ii+1)=(al+frp(ii+1))*up(ii+1)+(cf(ii+1)+fr(ii+1))*uu(ii+1)
    up(ii+1)=up(ii)+aio(upp,ii)     ! Adams-Moulton 校正
    uu(ii+1)=uu(ii)+aio(up,ii)
  end do
end do

! 内向积分：从远处到转折点
! 归一化：匹配外向和内向解
! 微扰修正能量
de=0.5d0*uout*(upout-upin)/(al*rr(mch))
```

**数学公式：**

1. **起始条件**（$r \to 0$）：$u(r) \sim r^\gamma$，其中

$$\gamma = \sqrt{1 - \alpha^2 Z^2} \quad (l=0)$$

2. **经典转折点**：$r_c$ 满足 $V(r_c) + l(l+1)/(2r_c^2) = \varepsilon$

3. **内向积分起始**（$r \to \infty$）：$u(r) \sim e^{-\kappa r}$，$\kappa = \sqrt{l(l+1)/r^2 + 2(V-\varepsilon)}$

4. **归一化**：

$$\int_0^\infty u^2(r) \frac{dr}{r} = 1$$

5. **能量修正**（微扰法）：

$$\Delta\varepsilon = \frac{u(r_m)}{2\alpha_L r_m}\left[\left.\frac{du}{dt}\right|_{\text{out}} - \left.\frac{du}{dt}\right|_{\text{in}}\right]$$

氢原子 1s 本征值：$\varepsilon_{1s} = -0.2386019$ Ha。

---

## 4. 量子阱态计算（wellstate）

**源代码位置：** [wellstate.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/wellstate.f90#L19-L212)

对于氢原子，只有一个价态（1s），但输入指定了 `nproj=2`，需要第二个投影子。第二个投影子对应正能散射态，需要用势垒约束来产生束缚态。

**调用入口：**

```fortran
call wellstate(npa(iprj,l1),ll,irc(l1),epa(iprj,l1),rr, &
               vfull,uu,up,zz,mmax,mch,srel)
```

### 4.1 势垒势构造

**数学公式：** 势垒势（论文 Eq.(17)）：

$$V_{\text{well}}(r) = V_{\text{AE}}(r) + V_{\text{barr}}(r)$$

$$V_{\text{barr}}(r) = c_{\text{well}} \frac{x^3}{1+x^3}, \quad x = \frac{r - r_{c+5}}{r_{\text{well}}}$$

其中 $c_{\text{well}} = \varepsilon_p + 0.5$ Ha 为势垒渐近值。

### 4.2 势垒宽度搜索

```fortran
do itrwell=1,100
  ! 构造势垒势
  ! 在势垒中求解束缚态 (lschfb)
  ! 区间折半法搜索合适的 r_well
  if(abs(et-ep)<eps) then
    ep=et; convg=.true.; exit
  end if
end do
```

**搜索策略：** 从 $r_{\text{well}} = 8r_c$ 开始，通过区间折半法调整 $r_{\text{well}}$，使得在势垒中的束缚态能量等于目标能量 $\varepsilon_p$。

氢原子结果：l=0, n=2, $\varepsilon = 1.0360$ Ha，$c_{\text{well}} = 1.5360$ Ha，半点半径 = 2.6839 Bohr。

---

## 5. 标量相对论修正势（vrel）

**源代码位置：** [vrel.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/vrel.f90#L19-L134)

```fortran
call vrel(ll,epa(iprj,l1),rr,vfull,vr(1,iprj,l1),uua(1,iprj),upa(1,iprj), &
          zz,mmax,irc(l1),srel)
```

**数学公式：** 计算有效势 $V_r(r)$，代表 Pauli 标量相对论算符的贡献：

$$V_r(r) = \frac{1}{2\alpha_L^2 r^2}\left[f_r(r) + f_r'(r)\frac{u'(r)}{u(r)}\right]$$

仅在 $[0.5r_c, 1.5r_c]$ 范围内计算，用于在投影子构造中平滑地强制投影子在 $r_c$ 处为零。

---

## 6. 全电子波函数重叠矩阵（fpovlp）

**源代码位置：** [fpovlp.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/fpovlp.f90#L19-L80)

```fortran
call fpovlp(uua(1,ii),uua(1,jj),irc(l1),ll,zz,qq(ii,jj),rr,srel)
```

**数学公式：** 计算两个全电子波函数在 $r_c$ 内的重叠积分：

$$Q_{ij} = \int_0^{r_c} u_i(r) u_j(r) \frac{dr}{r}$$

具体计算考虑标量相对论修正的 $r^\gamma$ 行为：

$$Q_{ij} = \frac{r_0^{2\gamma+1}}{2\gamma+1}\frac{u_i(1)u_j(1)}{r_1^{2\gamma}} + \alpha_L\sum_{k=1}^{n-3} u_i(k)u_j(k)r_k + \text{端点修正}$$

对氢原子 l=0，$\gamma = \sqrt{1-\alpha^2 Z^2} \approx 1$。

---

## 7. 优化赝波函数构造（run_optimize）

**源代码位置：** [run_optimize.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/run_optimize.f90#L19-L274)

这是 ONCV 方法的核心，对应论文 Sec. II。对每个角动量 l，依次构造优化赝波函数。

### 7.1 全电子波函数在 $r_c$ 处的导数（wf_rc_der）

**源代码位置：** [wf_rc_der.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/wf_rc_der.f90#L19-L92)

```fortran
call wf_rc_der(rr,uua(1,1),al,rc,irc,mmax,uord)
ulgd=uord(2)/uord(1)
```

**数学公式：** 计算 $u(r)/r$ 在 $r_c$ 处的值和 4 阶导数：

$$\left.\frac{u(r)}{r}\right|_{r_c}, \quad \left.\frac{d}{dr}\frac{u(r)}{r}\right|_{r_c}, \quad \left.\frac{d^2}{dr^2}\frac{u(r)}{r}\right|_{r_c}, \quad \ldots$$

使用 7 点数值微分公式逐次求导。对数导数：

$$\left.\frac{d\ln(u/r)}{dr}\right|_{r_c} = \frac{(u/r)'}{(u/r)}\bigg|_{r_c}$$

### 7.2 球贝塞尔函数波矢选择（qroots）

**源代码位置：** [qroots.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/qroots.f90#L19-L113)

```fortran
call qroots(ll,rc,ulgd,nbas,dq,qmax,qroot)
```

**数学公式：** 选择波矢 $q_i$ 使得球贝塞尔函数 $j_l(q_i r)$ 在 $r_c$ 处满足对数导数匹配条件：

$$\frac{j_l'(q_i r_c)}{j_l(q_i r_c)} = \left.\frac{d\ln(u/r)}{dr}\right|_{r_c}$$

通过区间折半法搜索满足此条件的根。额外添加两个补充波矢：
- $q_1 = 0.5 \times q_3$
- $q_3 = 0.5 \times (q_3^{\text{old}} + q_4^{\text{old}})$

### 7.3 正交化球贝塞尔基组（sbf_basis_con）

**源代码位置：** [sbf_basis_con.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/sbf_basis_con.f90#L19-L172)

**数学公式（论文 Eq.(3-4)）：**

球贝塞尔基函数（$\xi^B$ 基）：

$$\xi^B_i(r) = r \cdot j_l(q_i r), \quad i = 1, \ldots, N_{\text{bas}}$$

正交化基函数（$\xi^O$ 基，Eq.(4)）：

$$\xi^O_j(r) = \sum_{i=1}^{N_{\text{bas}}} O_{ij} \xi^B_i(r)$$

其中 $O_{ij}$ 通过重叠矩阵 $S_{ij} = \langle\xi^B_i|\xi^B_j\rangle$ 的本征值分解得到：

$$S = U \Sigma U^T, \quad O = U \Sigma^{-1/2}$$

同时计算 $\xi^O$ 在 $r_c$ 处的导数，以及与已有赝波函数的重叠（用于第二投影子的正交约束）。

### 7.4 约束基组构造（const_basis）

**源代码位置：** [const_basis.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/const_basis.f90#L19-L160)

**数学公式（论文 Eq.(6-10)）：**

约束矩阵 $C_{ij}$ 定义为：

$$C_{ij} = \left.\frac{d^{i-1}}{dr^{i-1}}\xi^O_j(r)\right|_{r_c}, \quad i = 1, \ldots, M_{\text{con}}$$

对约束矩阵做奇异值分解（SVD）：

$$C = U_C \Sigma_C V_C^T$$

**基本匹配赝波函数** $\tilde{\phi}_0$（满足约束但不归一化）：

$$\tilde{\phi}_0 = \sum_{j=1}^{N_{\text{bas}}} \left(\sum_{i=1}^{M_{\text{con}}} \frac{(U_C^T c)_i}{\sigma_i}\right) V_C(j,i) \cdot \xi^O_j$$

其中 $c$ 为约束向量（全电子波函数在 $r_c$ 处的值和导数）。

**零空间基函数** $\xi^N_k$（约束矩阵零空间中的正交基）：

$$\xi^N_k = \sum_{j=1}^{N_{\text{bas}}} V_C(j, M_{\text{con}}+k) \cdot \xi^O_j, \quad k = 1, \ldots, N_{\text{bas}}-M_{\text{con}}$$

氢原子 ncon=5, nbas=8，零空间维度 = 8-5 = 3。

### 7.5 残余动能矩阵元计算（eresid）

**源代码位置：** [eresid.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/eresid.f90#L19-L256)

**数学公式（论文 Eq.(1-2, 11)）：**

残余动能算子定义为：

$$E^r(q_c) = \frac{2}{\pi}\int_{q_c}^{\infty} q^4 |\tilde{\phi}(q)|^2 dq$$

其中 $\tilde{\phi}(q)$ 是赝波函数的傅里叶变换：

$$\tilde{\phi}(q) = \int_0^{\infty} r \cdot j_l(qr) \cdot \phi(r) \cdot r \, dr$$

矩阵元：

$$E^r_{00}(q_c) = \frac{2}{\pi}\int_{q_c}^{\infty} q^4 |\tilde{\phi}_0(q)|^2 dq$$

$$E^r_{0k}(q_c) = \frac{2}{\pi}\int_{q_c}^{\infty} q^4 \tilde{\phi}_0(q) \tilde{\xi}^N_k(q) dq$$

$$E^r_{kk'}(q_c) = \frac{2}{\pi}\int_{q_c}^{\infty} q^4 \tilde{\xi}^N_k(q) \tilde{\xi}^N_{k'}(q) dq$$

积分从 $q_{\max}$（视为无穷大）向内进行，在指定的 $q_c$ 处保存快照。

### 7.6 残余动能最小化（optimize）

**源代码位置：** [optimize.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/optimize.f90#L19-L232)

**数学公式（论文 Eq.(14-16)）：**

优化赝波函数表示为：

$$\tilde{\phi}_{\text{opt}} = \tilde{\phi}_0 + \sum_{k=1}^{N-M} x_k \xi^N_k$$

将 $E^r$ 矩阵对角化得到本征值 $e_i$ 和本征向量，在新基下：

$$E^r(x_1, \ldots, x_{N-M}) = E^r_{00} + \sum_{i=1}^{N-M}\left(2f_i x_i + e_i x_i^2\right)$$

其中 $f_i$ 为交叉项在新基下的表示。

**最小化过程（区间折半法）：**

1. $x_1$ 的符号与 $f_1$ 相反（保证极小值）
2. 对 $i \geq 2$：

$$x_i = -\frac{f_i}{e_i - e_1 + |f_1|/|x_1|}$$

3. 范数守恒约束（Eq.(15)）：

$$\sum_{i=1}^{N-M} x_i^2 = D_{\text{norm}} = Q_{ii} - \|\tilde{\phi}_0\|^2$$

4. 通过区间折半搜索 $|x_1|$ 满足范数约束

**氢原子结果：**
- 投影子 1：残余动能误差 = $1.36 \times 10^{-6}$ Ha
- 投影子 2：残余动能误差 = $3.35 \times 10^{-6}$ Ha

### 7.7 半局域赝势计算（pspot）

**源代码位置：** [pspot.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/pspot.f90#L19-L159)

```fortran
call pspot(iprj,ll,rr,irc,mmax,al,nbas,qroot,eig(iprj),uua(1,iprj), &
           pswfopt_sb,psopt(1,iprj),vae,work,vkb(1,iprj),ekin_num)
```

**数学公式：**

赝波函数在 $[0, r_c]$ 内由球贝塞尔函数展开构造：

$$\tilde{\phi}(r) = r \sum_{j=1}^{N_{\text{bas}}} c_j^{\text{opt}} j_l(q_j r), \quad r \leq r_c$$

在 $[r_c, \infty)$ 内等于全电子波函数。

半局域赝势由逆薛定谔方程得到：

$$V_l^{\text{ps}}(r) = \varepsilon_l - \frac{\hat{T}\tilde{\phi}_l(r)}{\tilde{\phi}_l(r)}, \quad r \leq r_c$$

其中 $\hat{T}$ 为动能算子（含离心势）：

$$\hat{T}\tilde{\phi} = \frac{1}{2}\left[\frac{l(l+1)}{r^2}\tilde{\phi} - \frac{d^2\tilde{\phi}}{dr^2}\right]$$

VKB 投影子的原始分量：

$$|\chi_i\rangle = (\varepsilon_i - \hat{T})|\tilde{\phi}_i\rangle$$

在 $r > r_c$ 区域，半局域势等于全电子势。

---

## 8. 局域势构造（vploc）

**源代码位置：** [vploc.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/vploc.f90#L19-L157)

由于 `lloc=4`，使用多项式外推全电子势到 $r=0$。

```fortran
if(lloc==4) then
  call vploc(rr,vfull,vp,dvloc0,irc(5),mmax,lpopt)
end if
```

**数学公式（lpopt=5）：**

在 $r_c$ 处匹配全电子势的值和 1-3 阶导数，构造多项式势：

$$V_{\text{loc}}(r) = a + b r^2 + c r^4 + d r^6, \quad r \leq r_c$$

系数由匹配条件确定：

$$V_{\text{loc}}(r_c) = V_{\text{AE}}(r_c), \quad V_{\text{loc}}'(r_c) = V_{\text{AE}}'(r_c), \quad V_{\text{loc}}''(r_c) = V_{\text{AE}}''(r_c), \quad V_{\text{loc}}'''(r_c) = V_{\text{AE}}'''(r_c)$$

额外添加平滑函数调整 $r=0$ 处的值：

$$V_{\text{loc}}(r) \to V_{\text{loc}}(r) + \Delta V_0 (1 - (r/r_c)^4)^4$$

氢原子 `dvloc0=0`，不添加额外修正。在 $r > r_c$ 区域，$V_{\text{loc}} = V_{\text{AE}}$。

---

## 9. Vanderbilt-Kleinman-Bylander 投影子构造（run_vkb）

**源代码位置：** [run_vkb.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/run_vkb.f90#L19-L315)

### 9.1 原始投影子构造

**数学公式（论文 Eq.(18)）：**

$$|\chi_i^{\text{raw}}\rangle = (\varepsilon_i - V_{\text{loc}} - V_r)\tilde{\phi}_i, \quad r \leq r_c$$

其中 $V_r$ 为标量相对论修正势（在 $r_c$ 附近最后 5% 范围内平滑加入）：

```fortran
do ii=1,irc(l1)
  xx=20.0d0*((rr(ii)/rr(irc(l1)))-1.0d0)
  if(xx>-1.0d0) then
    ff=(1.0d0-xx**2)**2
  else
    ff=0.0d0
  end if
  vkb(ii,jj,l1)=vkb(ii,jj,l1)-(vloc(ii)+ff*vr(ii,jj,l1))*pswf(ii,jj,l1)
end do
```

### 9.2 B 矩阵构造和对角化

**数学公式（论文 Eq.(21)）：**

$$B_{ij} = \langle\tilde{\phi}_i|\chi_j^{\text{raw}}\rangle$$

对 B 矩阵进行对称化后对角化：

$$B = U_B \Lambda_B U_B^T$$

构造对角化投影子：

$$|\chi_j\rangle = \sum_{i=1}^{n_{\text{proj}}} (U_B)_{ij} |\chi_i^{\text{raw}}\rangle$$

投影子系数：

$$e_j^{\text{VKB}} = 1/\lambda_j$$

### 9.3 正交归一化投影子

对于双投影子（nproj=2），进一步构造正交归一化投影子：

1. 计算投影子重叠矩阵 $S_{ij} = \langle\chi_i|\chi_j\rangle$
2. 对角化 $S$：$S = U_S \Sigma_S U_S^T$
3. 构造 $S^{1/2}$ 和 $S^{-1/2}$
4. 变换到正交基：$|\chi^{\text{orth}}\rangle = S^{-1/2}|\chi\rangle$
5. 构造 $B^{-1*} = S^{1/2,T} B^{-1} S^{1/2}$
6. 对角化 $B^{-1*}$ 得到最终正交投影子系数

**氢原子结果：** 正交投影子系数 = $-1.1258, -0.37415$

---

## 10. 赝价电荷密度和反屏蔽

**源代码位置：** [oncvpsp.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/oncvpsp.f90#L467-L568)

### 10.1 赝波函数求解（lschvkbb）

**源代码位置：** [lschvkbb.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/lschvkbb.f90#L19-L255)

使用 VKB 非局域势求解束缚态：

$$\left[-\frac{1}{2}\nabla^2 + V_{\text{loc}}(r) + \sum_{i=1}^{n_{\text{proj}}} e_i^{\text{VKB}} |\chi_i\rangle\langle\chi_i|\right]\tilde{\psi} = \varepsilon \tilde{\psi}$$

外向积分中使用 VKB 非局域势的矩阵元修正波函数。

### 10.2 电荷密度累积

```fortran
rhoae(:,kk)=(uu(:)/rr(:))**2
rhotae(:)=rhotae(:) + fa(nc+kk)*rhoae(:,kk)
...
rhops(:,kk)=(uu(:)/rr(:))**2
rho(:)=rho(:)+fa(nc+kk)*rhops(:,kk)
```

**数学公式：**

全电子价电荷密度：

$$\rho_{\text{AE}}(r) = \sum_{i} f_i \left|\frac{u_i(r)}{r}\right|^2$$

赝价电荷密度：

$$\rho_{\text{ps}}(r) = \sum_{i} f_i \left|\frac{\tilde{u}_i(r)}{r}\right|^2$$

### 10.3 模型芯电荷

氢原子 `icmod=0`，不使用非线性芯校正。

### 10.4 屏蔽势计算（vout）

**源代码位置：** [vout.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/vout.f90#L19-L194)

计算 Hartree 势和交换关联势：

$$V_{\text{scr}}(r) = V_H[\rho_{\text{ps}}](r) + V_{xc}[\rho_{\text{ps}}](r)$$

Hartree 势通过径向泊松方程求解：

$$V_H(r) = \frac{4\pi}{r}\int_0^r \rho(r')r'^2 dr' + 4\pi\int_r^{\infty} \rho(r')r' dr'$$

PBE 交换关联势通过 `excggc` 子程序计算。

### 10.5 赝原子总能量

```fortran
epstot = eeig + eexc - 0.5d0*eeel
```

$$E_{\text{ps}} = \sum_i f_i \varepsilon_i^{\text{ps}} + E_{xc}[\rho_{\text{ps}}] - \frac{1}{2}E_{e-e}[\rho_{\text{ps}}]$$

氢原子结果：$E_{\text{ps}} = -0.458761$ Ha。

### 10.6 反屏蔽

**数学公式：**

$$V_l^{\text{unscr}}(r) = V_l^{\text{scr}}(r) - V_{\text{scr}}(r)$$

```fortran
do l1=1,max(lmax+1,lloc+1)
  vpuns(:,l1)=vp(:,l1)-vo(:)
end do
```

在电荷密度为零的区域，强制反屏蔽势趋于 $-Z_{\text{ion}}/r$：

```fortran
do ii=mmax,1,-1
  if(rho(ii)==0.0d0) then
    do l1=1,max(lmax+1,lloc+1)
      vpuns(ii,l1)=-zion/rr(ii)
    end do
  else
    exit
  end if
end do
```

---

## 11. 诊断测试（run_diag）

**源代码位置：** [run_diag.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/run_diag.f90#L19-L153)

对每个角动量，比较全电子和赝势结果：

1. **束缚态**：比较本征值、波函数在 $r_c$ 处的值和斜率
2. **散射态**：比较 $r_c$ 处的对数导数

**氢原子结果：**

| l | r_core | r_match | e_in | Δe | norm_test | slope_test |
|---|--------|---------|------|-----|-----------|------------|
| 0 | 1.144 | 1.179 | -0.2386 | -2.2×10⁻⁶ | 1.000009 | 1.000019 |
| 0 | 1.144 | 1.179 | 1.0360 | -1.6×10⁻⁶ | 1.000053 | 1.000053 |

---

## 12. 幽灵态检测（run_ghosts）

**源代码位置：** [run_ghosts.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/run_ghosts.f90#L19-L292)

两种幽灵态测试：

### 12.1 负能幽灵态（GHOST(-)）

在局域势上加硬壁势垒（$3r_c$），构造基组并对角化哈密顿量：

$$H_{ij} = \varepsilon_i^{\text{barrier}}\delta_{ij} + \sum_{k=1}^{n_{\text{proj}}} e_k^{\text{VKB}} \langle\phi_i^{\text{barrier}}|\chi_k\rangle\langle\chi_k|\phi_j^{\text{barrier}}\rangle$$

如果对角化后的本征值低于全径向薛定谔方程的结果，则存在 GHOST(-)。

### 12.2 正能幽灵态（GHOST(+)）

计算正能态的平均半径 $\langle r \rangle$，若 $\langle r \rangle / r_c < 1$，则报告 GHOST(+)。

**氢原子结果：** 无幽灵态。

---

## 13. 测试组态计算（run_config）

**源代码位置：** [run_config.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/run_config.f90#L19-L293)

氢原子 `ncnf=0`，仅运行参考组态（组态 0）作为一致性检验。

**过程：**
1. 全电子原子自洽计算（sratom）
2. 赝原子自洽计算（psatom），使用 VKB 投影子
3. 比较本征值和总能量

**氢原子结果：**

| n | l | f | e_AE | e_PS | diff |
|---|---|---|------|------|------|
| 1 | 0 | 1.0 | -0.23860194 | -0.23860415 | -2.21×10⁻⁶ |

PSP 激发能误差 = $7.27 \times 10^{-15}$ Ha（参考组态，应为零）。

---

## 14. 对数导数分析（run_phsft）

**源代码位置：** [run_phsft.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/run_phsft.f90#L19-L102)

计算并比较全电子和赝势的对数导数：

$$\delta_l(\varepsilon) = \arctan\left(r_m \cdot \frac{d\psi/dr}{\psi}\bigg|_{r_m}\right)$$

其中 $r_m$ 为匹配半径（略大于 $r_c$）。

能量范围：$[\varepsilon_{\text{sh1}}, \varepsilon_{\text{sh2}}]$ = [-5.0, 3.0] Ha，步长 0.02 Ha。

---

## 15. 输出文件生成

### 15.1 绘图数据（run_plot）

输出势、电荷密度、波函数和收敛性曲线的绘图数据。

### 15.2 UPF 格式输出（upfout）

**源代码位置：** [upfout.f90](file:///home/linearline/project/oncvpsp-4.0.1/src/upfout.f90)

将赝势数据写入 Quantum ESPRESSO 兼容的 UPF v2.0.1 格式文件，包含：
- 局域势 $V_{\text{loc}}(r)$
- 非局域投影子 $\chi_i(r)$ 和系数 $e_i^{\text{VKB}}$
- 赝波函数 $\tilde{\phi}_i(r)$
- 赝价电荷密度 $\rho_{\text{ps}}(r)$
- 全电子电荷密度 $\rho_{\text{AE}}(r)$

---

## 16. 完整流程总结

```
┌─────────────────────────────────────────────────────────────────┐
│  1. 读取输入文件 H.dat                                          │
│     H, Z=1, nc=0, nv=1, iexc=4(PBE), lmax=0                   │
│     l=0: rc=1.13748, ncon=5, nbas=8, qcut=9.72141             │
│     lloc=4, lpopt=5, nproj=2, icmod=0                          │
├─────────────────────────────────────────────────────────────────┤
│  2. 构建对数径向网格 r_i = r_1·exp(α(i-1))                      │
│     α=ln(1.006), r_1=0.0005, M≈1470                            │
├─────────────────────────────────────────────────────────────────┤
│  3. 全电子自洽原子计算 (sratom)                                  │
│     Thomas-Fermi 初始势 → 自洽迭代                               │
│     lschfb 求解标量相对论薛定谔方程                               │
│     Anderson 混合法收敛 → ε_1s = -0.2386 Ha                    │
│     E_tot = -0.4589 Ha, 22 次迭代                               │
├─────────────────────────────────────────────────────────────────┤
│  4. 角动量循环 (l=0):                                           │
│  ├─ 4a. 获取全电子波函数 (lschfb)                                │
│  │     第1投影子: 1s 束缚态, ε=-0.2386 Ha                       │
│  │     第2投影子: 势阱态 (wellstate), ε=1.0360 Ha               │
│  ├─ 4b. 标量相对论修正势 (vrel)                                  │
│  ├─ 4c. 重叠矩阵 Q_ij (fpovlp)                                  │
│  ├─ 4d. 优化赝波函数 (run_optimize):                             │
│  │   ├─ wf_rc_der: r_c 处导数                                   │
│  │   ├─ qroots: 球贝塞尔波矢选择                                 │
│  │   ├─ sbf_basis_con: 正交化基组                                │
│  │   ├─ const_basis: 约束基组 (SVD)                              │
│  │   ├─ eresid: 残余动能矩阵元                                   │
│  │   ├─ optimize: 残余动能最小化                                  │
│  │   └─ pspot: 半局域赝势 + VKB原始投影子                        │
│  └─ 结果: ΔE^r_1=1.36e-6 Ha, ΔE^r_2=3.35e-6 Ha               │
├─────────────────────────────────────────────────────────────────┤
│  5. 局域势构造 (vploc)                                          │
│     lloc=4: 多项式外推 V_AE → V_loc(r) = a+br²+cr⁴+dr⁶       │
├─────────────────────────────────────────────────────────────────┤
│  6. VKB 投影子构造 (run_vkb)                                    │
│     B矩阵 → 对角化 → 正交归一化投影子                             │
│     系数: e₁=-1.1258, e₂=-0.37415                              │
├─────────────────────────────────────────────────────────────────┤
│  7. 赝价电荷密度计算 (lschvkbb)                                  │
│     反屏蔽: V_unscr = V_scr - V_Hartree - V_xc                  │
│     E_ps = -0.4588 Ha                                           │
├─────────────────────────────────────────────────────────────────┤
│  8. 诊断测试 (run_diag)                                         │
│     本征值误差 ~2×10⁻⁶ Ha, 范数/斜率比 ~1.00001                │
├─────────────────────────────────────────────────────────────────┤
│  9. 幽灵态检测 (run_ghosts)                                     │
│     无幽灵态 ✓                                                   │
├─────────────────────────────────────────────────────────────────┤
│  10. 测试组态 (run_config)                                      │
│      参考组态: PSP激发能误差 ≈ 0                                 │
├─────────────────────────────────────────────────────────────────┤
│  11. 对数导数分析 (run_phsft)                                    │
│      E ∈ [-5, 3] Ha, 步长 0.02 Ha                              │
├─────────────────────────────────────────────────────────────────┤
│  12. 输出 UPF 格式赝势文件 (upfout)                              │
│      H.upf → Quantum ESPRESSO 使用                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 参考文献

1. D. R. Hamann, Phys. Rev. B **88**, 085117 (2013) — ONCV 方法原始论文
2. D. R. Hamann, Phys. Rev. B **95**, 239906 (2017) — 勘误
3. D. Vanderbilt, Phys. Rev. B **41**, 7892 (1990) — 范数守恒广义条件
4. L. Kleinman and D. M. Bylander, Phys. Rev. Lett. **48**, 1425 (1982) — KB 分离形式
5. J. P. Perdew, K. Burke, and M. Ernzerhof, Phys. Rev. Lett. **77**, 3865 (1996) — PBE 泛函
