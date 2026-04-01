# Project 文件结构

## 项目概述

本项目是一个深度学习与密度泛函理论（DFT）计算相关的项目集合，包含多个子项目和依赖库。

## 文件结构

```
project/
├── 00_hydrogen_abacus/       # 氢原子 ABACUS 计算示例
├── DeePKS-L/                 # DeePKS-L 主项目（子模块）
│   ├── deepks/              # 核心 Python 包
│   ├── scripts/             # 辅助脚本
│   ├── tests/               # 测试文件
│   ├── examples/            # 使用示例
│   ├── docs/                # 文档
│   └── setup.py             # 安装配置
├── abacus-develop/           # ABACUS DFT 软件包（子模块）
│   ├── source/              # 源代码
│   ├── build/               # 编译目录
│   ├── docs/                # 文档
│   ├── examples/            # 示例
│   └── tests/               # 测试
├── deepks-kit/              # DeePKS 工具包
│   ├── deepks/              # Python 包
│   ├── examples/            # 示例
│   ├── docs/                # 文档
│   └── scripts/             # 脚本
├── LibComm/                 # 通信库
├── LibRI/                   # RI 积分库
├── libnpy/                  # NumPy 文件读写库
├── libtorch/                # PyTorch C++ 库
│   ├── bin/                 # 可执行文件
│   ├── include/             # 头文件
│   ├── lib/                 # 库文件
│   └── share/               # 共享文件
├── libxc/                   # LibXC 交换相关泛函库（编译后）
├── libxc-7.0.0/             # LibXC 7.0.0 源代码
│   ├── src/                 # 源代码
│   ├── build/               # 编译目录
│   └── examples/            # 示例
├── abacus.sh                # ABACUS 编译脚本
├── deepks.sh                # DeePKS 运行脚本
├── deepks-kit-analysis.md   # DeePKS 工具包分析文档
├── .gitignore               # Git 忽略配置
├── .gitmodules              # Git 子模块配置
└── README.md                # 本文件
```

## 包含的文件/文件夹

以下文件和文件夹已纳入 Git 版本控制：

### 核心项目
- `DeePKS-L/` - DeePKS-L 主项目
- `abacus-develop/` - ABACUS DFT 软件
- `deepks-kit/` - DeePKS 工具包
- `00_hydrogen_abacus/` - 计算示例

### 依赖库
- `LibComm/` - 通信库
- `LibRI/` - RI 积分库
- `libnpy/` - NumPy 文件读写库

### 编译后的库文件
- `libxc/` - LibXC 编译后的库文件
- `libtorch/` - PyTorch C++ 库

### 源代码和配置
- `libxc-7.0.0/` - LibXC 源代码（不包含编译产物）
- `*.sh` - 编译和运行脚本
- `*.md` - 文档文件

### Git 配置
- `.gitignore` - Git 忽略规则
- `.gitmodules` - 子模块配置

## 未包含的文件/文件夹

以下文件和文件夹被 `.gitignore` 排除，不在 Git 版本控制中：

### 大型压缩包
- `libtorch-shared-with-deps-2.11.0+cpu.zip` - libtorch 压缩包
- `libxc-7.0.0.tar.bz2` - libxc 源代码压缩包

### 编译生成的文件
- `*.o` - 目标文件
- `*.so` - 共享库文件
- `*.a` - 静态库文件

### 临时文件和目录
- `tmp/` - 临时文件目录
- `pip_cache/` - pip 缓存目录

### Python 相关文件
- `__pycache__/` - Python 缓存目录
- `*.pyc` - Python 字节码文件
- `*.egg-info/` - Python 包信息
- `build/` - Python 构建目录
- `dist/` - Python 分发目录

### IDE 相关文件
- `.vscode/` - VS Code 配置
- `.idea/` - PyCharm 配置

### 其他
- `*.log` - 日志文件
- `*.cfg` - 配置文件
- `*.ini` - 初始化文件
- `.DS_Store` - macOS 系统文件
- `Thumbs.db` - Windows 系统文件

## 子模块说明

本项目使用 Git 子模块管理以下项目：

1. **DeePKS-L** - 深度学习 Kohn-Sham 方法
2. **abacus-develop** - ABACUS 第一性原理计算软件

克隆项目时请使用：
```bash
git clone --recursive <repository-url>
```

或克隆后初始化子模块：
```bash
git submodule update --init --recursive
```

## 编译和安装

### ABACUS 编译
```bash
bash abacus.sh
```

### DeePKS 运行
```bash
bash deepks.sh
```

## 依赖说明

- **libtorch** - PyTorch C++ API，用于深度学习模型推理
- **libxc** - 交换相关泛函库，用于 DFT 计算
- **Python 依赖** - 见各子项目的 `requirements.txt`

## 许可证

各子项目有独立的许可证，请参考各自的 LICENSE 文件。
