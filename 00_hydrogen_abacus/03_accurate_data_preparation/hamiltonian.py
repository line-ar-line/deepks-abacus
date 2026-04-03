import numpy as np

# 方法1: 对角哈密顿量 (最简单, 最干净)
nlocal = 9
eps_exact_hartree = np.array([
    -0.5,        # n=1, 1s  (-13.6057 eV)
    -0.125,      # n=2, 2s/2p (-3.4014 eV) ×4个简并
    -0.125,
    -0.125,
    -0.125,
    -0.055556,   # n=3, 3s/3p (-1.5118 eV) ×4个简并
    -0.055556,
    -0.055556,
    -0.055556,
])

H = np.diag(eps_exact_hartree).astype(np.complex128)
hamiltonian = H[np.newaxis, np.newaxis, :, :]  # (1, 1, 9, 9)

np.save("/home/linearline/project/00_hydrogen_abacus/03_accurate_data_preparation/hamiltonian.npy", hamiltonian)

# 验证
eps_check = np.linalg.eigvalsh(hamiltonian[0,0])
print(f"Eigenvalues: {eps_check.real}")
print(f"Match exact: {np.allclose(eps_check.real, eps_exact_hartree)}")