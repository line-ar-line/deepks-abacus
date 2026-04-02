train_args:
  # 现有参数保持不变
  decay_rate: 0.5
  decay_steps: 1000
  display_epoch: 100:
  n_epoch: 5000
  start_lr: 0.0001
  
  # ✨ 添加这两行即可！
  band_factor: 1.0      # 启用能级能量损失
  band_occ: 1           # H原子只约束1个占据态
