# 阶段一基础实验

本目录实现《双边平台竞争_分阶段实验设置与实现方式.md》中的第一阶段实验 1-5。

主模型采用：

- Logit 选择概率；
- 用户份额和商户份额的动态调整方程；
- `scipy.integrate.solve_ivp` 数值求解。

推荐运行方式（已在 `ai2025` 环境验证）：

```powershell
& 'C:\Users\29090\.conda\envs\ai2025\python.exe' .\阶段一基础实验\run_stage1.py
```

如果当前终端已激活 `ai2025`，也可以运行：

```powershell
python .\阶段一基础实验\run_stage1.py
```

输出：

- `results/figures/`：实验图像；
- `results/tables/`：实验汇总 CSV 与参数扫描矩阵。
