# Mesa 多智能体仿真实验

本目录用于实现双边平台竞争的 Agent-Based Model。

## 基础类

- `UserAgent`：用户智能体，根据平台服务质量、商户份额、补贴、价格、切换成本和个体偏好噪声选择平台。
- `MerchantAgent`：商户智能体，根据平台服务质量、用户份额、补贴、抽成、切换成本和个体偏好噪声选择平台。
- `TwoSidedPlatformModel`：Mesa 模型类，维护用户和商户代理、计算平台 A 份额并记录时间序列。
- `ABMParams`：参数类，与前面 ODE 模型参数保持一致，并额外加入 `sigmaU`、`sigmaM` 表示个体偏好噪声强度。

## 运行

先安装依赖：

```powershell
& 'C:\Users\29090\.conda\envs\ai2025\python.exe' -m pip install -r .\AgentBasedModel_Mesa\requirements.txt
```

运行基础仿真：

```powershell
& 'C:\Users\29090\.conda\envs\ai2025\python.exe' .\AgentBasedModel_Mesa\run_abm_basic.py
```

输出：

- `results/tables/abm_basic_run.csv`
- `results/figures/abm_basic_dynamics.png`
