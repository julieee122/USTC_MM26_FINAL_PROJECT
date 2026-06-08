# Mesa ABM 实验

本目录用于实现双边平台竞争的 Agent-Based Model 实验。当前仅保留修订后的 ABM 实验代码，不再维护最初的基础 ABM 示例脚本。

## 核心文件

- `abm_experiment_model.py`：ABM 引擎，支持个体异质性、精准补贴、商户多归属、冷启动种子策略等机制。
- `run_abm_experiments.py`：运行 ABM 实验并生成图像、汇总表和细粒度参数扫描结果。
- `redraw_abm_figures_from_tables.py`：基于已有 CSV 结果表重新绘制报告图像，不重新运行仿真。
- `Mesa_ABM_实验报告.md`：Markdown 实验报告。
- `Mesa_ABM_实验报告.tex`：LaTeX 实验报告。

## 实验内容

当前 ABM 实验包括：

1. 临界区个体异质性实验；
2. 等预算补贴策略与临界预算实验；
3. 冷启动二维资源配比与种子质量实验；
4. 商户多归属成本阈值实验。

## 运行方式

安装依赖：

```powershell
pip install -r requirements.txt
```

运行完整 ABM 实验：

```powershell
python .\run_abm_experiments.py
```

仅根据已有结果表重绘图像：

```powershell
python .\redraw_abm_figures_from_tables.py
```

编译 LaTeX 报告：

```powershell
xelatex -interaction=nonstopmode -halt-on-error -jobname=Mesa_ABM_report "\input{Mesa_ABM_实验报告.tex}"
xelatex -interaction=nonstopmode -halt-on-error -jobname=Mesa_ABM_report "\input{Mesa_ABM_实验报告.tex}"
```

## 输出目录

- `results/figures/`：实验图像；
- `results/tables/`：实验结果表；
- `Mesa_ABM_report.pdf`：由 LaTeX 报告编译得到的 PDF。
