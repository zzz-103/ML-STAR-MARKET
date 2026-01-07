# SZU_w4

一个从“原始日线数据 → 因子面板 → 机器学习选股/权重输出”的本地化流水线项目。

## 目录结构

- `data_preprocess/`：原始 CSV 合并与数据补充（如行业、研发、流通市值等），产出 `pre_data/` 下的清洗/合并结果
- `generate_factors/`：因子计算与因子质量报告，产出 `factors_data/`
- `ml_models/`：多因子选股模型（XGBoost 为主），用因子面板训练/打分并输出逐日权重
- `pre_data/`：存放你的原始数据的地方（需要自行在根目录创建）
- `ml_results/`：模型权重生成文件，以及快速预览的结果的地方（需要自行在根目录创建）
- `requirement.txt`：项目运行环境依赖

## 环境准备
建议使用虚拟环境

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirement.txt
```

## 推荐的工作流

### 1) 数据合并与清洗（输出到 pre_data）

入口脚本：
- `python data_preprocess/merge_stockdata2.py --help`

它负责把分日/分股票的 CSV 合并为统一表，并支持把部分外部数据源合并到 CSV/Parquet（具体参数以 `--help` 为准）。

### 2) 计算因子（输出到 factors_data）

常用命令（详见 [generate_factors/readme.md](file:///Users/zhuzhuxia/Documents/SZU_w4/generate_factors/readme.md)）：

```bash
# 列出可用因子模块
python generate_factors/main.py --list-models

# 跑指定模块
python generate_factors/main.py --models ma_factors,vol_factors

# 只汇总/生成因子质量报告（不重算因子文件）
python generate_factors/main.py --summary-only --summary-start-date 20230101 --summary-end-date 20241231
```

### 3) 训练与生成权重（输出到 ml_results）

模型入口：
- `python ml_models/main.py --help`

常用方式是给定训练区间，滚动训练并输出逐日权重文件。更完整的输入/输出说明与参数示例见 [ml_models/readme.md](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/readme.md)。

## 说明

- 数据与产出路径默认写在各入口脚本顶部的常量里；迁移目录时优先改这些常量或用命令行参数覆盖。
- 若你希望依赖文件只包含“直接依赖”，需要改为手动维护；当前 `requirement.txt` 记录的是环境实际安装集合（含间接依赖），用于快速复现环境更省事。
