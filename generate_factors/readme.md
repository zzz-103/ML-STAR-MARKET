列出可用因子程序（模块）：
- python generate_factors/main.py --list-models

只跑新写的某个模块（例如你新增了 models/my_new_factor.py ）：
- python generate_factors/main.py --models my_new_factor

只跑某几个模块：
- python generate_factors/main.py --models ma_factors,vol_factors

只保存某几个因子（假设模块返回列名是 ma5_10_diff 等）：
- python generate_factors/main.py --factors ma5_10_diff

强制重跑并覆盖已有输出：
- python generate_factors/main.py --models ma_factors --force

只生成因子质量汇总（不重算因子文件）：
```bash
# 在项目根目录执行
python generate_factors/main.py \
  --summary-only \
  --summary-start-date 20230101 \
  --summary-end-date 20241231
```
