# ml_models：预测 / 训练 / 持仓 逻辑说明

本文档解释目录 [/ml_models](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models) 的核心流程：数据与标签构建、滚动训练与预测（打分）、以及如何把分数转为逐日持仓权重文件。

## 1. 程序入口与整体流程

推荐入口是模块化主程序：

- 入口：[main.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/main.py)
- 运行方式：`python -m ml_models.main ...`
- 参数解析：[model_functions/_01_cli.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_01_cli.py)
- 默认参数集中配置：[xgb_config.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/xgb_config.py)

主流程（按实际执行顺序）：

1. 解析参数并输出摘要日志（[main.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/main.py#L148-L250)，[ _16_run_params.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_16_run_params.py#L8-L52)）
2. 构建输出目录：保存持仓 CSV、临时分数 parquet、日志（[main.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/main.py#L82-L103)）
3. 构建训练/预测面板数据 df_ml（特征 + 标签 ret_next）与价格面板 df_price（[main.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/main.py#L253-L256)，[ _06_data_preprocessing.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_06_data_preprocessing.py#L55-L120)）
4. 特征过滤（drop list + turnover 特殊规则）（[main.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/main.py#L263-L272)，[ _04_feature_engineering.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_04_feature_engineering.py#L11-L29)）
5. 生成滚动任务列表 tasks：每个 target_date 对应一段训练窗口（[main.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/main.py#L273-L310)）
6. 并行滚动训练并对 target_date 打分，写入 temp_dir/YYYYMMDD.parquet（[main.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/main.py#L312-L354)，[ _09_scoring.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_09_scoring.py#L25-L153)）
7. 临时评估（同日 ret_next vs score）并输出日志报告（[main.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/main.py#L355-L366)）
8. 因子重要性（在最后一个 target_date 的训练窗拟合一次 XGB 并导出 gain 等）（[main.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/main.py#L368-L376)，[ _12_factor_importance.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_12_factor_importance.py#L20-L131)）
9. 把每日分数序列转成逐日持仓权重 CSV（TopK + 缓冲 + 调仓 + 择时/风控）（[main.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/main.py#L377-L379)，[ _11_portfolio.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L19-L342)）

另外目录下还保留了单文件版 runner：

- 单文件版：[xgb_knn_runner_tech.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/xgb_knn_runner_tech.py)  
  它与模块化版本的核心逻辑一致（prepare_dataset / process_single_day_score / generate_positions_with_buffer 的整体结构相同），但维护优先级建议以 main.py 为准。

## 2. 数据结构与关键约束

### 2.1 面板索引规范（MultiIndex）

整个流程依赖一个约定：

- 因子数据 df_factors：索引为 `MultiIndex(date, code)`，并按索引排序
- 价格数据 df_price：索引为 `MultiIndex(date, code)`，并按索引排序

在构建数据时会强制转换/整理：

- 因子索引整理：[ensure_factors_index](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_06_data_preprocessing.py#L28-L39)

### 2.2 股票池过滤（Universe）

仅使用指定前缀的股票代码（默认 `("300","688")`）：

- 默认配置：[xgb_config.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/xgb_config.py#L17-L20)
- 实际过滤：[ _06_data_preprocessing.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_06_data_preprocessing.py#L65-L68)

### 2.3 end-date 截断（防止“未来数据”进入标签）

当你传入 `--end-date` 时，程序会在构建标签前先把因子与价格数据截断到 `<= end_date`，确保 end_date 之后的数据不会被用于计算 `ret_5d/ret_10d`。

- 实现位置：[ _06_data_preprocessing.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_06_data_preprocessing.py#L70-L84)

这条规则直接决定了一个很重要的性质：

- 标签需要 `t+10` 的 open，因此在 `end_date=20241231` 时，`ret_10d` 在 2024-12-18 之后不可得，面板 df_ml 会自然截止到 **2024-12-17**（后面的行会因为 label 为 NaN 被 drop）。
- 你在日志里会看到 df_ml 的最大日期是 20241217（示例验证输出来自 prepare_dataset 的 log）。

## 3. 标签（Label）定义：预测什么

### 3.1 原始收益标签：0.5*5日 + 0.5*10日

标签是未来收益（用 open 计算）：

- `ret_5d(t) = (open(t+5) - open(t)) / open(t)`
- `ret_10d(t) = (open(t+10) - open(t)) / open(t)`
- `ret_next_raw(t) = 0.5*ret_5d(t) + 0.5*ret_10d(t)`

实现位置：

- [ _06_data_preprocessing.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_06_data_preprocessing.py#L90-L95)

### 3.2 超额收益：减去当日横截面基准

为了让模型学的是“相对强弱/超额”，会把 `ret_next_raw` 减去当日市场基准：

1. 先构建 tradable 掩码（open>0、非涨跌停、流动性满足阈值）
2. 基准取横截面 mean 或 median（由参数控制）
3. `ret_next = ret_next_raw - market_benchmark`

实现位置：

- tradable：[build_tradable_mask](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_06_data_preprocessing.py#L41-L52)
- benchmark + ret_next：[ _06_data_preprocessing.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_06_data_preprocessing.py#L96-L107)

其中 `label_benchmark_universe` 决定基准用“全市场”还是仅 “tradable” 子集：

- 参数入口：[ _01_cli.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_01_cli.py#L85-L90)

## 4. 特征（Feature）如何避免未来函数

### 4.1 特征滞后 shift(1)

因子特征按股票做 `shift(1)`，即在日期 t 预测时，输入特征是 t-1 的因子值：

- [ _06_data_preprocessing.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_06_data_preprocessing.py#L74-L76)

这条规则是最关键的反泄漏措施之一：即使因子原始数据在日期 t 里包含盘后信息，shift(1) 也会把它推迟到下一天使用。

### 4.2 特征过滤：drop list + turnover 规则

训练时使用的特征来自 df_ml 除 `ret_next` 外的全部列，然后应用过滤：

- 默认 drop 列表（可扩展）：[xgb_config.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/xgb_config.py#L114-L150)
- 过滤规则实现：[ _04_feature_engineering.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_04_feature_engineering.py#L11-L29)

注意：凡是以 `turnover_` 开头的特征，除 `turnover_bias_5` 外都会被剔除（这是一个硬编码的规则）。

## 5. 滚动训练与预测（打分）逻辑

### 5.1 训练窗口与预测日期序列

df_ml 里所有可用交易日为 `all_dates`。程序会从足够长的历史后开始预测：

- `predict_dates = all_dates[train_window + train_gap:]`
- 再按 `--start-date/--end-date` 过滤

实现位置：[main.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/main.py#L273-L285)

### 5.2 每个预测日对应的训练区间（train_gap 的意义）

对每个 `target_date`，训练区间为：

- `train_end_date = all_dates[pos - train_gap]`
- `train_start_date = all_dates[pos - train_gap - train_window + 1]`

实现位置：[main.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/main.py#L290-L305)

`train_gap` 的作用是把训练集与预测日拉开，降低标签计算（需要 t+10）对预测日附近数据的“穿越”风险：训练集最后一天的标签最多用到 `train_end_date+10`，当 `train_gap>=12` 时，这些未来 open 仍然严格早于 target_date。

### 5.3 Worker 训练与当日打分

每个 target_date 都会训练一个模型，并只对当天股票做预测分数（score）：

- worker 实现：[process_single_day_score](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_09_scoring.py#L25-L153)
- 训练数据：`df_ml[train_start_date:train_end_date]`
- 测试数据：`df_ml[target_date]`

模型默认是 XGBRegressor（也支持 rank: 目标）：

- 拟合封装：[fit_xgb_model](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_08_xgb_training.py#L12-L68)

### 5.4 样本权重（时间衰减）

训练时可给每条样本按“距离 train_end_date 的天数”做衰减：

- 权重生成：[build_sample_weights](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_05_weights.py#L11-L37)
- worker 调用入口：[ _09_scoring.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_09_scoring.py#L53-L72)

默认配置为指数衰减（`time_decay_exp`），具体参数在 [xgb_config.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/xgb_config.py#L28-L37)。

### 5.5 单调约束（Monotone Constraints）

如启用 `--use-constraints`，会把配置表映射成 XGBoost 的 `monotone_constraints` 向量：

- 构建约束字典：[build_constraints_dict](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_04_feature_engineering.py#L32-L48)
- 生成向量字符串：[build_monotone_constraints](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_04_feature_engineering.py#L51-L56)

默认约束集合在 [xgb_config.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/xgb_config.py#L152-L173)。

### 5.6 可选：KNN 融合

当 `--use-knn` 开启时，会训练一个 KNNRegressor 并把 XGB 与 KNN 的分数做 rank 融合：

- KNN 训练：缺失用训练集特征中位数填充 + StandardScaler 标准化
- 融合方式：分别对 `pred_xgb/pred_knn` 做横截面 `rank(pct=True)`，再按权重线性组合

实现位置：[ _09_scoring.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_09_scoring.py#L94-L118)

### 5.7 临时分数文件（temp_scores）

worker 输出为 `temp_dir/YYYYMMDD.parquet`，至少包含两列：

- `code`
- `score`

写入位置：[ _09_scoring.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_09_scoring.py#L119-L153)

## 6. 持仓生成：分数如何变成逐日权重

持仓生成由：

- [generate_positions_with_buffer](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L19-L342)

读取所有 `temp_dir/*.parquet`，按日期顺序逐日输出到 `save_dir/YYYYMMDD.csv`。

### 6.1 分数平滑（smooth_window）

为了减少换手，会用最近 `smooth_window` 天的 score 做均值平滑：

- 把每日 `score` 变成 `Series(code->score)` 加入 `history_scores`
- `mean_scores = mean(history_scores)` 得到平滑后的分数

实现位置：[ _11_portfolio.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L90-L99)

### 6.2 缓冲池（buffer_k）与惯性（inertia_ratio）

核心是“TopK 买入 + TopBuffer 持有”的双轨制：

1. 先对平滑分数降序排序得到 `ranked_raw`
2. 对昨日持仓给予“惯性加成”：把持仓股的 score 乘以 `inertia_ratio` 再重排
3. 选股：
   - 先保留昨日持仓中仍在 top `buffer_k` 的股票
   - 再从高分往下补足到 `top_k`

实现位置：

- 惯性加成：[ _11_portfolio.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L218-L224)
- 缓冲选股：[ _11_portfolio.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L225-L246)

### 6.3 调仓周期（rebalance_period）与非调仓日行为

只有在调仓日（或首次建仓/触发止损）才会真正改变持仓；否则按 `non_rebalance_action`：

- `empty`：当日输出空文件（表示不持仓）
- `carry`：沿用上一次持仓权重（并叠加择时缩放）

实现位置：[ _11_portfolio.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L202-L216)

### 6.4 止损（emergency_exit_rank）

如果持仓股在当日排序里掉出到 `emergency_exit_rank` 之后（排名更差），会触发“强制调仓日”：

- 计算持仓股在 `ranked_raw` 中的名次，若超过阈值，则 `is_rebalance_day=True`

实现位置：[ _11_portfolio.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L169-L201)

### 6.5 可交易与涨跌停/流动性过滤

先对候选集合（目标股 + 旧持仓）计算 buyable：

- open > 0
- turnover_prev 或 turnover > min_turnover
- 非涨跌停（通过 upper_limit/lower_limit 判断）

实现位置：[ _11_portfolio.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L247-L272)

### 6.6 limit_policy：冻结不可交易的旧持仓

当 `limit_policy=freeze` 时：

- 昨日持仓里如果今天不可交易，会被强制保留在组合里（不能卖出）
- 价格缺失的旧持仓也会被强制保留

实现位置：[ _11_portfolio.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L274-L282)

### 6.7 换手上限（rebalance_turnover_cap）

每次调仓最多允许买入 `max_new = floor(cap * top_k)` 只新股票：

- `cap=0` 表示完全不引入新票
- `cap=1` 表示最多可以买满 top_k

实现位置：[ _11_portfolio.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L283-L288)

### 6.8 权重分配、上限与最小权重

权重逻辑是“卖出释放预算 -> 等权买入新票”，并叠加约束：

- 卖出集合：旧持仓中不在 desired_set 的 + 止损集合，并且必须 buyable
- 买入预算：卖出权重之和
- 新买入等权：`buy_budget / len(buy_keep)`
- 单票上限：`max_w`（通过重分配实现）  
  实现：[apply_max_weight_cap](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_05_weights.py#L40-L70)
- 最小权重：`min_weight`（低于阈值直接删掉）
- band_threshold：极小权重过滤 / 极小卖出过滤

实现位置：[ _11_portfolio.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L296-L325)

### 6.9 择时（timing_method）与最终输出

择时会把当日组合整体乘以 `timing_scale`（例如风险关闭时降低仓位），当 scale<=0 时输出空仓。

支持的择时方法：

- `none`：永远满仓（scale=1）
- `index_ma20`：指数 close 与 MA20 比较（带 buffer），信号 **shift(1)**（用昨日收盘/均线作为今日信号）
- `index_ma_dual`：指数快慢均线比较，同样 **shift(1)**
- `score`：用 top30 平均分数与阈值/滞回判断 risk_on

实现位置：

- 风控数据加载（含 shift(1)）：[ _07_risk_signals.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_07_risk_signals.py#L12-L71)
- 择时状态机与 scale：[ _11_portfolio.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L99-L164)
- 最终写文件（无表头，两列 code/weight）：[ _11_portfolio.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_11_portfolio.py#L335-L342)

## 7. 输出文件与目录结构

基于 `--output-dir / --sub-dir-name / --temp-dir-name`：

- `output_dir/sub_dir_name/`
  - `YYYYMMDD.csv`：逐日权重文件（无表头），每行 `code,weight`
  - `eval_report.txt`：临时评估报告（如果生成）
- `output_dir/temp_dir_name/`
  - `YYYYMMDD.parquet`：逐日分数（code, score[, turnover_prev]）
- `output_dir/logs/`
  - 运行日志
- `factors_importance_dir/`
  - 因子重要性导出（csv/meta/brief）

默认路径配置见：[xgb_config.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/xgb_config.py#L7-L15)

## 8. 过拟合自检（可选）

如果使用 `--overfit-check`：

- 会在训练窗口内再切分出最近 `overfit_valid_days` 作为 validation
- 输出 train/valid 的 RMSE、IC 均值、IC IR、TopK 平均收益等，辅助判断“训练表现远好于验证”的过拟合迹象

实现位置：[ _14_overfit.py](file:///Users/zhuzhuxia/Documents/SZU_w4/ml_models/model_functions/_14_overfit.py#L48-L195)

## 9. 一句话总结“你在预测什么”

在日期 t：

- 输入：股票在 t-1 的因子特征（因子 shift(1)）
- 目标：股票从 t 开始未来 5/10 交易日的 open 收益加权（并减去当日基准后的超额收益）
- 模型：对历史滚动窗拟合的 XGB（可选 KNN 融合），输出当天横截面排序分数
- 组合：对分数做平滑 + 缓冲 + 调仓 + 可交易过滤 + 择时缩放，生成当日权重文件

