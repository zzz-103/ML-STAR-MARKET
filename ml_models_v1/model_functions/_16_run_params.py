# 位置: 16（运行参数摘要）| main.py 输出关键参数到日志
# 输入: args/objective/model_kind/timing_method
# 输出: (s1, s2) 两行字符串（便于日志检索与复现）
# 依赖: 无
from __future__ import annotations


def format_run_params(args, objective: str, model_kind: str, timing_method: str) -> tuple[str, str]:
    xgb_sub = float(getattr(args, "subsample", 0.8))
    xgb_rl = float(getattr(args, "reg_lambda", 1.0))
    sw_mode = str(getattr(args, "sample_weight_mode", "none"))

    # Mapping English terms to Chinese for better readability
    sw_mode_cn = sw_mode
    if sw_mode == "time_decay_exp":
        sw_mode_cn = "指数衰减"
    elif sw_mode == "time_decay_linear":
        sw_mode_cn = "线性衰减"
    elif sw_mode == "none":
        sw_mode_cn = "无"

    timing_cn = timing_method
    if timing_method == "index_ma20":
        timing_cn = "指数MA20"
    
    s1 = (
        f"[组合配置] "
        f"训练窗口={int(getattr(args, 'train_window', 0))}日 | "
        f"标签=5日收益 | "
        f"持仓(Top={int(getattr(args, 'top_k', 0))}, 缓冲={int(getattr(args, 'buffer_k', 0))}) | "
        f"调仓={int(getattr(args, 'rebalance_period', 0))}日(Cap={float(getattr(args, 'rebalance_turnover_cap', 0.0)):.2f}) | "
        f"平滑={int(getattr(args, 'smooth_window', 0))}日 | "
        f"惯性={float(getattr(args, 'inertia_ratio', 0.0)):.3g} | "
        f"止损排名={int(getattr(args, 'emergency_exit_rank', 0))} | "
        f"择时={timing_cn} | "
        f"风控指数={str(getattr(args, 'risk_index_code', ''))} | "
        f"MA={int(getattr(args, 'risk_ma_window', 0))} | "
        f"权重={sw_mode_cn}"
    )
    
    obj_cn = "回归(MSE)" if objective == "reg:squarederror" else objective
    if objective.startswith("rank:"):
        obj_cn = "排序(Rank)"
        
    s2 = (
        f"[模型参数] "
        f"目标={obj_cn} | 类型={model_kind} | "
        f"XGB(树={int(getattr(args, 'n_estimators', 0))}, lr={float(getattr(args, 'learning_rate', 0.0)):.3g}, "
        f"深度={int(getattr(args, 'max_depth', 0))}, 采样={xgb_sub:.2f}, 正则={xgb_rl:.3g}) | "
        f"KNN融合={'开启' if bool(getattr(args, 'use_knn', False)) else '关闭'}"
    )
    return s1, s2
