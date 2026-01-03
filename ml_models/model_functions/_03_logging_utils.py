# 位置: 03（日志）| main.py 创建运行日志（终端+文件，毫秒时间戳）
# 输入: log_dir(str), run_name(str)
# 输出: logging.Logger；并提供 log_section 分段输出
# 依赖: 标准库 logging/os/datetime
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime


class ConsoleFormatter(logging.Formatter):
    """
    终端专用格式化器：
    - INFO 级别：直接输出消息内容（不带时间/级别），保持清爽。
    - 其他级别（WARNING/ERROR）：带上 [LEVEL] 前缀以示区分。
    """
    def format(self, record):
        if record.levelno == logging.INFO:
            return record.getMessage()
        # 对于非 INFO 级别，保留一些警示信息
        return f"[{record.levelname}] {record.getMessage()}"


class FileFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__()
        self._is_first_record = True
        self._first_formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._rest_formatter = logging.Formatter(fmt="%(levelname)s | %(message)s")

    def format(self, record):
        if self._is_first_record:
            self._is_first_record = False
            return self._first_formatter.format(record)
        return self._rest_formatter.format(record)


def _is_console_handler(h: logging.Handler) -> bool:
    return isinstance(h, logging.StreamHandler) and (not isinstance(h, logging.FileHandler))


def _is_file_handler(h: logging.Handler) -> bool:
    return isinstance(h, logging.FileHandler)


def _emit_line(logger: logging.Logger, h: logging.Handler, line: str) -> None:
    record = logger.makeRecord(
        name=logger.name,
        level=logging.INFO,
        fn="",
        lno=0,
        msg=line,
        args=(),
        exc_info=None,
    )
    h.handle(record)


def build_logger(log_dir: str, run_name: str) -> logging.Logger:
    """创建同时输出到终端与文件的 INFO 级日志器（时间戳精确到毫秒）。"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    file_fmt = FileFormatter()

    # 终端日志：使用自定义精简格式
    console_fmt = ConsoleFormatter()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(console_fmt)
    logger.addHandler(ch)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh_path = os.path.join(log_dir, f"{run_name}_{ts}.log")
    fh = logging.FileHandler(fh_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    logger.info("log_file=%s", fh_path)
    return logger


def log_section(logger: logging.Logger, title: str) -> None:
    """输出分段标题，便于在终端与日志文件中定位运行阶段。"""
    logger.info("========== %s ==========", str(title))


def log_data_grid(logger: logging.Logger, data: dict, title: str = "Config") -> None:
    """
    以紧凑的网格形式输出字典内容，便于快速浏览关键参数。
    """
    if not data:
        return

    keys = sorted(data.keys())
    # Format "key: value" pairs
    items = []
    for k in keys:
        v = data[k]
        s_v = str(v)
        if len(s_v) > 50:  # Truncate long values
            s_v = s_v[:47] + "..."
        items.append(f"{k}: {s_v}")

    lines: list[str] = [f"┌── {title}"]

    current_line: list[str] = []
    current_len = 0
    max_width = 100

    for item in items:
        if current_len + len(item) + 4 > max_width:
            if current_line:
                lines.append("│ " + "   ".join(current_line))
            current_line = [item]
            current_len = len(item)
        else:
            current_line.append(item)
            current_len += len(item) + 4

    if current_line:
        lines.append("│ " + "   ".join(current_line))

    max_console_lines = 4
    if len(lines) <= max_console_lines:
        console_lines = lines
    else:
        kept = max(1, max_console_lines - 1)
        omitted = max(0, len(lines) - kept)
        console_lines = lines[:kept]
        console_lines.append(f"│ ... (省略{omitted}行，详见日志文件)")

    file_block = "\n".join(lines)
    console_block = "\n".join(console_lines)
    for h in list(getattr(logger, "handlers", [])):
        if _is_file_handler(h):
            _emit_line(logger, h, file_block)
        elif _is_console_handler(h):
            _emit_line(logger, h, console_block)
        else:
            _emit_line(logger, h, file_block)
