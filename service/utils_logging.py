# service/utils_logging.py
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

def append_jsonl(file_path: str, record: Dict[str, Any]) -> None:
    """
    将单条记录以 JSONL 的形式追加写入到 file_path。
    如果父目录不存在则自动创建。
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 标准 UTC 时间戳，便于后续分析
    if "timestamp" not in record:
        record["timestamp"] = datetime.now(timezone.utc).isoformat()

    line = json.dumps(record, ensure_ascii=False)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")