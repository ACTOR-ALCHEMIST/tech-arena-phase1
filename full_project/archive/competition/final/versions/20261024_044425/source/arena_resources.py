"""Append a small local resource sample without importing training frameworks."""
import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import psutil

root=Path(__file__).resolve().parent/"competition"
memory=psutil.virtual_memory()
processes=[]
for process in psutil.process_iter(["pid","cmdline","memory_info"]):
    try:
        args=process.info["cmdline"] or []
        if any("arena_" in value and value.endswith(".py") for value in args):
            processes.append({"pid":process.pid,"rss_gb":process.info["memory_info"].rss/1e9,"command":args[-3:]})
    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
record={"time":datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds"),
        "cpu_percent":psutil.cpu_percent(interval=0.2),"available_gb":memory.available/1e9,
        "swap_used_gb":psutil.swap_memory().used/1e9,"processes":processes}
with (root/"resources.jsonl").open("a") as handle:handle.write(json.dumps(record)+"\n")
print(json.dumps(record))
