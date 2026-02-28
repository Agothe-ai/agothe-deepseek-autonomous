# skill: system_report
# Returns a full system health report
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

lines = []
lines.append(f"=== SYSTEM REPORT | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

# Python
lines.append(f"Python: {sys.version.split()[0]} at {sys.executable}")

# Disk
t, u, f = shutil.disk_usage(".")
lines.append(f"Disk: {f//10**9:.1f}GB free / {t//10**9:.1f}GB total ({100*u//t}% used)")

# Repo status
try:
    git = subprocess.run(["git", "log", "--oneline", "-3"], capture_output=True, text=True, timeout=5)
    if git.returncode == 0:
        lines.append("Last 3 commits:")
        for l in git.stdout.strip().split("\n"):
            lines.append(f"  {l}")
except Exception:
    lines.append("Git: not available")

# Skills count
sk_dir = Path("skills")
if sk_dir.exists():
    count = sum(1 for p in sk_dir.iterdir() if p.is_dir())
    lines.append(f"Skills loaded: {count}")

result = "\n".join(lines)
