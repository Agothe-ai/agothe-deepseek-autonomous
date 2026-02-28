# skill: git_summary
import subprocess

n = context.get("n", 10)
lines_out = []

try:
    # Recent commits
    log = subprocess.run(
        ["git", "log", f"--oneline", f"-{n}", "--stat"],
        capture_output=True, text=True, timeout=10,
        encoding="utf-8", errors="replace"
    )
    if log.returncode == 0:
        lines_out.append(f"=== GIT SUMMARY | Last {n} commits ===")
        lines_out.append(log.stdout.strip())
    else:
        lines_out.append("[git_summary] Not a git repo or git not found.")

    # Branch status
    status = subprocess.run(
        ["git", "status", "--short"],
        capture_output=True, text=True, timeout=5,
        encoding="utf-8", errors="replace"
    )
    if status.returncode == 0:
        s = status.stdout.strip()
        lines_out.append(f"\nWorking tree: {'clean' if not s else s}")

except Exception as e:
    lines_out.append(f"[git_summary error] {e}")

result = "\n".join(lines_out)
