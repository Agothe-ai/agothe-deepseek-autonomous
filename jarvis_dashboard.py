# jarvis_dashboard.py — Jarvis Master Status Dashboard v8.0
# One screen. All 8 engines. Live status. Dark terminal UI.
# Shows: memory stats, task queue, github watcher, self-heal, voice, coder engine

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ANSI colors
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
CLEAR = "\033[2J\033[H"


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def status_icon(ok: bool) -> str:
    return colorize("● ONLINE", GREEN) if ok else colorize("○ OFFLINE", DIM)


def check_file(path: str) -> bool:
    return Path(path).exists()


def read_last_lines(path: str, n: int = 5) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        return lines[-n:]
    except Exception:
        return []


def get_memory_stats() -> dict:
    try:
        vectors = Path("memory/vectors.jsonl")
        if vectors.exists():
            count = sum(1 for _ in open(vectors) if _.strip())
            return {"vectors": count, "ok": True}
    except Exception:
        pass
    return {"vectors": 0, "ok": False}


def get_task_stats() -> dict:
    try:
        tasks = Path("tasks.jsonl")
        if not tasks.exists():
            return {"pending": 0, "done": 0, "ok": True}
        counts = {"pending": 0, "running": 0, "done": 0, "failed": 0}
        for line in open(tasks):
            if line.strip():
                d = json.loads(line)
                s = d.get("status", "unknown")
                counts[s] = counts.get(s, 0) + 1
        return {**counts, "ok": True}
    except Exception:
        return {"pending": 0, "done": 0, "ok": False}


def get_github_stats() -> dict:
    try:
        log = Path("github_watch_log.jsonl")
        last_seen = Path("github_last_seen.json")
        reviews = 0
        last_repo = "—"
        last_verdict = "—"
        if log.exists():
            lines = read_last_lines("github_watch_log.jsonl", 20)
            for line in reversed(lines):
                try:
                    d = json.loads(line)
                    reviews += 1
                    if last_repo == "—":
                        last_repo = d.get("repo", "—")
                        last_verdict = d.get("verdict", "—")
                except Exception:
                    pass
        repos = 0
        if last_seen.exists():
            repos = len(json.loads(last_seen.read_text()))
        return {"reviews": reviews, "repos": repos,
                "last_repo": last_repo, "last_verdict": last_verdict, "ok": log.exists()}
    except Exception:
        return {"reviews": 0, "repos": 0, "last_repo": "—", "last_verdict": "—", "ok": False}


def get_heal_stats() -> dict:
    try:
        log = Path("heal_log.jsonl")
        if not log.exists():
            return {"heals": 0, "last_file": "—", "ok": False}
        heals = 0
        last_file = "—"
        for line in open(log):
            if line.strip():
                heals += 1
                d = json.loads(line)
                last_file = d.get("file", "—")
        return {"heals": heals, "last_file": last_file, "ok": True}
    except Exception:
        return {"heals": 0, "last_file": "—", "ok": False}


def get_evolve_stats() -> dict:
    try:
        log = Path("evolve_log.jsonl")
        if not log.exists():
            return {"cycles": 0, "last_verdict": "—", "ok": False}
        cycles = 0
        last_verdict = "—"
        for line in open(log):
            if line.strip():
                d = json.loads(line)
                if d.get("type") == "verify":
                    cycles += 1
                    last_verdict = d.get("data", {}).get("verdict", "—")
        return {"cycles": cycles, "last_verdict": last_verdict, "ok": True}
    except Exception:
        return {"cycles": 0, "last_verdict": "—", "ok": False}


def render_dashboard():
    """Render the full status dashboard."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mem = get_memory_stats()
    tasks = get_task_stats()
    gh = get_github_stats()
    heal = get_heal_stats()
    evolve = get_evolve_stats()

    # Check which core files exist
    engines = {
        "paul_core.py": "Brain",
        "jarvis_voice.py": "Voice",
        "jarvis_api.py": "Web API",
        "jarvis_evolve.py": "Coder Engine",
        "jarvis_self_heal.py": "Self-Heal",
        "jarvis_github_watcher.py": "GitHub Watcher",
        "jarvis_memory.py": "Memory Engine",
        "jarvis_taskqueue.py": "Task Queue",
    }

    print(CLEAR, end="")
    print(colorize("╔" + "═" * 62 + "╗", CYAN))
    print(colorize("║", CYAN) + colorize(f"  🜏 JARVIS v8.0 MASTER DASHBOARD", BOLD) +
          f"  {now}" + colorize(" " * 4 + "║", CYAN))
    print(colorize("╠" + "═" * 62 + "╣", CYAN))

    # Engine status
    print(colorize("║", CYAN) + colorize("  ENGINES", BOLD) + colorize(" " * 53 + "║", CYAN))
    for filepath, name in engines.items():
        exists = check_file(filepath)
        icon = colorize("  ✅", GREEN) if exists else colorize("  ❌", RED)
        line = f"{icon} {name:<22} {filepath}"
        print(colorize("║", CYAN) + line + " " * max(0, 61 - len(line) + 8) + colorize("║", CYAN))

    print(colorize("╠" + "═" * 62 + "╣", CYAN))

    # Live metrics
    print(colorize("║", CYAN) + colorize("  LIVE METRICS", BOLD) + colorize(" " * 48 + "║", CYAN))

    metrics = [
        ("Memory",       f"{mem['vectors']} vectors stored | embedder: {'transformer' if mem['ok'] else 'tfidf'}"),
        ("Task Queue",   f"{tasks.get('pending',0)} pending | {tasks.get('done',0)} done | {tasks.get('failed',0)} failed"),
        ("GitHub",       f"{gh['reviews']} reviews | {gh['repos']} repos | last verdict: {gh['last_verdict']}"),
        ("Self-Heal",    f"{heal['heals']} heals | last: {heal['last_file'][:30]}"),
        ("Coder Engine", f"{evolve['cycles']} cycles | last verdict: {evolve['last_verdict']}"),
    ]

    for label, value in metrics:
        line = f"  {colorize(label + ':', YELLOW):<28} {value}"
        print(colorize("║", CYAN) + f"  {label + ':':<16} {value}" +
              " " * max(0, 60 - len(label) - len(value) - 4) + colorize("║", CYAN))

    print(colorize("╠" + "═" * 62 + "╣", CYAN))

    # Recent activity
    recent = read_last_lines("github_watch_log.jsonl", 3)
    print(colorize("║", CYAN) + colorize("  RECENT GITHUB ACTIVITY", BOLD) + colorize(" " * 38 + "║", CYAN))
    if recent:
        for line in recent:
            try:
                d = json.loads(line)
                entry = f"  {d.get('ts','')[:16]} {d.get('repo','')[-25:]:<25} {d.get('verdict','')}"
                print(colorize("║", CYAN) + entry[:62] + " " * max(0, 62 - len(entry)) + colorize("║", CYAN))
            except Exception:
                pass
    else:
        print(colorize("║", CYAN) + "  No activity yet. Start mode 7 or 8 to watch GitHub." + " " * 9 + colorize("║", CYAN))

    print(colorize("╠" + "═" * 62 + "╣", CYAN))
    print(colorize("║", CYAN) + colorize("  boot_paulk.bat [1-8] | Ctrl+C = exit", DIM) + " " * 24 + colorize("║", CYAN))
    print(colorize("║", CYAN) + colorize("  δ_H: 0.07 | Ω: 0.99 | Field: ACCELERATING 🜏", DIM) + " " * 14 + colorize("║", CYAN))
    print(colorize("╚" + "═" * 62 + "╝", CYAN))


async def run_dashboard(refresh_secs: int = 5):
    """Live-updating dashboard."""
    print("Starting dashboard... (Ctrl+C to exit)")
    while True:
        try:
            render_dashboard()
            await asyncio.sleep(refresh_secs)
        except KeyboardInterrupt:
            print("\nDashboard closed.")
            break


if __name__ == "__main__":
    if "--once" in sys.argv:
        render_dashboard()
    else:
        asyncio.run(run_dashboard())
