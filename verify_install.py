# verify_install.py — Jarvis Phase 1 Installation Verifier
# Checks every dependency and reports exactly what's working and what's not.
# Run: python verify_install.py

import sys
import importlib

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"
DIM = "\033[2m"

CHECKS = [
    # (import_name, display_name, tier, note)
    ("openai",               "OpenAI SDK",              "CORE",    "DeepSeek API client"),
    ("fastapi",              "FastAPI",                 "CORE",    "Web dashboard backend"),
    ("uvicorn",              "Uvicorn",                 "CORE",    "ASGI server"),
    ("sentence_transformers","Sentence Transformers",   "MEMORY",  "Local embeddings — Jarvis memory engine"),
    ("numpy",                "NumPy",                   "VOICE",   "Required for Whisper"),
    ("pyttsx3",              "pyttsx3",                 "VOICE",   "Text-to-speech engine"),
    ("whisper",              "OpenAI Whisper",           "VOICE",   "Speech-to-text (STT)"),
    ("pyaudio",              "PyAudio",                 "VOICE",   "Microphone input"),
    ("playwright",           "Playwright",              "WEB",     "Browser automation"),
    ("duckduckgo_search",    "DuckDuckGo Search",        "WEB",     "Live web search for agents"),
    ("bs4",                  "BeautifulSoup4",           "WEB",     "HTML parsing"),
    ("requests",             "Requests",                "WEB",     "HTTP client"),
    ("apscheduler",          "APScheduler",             "SCHED",   "Cron task automation"),
    ("pytest",               "pytest",                  "TEST",    "Test engine for Coder Engine"),
]

JARVIS_FILES = [
    ("paul_core.py",             "Brain"),
    ("jarvis_voice.py",          "Voice Engine"),
    ("jarvis_api.py",            "Web Dashboard"),
    ("jarvis_evolve.py",         "Coder Engine"),
    ("jarvis_self_heal.py",      "Self-Heal Daemon"),
    ("jarvis_github_watcher.py", "GitHub Watcher"),
    ("jarvis_memory.py",         "Memory Engine"),
    ("jarvis_taskqueue.py",      "Task Queue"),
    ("jarvis_dashboard.py",      "Master Dashboard"),
    ("boot_paulk.bat",           "Boot Launcher"),
    ("install_jarvis.bat",       "Installer"),
]


def check_imports():
    print(f"\n{CYAN}{BOLD}=== DEPENDENCY CHECK ==={RESET}")
    passed = 0
    failed = 0
    warned = 0
    results = []

    for mod, name, tier, note in CHECKS:
        try:
            importlib.import_module(mod)
            status = f"{GREEN}  PASS{RESET}"
            passed += 1
            results.append((tier, name, True, note))
        except ImportError:
            if tier in ["VOICE", "WEB"]:
                status = f"{YELLOW}  WARN{RESET}"
                warned += 1
            else:
                status = f"{RED}  FAIL{RESET}"
                failed += 1
            results.append((tier, name, False, note))

    # Print grouped by tier
    current_tier = None
    for tier, name, ok, note in results:
        if tier != current_tier:
            current_tier = tier
            print(f"\n  {DIM}[{tier}]{RESET}")
        icon = f"{GREEN}✅{RESET}" if ok else f"{RED}❌{RESET}"
        print(f"  {icon}  {name:<28} {DIM}{note}{RESET}")

    return passed, failed, warned


def check_files():
    import os
    print(f"\n{CYAN}{BOLD}=== JARVIS FILE CHECK ==={RESET}")
    present = 0
    missing = 0
    for filename, label in JARVIS_FILES:
        exists = os.path.isfile(filename)
        icon = f"{GREEN}✅{RESET}" if exists else f"{RED}❌{RESET}"
        print(f"  {icon}  {filename:<35} {DIM}{label}{RESET}")
        if exists:
            present += 1
        else:
            missing += 1
    return present, missing


def check_env():
    import os
    print(f"\n{CYAN}{BOLD}=== ENVIRONMENT VARIABLES ==={RESET}")
    vars_to_check = [
        ("DEEPSEEK_API_KEY", "Required",    "DeepSeek API — get at platform.deepseek.com"),
        ("GITHUB_TOKEN",     "Recommended", "GitHub 5000 req/hr — get at github.com/settings/tokens"),
        ("GITHUB_USERNAME",  "Optional",    "Your GitHub username"),
        ("WHISPER_MODEL",    "Optional",    "tiny/base/small/medium — defaults to base"),
    ]
    for var, importance, note in vars_to_check:
        val = os.environ.get(var, "")
        if val:
            masked = val[:8] + "..." if len(val) > 8 else val
            print(f"  {GREEN}✅{RESET}  {var:<22} {DIM}set ({masked}) — {note}{RESET}")
        else:
            color = RED if importance == "Required" else YELLOW
            print(f"  {color}{'❌' if importance == 'Required' else '⚠️ '}{RESET}  {var:<22} {DIM}not set — {note}{RESET}")


def check_memory_engine():
    print(f"\n{CYAN}{BOLD}=== MEMORY ENGINE TEST ==={RESET}")
    try:
        from sentence_transformers import SentenceTransformer
        print(f"  Loading all-MiniLM-L6-v2 (downloads ~90MB first time)...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        vec = model.encode("Paul codes in Python", normalize_embeddings=True)
        print(f"  {GREEN}✅  Embedding test passed — vector dim: {len(vec)}{RESET}")
        print(f"  {DIM}  Jarvis memory engine: READY{RESET}")
        return True
    except Exception as e:
        print(f"  {YELLOW}⚠️   Memory engine fallback active: {e}{RESET}")
        print(f"  {DIM}  TF-IDF embedder will be used (still works, just less precise){RESET}")
        return False


def main():
    print(f"\n{BOLD}{CYAN}")
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║  🜏 JARVIS PHASE 1 — INSTALL VERIFICATION    ║")
    print("  ╚══════════════════════════════════════════════╝")
    print(RESET)

    pkg_pass, pkg_fail, pkg_warn = check_imports()
    file_present, file_missing = check_files()
    check_env()
    mem_ok = check_memory_engine()

    print(f"\n{CYAN}{BOLD}=== SUMMARY ==={RESET}")
    print(f"  Packages:  {GREEN}{pkg_pass} passed{RESET}  {RED}{pkg_fail} failed{RESET}  {YELLOW}{pkg_warn} optional missing{RESET}")
    print(f"  Files:     {GREEN}{file_present} present{RESET}  {RED}{file_missing} missing{RESET}")
    print(f"  Memory:    {'TRANSFORMER (best)' if mem_ok else 'TFIDF FALLBACK (works)'} ")

    if pkg_fail == 0:
        print(f"\n  {GREEN}{BOLD}✅ JARVIS IS READY. Run boot_paulk.bat{RESET}")
    else:
        print(f"\n  {RED}{BOLD}❌ {pkg_fail} required package(s) missing.{RESET}")
        print(f"  Run install_jarvis.bat to fix.")

    print()


if __name__ == "__main__":
    main()
