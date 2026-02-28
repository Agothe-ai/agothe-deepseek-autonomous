# system_scan.py — Jarvis skill: deep system scan for Paul's PC health
# Loaded via: TOOL: load_skill(system_scan)

import subprocess
import platform
import sys
from pathlib import Path

def system_scan():
    print(f"🜏 SYSTEM SCAN — {platform.node()}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Disk
    import shutil
    total, used, free = shutil.disk_usage(".")
    print(f"Disk (current drive): {free//1e9:.1f}GB free / {total//1e9:.1f}GB total")
    
    # Check GPU (NVIDIA)
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.free,memory.total,temperature.gpu",
                             "--format=csv,noheader"], capture_output=True, text=True, timeout=5)
        if gpu.returncode == 0:
            print(f"GPU: {gpu.stdout.strip()}")
    except Exception:
        print("GPU: Not detected or nvidia-smi unavailable")
    
    # Running Python processes
    try:
        procs = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"],
                              capture_output=True, text=True, timeout=5)
        lines = [l for l in procs.stdout.split("\n") if "python" in l.lower()]
        print(f"Python processes running: {len(lines)}")
    except Exception:
        pass
    
    # Check key services
    services = ["ollama"]
    for svc in services:
        try:
            check = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {svc}.exe"],
                                  capture_output=True, text=True, timeout=5)
            running = svc in check.stdout.lower()
            print(f"  {svc}: {'🟢 running' if running else '🔴 not running'}")
        except Exception:
            pass
    
    print("\n🜏 Scan complete.")

system_scan()
