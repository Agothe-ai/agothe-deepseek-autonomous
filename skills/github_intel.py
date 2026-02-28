# github_intel.py — Jarvis skill: GitHub intelligence from chat
# Paul says 'review my last commit' → Jarvis fetches + R1 reviews + speaks it

import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("""
🜏 GITHUB INTELLIGENCE SKILL LOADED

Jarvis can now:
  - Watch all your repos for new pushes (real-time)
  - Review any commit with DeepSeek R1 in <30s
  - Review open PRs and speak the verdict
  - Auto-discover all your GitHub repos
  - Speak code review feedback out loud

Commands from Jarvis chat:
  'review my last commit'
  'review repo [name]'
  'review my open PRs'
  'start watching github'
  'github status'

Direct CLI: python jarvis_github_watcher.py
""")

try:
    from jarvis_github_watcher import GitHubWatcherDaemon, GITHUB_USERNAME
    print(f"Watcher: ✅ Ready for {GITHUB_USERNAME}")
except ImportError as e:
    print(f"Watcher: ⚠️  {e}")
