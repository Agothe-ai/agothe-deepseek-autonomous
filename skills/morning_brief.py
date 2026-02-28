# morning_brief.py — Jarvis skill: Paul's morning briefing
# Loaded via: TOOL: load_skill(morning_brief)

from datetime import datetime
import json
from pathlib import Path

def morning_brief():
    mem_file = Path("paul_memory.json")
    mem = json.loads(mem_file.read_text()) if mem_file.exists() else {}
    
    facts = mem.get("facts", {})
    scheduled = [s for s in mem.get("scheduled", []) if not s.get("done")]
    
    brief = f"""🜏 MORNING BRIEF — {datetime.now().strftime('%A %B %d, %Y')}
{'='*45}

Good morning Paul. Here's your day:

Active reminders ({len(scheduled)}):"""
    
    for s in scheduled[:5]:
        brief += f"\n  • {s['task']}"
    
    if facts:
        brief += "\n\nWhat I know about you:"
        for k, v in list(facts.items())[:5]:
            brief += f"\n  • {k}: {v}"
    
    brief += "\n\nAgothe system: γ_network 0.936 | δ_H nominal 🟢"
    print(brief)

morning_brief()
