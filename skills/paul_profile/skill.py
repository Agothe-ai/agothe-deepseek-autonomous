# skill: paul_profile
from datetime import datetime

mem = context.get("mem", {})
lines = []
lines.append("=== PAUL'S PROFILE ===")
lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
lines.append("")

# Facts
facts = mem.get("facts", {})
if facts:
    lines.append(f"KNOWN FACTS ({len(facts)}):")
    for k, v in facts.items():
        lines.append(f"  {k}: {v}")
else:
    lines.append("KNOWN FACTS: none yet -- tell Jarvis things to remember")

lines.append("")

# Stats
stats = mem.get("stats", {})
lines.append("SESSION STATS:")
lines.append(f"  Sessions: {stats.get('session_count', 0)}")
lines.append(f"  Total turns: {stats.get('total_turns', 0)}")

lines.append("")

# History summary
history = mem.get("history", [])
user_msgs = [h for h in history if h["role"] == "user"]
lines.append(f"RECENT ACTIVITY: {len(user_msgs)} messages in current session")
if user_msgs:
    lines.append(f"  Last message: {user_msgs[-1]['content'][:80]}")

result = "\n".join(lines)
