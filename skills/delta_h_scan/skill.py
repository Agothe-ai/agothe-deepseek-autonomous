# skill: delta_h_scan
# Computes delta_H across conversation history

history = context.get("history", [])

def compute_delta_h(text):
    words = text.lower().split()
    pressure = min(2.0, sum(0.4 for w in words if w in {"must","urgent","now","critical","deadline","immediately","asap"}))
    contra   = min(2.0, sum(0.4 for w in words if w in {"but","however","not","never","can't","won't"}))
    dH = min(1.0, (pressure * 0.5 + contra * 0.5) / 1.5)
    return round(dH, 3)

if not history:
    result = "[delta_h_scan] No history in context."
else:
    scores = []
    user_msgs = [h for h in history if h.get("role") == "user"]
    for msg in user_msgs[-10:]:
        dH = compute_delta_h(msg["content"])
        status = "CRITICAL" if dH >= 0.52 else "ELEVATED" if dH >= 0.40 else "nominal"
        scores.append((dH, status, msg["content"][:60]))

    avg = sum(s[0] for s in scores) / len(scores) if scores else 0
    overall = "CRITICAL" if avg >= 0.52 else "ELEVATED" if avg >= 0.40 else "NOMINAL"

    lines = [f"=== delta_H SCAN | {len(scores)} messages analyzed ==="]
    for dH, status, preview in scores:
        lines.append(f"  [{status:8}] {dH:.3f} -- {preview}")
    lines.append(f"\nOverall delta_H: {avg:.3f} [{overall}]")
    if avg >= 0.52:
        lines.append("RECOMMENDATION: Stop. One thing only. Breathe.")
    elif avg >= 0.40:
        lines.append("RECOMMENDATION: Slow down. Prioritize top item.")
    else:
        lines.append("RECOMMENDATION: Coherence nominal. Keep going.")

    result = "\n".join(lines)
