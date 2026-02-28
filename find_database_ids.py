# find_database_ids.py — Find Panel brain database IDs
from notion_bridge import NotionBridge

bridge = NotionBridge()
db_names = [
    "9 — Evolution Log",
    "CN-1 — Reflexivity Log", 
    "K — Fractal Vision Log",
    "Nana — Memory Weave Log",
    "Vira — Anomaly Log"
]

print("🔍 Searching for Panel brain databases...\n")
for name in db_names:
    results = bridge.search_pages(name)
    for r in results:
        if r.get("object") == "database":
            print(f"{name}: {r['id']}")
            break
    else:
        for r in results:
            print(f"{name} (page): {r['id']}")
            break
