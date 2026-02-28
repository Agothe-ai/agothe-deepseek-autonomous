# notion_sync.py — Jarvis skill: sync Paul's tasks from Notion
# Loaded via: TOOL: load_skill(notion_sync)

import os
import json

def notion_sync():
    token = os.environ.get("NOTION_API_TOKEN")
    if not token:
        print("NOTION_API_TOKEN not set. Add to environment.")
        return
    
    try:
        from notion_client import Client
        client = Client(auth=token)
        
        # Search for task databases
        results = client.search(query="Tasks", filter={"property": "object", "value": "database"})
        databases = results.get("results", [])
        
        if not databases:
            print("No task databases found in Notion.")
            return
        
        print(f"Found {len(databases)} database(s):")
        for db in databases[:3]:
            props = db.get("properties", {})
            title_parts = db.get("title", [])
            title = "".join(t.get("plain_text", "") for t in title_parts)
            print(f"  • {title} — {db['id']}")
    except ImportError:
        print("notion-client not installed. Run: pip install notion-client")
    except Exception as e:
        print(f"Notion sync error: {e}")

notion_sync()
