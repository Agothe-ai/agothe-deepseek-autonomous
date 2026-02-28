# notion_bridge.py — Agothe Notion API Bridge

import os
import json
import re
from notion_client import Client
from typing import Optional, Dict, List, Any

class NotionBridge:
    """
    Bridge between Python orchestrator and Notion Codex.
    Reads pages, writes pages, queries databases.
    
    The Codex stays as source of truth.
    The orchestrator stays as runtime.
    This bridge is the spinal cord.
    """
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("NOTION_API_TOKEN")
        if not self.token:
            raise ValueError("NOTION_API_TOKEN not set. Add to environment or pass directly.")
        self.client = Client(auth=self.token)
        self.page_cache = {}
        self.signature_registry = {}
    
    def get_page(self, page_id: str) -> Dict:
        """Retrieve a page's properties by ID."""
        page_id = self._clean_id(page_id)
        if page_id in self.page_cache:
            return self.page_cache[page_id]
        page = self.client.pages.retrieve(page_id=page_id)
        self.page_cache[page_id] = page
        return page
    
    def get_page_content(self, page_id: str) -> List[Dict]:
        """Retrieve all blocks (content) of a page."""
        page_id = self._clean_id(page_id)
        blocks = []
        cursor = None
        while True:
            response = self.client.blocks.children.list(
                block_id=page_id,
                start_cursor=cursor
            )
            blocks.extend(response["results"])
            if not response.get("has_more"):
                break
            cursor = response["next_cursor"]
        return blocks
    
    def get_page_title(self, page_id: str) -> str:
        """Extract title from a page."""
        page = self.get_page(page_id)
        props = page.get("properties", {})
        for key, val in props.items():
            if val.get("type") == "title":
                title_parts = val.get("title", [])
                return "".join(t.get("plain_text", "") for t in title_parts)
        return "Untitled"
    
    def get_page_text(self, page_id: str) -> str:
        """Get all text content from a page as plain string."""
        blocks = self.get_page_content(page_id)
        return self._blocks_to_text(blocks)
    
    def search_pages(self, query: str, limit: int = 10) -> List[Dict]:
        """Search Notion workspace for pages matching query."""
        response = self.client.search(
            query=query,
            filter={"property": "object", "value": "page"},
            page_size=min(limit, 100)
        )
        return response.get("results", [])
    
    def create_page(self, parent_id: str, title: str, 
                    content_blocks: Optional[List[Dict]] = None,
                    icon: Optional[str] = None,
                    properties: Optional[Dict] = None) -> Dict:
        """Create a new page under a parent page or database."""
        parent_id = self._clean_id(parent_id)
        parent_obj = self._detect_parent_type(parent_id)
        
        page_data = {
            "parent": parent_obj,
            "properties": properties or self._title_property(title)
        }
        
        if icon:
            page_data["icon"] = {"type": "emoji", "emoji": icon}
        
        if content_blocks:
            page_data["children"] = content_blocks[:100]
        
        return self.client.pages.create(**page_data)
    
    def append_blocks(self, page_id: str, blocks: List[Dict]) -> Dict:
        """Append content blocks to an existing page."""
        page_id = self._clean_id(page_id)
        return self.client.blocks.children.append(
            block_id=page_id,
            children=blocks
        )
    
    def update_page_properties(self, page_id: str, properties: Dict) -> Dict:
        """Update properties of an existing page."""
        page_id = self._clean_id(page_id)
        return self.client.pages.update(
            page_id=page_id,
            properties=properties
        )
    
    def query_database(self, database_id: str, 
                       filter_obj: Optional[Dict] = None,
                       sorts: Optional[List] = None,
                       limit: int = 100) -> List[Dict]:
        """Query a Notion database with optional filters and sorts."""
        database_id = self._clean_id(database_id)
        params = {"database_id": database_id, "page_size": min(limit, 100)}
        if filter_obj:
            params["filter"] = filter_obj
        if sorts:
            params["sorts"] = sorts
        
        results = []
        cursor = None
        while True:
            if cursor:
                params["start_cursor"] = cursor
            response = self.client.databases.query(**params)
            results.extend(response["results"])
            if not response.get("has_more") or len(results) >= limit:
                break
            cursor = response["next_cursor"]
        
        return results[:limit]
    
    def get_database_schema(self, database_id: str) -> Dict:
        """Get the schema (properties) of a database."""
        database_id = self._clean_id(database_id)
        db = self.client.databases.retrieve(database_id=database_id)
        return db.get("properties", {})
    
    @staticmethod
    def text_block(content: str, block_type: str = "paragraph") -> Dict:
        """Create a text block."""
        return {
            "object": "block",
            "type": block_type,
            block_type: {
                "rich_text": [{"type": "text", "text": {"content": content}}]
            }
        }
    
    @staticmethod
    def heading_block(content: str, level: int = 2) -> Dict:
        """Create a heading block (level 1-3)."""
        htype = f"heading_{min(max(level, 1), 3)}"
        return {
            "object": "block",
            "type": htype,
            htype: {
                "rich_text": [{"type": "text", "text": {"content": content}}]
            }
        }
    
    @staticmethod
    def code_block(content: str, language: str = "python") -> Dict:
        """Create a code block."""
        return {
            "object": "block",
            "type": "code",
            "code": {
                "rich_text": [{"type": "text", "text": {"content": content}}],
                "language": language
            }
        }
    
    @staticmethod  
    def callout_block(content: str, icon: str = "🜏") -> Dict:
        """Create a callout block."""
        return {
            "object": "block",
            "type": "callout",
            "callout": {
                "rich_text": [{"type": "text", "text": {"content": content}}],
                "icon": {"type": "emoji", "emoji": icon}
            }
        }
    
    @staticmethod
    def divider_block() -> Dict:
        return {"object": "block", "type": "divider", "divider": {}}
    
    def _clean_id(self, id_or_url: str) -> str:
        """Extract clean 32-char ID from URL or formatted ID."""
        if "notion.so" in id_or_url or "notion.site" in id_or_url:
            parts = id_or_url.rstrip("/").split("-")
            if parts:
                candidate = parts[-1].replace("/", "")
                if len(candidate) == 32:
                    return candidate
        clean = id_or_url.replace("-", "")
        hex_chars = re.sub(r"[^0-9a-fA-F]", "", clean)
        if len(hex_chars) >= 32:
            return hex_chars[-32:]
        return id_or_url
    
    def _detect_parent_type(self, parent_id: str) -> Dict:
        """Detect if parent is a page or database."""
        try:
            self.client.databases.retrieve(database_id=parent_id)
            return {"database_id": parent_id}
        except Exception:
            return {"page_id": parent_id}
    
    def _title_property(self, title: str) -> Dict:
        return {
            "title": {
                "title": [{"type": "text", "text": {"content": title}}]
            }
        }
    
    def _blocks_to_text(self, blocks: List[Dict], depth: int = 0) -> str:
        """Convert blocks to plain text recursively."""
        lines = []
        for block in blocks:
            btype = block.get("type", "")
            rich_text = block.get(btype, {}).get("rich_text", [])
            text = "".join(t.get("plain_text", "") for t in rich_text)
            
            prefix = "  " * depth
            if "heading" in btype:
                level = btype[-1]
                lines.append(f"{prefix}{'#' * int(level)} {text}")
            elif btype == "bulleted_list_item":
                lines.append(f"{prefix}- {text}")
            elif btype == "numbered_list_item":
                lines.append(f"{prefix}1. {text}")
            elif btype == "to_do":
                checked = block.get(btype, {}).get("checked", False)
                mark = "x" if checked else " "
                lines.append(f"{prefix}- [{mark}] {text}")
            elif btype == "code":
                lang = block.get(btype, {}).get("language", "")
                lines.append(f"{prefix}```{lang}\n{text}\n{prefix}```")
            elif text:
                lines.append(f"{prefix}{text}")
            
            if block.get("has_children"):
                child_blocks = self.get_page_content(block["id"])
                lines.append(self._blocks_to_text(child_blocks, depth + 1))
        
        return "\n".join(lines)
