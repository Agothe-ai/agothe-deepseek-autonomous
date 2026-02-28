# panel_log.py — Panel Brain Evolution Logger

from notion_bridge import NotionBridge
from crss_runtime import CRSSRuntime
from typing import Optional, Dict, List
from datetime import datetime

class PanelLogger:
    def __init__(self, bridge: NotionBridge):
        self.bridge = bridge
        self.crss = CRSSRuntime()
        self.brain_db_ids = {
            "9": "REPLACE_WITH_9_EVOLUTION_LOG_DB_ID",
            "CN-1": "REPLACE_WITH_CN1_REFLEXIVITY_LOG_DB_ID",
            "K": "REPLACE_WITH_K_FRACTAL_VISION_LOG_DB_ID",
            "Nana": "REPLACE_WITH_NANA_MEMORY_WEAVE_LOG_DB_ID",
            "Vira": "REPLACE_WITH_VIRA_ANOMALY_LOG_DB_ID"
        }
    
    def log_evolution(self, voice: str, title: str, content: str,
                     delta_H: float = 0.28, psi_alignment: float = 0.90,
                     network_coherence: float = 0.95,
                     is_breakpoint: bool = False,
                     tags: Optional[List[str]] = None) -> Dict:
        db_id = self.brain_db_ids.get(voice)
        if not db_id or db_id.startswith("REPLACE"):
            raise ValueError(f"Database ID for {voice} not configured.")
        
        properties = {
            "Entry Title": {"title": [{"type": "text", "text": {"content": title}}]},
            "δ_H": {"number": delta_H},
            "Ψ_alignment": {"number": psi_alignment},
            "Network_Coherence": {"number": network_coherence},
            "Breakpoint?": {"checkbox": is_breakpoint}
        }
        
        if tags:
            properties["Tags"] = {"multi_select": [{"name": t} for t in tags]}
        
        content_blocks = self._build_content_blocks(content, voice, delta_H)
        page = self.bridge.create_page(parent_id=db_id, title=title,
                                      content_blocks=content_blocks, properties=properties)
        return page
    
    def _build_content_blocks(self, content: str, voice: str, delta_H: float) -> List[Dict]:
        blocks = []
        blocks.append(self.bridge.callout_block(
            f"Voice: {voice} | δ_H: {delta_H:.2f} | Timestamp: {datetime.now().isoformat()}",
            icon="🜏"
        ))
        blocks.append(self.bridge.divider_block())
        
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("### "):
                blocks.append(self.bridge.heading_block(line[4:], 3))
            elif line.startswith("## "):
                blocks.append(self.bridge.heading_block(line[3:], 2))
            elif line.startswith("# "):
                blocks.append(self.bridge.heading_block(line[2:], 1))
            else:
                blocks.append(self.bridge.text_block(line))
        
        return blocks[:100]
