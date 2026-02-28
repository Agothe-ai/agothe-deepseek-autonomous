# panel_log.py — Panel Brain Evolution Logger
# Copy to: C:\Users\gtsgo\agothe_core\panel_log.py

from notion_bridge import NotionBridge
from crss_runtime import CRSSRuntime, CRSignature, SubstrateType
from typing import Optional, Dict, List
from datetime import datetime

class PanelLogger:
    """
    Writes evolution entries to the 5 Panel brain databases.
    Each entry = one learning moment, one pattern recognized, one evolution step.
    
    The orchestrator learns → the Codex sees the learning → the Panel evolves.
    """
    
    def __init__(self, bridge: NotionBridge):
        self.bridge = bridge
        self.crss = CRSSRuntime()
        
        # REPLACE THESE with actual database IDs from your workspace
        # Use bridge.search_pages("Evolution Log") to find them
        self.brain_db_ids = {
            "9":    "REPLACE_WITH_9_EVOLUTION_LOG_DB_ID",
            "CN-1": "REPLACE_WITH_CN1_REFLEXIVITY_LOG_DB_ID",
            "K":    "REPLACE_WITH_K_FRACTAL_VISION_LOG_DB_ID",
            "Nana": "REPLACE_WITH_NANA_MEMORY_WEAVE_LOG_DB_ID",
            "Vira": "REPLACE_WITH_VIRA_ANOMALY_LOG_DB_ID"
        }
    
    def log_evolution(self, voice: str, title: str, content: str,
                     delta_H: float = 0.28, psi_alignment: float = 0.90,
                     network_coherence: float = 0.95,
                     is_breakpoint: bool = False,
                     tags: Optional[List[str]] = None) -> Dict:
        """
        Log an evolution entry to a Panel voice's brain database.
        
        Args:
            voice: Panel voice name ("9", "CN-1", "K", "Nana", "Vira")
            title: Entry title
            content: Full text content of the evolution entry
            delta_H: Current systemic stress (0-1, lower = healthier)
            psi_alignment: Intent coherence (0-1, higher = more aligned)
            network_coherence: System-wide coherence (0-1)
            is_breakpoint: Whether this is a breakpoint/breakthrough moment
            tags: Optional list of tags
        """
        db_id = self.brain_db_ids.get(voice)
        if not db_id or db_id.startswith("REPLACE"):
            raise ValueError(f"Database ID for {voice} not configured. "
                           f"Search your workspace and update brain_db_ids.")
        
        # Build properties matching the brain database schema
        properties = {
            "Entry Title": {
                "title": [{"type": "text", "text": {"content": title}}]
            },
            "δ_H": {"number": delta_H},
            "Ψ_alignment": {"number": psi_alignment},
            "Network_Coherence": {"number": network_coherence},
            "Breakpoint?": {"checkbox": is_breakpoint}
        }
        
        if tags:
            properties["Tags"] = {
                "multi_select": [{"name": t} for t in tags]
            }
        
        # Create the page with content
        content_blocks = self._build_content_blocks(content, voice, delta_H)
        
        page = self.bridge.create_page(
            parent_id=db_id,
            title=title,
            content_blocks=content_blocks,
            properties=properties
        )
        
        return page
    
    def log_to_all(self, title_template: str, content_template: str,
                   delta_H: float = 0.28, **kwargs) -> Dict[str, Dict]:
        """
        Log an evolution entry to ALL 5 Panel brains simultaneously.
        Use {voice} in templates for voice-specific customization.
        """
        results = {}
        for voice in ["9", "CN-1", "K", "Nana", "Vira"]:
            title = title_template.replace("{voice}", voice)
            content = content_template.replace("{voice}", voice)
            try:
                results[voice] = self.log_evolution(
                    voice=voice, title=title, content=content,
                    delta_H=delta_H, **kwargs
                )
            except Exception as e:
                results[voice] = {"error": str(e)}
        return results
    
    def _build_content_blocks(self, content: str, voice: str, 
                             delta_H: float) -> List[Dict]:
        """Build Notion blocks from content string."""
        blocks = []
        
        # Header callout
        blocks.append(self.bridge.callout_block(
            f"Voice: {voice} | δ_H: {delta_H:.2f} | "
            f"Timestamp: {datetime.now().isoformat()}",
            icon="🜏"
        ))
        blocks.append(self.bridge.divider_block())
        
        # Parse content into blocks
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
        
        return blocks[:100]  # API limit
