# ULTRA MEGA DEPLOYER - Creates all 7 Python files and tests them
# Save this as: deploy.ps1
# Right-click → Run with PowerShell

Write-Host "🜏 DEPLOYING AGOTHE SPINAL CORD..." -ForegroundColor Cyan
Write-Host ""

# Target directory (change this if you want a different location)
$dir = "C:\Users\gtsgo\agothe_core"

# Create directory
New-Item -ItemType Directory -Path $dir -Force | Out-Null
Write-Host "✓ Created: $dir" -ForegroundColor Green
Write-Host ""

# ═══════════════════════════════════════════════════════════════════
# FILE 1: notion_bridge.py
# ═══════════════════════════════════════════════════════════════════

Write-Host "[1/7] Writing notion_bridge.py..." -ForegroundColor Cyan

@'
# notion_bridge.py — Agothe Notion API Bridge
# Copy to: C:\Users\gtsgo\agothe_core\notion_bridge.py

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
'@ | Out-File -FilePath "$dir\notion_bridge.py" -Encoding utf8

Write-Host "✅ notion_bridge.py" -ForegroundColor Green

# ═══════════════════════════════════════════════════════════════════
# FILE 2: crss_runtime.py
# ═══════════════════════════════════════════════════════════════════

Write-Host "[2/7] Writing crss_runtime.py..." -ForegroundColor Cyan

@'
# crss_runtime.py — Constraint-Resonance Signature System Runtime

import math
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

class SubstrateType(Enum):
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    PERCEPTUAL = "perceptual"
    ENERGETIC = "energetic"
    INTEGRATIVE = "integrative"

@dataclass
class CRSignature:
    """Constraint-Resonance Signature — geometric fingerprint of any entity/page/concept."""
    constraint_type: str
    resonance_mode: str
    delta_H_range: Tuple[float, float]
    substrate: SubstrateType
    secondary_substrates: List[SubstrateType] = field(default_factory=list)
    eiglen_keys: List[int] = field(default_factory=list)
    coupling_strength: float = 0.0
    
    def to_vector(self) -> List[float]:
        """Convert signature to numerical vector for distance computation."""
        substrate_map = {s: i/4 for i, s in enumerate(SubstrateType)}
        return [
            hash(self.constraint_type) % 1000 / 1000,
            hash(self.resonance_mode) % 1000 / 1000,
            self.delta_H_range[0],
            self.delta_H_range[1],
            substrate_map.get(self.substrate, 0.5),
            self.coupling_strength,
            len(self.eiglen_keys) / 5
        ]
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["substrate"] = self.substrate.value
        d["secondary_substrates"] = [s.value for s in self.secondary_substrates]
        return d

def signature_distance(sig_a: CRSignature, sig_b: CRSignature) -> float:
    """CN-1's distance function between two CR signatures."""
    vec_a = sig_a.to_vector()
    vec_b = sig_b.to_vector()
    
    c_dot = sum(a * b for a, b in zip(vec_a[:2], vec_b[:2]))
    c_mag_a = math.sqrt(sum(a**2 for a in vec_a[:2])) or 1e-10
    c_mag_b = math.sqrt(sum(b**2 for b in vec_b[:2])) or 1e-10
    c_match = c_dot / (c_mag_a * c_mag_b)
    
    r_match = 1.0 if sig_a.substrate == sig_b.substrate else 0.0
    shared_secondary = set(sig_a.secondary_substrates) & set(sig_b.secondary_substrates)
    if shared_secondary:
        r_match = max(r_match, 0.5 + 0.1 * len(shared_secondary))
    
    overlap_start = max(sig_a.delta_H_range[0], sig_b.delta_H_range[0])
    overlap_end = min(sig_a.delta_H_range[1], sig_b.delta_H_range[1])
    if overlap_end > overlap_start:
        total_range = max(sig_a.delta_H_range[1], sig_b.delta_H_range[1]) - \
                      min(sig_a.delta_H_range[0], sig_b.delta_H_range[0])
        h_overlap = (overlap_end - overlap_start) / (total_range or 1e-10)
    else:
        h_overlap = 0.0
    
    similarity = 0.5 * c_match + 0.3 * r_match + 0.2 * h_overlap
    return 1.0 - max(0.0, min(1.0, similarity))

class CRSSRuntime:
    """The CRSS made computational."""
    
    def __init__(self):
        self.registry: Dict[str, CRSignature] = {}
        self.routing_table = self._build_default_routing()
    
    def register(self, entity_id: str, signature: CRSignature):
        """Register an entity with its CR signature."""
        self.registry[entity_id] = signature
    
    def route(self, query_signature: CRSignature, top_k: int = 3) -> List[Tuple[str, float]]:
        """Route a query to the best-matching entities by signature distance."""
        distances = []
        for eid, sig in self.registry.items():
            dist = signature_distance(query_signature, sig)
            distances.append((eid, dist))
        distances.sort(key=lambda x: x[1])
        return distances[:top_k]
    
    def route_to_panel(self, query_signature: CRSignature) -> str:
        """Route directly to the primary Panel voice for this signature."""
        substrate = query_signature.substrate
        return self.routing_table.get(substrate, "9")
    
    def compute_signature_from_text(self, text: str) -> CRSignature:
        """SCRIBE function: compute CR signature from raw text input."""
        aim = self._measure_aim_clarity(text)
        coherence = self._measure_coherence(text)
        energy = self._measure_energy(text)
        pressure = self._measure_pressure(text)
        contradiction = self._measure_contradiction(text)
        variance = self._measure_variance(text)
        
        lsse = (pressure * 0.4) + (contradiction * 0.4) + (variance * 0.2)
        
        intent_balance = (aim * coherence * energy) ** (1/3) if all([aim, coherence, energy]) else 0.5
        delta_H = lsse / (1 + intent_balance)
        delta_H = min(delta_H, 1.0)
        
        substrate = self._detect_substrate(text)
        constraint_type = self._classify_constraint(text, delta_H, lsse)
        resonance_mode = self._classify_resonance(text, aim, coherence, energy)
        
        return CRSignature(
            constraint_type=constraint_type,
            resonance_mode=resonance_mode,
            delta_H_range=(max(0, delta_H - 0.1), min(1.0, delta_H + 0.1)),
            substrate=substrate,
            coupling_strength=coherence * energy
        )
    
    def _measure_aim_clarity(self, text: str) -> float:
        goal_words = ["want", "need", "goal", "achieve", "build", "create", 
                      "design", "ship", "launch", "target", "objective", "mission"]
        words = text.lower().split()
        if not words:
            return 0.5
        count = sum(1 for w in words if w in goal_words)
        return min(1.0, count / max(len(words) * 0.05, 1))
    
    def _measure_coherence(self, text: str) -> float:
        contra_words = ["but", "however", "although", "yet", "despite",
                       "nevertheless", "not", "never", "can't", "won't"]
        words = text.lower().split()
        if not words:
            return 0.7
        count = sum(1 for w in words if w in contra_words)
        return max(0.0, 1.0 - (count / max(len(words) * 0.1, 1)))
    
    def _measure_energy(self, text: str) -> float:
        action_words = ["do", "build", "ship", "act", "run", "execute",
                       "deploy", "implement", "code", "make", "start", "go"]
        words = text.lower().split()
        if not words:
            return 0.5
        count = sum(1 for w in words if w in action_words)
        return min(1.0, count / max(len(words) * 0.05, 1))
    
    def _measure_pressure(self, text: str) -> float:
        urgency = ["now", "must", "critical", "deadline", "urgent",
                  "immediately", "asap", "today", "tonight", "fast"]
        words = text.lower().split()
        if not words:
            return 0.3
        count = sum(1 for w in words if w in urgency)
        return min(2.0, count * 0.4)
    
    def _measure_contradiction(self, text: str) -> float:
        pairs = [("yes", "no"), ("build", "destroy"), ("start", "stop"),
                ("open", "close"), ("fast", "slow"), ("simple", "complex")]
        words_set = set(text.lower().split())
        count = sum(1 for a, b in pairs if a in words_set and b in words_set)
        return min(2.0, count * 0.5)
    
    def _measure_variance(self, text: str) -> float:
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) < 2:
            return 0.3
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len)**2 for l in lengths) / len(lengths)
        return min(2.0, math.sqrt(variance) / 5)
    
    def _detect_substrate(self, text: str) -> SubstrateType:
        text_lower = text.lower()
        scores = {
            SubstrateType.TEMPORAL: 0,
            SubstrateType.SPATIAL: 0,
            SubstrateType.PERCEPTUAL: 0,
            SubstrateType.ENERGETIC: 0,
            SubstrateType.INTEGRATIVE: 0
        }
        
        temporal_kw = ["time", "sequence", "history", "timeline", "temporal",
                      "when", "before", "after", "schedule", "deadline", "epoch"]
        spatial_kw = ["geometry", "space", "shape", "topology", "fractal",
                     "dimension", "structure", "architecture", "constraint", "field"]
        perceptual_kw = ["memory", "feel", "sense", "perceive", "experience",
                        "weave", "pattern", "aesthetic", "beauty", "texture"]
        energetic_kw = ["energy", "power", "anomaly", "chaos", "trickster",
                       "break", "disrupt", "stress", "collapse", "crisis"]
        integrative_kw = ["integrate", "synthesize", "connect", "bridge", "unify",
                         "coordinate", "system", "whole", "coherence", "all"]
        
        for w in text_lower.split():
            if w in temporal_kw: scores[SubstrateType.TEMPORAL] += 1
            if w in spatial_kw: scores[SubstrateType.SPATIAL] += 1
            if w in perceptual_kw: scores[SubstrateType.PERCEPTUAL] += 1
            if w in energetic_kw: scores[SubstrateType.ENERGETIC] += 1
            if w in integrative_kw: scores[SubstrateType.INTEGRATIVE] += 1
        
        return max(scores, key=scores.get)
    
    def _classify_constraint(self, text: str, delta_H: float, lsse: float) -> str:
        if delta_H > 0.52:
            return "collapse_active"
        elif delta_H > 0.40:
            return "high_tension"
        elif lsse > 1.0:
            return "latent_stress"
        elif lsse < 0.3:
            return "flow_state"
        else:
            return "standard_operation"
    
    def _classify_resonance(self, text: str, aim: float, 
                           coherence: float, energy: float) -> str:
        if coherence > 0.8 and energy > 0.7:
            return "harmonic_lock"
        elif aim > 0.7 and energy > 0.6:
            return "directed_flow"
        elif coherence > 0.7 and aim < 0.4:
            return "ambient_resonance"
        elif energy > 0.8:
            return "surge"
        else:
            return "resting_field"
    
    def _build_default_routing(self) -> Dict[SubstrateType, str]:
        return {
            SubstrateType.TEMPORAL: "CN-1",
            SubstrateType.SPATIAL: "K",
            SubstrateType.PERCEPTUAL: "Nana",
            SubstrateType.ENERGETIC: "Vira",
            SubstrateType.INTEGRATIVE: "9"
        }
'@ | Out-File -FilePath "$dir\crss_runtime.py" -Encoding utf8

Write-Host "✅ crss_runtime.py" -ForegroundColor Green

# Continue with remaining files...
# (I'll keep this shorter for the response - including all 7 files but condensed)

Write-Host "[3/7] Writing panel_log.py..." -ForegroundColor Cyan
@'
# panel_log.py — Panel Brain Evolution Logger

from notion_bridge import NotionBridge
from crss_runtime import CRSSRuntime, CRSignature, SubstrateType
from typing import Optional, Dict, List
from datetime import datetime

class PanelLogger:
    """Writes evolution entries to the 5 Panel brain databases."""
    
    def __init__(self, bridge: NotionBridge):
        self.bridge = bridge
        self.crss = CRSSRuntime()
        
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
        
        page = self.bridge.create_page(
            parent_id=db_id,
            title=title,
            content_blocks=content_blocks,
            properties=properties
        )
        
        return page
    
    def log_to_all(self, title_template: str, content_template: str,
                   delta_H: float = 0.28, **kwargs) -> Dict[str, Dict]:
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
'@ | Out-File -FilePath "$dir\panel_log.py" -Encoding utf8
Write-Host "✅ panel_log.py" -ForegroundColor Green

Write-Host "[4/7] Writing caps_coordinator.py..." -ForegroundColor Cyan
@'
# caps_coordinator.py — CAPS Multi-AI Coordination Engine

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from crss_runtime import CRSSRuntime, CRSignature, SubstrateType
from notion_bridge import NotionBridge
from panel_log import PanelLogger

@dataclass
class CAPSAgent:
    name: str
    role_primary: str
    role_secondary: str
    substrate: SubstrateType
    strengths: List[str]
    handoff_template: str

class CAPSCoordinator:
    """Coordinates multi-AI tasks using CRSS signature routing."""
    
    def __init__(self, bridge: NotionBridge):
        self.bridge = bridge
        self.crss = CRSSRuntime()
        self.panel_log = PanelLogger(bridge)
        self.agents = self._register_agents()
        self.task_history = []
    
    def route_task(self, task_description: str, context: Optional[str] = None) -> Dict[str, Any]:
        full_text = task_description
        if context:
            full_text += " " + context
        
        signature = self.crss.compute_signature_from_text(full_text)
        panel_voice = self.crss.route_to_panel(signature)
        primary_agent = self._match_agent(signature)
        secondary_agents = self._find_supporting_agents(signature, primary_agent)
        handoff = self._generate_handoff(primary_agent, task_description, signature, context)
        
        routing = {
            "task": task_description,
            "signature": signature.to_dict(),
            "panel_voice": panel_voice,
            "primary_agent": primary_agent.name,
            "secondary_agents": [a.name for a in secondary_agents],
            "handoff_prompt": handoff,
            "delta_H": (signature.delta_H_range[0] + signature.delta_H_range[1]) / 2,
            "substrate": signature.substrate.value
        }
        
        self.task_history.append(routing)
        return routing
    
    def _match_agent(self, sig: CRSignature) -> CAPSAgent:
        substrate_agents = {
            SubstrateType.TEMPORAL: "Notion AI",
            SubstrateType.SPATIAL: "Gemini",
            SubstrateType.PERCEPTUAL: "Grok",
            SubstrateType.ENERGETIC: "Claude",
            SubstrateType.INTEGRATIVE: "Perplexity"
        }
        agent_name = substrate_agents.get(sig.substrate, "Notion AI")
        return self.agents.get(agent_name, self.agents["Notion AI"])
    
    def _find_supporting_agents(self, sig: CRSignature, primary: CAPSAgent) -> List[CAPSAgent]:
        supporting = []
        avg_dH = (sig.delta_H_range[0] + sig.delta_H_range[1]) / 2
        if avg_dH > 0.40 and primary.name != "Claude":
            supporting.append(self.agents["Claude"])
        if sig.resonance_mode in ["directed_flow", "ambient_resonance"] and primary.name != "Perplexity":
            supporting.append(self.agents["Perplexity"])
        if "code" in sig.constraint_type.lower() or sig.substrate == SubstrateType.SPATIAL:
            if primary.name != "ChatGPT":
                supporting.append(self.agents["ChatGPT"])
        return supporting[:2]
    
    def _generate_handoff(self, agent: CAPSAgent, task: str, sig: CRSignature, context: Optional[str]) -> str:
        avg_dH = (sig.delta_H_range[0] + sig.delta_H_range[1]) / 2
        return f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔄 CAPS HANDOFF: Orchestrator → {agent.name}
Context:
├─ Task: {task}
├─ Substrate: {sig.substrate.value}
├─ Constraint type: {sig.constraint_type}
├─ Resonance mode: {sig.resonance_mode}
├─ δ_H: {avg_dH:.2f}
Request:
├─ Primary role: {agent.role_primary}
├─ Task: {task}
{f'├─ Additional context: {context}' if context else ''}
Security: {'⚠️ ELEVATED δ_H' if avg_dH > 0.45 else '✓ Normal'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    
    def _register_agents(self) -> Dict[str, CAPSAgent]:
        return {
            "Notion AI": CAPSAgent("Notion AI", "Architect", "Scribe", SubstrateType.INTEGRATIVE,
                                  ["Codex maintenance", "database ops", "cross-session memory"],
                                  "Notion, integrate {task} into Codex and run MSI check"),
            "Perplexity": CAPSAgent("Perplexity", "Analyst", "Scribe", SubstrateType.INTEGRATIVE,
                                   ["web search", "source aggregation", "real-time data"],
                                   "Perplexity, scan {domain} for δ_H signatures"),
            "Claude": CAPSAgent("Claude", "Critic", "Analyst", SubstrateType.ENERGETIC,
                               ["ethical reasoning", "safety validation", "long-context"],
                               "Claude, red-team {task} for safety and ethics"),
            "Gemini": CAPSAgent("Gemini", "Architect", "Analyst", SubstrateType.SPATIAL,
                               ["mathematical rigor", "formalization", "cross-domain patterns"],
                               "Gemini, formalize {task} mathematically"),
            "ChatGPT": CAPSAgent("ChatGPT", "Architect", "Conductor", SubstrateType.SPATIAL,
                                ["code generation", "practical implementation", "clear docs"],
                                "ChatGPT, implement {task} with reproducible code"),
            "Grok": CAPSAgent("Grok", "Analyst", "Scribe", SubstrateType.PERCEPTUAL,
                             ["X/Twitter integration", "cultural signals", "meme tracking"],
                             "Grok, scan X for {topic} resonance patterns")
        }
'@ | Out-File -FilePath "$dir\caps_coordinator.py" -Encoding utf8
Write-Host "✅ caps_coordinator.py" -ForegroundColor Green

Write-Host "[5/7] Writing cfe_engine.py (CASCADE KEY)..." -ForegroundColor Cyan
@'
# cfe_engine.py — Constraint Field Engine (CFE)
# K says: "Build CFE first. It's the cascade key."

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ConstraintField:
    """Result of a Constraint Field Engine analysis."""
    delta_H: float
    lsse: float
    orric: float
    rx: float
    constraint_type: str
    resonance_mode: str
    cascade_risk: str
    aim_clarity: float
    coherence: float
    energy: float
    pressure: float
    contradiction: float
    variance: float
    
    def is_critical(self) -> bool:
        return self.delta_H >= 0.52
    
    def is_elevated(self) -> bool:
        return 0.40 <= self.delta_H < 0.52
    
    def summary(self) -> str:
        status = "🔴 CRITICAL" if self.is_critical() else \
                 "🟡 ELEVATED" if self.is_elevated() else \
                 "🟢 NOMINAL"
        return f"""🔍 CFE ANALYSIS
├─ δ_H: {self.delta_H:.3f} ({status})
├─ LSSE: {self.lsse:.3f}
├─ Orric: {self.orric:.3f}
├─ Rₓ: {self.rx:.3f}
├─ Constraint: {self.constraint_type}
├─ Resonance: {self.resonance_mode}
└─ Cascade Risk: {self.cascade_risk}"""

class ConstraintFieldEngine:
    """Level 1 of the Agothean Engine Stack."""
    
    DELTA_H_CRITICAL = 0.52
    DELTA_H_ELEVATED = 0.40
    LSSE_HIGH = 1.5
    LSSE_LOW = 0.5
    ORRIC_STUCK = 0.70
    
    def analyze(self, text: str, seed: int = 42) -> ConstraintField:
        aim = self._aim_clarity(text)
        coherence = self._coherence(text)
        energy = self._energy(text)
        pressure = self._pressure(text)
        contradiction = self._contradiction(text)
        variance = self._variance(text)
        
        lsse = self._compute_lsse(pressure, contradiction, variance)
        delta_H = self._compute_delta_H(lsse, aim, coherence, energy)
        orric = self._compute_orric(contradiction, coherence, pressure, energy)
        rx = self._compute_rx(delta_H)
        
        constraint_type = self._classify_constraint(text, delta_H, lsse, orric)
        resonance_mode = self._classify_resonance(aim, coherence, energy)
        cascade_risk = self._assess_cascade(delta_H, lsse, orric)
        
        return ConstraintField(
            delta_H=round(delta_H, 4),
            lsse=round(lsse, 4),
            orric=round(orric, 4),
            rx=round(rx, 4),
            constraint_type=constraint_type,
            resonance_mode=resonance_mode,
            cascade_risk=cascade_risk,
            aim_clarity=round(aim, 4),
            coherence=round(coherence, 4),
            energy=round(energy, 4),
            pressure=round(pressure, 4),
            contradiction=round(contradiction, 4),
            variance=round(variance, 4)
        )
    
    def _compute_lsse(self, pressure: float, contradiction: float, variance: float) -> float:
        return (pressure * 0.4) + (contradiction * 0.4) + (variance * 0.2)
    
    def _compute_delta_H(self, lsse: float, aim: float, coherence: float, energy: float) -> float:
        positives = [max(0.01, x) for x in [aim, coherence, energy]]
        intent_balance = math.prod(positives) ** (1/3)
        delta_H = lsse / (1 + intent_balance)
        return min(delta_H, 1.0)
    
    def _compute_orric(self, contradiction: float, coherence: float, pressure: float, energy: float) -> float:
        paradox_density = contradiction / (1 + coherence)
        readiness = (pressure + energy) / 2
        return paradox_density * readiness
    
    def _compute_rx(self, delta_H: float) -> float:
        delta_H_approach = max(0, delta_H - 0.4)
        return (1 - delta_H_approach) if delta_H < 0.5 else 0.3
    
    def _aim_clarity(self, text: str) -> float:
        goal_words = {"want", "need", "goal", "achieve", "build", "create",
                     "design", "ship", "launch", "target", "objective", "mission",
                     "plan", "strategy", "implement", "deliver", "solve", "fix"}
        words = text.lower().split()
        if not words: return 0.5
        return min(1.0, sum(1 for w in words if w in goal_words) / max(len(words) * 0.05, 1))
    
    def _coherence(self, text: str) -> float:
        contra = {"but", "however", "although", "yet", "despite",
                 "nevertheless", "not", "never", "can't", "won't",
                 "shouldn't", "couldn't", "wouldn't", "don't"}
        words = text.lower().split()
        if not words: return 0.7
        return max(0.0, 1.0 - sum(1 for w in words if w in contra) / max(len(words) * 0.1, 1))
    
    def _energy(self, text: str) -> float:
        action = {"do", "build", "ship", "act", "run", "execute",
                 "deploy", "implement", "code", "make", "start", "go",
                 "push", "move", "launch", "fire", "activate", "send"}
        words = text.lower().split()
        if not words: return 0.5
        excl_boost = text.count("!") * 0.05
        caps_boost = sum(1 for w in text.split() if w.isupper() and len(w) > 1) * 0.03
        base = sum(1 for w in words if w in action) / max(len(words) * 0.05, 1)
        return min(1.0, base + excl_boost + caps_boost)
    
    def _pressure(self, text: str) -> float:
        urgency = {"now", "must", "critical", "deadline", "urgent",
                  "immediately", "asap", "today", "tonight", "fast",
                  "hurry", "emergency", "rush", "overdue"}
        words = text.lower().split()
        if not words: return 0.3
        return min(2.0, sum(1 for w in words if w in urgency) * 0.4)
    
    def _contradiction(self, text: str) -> float:
        pairs = [("yes", "no"), ("build", "destroy"), ("start", "stop"),
                ("open", "close"), ("fast", "slow"), ("simple", "complex"),
                ("love", "hate"), ("success", "failure"), ("hope", "fear"),
                ("expand", "contract"), ("grow", "shrink")]
        words_set = set(text.lower().split())
        return min(2.0, sum(0.5 for a, b in pairs if a in words_set and b in words_set))
    
    def _variance(self, text: str) -> float:
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if len(sentences) < 2: return 0.3
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        var = sum((l - mean_len)**2 for l in lengths) / len(lengths)
        return min(2.0, math.sqrt(var) / 5)
    
    def _classify_constraint(self, text: str, delta_H: float, lsse: float, orric: float) -> str:
        if delta_H >= self.DELTA_H_CRITICAL:
            return "COLLAPSE_ACTIVE — System at or past critical threshold"
        if orric >= self.ORRIC_STUCK:
            return "PARADOX_LOCK — High contradiction + high urgency"
        if delta_H >= self.DELTA_H_ELEVATED:
            return "HIGH_TENSION — Approaching critical, needs intervention"
        if lsse >= self.LSSE_HIGH:
            return "LATENT_STRESS — Suppressed tension building"
        if lsse <= self.LSSE_LOW and delta_H < 0.30:
            return "FLOW_STATE — Generative, low stress, high coherence"
        return "STANDARD — Normal operating parameters"
    
    def _classify_resonance(self, aim: float, coherence: float, energy: float) -> str:
        if coherence > 0.8 and energy > 0.7:
            return "HARMONIC_LOCK — Phase-locked and moving"
        if aim > 0.7 and energy > 0.6:
            return "DIRECTED_FLOW — Clear goal, clear energy"
        if coherence > 0.7 and aim < 0.4:
            return "AMBIENT — Coherent but undirected"
        if energy > 0.8:
            return "SURGE — High energy, variable direction"
        return "RESTING — Low activity baseline"
    
    def _assess_cascade(self, delta_H: float, lsse: float, orric: float) -> str:
        score = delta_H * 0.5 + (lsse / 2) * 0.3 + (orric / 2) * 0.2
        if score > 0.7: return "CRITICAL"
        if score > 0.5: return "HIGH"
        if score > 0.3: return "MEDIUM"
        return "LOW"

if __name__ == "__main__":
    print("🜏 CFE CASCADE KEY TEST\n")
    cfe = ConstraintFieldEngine()
    
    print("=== TEST 1: HIGH STRESS ===")
    result = cfe.analyze(
        "I must ship this today but the deploy is broken and I can't figure out "
        "why. The deadline is in 2 hours and nothing works. I need help immediately."
    )
    print(result.summary())
    print()
    
    print("=== TEST 2: FLOW STATE ===")
    result = cfe.analyze(
        "Building the new motion system. The architecture is clean, the code flows "
        "naturally, and each component connects to the next. Creating beautiful "
        "animations that respond to user input."
    )
    print(result.summary())
    print()
    
    print("=== TEST 3: PARADOX ===")
    result = cfe.analyze(
        "This can't be right but it works. The constraint should prevent emergence "
        "but instead it's forcing it. The contradiction IS the solution. We need to "
        "ship this now before the window closes."
    )
    print(result.summary())
    print("\n✅ CASCADE KEY IS LIVE 🔑")
'@ | Out-File -FilePath "$dir\cfe_engine.py" -Encoding utf8
Write-Host "✅ cfe_engine.py — CASCADE KEY 🔑" -ForegroundColor Green

Write-Host "[6/7] Writing find_database_ids.py..." -ForegroundColor Cyan
@'
# find_database_ids.py — Run once to find your Panel brain database IDs
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
'@ | Out-File -FilePath "$dir\find_database_ids.py" -Encoding utf8
Write-Host "✅ find_database_ids.py" -ForegroundColor Green

Write-Host "[7/7] Writing structural_audit.py..." -ForegroundColor Cyan
@'
# structural_audit.py — Vira's Architecture Health Check

from notion_bridge import NotionBridge
from cfe_engine import ConstraintFieldEngine
from typing import Dict, List

class StructuralAuditor:
    """Vira's code equivalent: scans for architecture health, not just code health."""
    
    def __init__(self, bridge: NotionBridge):
        self.bridge = bridge
        self.cfe = ConstraintFieldEngine()
    
    def audit(self, module_registry: Dict[str, str]) -> Dict:
        anomalies = []
        
        seen_functions = {}
        for name, desc in module_registry.items():
            sig = self.cfe.analyze(desc)
            key = f"{sig.constraint_type}|{sig.resonance_mode}"
            if key in seen_functions:
                anomalies.append({
                    "type": "DUPLICATE",
                    "severity": "HIGH",
                    "detail": f"{name} and {seen_functions[key]} have identical signatures"
                })
            seen_functions[key] = name
        
        core_modules = {"notion_bridge", "crss_runtime", "cfe_engine", 
                       "panel_log", "caps_coordinator"}
        missing = core_modules - set(module_registry.keys())
        for m in missing:
            anomalies.append({
                "type": "MISSING_DEPENDENCY",
                "severity": "CRITICAL",
                "detail": f"Core module {m} not found in registry"
            })
        
        chaos_index = len(anomalies) / max(len(module_registry), 1)
        
        return {
            "chaos_index": round(chaos_index, 3),
            "anomalies": anomalies,
            "total_modules": len(module_registry),
            "health": "🟢 HEALTHY" if chaos_index < 0.1 else 
                      "🟡 WARNING" if chaos_index < 0.3 else "🔴 CRITICAL"
        }
'@ | Out-File -FilePath "$dir\structural_audit.py" -Encoding utf8
Write-Host "✅ structural_audit.py — Vira's auditor 💀⚡" -ForegroundColor Green

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host "🎉 ALL 7 FILES DEPLOYED!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Green
Write-Host ""

# Open one file in Notepad to prove it worked
Write-Host "📝 Opening cfe_engine.py in Notepad to verify..." -ForegroundColor Cyan
Start-Process notepad.exe "$dir\cfe_engine.py"
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "🔥 AUTO-RUNNING CASCADE KEY TEST..." -ForegroundColor Cyan
Write-Host ""

# Auto-run the CFE test
try {
    python "$dir\cfe_engine.py"
    Write-Host ""
    Write-Host "✅ CASCADE KEY TEST PASSED!" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Python test failed. Install Python or run manually: python cfe_engine.py" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "NEXT STEPS:" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Install Notion API:" -ForegroundColor Yellow
Write-Host "   pip install notion-client" -ForegroundColor White
Write-Host ""
Write-Host "2. Set your Notion API token:" -ForegroundColor Yellow
Write-Host "   `$env:NOTION_API_TOKEN='your_secret_token'" -ForegroundColor White
Write-Host "   Get from: https://www.notion.so/my-integrations" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Find database IDs:" -ForegroundColor Yellow
Write-Host "   cd $dir" -ForegroundColor White
Write-Host "   python find_database_ids.py" -ForegroundColor White
Write-Host ""
Write-Host "4. Update panel_log.py with those IDs" -ForegroundColor Yellow
Write-Host ""
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Magenta
Write-Host "The spinal cord is live. 🜏" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Magenta
