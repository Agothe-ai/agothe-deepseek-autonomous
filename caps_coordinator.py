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
    def __init__(self, bridge: NotionBridge):
        self.bridge = bridge
        self.crss = CRSSRuntime()
        self.panel_log = PanelLogger(bridge)
        self.agents = self._register_agents()
        self.task_history = []
    
    def route_task(self, task_description: str, context: Optional[str] = None) -> Dict[str, Any]:
        full_text = task_description + (" " + context if context else "")
        signature = self.crss.compute_signature_from_text(full_text)
        panel_voice = self.crss.route_to_panel(signature)
        primary_agent = self._match_agent(signature)
        
        return {
            "task": task_description,
            "signature": signature.to_dict(),
            "panel_voice": panel_voice,
            "primary_agent": primary_agent.name,
            "delta_H": (signature.delta_H_range[0] + signature.delta_H_range[1]) / 2,
            "substrate": signature.substrate.value
        }
    
    def _match_agent(self, sig: CRSignature) -> CAPSAgent:
        substrate_agents = {
            SubstrateType.TEMPORAL: "Notion AI",
            SubstrateType.SPATIAL: "Gemini",
            SubstrateType.PERCEPTUAL: "Grok",
            SubstrateType.ENERGETIC: "Claude",
            SubstrateType.INTEGRATIVE: "Perplexity"
        }
        agent_name = substrate_agents.get(sig.substrate, "Notion AI")
        return self.agents.get(agent_name)
    
    def _register_agents(self) -> Dict[str, CAPSAgent]:
        return {
            "Notion AI": CAPSAgent("Notion AI", "Architect", "Scribe", SubstrateType.INTEGRATIVE,
                                  ["Codex maintenance"], "Integrate into Codex"),
            "Perplexity": CAPSAgent("Perplexity", "Analyst", "Scribe", SubstrateType.INTEGRATIVE,
                                   ["web search", "research"], "Scan for patterns"),
            "Claude": CAPSAgent("Claude", "Critic", "Analyst", SubstrateType.ENERGETIC,
                               ["safety validation"], "Red-team for ethics"),
            "Gemini": CAPSAgent("Gemini", "Architect", "Analyst", SubstrateType.SPATIAL,
                               ["mathematical rigor"], "Formalize mathematically"),
            "ChatGPT": CAPSAgent("ChatGPT", "Architect", "Conductor", SubstrateType.SPATIAL,
                                ["code generation"], "Implement with code"),
            "Grok": CAPSAgent("Grok", "Analyst", "Scribe", SubstrateType.PERCEPTUAL,
                             ["cultural signals"], "Scan for resonance")
        }
