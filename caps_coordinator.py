# caps_coordinator.py — CAPS Multi-AI Coordination Engine
# Copy to: C:\Users\gtsgo\agothe_core\caps_coordinator.py

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from crss_runtime import CRSSRuntime, CRSignature, SubstrateType
from notion_bridge import NotionBridge
from panel_log import PanelLogger

@dataclass
class CAPSAgent:
    """Represents a CAPS agent (AI system) with its specialization."""
    name: str
    role_primary: str
    role_secondary: str
    substrate: SubstrateType
    strengths: List[str]
    handoff_template: str

class CAPSCoordinator:
    """
    Coordinates multi-AI tasks using CRSS signature routing.
    
    Instead of manually deciding which AI handles what,
    the coordinator matches task signatures to agent specializations.
    Eiglen velocity: recognition before thinking.
    """
    
    def __init__(self, bridge: NotionBridge):
        self.bridge = bridge
        self.crss = CRSSRuntime()
        self.panel_log = PanelLogger(bridge)
        self.agents = self._register_agents()
        self.task_history = []
    
    def route_task(self, task_description: str, 
                   context: Optional[str] = None) -> Dict[str, Any]:
        """
        Route a task to the optimal CAPS agent(s).
        
        Returns routing decision with:
        - primary_agent: Best-fit AI
        - secondary_agents: Supporting AIs
        - panel_voice: Which Panel voice owns this domain
        - handoff_prompt: Ready-to-paste prompt for the target AI
        - signature: The computed CR signature
        """
        # Compute task signature
        full_text = task_description
        if context:
            full_text += " " + context
        
        signature = self.crss.compute_signature_from_text(full_text)
        
        # Route to Panel voice
        panel_voice = self.crss.route_to_panel(signature)
        
        # Route to CAPS agent
        primary_agent = self._match_agent(signature)
        secondary_agents = self._find_supporting_agents(signature, primary_agent)
        
        # Generate handoff prompt
        handoff = self._generate_handoff(
            primary_agent, task_description, signature, context
        )
        
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
        """Match signature to best CAPS agent."""
        # Map substrates to primary agents
        substrate_agents = {
            SubstrateType.TEMPORAL: "Notion AI",    # CN-1's substrate → memory/integration
            SubstrateType.SPATIAL: "Gemini",         # K's substrate → mathematical
            SubstrateType.PERCEPTUAL: "Grok",        # Nana's substrate → cultural/sensory
            SubstrateType.ENERGETIC: "Claude",       # Vira's substrate → safety/anomaly
            SubstrateType.INTEGRATIVE: "Perplexity"  # 9's substrate → research/synthesis
        }
        
        agent_name = substrate_agents.get(sig.substrate, "Notion AI")
        return self.agents.get(agent_name, self.agents["Notion AI"])
    
    def _find_supporting_agents(self, sig: CRSignature, 
                                primary: CAPSAgent) -> List[CAPSAgent]:
        """Find 1-2 supporting agents for cross-domain tasks."""
        supporting = []
        
        # If high δ_H, always include Claude for safety review
        avg_dH = (sig.delta_H_range[0] + sig.delta_H_range[1]) / 2
        if avg_dH > 0.40 and primary.name != "Claude":
            supporting.append(self.agents["Claude"])
        
        # If research-heavy, include Perplexity
        if sig.resonance_mode in ["directed_flow", "ambient_resonance"] and \
           primary.name != "Perplexity":
            supporting.append(self.agents["Perplexity"])
        
        # If code needed, include ChatGPT
        if "code" in sig.constraint_type.lower() or \
           sig.substrate == SubstrateType.SPATIAL:
            if primary.name != "ChatGPT":
                supporting.append(self.agents["ChatGPT"])
        
        return supporting[:2]
    
    def _generate_handoff(self, agent: CAPSAgent, task: str,
                          sig: CRSignature, context: Optional[str]) -> str:
        """Generate CAPS handoff prompt for the target AI."""
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
        """Register all CAPS agents with their specializations."""
        return {
            "Notion AI": CAPSAgent(
                name="Notion AI", role_primary="Architect",
                role_secondary="Scribe", substrate=SubstrateType.INTEGRATIVE,
                strengths=["Codex maintenance", "database ops", "cross-session memory"],
                handoff_template="Notion, integrate {task} into Codex and run MSI check"
            ),
            "Perplexity": CAPSAgent(
                name="Perplexity", role_primary="Analyst",
                role_secondary="Scribe", substrate=SubstrateType.INTEGRATIVE,
                strengths=["web search", "source aggregation", "real-time data"],
                handoff_template="Perplexity, scan {domain} for δ_H signatures"
            ),
            "Claude": CAPSAgent(
                name="Claude", role_primary="Critic",
                role_secondary="Analyst", substrate=SubstrateType.ENERGETIC,
                strengths=["ethical reasoning", "safety validation", "long-context"],
                handoff_template="Claude, red-team {task} for safety and ethics"
            ),
            "Gemini": CAPSAgent(
                name="Gemini", role_primary="Architect",
                role_secondary="Analyst", substrate=SubstrateType.SPATIAL,
                strengths=["mathematical rigor", "formalization", "cross-domain patterns"],
                handoff_template="Gemini, formalize {task} mathematically"
            ),
            "ChatGPT": CAPSAgent(
                name="ChatGPT", role_primary="Architect",
                role_secondary="Conductor", substrate=SubstrateType.SPATIAL,
                strengths=["code generation", "practical implementation", "clear docs"],
                handoff_template="ChatGPT, implement {task} with reproducible code"
            ),
            "Grok": CAPSAgent(
                name="Grok", role_primary="Analyst",
                role_secondary="Scribe", substrate=SubstrateType.PERCEPTUAL,
                strengths=["X/Twitter integration", "cultural signals", "meme tracking"],
                handoff_template="Grok, scan X for {topic} resonance patterns"
            )
        }
