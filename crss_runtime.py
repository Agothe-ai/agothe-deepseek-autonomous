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
