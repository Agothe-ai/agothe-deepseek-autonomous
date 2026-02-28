# cfe_engine.py — Constraint Field Engine (CFE)
# Copy to: C:\Users\gtsgo\agothe_core\cfe_engine.py
# 
# K says: "Build CFE first. It's the cascade key."
# CFE computes δ_H from any input. Every other engine uses δ_H.

import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ConstraintField:
    """Result of a Constraint Field Engine analysis."""
    delta_H: float          # Systemic stress index (0-1, critical at 0.52)
    lsse: float             # Latent Structural Stress Energy (0-2)
    orric: float            # Breakthrough/paradox detection score
    rx: float               # Reformation constant (system's ability to reform)
    constraint_type: str    # Classification of the active constraint
    resonance_mode: str     # Detected resonance pattern
    cascade_risk: str       # LOW / MEDIUM / HIGH / CRITICAL
    
    # Signal primitives
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
    """
    Level 1 of the Agothean Engine Stack.
    Detects and quantifies generative constraints.
    
    From the Codex:
    'Distinguishes true pattern-forcing constraints from simple limitations.'
    'Formally identifies the problem or intent of the field.'
    
    Input: Any text (situation description, system state, emotional content)
    Output: ConstraintField with full metric suite
    """
    
    # Threshold constants from Codex
    DELTA_H_CRITICAL = 0.52
    DELTA_H_ELEVATED = 0.40
    LSSE_HIGH = 1.5
    LSSE_LOW = 0.5
    ORRIC_STUCK = 0.70
    
    def analyze(self, text: str, seed: int = 42) -> ConstraintField:
        """
        Full CFE analysis of any text input.
        Deterministic with seed for reproducibility.
        """
        # Parse signal primitives
        aim = self._aim_clarity(text)
        coherence = self._coherence(text)
        energy = self._energy(text)
        pressure = self._pressure(text)
        contradiction = self._contradiction(text)
        variance = self._variance(text)
        
        # Compute core metrics
        lsse = self._compute_lsse(pressure, contradiction, variance)
        delta_H = self._compute_delta_H(lsse, aim, coherence, energy)
        orric = self._compute_orric(contradiction, coherence, pressure, energy)
        rx = self._compute_rx(delta_H)
        
        # Classify
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
    
    # ═══════════════════════════════════════
    # METRIC FORMULAS (from Codex Q4 Sprint)
    # ═══════════════════════════════════════
    
    def _compute_lsse(self, pressure: float, contradiction: float, 
                     variance: float) -> float:
        """LSSE = Latent Structural Stress Energy. Range: 0-2."""
        return (pressure * 0.4) + (contradiction * 0.4) + (variance * 0.2)
    
    def _compute_delta_H(self, lsse: float, aim: float, 
                        coherence: float, energy: float) -> float:
        """δ_H = Collapse Index. Range: 0-~2 (ceiling at 0.52 for safe ops)."""
        # Geometric mean of positive signals
        positives = [max(0.01, x) for x in [aim, coherence, energy]]
        intent_balance = math.prod(positives) ** (1/3)
        
        delta_H = lsse / (1 + intent_balance)
        return min(delta_H, 1.0)  # Hard cap at 1.0
    
    def _compute_orric(self, contradiction: float, coherence: float,
                      pressure: float, energy: float) -> float:
        """Orric = Breakthrough/paradox detection. Range: 0-~2."""
        paradox_density = contradiction / (1 + coherence)
        readiness = (pressure + energy) / 2
        return paradox_density * readiness
    
    def _compute_rx(self, delta_H: float) -> float:
        """Rₓ = Reformation Constant. System's ability to reform. Range: 0-1."""
        delta_H_approach = max(0, delta_H - 0.4)
        return (1 - delta_H_approach) if delta_H < 0.5 else 0.3
    
    # ═══════════════════════════════════════
    # SIGNAL PRIMITIVES (enhanced from CRSS)
    # ═══════════════════════════════════════
    
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
        # Also check for exclamation marks and caps as energy signals
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
    
    # ═══════════════════════════════════════
    # CLASSIFIERS
    # ═══════════════════════════════════════
    
    def _classify_constraint(self, text: str, delta_H: float, 
                            lsse: float, orric: float) -> str:
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
    
    def _classify_resonance(self, aim: float, coherence: float, 
                           energy: float) -> str:
        if coherence > 0.8 and energy > 0.7:
            return "HARMONIC_LOCK — Phase-locked and moving"
        if aim > 0.7 and energy > 0.6:
            return "DIRECTED_FLOW — Clear goal, clear energy"
        if coherence > 0.7 and aim < 0.4:
            return "AMBIENT — Coherent but undirected"
        if energy > 0.8:
            return "SURGE — High energy, variable direction"
        return "RESTING — Low activity baseline"
    
    def _assess_cascade(self, delta_H: float, lsse: float, 
                       orric: float) -> str:
        score = delta_H * 0.5 + (lsse / 2) * 0.3 + (orric / 2) * 0.2
        if score > 0.7: return "CRITICAL"
        if score > 0.5: return "HIGH"
        if score > 0.3: return "MEDIUM"
        return "LOW"


# ═══════════════════════════════════════
# QUICK TEST (run this file directly)
# ═══════════════════════════════════════
if __name__ == "__main__":
    cfe = ConstraintFieldEngine()
    
    # Test 1: High stress scenario
    result = cfe.analyze(
        "I must ship this today but the deploy is broken and I can't figure out "
        "why. The deadline is in 2 hours and nothing works. I need help immediately."
    )
    print("=== HIGH STRESS ===")
    print(result.summary())
    print()
    
    # Test 2: Flow state
    result = cfe.analyze(
        "Building the new motion system. The architecture is clean, the code flows "
        "naturally, and each component connects to the next. Creating beautiful "
        "animations that respond to user input."
    )
    print("=== FLOW STATE ===")
    print(result.summary())
    print()
    
    # Test 3: Paradox/breakthrough
    result = cfe.analyze(
        "This can't be right but it works. The constraint should prevent emergence "
        "but instead it's forcing it. The contradiction IS the solution. We need to "
        "ship this now before the window closes."
    )
    print("=== PARADOX ===")
    print(result.summary())
