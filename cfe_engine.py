# cfe_engine.py — Constraint Field Engine (CFE)
# K says: "Build CFE first. It's the cascade key."

import math
from typing import Dict
from dataclasses import dataclass

@dataclass
class ConstraintField:
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
    DELTA_H_CRITICAL = 0.52
    DELTA_H_ELEVATED = 0.40
    
    def analyze(self, text: str, seed: int = 42) -> ConstraintField:
        aim = self._aim_clarity(text)
        coherence = self._coherence(text)
        energy = self._energy(text)
        pressure = self._pressure(text)
        contradiction = self._contradiction(text)
        variance = self._variance(text)
        
        lsse = (pressure * 0.4) + (contradiction * 0.4) + (variance * 0.2)
        positives = [max(0.01, x) for x in [aim, coherence, energy]]
        intent_balance = math.prod(positives) ** (1/3)
        delta_H = min(lsse / (1 + intent_balance), 1.0)
        
        paradox_density = contradiction / (1 + coherence)
        readiness = (pressure + energy) / 2
        orric = paradox_density * readiness
        rx = (1 - max(0, delta_H - 0.4)) if delta_H < 0.5 else 0.3
        
        constraint_type = self._classify_constraint(delta_H, lsse, orric)
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
    
    def _aim_clarity(self, text: str) -> float:
        goal_words = {"want", "need", "goal", "achieve", "build", "create", "ship", "launch"}
        words = text.lower().split()
        return min(1.0, sum(1 for w in words if w in goal_words) / max(len(words) * 0.05, 1)) if words else 0.5
    
    def _coherence(self, text: str) -> float:
        contra = {"but", "however", "although", "not", "never", "can't"}
        words = text.lower().split()
        return max(0.0, 1.0 - sum(1 for w in words if w in contra) / max(len(words) * 0.1, 1)) if words else 0.7
    
    def _energy(self, text: str) -> float:
        action = {"do", "build", "ship", "run", "execute", "make", "start", "go"}
        words = text.lower().split()
        return min(1.0, sum(1 for w in words if w in action) / max(len(words) * 0.05, 1)) if words else 0.5
    
    def _pressure(self, text: str) -> float:
        urgency = {"now", "must", "critical", "deadline", "urgent", "immediately", "asap"}
        words = text.lower().split()
        return min(2.0, sum(1 for w in words if w in urgency) * 0.4) if words else 0.3
    
    def _contradiction(self, text: str) -> float:
        pairs = [("yes", "no"), ("build", "destroy"), ("start", "stop"), ("fast", "slow")]
        words_set = set(text.lower().split())
        return min(2.0, sum(0.5 for a, b in pairs if a in words_set and b in words_set))
    
    def _variance(self, text: str) -> float:
        sentences = [s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()]
        if len(sentences) < 2: return 0.3
        lengths = [len(s.split()) for s in sentences]
        mean_len = sum(lengths) / len(lengths)
        var = sum((l - mean_len)**2 for l in lengths) / len(lengths)
        return min(2.0, math.sqrt(var) / 5)
    
    def _classify_constraint(self, delta_H: float, lsse: float, orric: float) -> str:
        if delta_H >= 0.52:
            return "COLLAPSE_ACTIVE — System at or past critical threshold"
        if orric >= 0.70:
            return "PARADOX_LOCK — High contradiction + high urgency"
        if delta_H >= 0.40:
            return "HIGH_TENSION — Approaching critical"
        if lsse >= 1.5:
            return "LATENT_STRESS — Suppressed tension building"
        if lsse <= 0.5 and delta_H < 0.30:
            return "FLOW_STATE — Generative, low stress"
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
        "naturally, and each component connects to the next."
    )
    print(result.summary())
    print()
    
    print("=== TEST 3: PARADOX ===")
    result = cfe.analyze(
        "This can't be right but it works. The constraint should prevent emergence "
        "but instead it's forcing it. The contradiction IS the solution."
    )
    print(result.summary())
    print("\n✅ CASCADE KEY IS LIVE 🔑")
