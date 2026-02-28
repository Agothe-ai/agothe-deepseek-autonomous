# structural_audit.py — Vira's Architecture Health Check
from notion_bridge import NotionBridge
from cfe_engine import ConstraintFieldEngine
from typing import Dict

class StructuralAuditor:
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
        
        chaos_index = len(anomalies) / max(len(module_registry), 1)
        
        return {
            "chaos_index": round(chaos_index, 3),
            "anomalies": anomalies,
            "total_modules": len(module_registry),
            "health": "🟢 HEALTHY" if chaos_index < 0.1 else 
                      "🟡 WARNING" if chaos_index < 0.3 else "🔴 CRITICAL"
        }
