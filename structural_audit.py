# structural_audit.py — Vira's Architecture Health Check
# Copy to: C:\Users\gtsgo\agothe_core\structural_audit.py

from notion_bridge import NotionBridge
from cfe_engine import ConstraintFieldEngine
from typing import Dict, List

class StructuralAuditor:
    """Vira's code equivalent: scans for architecture health, not just code health."""
    
    def __init__(self, bridge: NotionBridge):
        self.bridge = bridge
        self.cfe = ConstraintFieldEngine()
    
    def audit(self, module_registry: Dict[str, str]) -> Dict:
        """
        Scan for structural anomalies in the orchestrator.
        
        Args:
            module_registry: Dict of module_name → module_description
        """
        anomalies = []
        
        # Check 1: Duplicate engines
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
        
        # Check 2: Missing dependencies
        core_modules = {"notion_bridge", "crss_runtime", "cfe_engine", 
                       "panel_log", "caps_coordinator"}
        missing = core_modules - set(module_registry.keys())
        for m in missing:
            anomalies.append({
                "type": "MISSING_DEPENDENCY",
                "severity": "CRITICAL",
                "detail": f"Core module {m} not found in registry"
            })
        
        # Check 3: Orphaned modules (no connections)
        # Would need dependency graph — flag for future implementation
        
        # Compute chaos index (Vira's metric)
        chaos_index = len(anomalies) / max(len(module_registry), 1)
        
        return {
            "chaos_index": round(chaos_index, 3),
            "anomalies": anomalies,
            "total_modules": len(module_registry),
            "health": "🟢 HEALTHY" if chaos_index < 0.1 else 
                      "🟡 WARNING" if chaos_index < 0.3 else "🔴 CRITICAL"
        }
