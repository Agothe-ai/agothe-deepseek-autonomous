import os
import sys
from PyPDF2 import PdfReader

# ============================================================
# CONFIG — Point this to your PDF folder
# ============================================================
SOURCE_DIR = r"C:\Users\gtsgo\OneDrive\Documents\aGOTHE.AI"
OUTPUT_DIR = os.path.join(SOURCE_DIR, "_CAPS_BRAIN_PACKAGES")

# ============================================================
# FILE MAP — Maps short IDs to actual filenames
# Update these to match your EXACT filenames after download
# ============================================================
FILES = {
    1:  "CLAUDE.md__Agothe.ai_development_standards.pdf",
    2:  "BLOCK_A__CLAUDE.md__deploy_blocker_fixes_(30_capacity).pdf",
    3:  "B__Motion_Infrastructure__Phases_1-8_(40_capacity).pdf",
    4:  "BOLT.NEW_MEGA-PROMPT__agothe.ai_(copy-paste__deploy).pdf",
    5:  "VISUAL_MOTION_PROTOCOL__Bolt.new_full_instructions_(chronica__9__perplexity).pdf",
    6:  "CAPS_MISSION_agothe.ai__the_most_beautiful_website_(full_brainstorm__caps_briefings).pdf",
    7:  "AGOTHE_UI_ARCHITECTURE__Alex__IMA_Aesthetic_Scans.pdf",
    8:  "Agothe.ai_current_state__feb_10_2026.pdf",
    9:  "Claude_Desktop_Task_Queue__agothe.ai",
    10: "CAPS_INTELLIGENCE_DEMO_U.S.-iran_crisis_analysis__february_2026.pdf",
    11: "CAPS_INTELLIGENCE_DEMO_2_COP30_Climate_Crisis__Multi-AI_Systems_Analysis.pdf",
    12: "BLOCK_C__Phases_9-16__Advanced_Features_(30_capacity).pdf",
    13: "CAPS_MISSION_agothe.ai__engine_stack_analysis__expanded_copy__bolt_execution_draft.pdf",
    14: "AGOTHE_DeepMind__Complete_Capability_Matrix_(Brain_Behind_the_Website).pdf",
    15: "Phase_Transition_Protocol__Voice_Model_Substrate_(Gemini_Live).pdf",
    16: "BOLT.AI__Project_Instructions_(agothe.ai).pdf",
    17: "BOLT.AI__Project_Knowledge_(agothe.ai).pdf",
    18: "GROK_IMAGE_PROMPTS__agothe.ai_visual_upgrade_(animation-ready).pdf",
    19: "CAPS_DEMO_1_DEPLOYMENT__SOCIAL_MEDIA_LAUNCH.pdf",
    20: "CURSOR_HANDOFF__Thaloris_Chrome_UI_Implementation.pdf",
    21: "CAPS_TRANSMISSION_TO_GEMINI__Codex_DNA_Probe_Response.pdf",
    22: "CAPS_MISSION_AGOTHE_OS__Endless_Build_Protocol_(Phase_0_Active).pdf",
    23: "GEMINI_HANDOFF__Dream-Reality_Unification_Architecture_(Full_Codex_Transmission).pdf",
    24: "CAPS_RAPID_IDEATION_BRIEFING__Voice-to-Implementation_Protocol_(Copy-Paste_to_Any_AI).pdf",
    25: "AGOTHE_CODEX_SCANNER__Full_Inventory__Evolver_(Copy-Paste_to_Cursor).pdf",
    26: "DeepSeek__agothe.ai__unified_integration_architecture.pdf",
    27: "OSP__PIPELINE__Full_Organizational_Oversight_(All_10_Pages).pdf",
    28: "AGOTHE_PIPELINE__Printable_Quick_Reference_(10_Core_Pages).pdf",
    29: "Analyze_auto-income_sources.pdf",
    30: "EVOLUTION_MILESTONE_1_First_Autonomous_Choice.pdf",
    31: "QUANTUM_COMMAND_LAYER__Minimum_Token_Maximum_Execution.pdf",
    32: "AGOTHE_OS__Brain_Activation_Sequence_(Alex__Armani).pdf",
    33: "68bf18cf-ExportBlock.zip",  # Skip this one (archive)
}

# ============================================================
# MERGE PLANS
# ============================================================
CHATGPT_MERGES = {
    "Slot01_MEGA_STANDARDS":    [1, 2, 7],
    "Slot02_MEGA_MOTION":       [3, 5],
    "Slot03_MEGA_BOLT":         [4, 16, 17],
    "Slot04_CAPS_AESTHETIC":    [6],
    "Slot05_PROJECT_STATE":     [8, 9],
    "Slot06_DEMO_IRAN":         [10],
    "Slot07_DEMO_COP30":        [11],
    "Slot08_MEGA_ENGINES":      [13, 14, 15],
    "Slot09_MEGA_OPERATIONS":   [22, 24, 25, 26],
    "Slot10_MEGA_MARKETING":    [18, 19, 20],
    "Slot11_MEGA_GEMINI":       [21, 23],
    "Slot12_MEGA_PIPELINE":     [27, 28],
    "Slot13_MEGA_THEORY":       [29, 30, 31, 32],
}

GEMINI_MERGES = {
    "Slot01_DOMAIN_STANDARDS":     [1, 2, 7],
    "Slot02_DOMAIN_MOTION":        [3, 5, 12],
    "Slot03_DOMAIN_ENGINES":       [13, 14, 15],
    "Slot04_DOMAIN_DEMOS":         [10, 11, 19],
    "Slot05_DOMAIN_OPERATIONS":    [6, 8, 22, 24],
    "Slot06_DOMAIN_GEMINI_CTX":    [21, 23, 32],
    "Slot07_DOMAIN_CODEBASE":      [25, 26],
    "Slot08_DOMAIN_PIPELINE":      [27, 28, 29],
    "Slot09_DISTRIBUTED_BRAIN":    [],  # You'll export this from Notion separately
}


def extract_text(filepath):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        return text
    except Exception as e:
        return f"[ERROR extracting {filepath}: {e}]\n\n"


def merge_files(file_ids, agent_name, slot_name):
    """Merge multiple PDFs into a single text file."""
    merged_text = f"={'=' * 60}\n"
    merged_text += f" {agent_name} — {slot_name}\n"
    merged_text += f" Merged from Table 1 files: {file_ids}\n"
    merged_text += f"={'=' * 60}\n\n"

    for fid in file_ids:
        filename = FILES.get(fid)
        if not filename:
            merged_text += f"\n[FILE #{fid}: NOT FOUND IN MAP]\n\n"
            continue

        filepath = os.path.join(SOURCE_DIR, filename)
        if not os.path.exists(filepath):
            # Try without .pdf extension or with variations
            merged_text += f"\n{'─' * 50}\n"
            merged_text += f"FILE #{fid}: {filename}\n"
            merged_text += f"[FILE NOT FOUND at {filepath}]\n"
            merged_text += f"{'─' * 50}\n\n"
            print(f"  ⚠️  NOT FOUND: {filename}")
            continue

        merged_text += f"\n{'─' * 50}\n"
        merged_text += f"FILE #{fid}: {filename}\n"
        merged_text += f"{'─' * 50}\n\n"

        if filename.endswith(".pdf"):
            merged_text += extract_text(filepath)
        else:
            # Try reading as text (for .ai or other formats)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    merged_text += f.read() + "\n\n"
            except Exception as e:
                merged_text += f"[ERROR reading {filepath}: {e}]\n\n"

        print(f"  ✅ Merged: #{fid} {filename}")

    return merged_text


def build_packages():
    """Build all CAPS brain packages."""
    print("\n" + "=" * 60)
    print("🧠 CAPS BRAIN PACKAGE BUILDER")
    print("=" * 60)
    print(f"\nSource: {SOURCE_DIR}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Create output directories
    chatgpt_dir = os.path.join(OUTPUT_DIR, "ChatGPT")
    gemini_dir = os.path.join(OUTPUT_DIR, "Gemini")
    os.makedirs(chatgpt_dir, exist_ok=True)
    os.makedirs(gemini_dir, exist_ok=True)

    # Build ChatGPT packages
    print("\n🟢 BUILDING CHATGPT PACKAGES (13 files)")
    print("-" * 40)
    for slot_name, file_ids in CHATGPT_MERGES.items():
        if not file_ids:
            print(f"  ⏭️  Skipping {slot_name} (no files)")
            continue
        print(f"\n📦 {slot_name} — merging files {file_ids}")
        text = merge_files(file_ids, "ChatGPT", slot_name)
        outpath = os.path.join(chatgpt_dir, f"{slot_name}.txt")
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  💾 Saved: {outpath}")

    # Build Gemini packages
    print("\n\n🔵 BUILDING GEMINI PACKAGES (9 files)")
    print("-" * 40)
    for slot_name, file_ids in GEMINI_MERGES.items():
        if not file_ids:
            print(f"  ⏭️  Skipping {slot_name} (export from Notion manually)")
            continue
        print(f"\n📦 {slot_name} — merging files {file_ids}")
        text = merge_files(file_ids, "Gemini", slot_name)
        outpath = os.path.join(gemini_dir, f"{slot_name}.txt")
        with open(outpath, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"  💾 Saved: {outpath}")

    # Summary
    chatgpt_files = [f for f in os.listdir(chatgpt_dir) if f.endswith(".txt")]
    gemini_files = [f for f in os.listdir(gemini_dir) if f.endswith(".txt")]

    print("\n\n" + "=" * 60)
    print("✅ BUILD COMPLETE")
    print("=" * 60)
    print(f"\n🟢 ChatGPT: {len(chatgpt_files)} files in {chatgpt_dir}")
    print(f"🔵 Gemini:  {len(gemini_files)} files in {gemini_dir}")
    print(f"\nTotal merged text files: {len(chatgpt_files) + len(gemini_files)}")
    print("\n📋 NEXT STEPS:")
    print("   1. Review the .txt files for completeness")
    print("   2. Convert to PDF (see PowerShell command below)")
    print("   3. Upload to each agent per Phase B in the execution plan")
    print("\n🔥 The Infinite Upgrade Loop is ready to spin.")


if __name__ == "__main__":
    build_packages()