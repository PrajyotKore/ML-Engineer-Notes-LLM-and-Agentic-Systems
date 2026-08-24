import os
import json
import re
import html
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Complete curriculum definition in strict pedagogical order
DOC_FILES = [
    {"id": "00_ROLE_ANALYSIS", "file": "00_ROLE_ANALYSIS.md", "title": "00_ROLE_ANALYSIS", "label": "Role Analysis & Competency Map", "priority": "P1", "phase": "Phase 0: Architecture"},
    {"id": "01_MATHEMATICAL_FOUNDATIONS", "file": "01_MATHEMATICAL_FOUNDATIONS.md", "title": "01_MATHEMATICAL_FOUNDATIONS", "label": "Mathematical Foundations (SVD, Low-Rank, AdamW)", "priority": "P1", "phase": "Phase 1: Foundations"},
    {"id": "02_03_ML_AND_DL_FOUNDATIONS", "file": "02_03_ML_AND_DL_FOUNDATIONS.md", "title": "02_03_ML_AND_DL_FOUNDATIONS", "label": "ML & Deep Learning (Backprop, RMSNorm)", "priority": "P1", "phase": "Phase 1: Foundations"},
    {"id": "04_TRANSFORMERS_AND_LLMS", "file": "04_TRANSFORMERS_AND_LLMS.md", "title": "04_TRANSFORMERS_AND_LLMS", "label": "Transformers & Modern LLMs (MLA, RoPE, MoE)", "priority": "P0", "phase": "Phase 2: Core LLM"},
    {"id": "05_POST_TRAINING", "file": "05_POST_TRAINING.md", "title": "05_POST_TRAINING", "label": "Post-Training & Alignment (LoRA, DPO, GRPO)", "priority": "P0", "phase": "Phase 3: Alignment & Reasoning"},
    {"id": "06_DATA_AND_SYNTHETIC_DATA", "file": "06_DATA_AND_SYNTHETIC_DATA.md", "title": "06_DATA_AND_SYNTHETIC_DATA", "label": "Data Engineering & Synthetic Flywheels (MinHash)", "priority": "P1", "phase": "Phase 3: Alignment & Reasoning"},
    {"id": "08_GPU_AND_PERFORMANCE", "file": "08_GPU_AND_PERFORMANCE.md", "title": "08_GPU_AND_PERFORMANCE", "label": "GPU Architecture & FlashAttention-1/2/3", "priority": "P0", "phase": "Phase 4: Hardware & Inference"},
    {"id": "09_INFERENCE_SYSTEMS", "file": "09_INFERENCE_SYSTEMS.md", "title": "09_INFERENCE_SYSTEMS", "label": "Inference Systems (PagedAttention, SGLang, PD Split)", "priority": "P0", "phase": "Phase 4: Hardware & Inference"},
    {"id": "07_TRAINING_SYSTEMS", "file": "07_TRAINING_SYSTEMS.md", "title": "07_TRAINING_SYSTEMS", "label": "Training Systems (FSDP-2, 3D Parallelism)", "priority": "P1", "phase": "Phase 5: Distributed Scaling"},
    {"id": "18_DISTRIBUTED_SYSTEMS", "file": "18_DISTRIBUTED_SYSTEMS.md", "title": "18_DISTRIBUTED_SYSTEMS", "label": "Distributed Systems (Ring All-Reduce, RDMA)", "priority": "P1", "phase": "Phase 5: Distributed Scaling"},
    {"id": "10_AGENTIC_ML_SYSTEMS", "file": "10_AGENTIC_ML_SYSTEMS.md", "title": "10_AGENTIC_ML_SYSTEMS", "label": "Agentic ML Systems (FSM JSON, MCP, Hybrid RAG)", "priority": "P0", "phase": "Phase 6: Agentic Systems"},
    {"id": "11_LONG_RUNNING_WORKFLOW_RELIABILITY", "file": "11_LONG_RUNNING_WORKFLOW_RELIABILITY.md", "title": "11_LONG_RUNNING_WORKFLOW_RELIABILITY", "label": "Workflow Reliability (Temporal, Sagas, Jitter)", "priority": "P0", "phase": "Phase 6: Agentic Systems"},
    {"id": "15_SAFETY_AND_ROBUSTNESS", "file": "15_SAFETY_AND_ROBUSTNESS.md", "title": "15_SAFETY_AND_ROBUSTNESS", "label": "Safety & Robustness (Firecracker, Injections)", "priority": "P0", "phase": "Phase 6: Agentic Systems"},
    {"id": "12_EVALUATION", "file": "12_EVALUATION.md", "title": "12_EVALUATION", "label": "Evaluation Systems (Z-Tests, ELO, SWE-bench)", "priority": "P0", "phase": "Phase 7: Production & MLOps"},
    {"id": "13_PRODUCTION_ML", "file": "13_PRODUCTION_ML.md", "title": "13_PRODUCTION_ML", "label": "Production MLOps (PSI Drift, Canary, Pinning)", "priority": "P1", "phase": "Phase 7: Production & MLOps"},
    {"id": "14_OBSERVABILITY_AND_DEBUGGING", "file": "14_OBSERVABILITY_AND_DEBUGGING.md", "title": "14_OBSERVABILITY_AND_DEBUGGING", "label": "Observability & Debugging (Little's Law, MFU)", "priority": "P1", "phase": "Phase 7: Production & MLOps"},
    {"id": "16_SYSTEM_DESIGN", "file": "16_SYSTEM_DESIGN.md", "title": "16_SYSTEM_DESIGN", "label": "System Design Blueprints (100k QPS Serving)", "priority": "P1", "phase": "Phase 8: Synthesis & Interview Prep"},
    {"id": "17_PYTHON_AND_CODING", "file": "17_PYTHON_AND_CODING.md", "title": "17_PYTHON_AND_CODING", "label": "Production Code (MLA, FSM, Batcher)", "priority": "P2", "phase": "Phase 8: Synthesis & Interview Prep"},
    {"id": "19_LEADERSHIP_AND_TECHNICAL_JUDGMENT", "file": "19_LEADERSHIP_AND_TECHNICAL_JUDGMENT.md", "title": "19_LEADERSHIP_AND_TECHNICAL_JUDGMENT", "label": "Leadership & Technical Judgment Frameworks", "priority": "P1", "phase": "Phase 8: Synthesis & Interview Prep"},
    {"id": "20_INTERVIEW_QUESTION_BANK", "file": "20_INTERVIEW_QUESTION_BANK.md", "title": "20_INTERVIEW_QUESTION_BANK", "label": "Interview Question Bank (50+ Graded L1-L10)", "priority": "P0", "phase": "Phase 8: Synthesis & Interview Prep"},
    {"id": "21_CASE_STUDIES", "file": "21_CASE_STUDIES.md", "title": "21_CASE_STUDIES", "label": "Production Incident Case Studies & RCAs", "priority": "P0", "phase": "Phase 8: Synthesis & Interview Prep"},
    {"id": "22_FINAL_SYNTHESIS_PLAYBOOKS", "file": "22_FINAL_SYNTHESIS_PLAYBOOKS.md", "title": "22_FINAL_SYNTHESIS_PLAYBOOKS", "label": "2-Hour Final Synthesis Playbooks & Formulas", "priority": "P0", "phase": "Phase 8: Synthesis & Interview Prep"},
    {"id": "README", "file": "README.md", "title": "README", "label": "README & Master Curriculum", "priority": "P0", "phase": "Overview"}
]

def convert_md_to_html(raw_md: str) -> tuple[str, list]:
    """
    Converts raw Markdown to clean HTML with KaTeX math markers and extracts TOC headings.
    """
    # 1. Protect block math $$...$$
    block_math_list = []
    def save_block_math(match):
        block_math_list.append(match.group(1).strip())
        return f"<!--BLOCK_MATH_{len(block_math_list)-1}-->"
    
    text = re.sub(r'\$\$([\s\S]*?)\$\$', save_block_math, raw_md)

    # 2. Protect inline math $...$
    inline_math_list = []
    def save_inline_math(match):
        inline_math_list.append(match.group(1).strip())
        return f"<!--INLINE_MATH_{len(inline_math_list)-1}-->"
    
    text = re.sub(r'\$([^\$\n]+?)\$', save_inline_math, text)

    # 3. Handle Mermaid blocks
    mermaid_blocks = []
    def save_mermaid(match):
        mermaid_blocks.append(match.group(1).strip())
        return f"<!--MERMAID_BLOCK_{len(mermaid_blocks)-1}-->"
    
    text = re.sub(r'```mermaid\s*([\s\S]*?)```', save_mermaid, text)

    # 4. Convert markdown to HTML using python-markdown
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br', 'sane_lists'])
    rendered_html = md.convert(text)

    # 5. Restore Mermaid blocks
    for i, m_code in enumerate(mermaid_blocks):
        escaped_mermaid = html.escape(m_code)
        rendered_html = rendered_html.replace(f"<!--MERMAID_BLOCK_{i}-->", f'<div class="mermaid">{escaped_mermaid}</div>')

    # 6. Restore block math with KaTeX markup
    for i, b_math in enumerate(block_math_list):
        escaped_math = html.escape(b_math)
        rendered_html = rendered_html.replace(f"<!--BLOCK_MATH_{i}-->", f'<div class="katex-display">$${escaped_math}$$</div>')

    # 7. Restore inline math
    for i, in_math in enumerate(inline_math_list):
        escaped_math = html.escape(in_math)
        rendered_html = rendered_html.replace(f"<!--INLINE_MATH_{i}-->", f'<span class="katex-inline">${escaped_math}$</span>')

    # 8. Inject IDs into <h2> and <h3> for Table of Contents
    toc_items = []
    heading_counter = 0

    def add_heading_id(match):
        nonlocal heading_counter
        tag = match.group(1)
        content = match.group(2)
        h_id = f"heading-{heading_counter}"
        heading_counter += 1
        
        # Clean text for TOC label
        clean_label = re.sub(r'<[^>]+>', '', content).strip()
        clean_label = re.sub(r'^[0-9.]+\s*', '', clean_label)
        toc_items.append({"id": h_id, "tag": tag, "label": clean_label})
        
        return f'<{tag} id="{h_id}">{content}</{tag}>'

    rendered_html = re.sub(r'<(h[23])>(.*?)</\1>', add_heading_id, rendered_html, flags=re.DOTALL)

    return rendered_html, toc_items

# Master Page Template for Static Site
PAGE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{PAGE_TITLE}} — ML Engineer (SSK)</title>
  
  <!-- Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
  
  <!-- KaTeX for LaTeX Math -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>

  <!-- Prism.js for Syntax Highlighting -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-c.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-cpp.min.js"></script>

  <!-- Mermaid.js for Architecture Diagrams -->
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10.8.0/dist/mermaid.min.js"></script>

  <style>
    :root {
      --bg-primary: #0a0d14;
      --bg-secondary: #111622;
      --bg-card: rgba(22, 28, 42, 0.75);
      --bg-hover: #1c2436;
      --border-color: rgba(255, 255, 255, 0.08);
      --border-focus: rgba(99, 102, 241, 0.5);
      --text-primary: #f1f5f9;
      --text-secondary: #94a3b8;
      --text-muted: #64748b;
      --accent-primary: #6366f1;
      --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
      --accent-glow: rgba(99, 102, 241, 0.25);
      --p0-color: #ef4444;
      --p0-bg: rgba(239, 68, 68, 0.15);
      --p1-color: #3b82f6;
      --p1-bg: rgba(59, 130, 246, 0.15);
      --p2-color: #10b981;
      --p2-bg: rgba(16, 185, 129, 0.15);
      --sidebar-width: 320px;
      --toc-width: 260px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-primary);
      line-height: 1.65;
      overflow-x: hidden;
      display: flex;
      height: 100vh;
    }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.3); }

    /* Left Sidebar Navigation */
    aside.sidebar {
      width: var(--sidebar-width);
      min-width: var(--sidebar-width);
      background: var(--bg-secondary);
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      height: 100vh;
      z-index: 50;
    }

    .brand-header {
      padding: 20px 24px;
      border-bottom: 1px solid var(--border-color);
      background: rgba(10, 13, 20, 0.5);
      backdrop-filter: blur(10px);
    }

    .brand-title {
      font-family: 'Outfit', sans-serif;
      font-size: 1.15rem;
      font-weight: 700;
      background: var(--accent-gradient);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      display: flex;
      align-items: center;
      gap: 8px;
      text-decoration: none;
    }

    .brand-subtitle {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    .search-box {
      padding: 14px 20px;
      border-bottom: 1px solid var(--border-color);
    }

    .search-input {
      width: 100%;
      background: rgba(0, 0, 0, 0.3);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 10px 14px;
      color: var(--text-primary);
      font-size: 0.85rem;
      outline: none;
      transition: all 0.2s;
    }
    .search-input:focus {
      border-color: var(--accent-primary);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }

    .nav-list {
      flex: 1;
      overflow-y: auto;
      padding: 12px 14px;
      list-style: none;
    }

    .nav-phase-header {
      font-size: 0.7rem;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.08em;
      padding: 14px 10px 6px 10px;
    }

    .nav-item {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 8px 12px;
      border-radius: 6px;
      color: var(--text-secondary);
      font-size: 0.82rem;
      text-decoration: none;
      transition: all 0.15s ease;
      margin-bottom: 2px;
      border: 1px solid transparent;
    }
    .nav-item:hover {
      background: var(--bg-hover);
      color: var(--text-primary);
    }
    .nav-item.active {
      background: rgba(99, 102, 241, 0.15);
      color: #818cf8;
      border-color: rgba(99, 102, 241, 0.3);
      font-weight: 600;
    }

    .priority-badge {
      font-size: 0.65rem;
      font-weight: 700;
      padding: 2px 6px;
      border-radius: 4px;
      text-transform: uppercase;
    }
    .priority-p0 { color: var(--p0-color); background: var(--p0-bg); }
    .priority-p1 { color: var(--p1-color); background: var(--p1-bg); }
    .priority-p2 { color: var(--p2-color); background: var(--p2-bg); }

    /* Main Content Area */
    main.main-viewport {
      flex: 1;
      height: 100vh;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      position: relative;
    }

    .top-toolbar {
      position: sticky;
      top: 0;
      background: rgba(10, 13, 20, 0.85);
      backdrop-filter: blur(12px);
      border-bottom: 1px solid var(--border-color);
      padding: 12px 36px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      z-index: 40;
    }

    .breadcrumbs {
      font-size: 0.85rem;
      color: var(--text-secondary);
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .breadcrumbs a {
      color: var(--text-secondary);
      text-decoration: none;
    }
    .breadcrumbs a:hover {
      color: #818cf8;
    }
    .breadcrumbs span.current {
      color: var(--text-primary);
      font-weight: 600;
    }

    .tool-actions {
      display: flex;
      gap: 12px;
      align-items: center;
    }

    .btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--border-color);
      color: var(--text-primary);
      padding: 6px 14px;
      border-radius: 6px;
      font-size: 0.8rem;
      font-weight: 500;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      text-decoration: none;
      transition: all 0.2s;
    }
    .btn:hover {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.2);
    }

    .content-container {
      display: flex;
      flex: 1;
      max-width: 1600px;
      margin: 0 auto;
      width: 100%;
    }

    .markdown-body {
      flex: 1;
      padding: 40px 48px 100px 48px;
      max-width: 960px;
    }

    /* Markdown Styling */
    .markdown-body h1 {
      font-family: 'Outfit', sans-serif;
      font-size: 2.2rem;
      font-weight: 800;
      margin-bottom: 24px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--border-color);
      letter-spacing: -0.02em;
      color: #fff;
    }

    .markdown-body h2 {
      font-family: 'Outfit', sans-serif;
      font-size: 1.5rem;
      font-weight: 700;
      margin-top: 40px;
      margin-bottom: 16px;
      color: #e2e8f0;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
      padding-bottom: 8px;
    }

    .markdown-body h3 {
      font-size: 1.15rem;
      font-weight: 600;
      margin-top: 28px;
      margin-bottom: 12px;
      color: #cbd5e1;
    }

    .markdown-body h4 {
      font-size: 1.0rem;
      font-weight: 600;
      margin-top: 20px;
      margin-bottom: 8px;
      color: #94a3b8;
    }

    .markdown-body p {
      margin-bottom: 16px;
      color: #cbd5e1;
      font-size: 0.96rem;
    }

    .markdown-body ul, .markdown-body ol {
      margin-bottom: 18px;
      padding-left: 24px;
      color: #cbd5e1;
    }
    .markdown-body li {
      margin-bottom: 6px;
    }

    .markdown-body blockquote {
      border-left: 4px solid var(--accent-primary);
      background: rgba(99, 102, 241, 0.08);
      padding: 14px 20px;
      border-radius: 0 8px 8px 0;
      margin-bottom: 20px;
      color: #e2e8f0;
      font-size: 0.92rem;
    }

    .markdown-body table {
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 24px;
      font-size: 0.88rem;
      background: rgba(17, 22, 34, 0.6);
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border-color);
    }
    .markdown-body th {
      background: rgba(30, 41, 59, 0.8);
      padding: 12px 16px;
      text-align: left;
      font-weight: 600;
      color: #f1f5f9;
      border-bottom: 1px solid var(--border-color);
    }
    .markdown-body td {
      padding: 10px 16px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      color: #cbd5e1;
    }
    .markdown-body tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    .markdown-body code {
      font-family: 'Fira Code', monospace;
      font-size: 0.85em;
      background: rgba(255, 255, 255, 0.08);
      color: #f472b6;
      padding: 2px 6px;
      border-radius: 4px;
    }

    .markdown-body pre {
      background: #0f141f !important;
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 18px !important;
      margin-bottom: 24px;
      overflow-x: auto;
      position: relative;
    }
    .markdown-body pre code {
      background: transparent !important;
      color: #e2e8f0;
      padding: 0;
      font-size: 0.88rem;
    }

    /* Math Formulas */
    .katex-display {
      margin: 20px 0;
      padding: 14px 20px;
      background: rgba(17, 22, 34, 0.5);
      border-radius: 8px;
      border: 1px solid rgba(255, 255, 255, 0.05);
      overflow-x: auto;
      overflow-y: hidden;
    }

    /* Mermaid diagrams */
    .mermaid {
      background: rgba(15, 20, 31, 0.9);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 24px;
      display: flex;
      justify-content: center;
    }

    /* Navigation Footer */
    .doc-nav-footer {
      display: flex;
      justify-content: space-between;
      margin-top: 60px;
      padding-top: 24px;
      border-top: 1px solid var(--border-color);
    }
    .doc-nav-btn {
      display: flex;
      flex-direction: column;
      text-decoration: none;
      padding: 12px 20px;
      background: var(--bg-secondary);
      border: 1px solid var(--border-color);
      border-radius: 8px;
      transition: all 0.2s;
    }
    .doc-nav-btn:hover {
      border-color: var(--accent-primary);
      background: var(--bg-hover);
    }
    .doc-nav-btn .label {
      font-size: 0.72rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .doc-nav-btn .title {
      font-size: 0.9rem;
      color: #818cf8;
      font-weight: 600;
      margin-top: 2px;
    }

    /* Right TOC Sidebar */
    aside.toc-sidebar {
      width: var(--toc-width);
      padding: 40px 24px 40px 12px;
      position: sticky;
      top: 60px;
      height: calc(100vh - 60px);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }

    .toc-title {
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--text-muted);
      margin-bottom: 14px;
    }

    .toc-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 6px;
      border-left: 1px solid var(--border-color);
    }
    .toc-item {
      padding-left: 14px;
    }
    .toc-item.h3 {
      padding-left: 24px;
    }
    .toc-link {
      color: var(--text-secondary);
      font-size: 0.8rem;
      text-decoration: none;
      display: block;
      transition: color 0.15s;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .toc-link:hover {
      color: #818cf8;
    }

    /* Responsive */
    @media (max-width: 1200px) {
      aside.toc-sidebar { display: none; }
    }
    @media (max-width: 768px) {
      aside.sidebar { position: fixed; transform: translateX(-100%); transition: transform 0.3s; }
      aside.sidebar.open { transform: translateX(0); }
      .markdown-body { padding: 24px 20px; }
    }
  </style>
</head>
<body>

  <!-- Left Sidebar Navigation -->
  <aside class="sidebar">
    <div class="brand-header">
      <a href="index.html" class="brand-title">
        <span>⚡</span> ML Engineer (SSK)
      </a>
      <div class="brand-subtitle">LLM & Agentic Systems</div>
    </div>
    
    <div class="search-box">
      <input type="text" id="searchInput" class="search-input" placeholder="Quick search topics & math...">
    </div>

    <ul class="nav-list" id="navList">
      {{NAV_LIST_HTML}}
    </ul>
  </aside>

  <!-- Main Viewport -->
  <main class="main-viewport">
    <div class="top-toolbar">
      <div class="breadcrumbs">
        <a href="index.html">Curriculum</a>
        <span>/</span>
        <span>{{PHASE_NAME}}</span>
        <span>/</span>
        <span class="current">{{PAGE_TITLE}}</span>
      </div>
      <div class="tool-actions">
        <button class="btn" onclick="window.print()">🖨️ Print / PDF</button>
      </div>
    </div>

    <div class="content-container">
      <article class="markdown-body" id="docContent">
        {{DOCUMENT_HTML}}
        
        <!-- Navigation Footer -->
        <div class="doc-nav-footer">
          {{PREV_BUTTON_HTML}}
          {{NEXT_BUTTON_HTML}}
        </div>
      </article>

      <aside class="toc-sidebar">
        <div class="toc-title">On This Page</div>
        <ul class="toc-list">
          {{TOC_LIST_HTML}}
        </ul>
      </aside>
    </div>
  </main>

  <script>
    // Search filter for sidebar
    document.getElementById('searchInput').addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase();
      document.querySelectorAll('.nav-item').forEach(item => {
        const text = item.textContent.toLowerCase();
        item.style.display = text.includes(query) ? 'flex' : 'none';
      });
    });

    // Initialize KaTeX, Prism, and Mermaid on static DOM
    document.addEventListener('DOMContentLoaded', () => {
      // 1. KaTeX Math Render
      try {
        if (typeof renderMathInElement !== 'undefined') {
          renderMathInElement(document.getElementById('docContent'), {
            delimiters: [
              { left: '$$', right: '$$', display: true },
              { left: '$', right: '$', display: false }
            ],
            throwOnError: false
          });
        }
      } catch (e) {
        console.warn('KaTeX render error:', e);
      }

      // 2. Prism Code Highlight
      try {
        if (typeof Prism !== 'undefined') {
          Prism.highlightAll();
        }
      } catch (e) {
        console.warn('Prism highlight error:', e);
      }

      // 3. Mermaid Diagrams Render
      try {
        if (typeof mermaid !== 'undefined') {
          mermaid.initialize({ startOnLoad: true, theme: 'dark', securityLevel: 'loose' });
        }
      } catch (e) {
        console.warn('Mermaid render error:', e);
      }
    });
  </script>
</body>
</html>
"""

# Build all pages
print("Reading and pre-compiling all Markdown files...")

processed_docs = []
for item in DOC_FILES:
    file_path = os.path.join(ROOT_DIR, item["file"])
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
        
        html_content, toc_items = convert_md_to_html(raw_content)
        processed_docs.append({
            **item,
            "html": html_content,
            "toc": toc_items
        })
    else:
        print(f"Warning: {item['file']} not found.")

print(f"Compiled {len(processed_docs)} documents.")

# Generate each standalone root HTML page
for idx, doc in enumerate(processed_docs):
    # 1. Build Navigation Sidebar HTML with active class
    nav_html = []
    current_phase = ""
    for nav_doc in processed_docs:
        if nav_doc["phase"] != current_phase:
            current_phase = nav_doc["phase"]
            nav_html.append(f'<li class="nav-phase-header">{current_phase}</li>')
        
        is_active = (nav_doc["id"] == doc["id"])
        active_class = " active" if is_active else ""
        priority_class = f"priority-{nav_doc['priority'].lower()}"
        target_file = f"{nav_doc['id']}.html"
        
        nav_html.append(f'''
        <li>
          <a href="{target_file}" class="nav-item{active_class}">
            <span>{nav_doc["title"]}</span>
            <span class="priority-badge {priority_class}">{nav_doc["priority"]}</span>
          </a>
        </li>
        ''')
    
    # 2. Build TOC List HTML
    toc_html = []
    for toc in doc["toc"]:
        h3_class = " h3" if toc["tag"] == "h3" else ""
        toc_html.append(f'''
        <li class="toc-item{h3_class}">
          <a href="#{toc["id"]}" class="toc-link">{toc["label"]}</a>
        </li>
        ''')
    if not toc_html:
        toc_html.append('<li class="toc-item"><span style="color: var(--text-muted); font-size: 0.8rem;">No subsections</span></li>')

    # 3. Build Previous / Next buttons
    prev_doc = processed_docs[idx - 1] if idx > 0 else None
    next_doc = processed_docs[idx + 1] if idx < len(processed_docs) - 1 else None

    prev_btn = ""
    if prev_doc:
        prev_btn = f'''
        <a href="{prev_doc['id']}.html" class="doc-nav-btn">
          <span class="label">← Previous</span>
          <span class="title">{prev_doc['title']}</span>
        </a>
        '''
    else:
        prev_btn = '<div></div>'

    next_btn = ""
    if next_doc:
        next_btn = f'''
        <a href="{next_doc['id']}.html" class="doc-nav-btn" style="text-align: right;">
          <span class="label">Next →</span>
          <span class="title">{next_doc['title']}</span>
        </a>
        '''
    else:
        next_btn = '<div></div>'

    # 4. Render Page Template
    page_html = (PAGE_TEMPLATE
                 .replace("{{PAGE_TITLE}}", doc["title"])
                 .replace("{{PHASE_NAME}}", doc["phase"])
                 .replace("{{NAV_LIST_HTML}}", "".join(nav_html))
                 .replace("{{DOCUMENT_HTML}}", doc["html"])
                 .replace("{{TOC_LIST_HTML}}", "".join(toc_html))
                 .replace("{{PREV_BUTTON_HTML}}", prev_btn)
                 .replace("{{NEXT_BUTTON_HTML}}", next_btn))

    # Write out DOC_ID.html in root
    out_file = os.path.join(ROOT_DIR, f"{doc['id']}.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(page_html)

    # If this is 00_ROLE_ANALYSIS, also write it out as index.html
    if doc["id"] == "00_ROLE_ANALYSIS":
        index_file = os.path.join(ROOT_DIR, "index.html")
        with open(index_file, "w", encoding="utf-8") as f:
            f.write(page_html)

print(f"Successfully generated all {len(processed_docs)} static HTML pages + index.html at root.")
