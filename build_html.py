import os
import json
import re

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
HTML_DIR = os.path.join(ROOT_DIR, "html")
os.makedirs(HTML_DIR, exist_ok=True)

# List of all markdown documentation files in logical order
DOC_FILES = [
    {"id": "00_ROLE_ANALYSIS", "file": "00_ROLE_ANALYSIS.md", "title": "Role Analysis & Competency Map", "priority": "P1", "phase": "Phase 0: Architecture"},
    {"id": "01_MATHEMATICAL_FOUNDATIONS", "file": "01_MATHEMATICAL_FOUNDATIONS.md", "title": "Mathematical Foundations (SVD, Low-Rank, AdamW)", "priority": "P1", "phase": "Phase 1: Foundations"},
    {"id": "02_03_ML_AND_DL_FOUNDATIONS", "file": "02_03_ML_AND_DL_FOUNDATIONS.md", "title": "ML & Deep Learning Foundations (Backprop, RMSNorm)", "priority": "P1", "phase": "Phase 1: Foundations"},
    {"id": "04_TRANSFORMERS_AND_LLMS", "file": "04_TRANSFORMERS_AND_LLMS.md", "title": "Transformers & Modern LLMs (MLA, RoPE, SwiGLU, MoE)", "priority": "P0", "phase": "Phase 2: Core LLM"},
    {"id": "05_POST_TRAINING", "file": "05_POST_TRAINING.md", "title": "Post-Training & Alignment (LoRA, DPO, GRPO)", "priority": "P0", "phase": "Phase 3: Alignment & Reasoning"},
    {"id": "06_DATA_AND_SYNTHETIC_DATA", "file": "06_DATA_AND_SYNTHETIC_DATA.md", "title": "Data Engineering & Synthetic Flywheels (MinHash LSH)", "priority": "P1", "phase": "Phase 3: Alignment & Reasoning"},
    {"id": "08_GPU_AND_PERFORMANCE", "file": "08_GPU_AND_PERFORMANCE.md", "title": "GPU Architecture & FlashAttention-1/2/3", "priority": "P0", "phase": "Phase 4: Hardware & Inference"},
    {"id": "09_INFERENCE_SYSTEMS", "file": "09_INFERENCE_SYSTEMS.md", "title": "Inference Systems (PagedAttention, SGLang, PD Split)", "priority": "P0", "phase": "Phase 4: Hardware & Inference"},
    {"id": "07_TRAINING_SYSTEMS", "file": "07_TRAINING_SYSTEMS.md", "title": "Training Systems (FSDP-2, 3D Parallelism, Bubbles)", "priority": "P1", "phase": "Phase 5: Distributed Scaling"},
    {"id": "18_DISTRIBUTED_SYSTEMS", "file": "18_DISTRIBUTED_SYSTEMS.md", "title": "Distributed Systems (Ring All-Reduce, RDMA)", "priority": "P1", "phase": "Phase 5: Distributed Scaling"},
    {"id": "10_AGENTIC_ML_SYSTEMS", "file": "10_AGENTIC_ML_SYSTEMS.md", "title": "Agentic ML Systems (FSM JSON, MCP, Hybrid RAG)", "priority": "P0", "phase": "Phase 6: Agentic Systems"},
    {"id": "11_LONG_RUNNING_WORKFLOW_RELIABILITY", "file": "11_LONG_RUNNING_WORKFLOW_RELIABILITY.md", "title": "Long-Running Workflow Reliability (Temporal, Sagas)", "priority": "P0", "phase": "Phase 6: Agentic Systems"},
    {"id": "15_SAFETY_AND_ROBUSTNESS", "file": "15_SAFETY_AND_ROBUSTNESS.md", "title": "Safety & Robustness (Firecracker, Injections, IAM)", "priority": "P0", "phase": "Phase 6: Agentic Systems"},
    {"id": "12_EVALUATION", "file": "12_EVALUATION.md", "title": "Evaluation Systems (Z-Tests, ELO, SWE-bench)", "priority": "P0", "phase": "Phase 7: Production & MLOps"},
    {"id": "13_PRODUCTION_ML", "file": "13_PRODUCTION_ML.md", "title": "Production MLOps (PSI Drift, Canary, Pinning)", "priority": "P1", "phase": "Phase 7: Production & MLOps"},
    {"id": "14_OBSERVABILITY_AND_DEBUGGING", "file": "14_OBSERVABILITY_AND_DEBUGGING.md", "title": "Observability & Debugging (Little's Law, MFU, Traces)", "priority": "P1", "phase": "Phase 7: Production & MLOps"},
    {"id": "16_SYSTEM_DESIGN", "file": "16_SYSTEM_DESIGN.md", "title": "System Design Blueprints (100k QPS Serving, Agents)", "priority": "P1", "phase": "Phase 8: Synthesis & Interview Prep"},
    {"id": "17_PYTHON_AND_CODING", "file": "17_PYTHON_AND_CODING.md", "title": "Production Code Implementations (MLA, FSM, Batcher)", "priority": "P2", "phase": "Phase 8: Synthesis & Interview Prep"},
    {"id": "19_LEADERSHIP_AND_TECHNICAL_JUDGMENT", "file": "19_LEADERSHIP_AND_TECHNICAL_JUDGMENT.md", "title": "Leadership & Technical Judgment Frameworks", "priority": "P1", "phase": "Phase 8: Synthesis & Interview Prep"},
    {"id": "20_INTERVIEW_QUESTION_BANK", "file": "20_INTERVIEW_QUESTION_BANK.md", "title": "Interview Question Bank (50+ Graded L1-L10 Questions)", "priority": "P0", "phase": "Phase 8: Synthesis & Interview Prep"},
    {"id": "21_CASE_STUDIES", "file": "21_CASE_STUDIES.md", "title": "Production Incident Case Studies & RCAs", "priority": "P0", "phase": "Phase 8: Synthesis & Interview Prep"},
    {"id": "22_FINAL_SYNTHESIS_PLAYBOOKS", "file": "22_FINAL_SYNTHESIS_PLAYBOOKS.md", "title": "The 2-Hour Final Synthesis Playbooks & Formulas", "priority": "P0", "phase": "Phase 8: Synthesis & Interview Prep"},
    {"id": "README", "file": "README.md", "title": "README & Master Curriculum", "priority": "P0", "phase": "Overview"}
]

# Read all markdown contents
docs_data = []
for item in DOC_FILES:
    file_path = os.path.join(ROOT_DIR, item["file"])
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        docs_data.append({
            **item,
            "content": content
        })
    else:
        print(f"Warning: {item['file']} not found.")

print(f"Loaded {len(docs_data)} markdown files.")

# Generate Master Interactive Single-Page App (index.html)
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ML Engineer (LLM & Agentic Systems) — Master Reference</title>
  
  <!-- Fonts -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
  
  <!-- KaTeX for LaTeX Math -->
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>

  <!-- Marked (Markdown Parser) -->
  <script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>

  <!-- Prism.js for Syntax Highlighting -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-c.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-cpp.min.js"></script>

  <!-- Mermaid.js for Diagrams -->
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

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.3); }

    /* Left Sidebar */
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
      cursor: pointer;
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
      <div class="brand-title">
        <span>⚡</span> ML Engineer (SSK)
      </div>
      <div class="brand-subtitle">LLM & Agentic Systems</div>
    </div>
    
    <div class="search-box">
      <input type="text" id="searchInput" class="search-input" placeholder="Quick search topics & math...">
    </div>

    <ul class="nav-list" id="navList">
      <!-- Generated dynamically by JS -->
    </ul>
  </aside>

  <!-- Main Viewport -->
  <main class="main-viewport">
    <div class="top-toolbar">
      <div class="breadcrumbs">
        <span>Curriculum</span>
        <span>/</span>
        <span id="breadcrumbPhase">Phase 0</span>
        <span>/</span>
        <span id="breadcrumbDoc" class="current">00_ROLE_ANALYSIS</span>
      </div>
      <div class="tool-actions">
        <button class="btn" onclick="window.print()">🖨️ Print / PDF</button>
      </div>
    </div>

    <div class="content-container">
      <div class="markdown-body" id="docContent">
        <!-- Rendered markdown content -->
      </div>

      <aside class="toc-sidebar">
        <div class="toc-title">On This Page</div>
        <ul class="toc-list" id="tocList">
          <!-- Dynamic TOC -->
        </ul>
      </aside>
    </div>
  </main>

  <script>
    // Embedded Knowledge Data
    const DOCS = __DOCS_DATA_PLACEHOLDER__;

    let currentDocIndex = 0;

    // Initialize Navigation List
    function initNav() {
      const navList = document.getElementById('navList');
      navList.innerHTML = '';

      let currentPhase = '';

      DOCS.forEach((doc, idx) => {
        if (doc.phase !== currentPhase) {
          currentPhase = doc.phase;
          const phaseHeader = document.createElement('li');
          phaseHeader.className = 'nav-phase-header';
          phaseHeader.textContent = currentPhase;
          navList.appendChild(phaseHeader);
        }

        const li = document.createElement('li');
        const a = document.createElement('a');
        a.className = `nav-item ${idx === currentDocIndex ? 'active' : ''}`;
        a.onclick = () => loadDoc(idx);

        const titleSpan = document.createElement('span');
        titleSpan.textContent = doc.file.replace('.md', '');

        const badge = document.createElement('span');
        badge.className = `priority-badge priority-${doc.priority.toLowerCase()}`;
        badge.textContent = doc.priority;

        a.appendChild(titleSpan);
        a.appendChild(badge);
        li.appendChild(a);
        navList.appendChild(li);
      });
    }

    // Search filter
    document.getElementById('searchInput').addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase();
      const items = document.querySelectorAll('.nav-item');
      items.forEach((item, idx) => {
        const doc = DOCS[idx];
        const match = doc.title.toLowerCase().includes(query) || 
                      doc.file.toLowerCase().includes(query) || 
                      doc.content.toLowerCase().includes(query);
        item.style.display = match ? 'flex' : 'none';
      });
    });

    // Custom Marked extension to protect LaTeX math blocks from markdown processing
    function renderMarkdownWithKaTeX(rawText) {
      // 1. Protect block math $$...$$
      const blockMath = [];
      let text = rawText.replace(/\$\$([\s\S]*?)\$\$/g, (match, math) => {
        blockMath.push(math);
        return `%%%BLOCK_MATH_${blockMath.length - 1}%%%`;
      });

      // 2. Protect inline math $...$
      const inlineMath = [];
      text = text.replace(/\$([^\$\n]+?)\$/g, (match, math) => {
        inlineMath.push(math);
        return `%%%INLINE_MATH_${inlineMath.length - 1}%%%`;
      });

      // 3. Parse Markdown
      let html = marked.parse(text);

      // 4. Restore block math
      html = html.replace(/%%%BLOCK_MATH_(\d+)%%%/g, (match, id) => {
        try {
          return katex.renderToString(blockMath[parseInt(id)], { displayMode: true, throwOnError: false });
        } catch (e) {
          return `<pre class="math-error">${blockMath[parseInt(id)]}</pre>`;
        }
      });

      // 5. Restore inline math
      html = html.replace(/%%%INLINE_MATH_(\d+)%%%/g, (match, id) => {
        try {
          return katex.renderToString(inlineMath[parseInt(id)], { displayMode: false, throwOnError: false });
        } catch (e) {
          return `<code>${inlineMath[parseInt(id)]}</code>`;
        }
      });

      return html;
    }

    function generateTOC() {
      const tocList = document.getElementById('tocList');
      tocList.innerHTML = '';
      const headers = document.querySelectorAll('#docContent h2, #docContent h3');
      
      headers.forEach((h, i) => {
        const id = `heading-${i}`;
        h.id = id;

        const li = document.createElement('li');
        li.className = 'toc-item';
        if (h.tagName === 'H3') li.style.paddingLeft = '24px';

        const a = document.createElement('a');
        a.className = 'toc-link';
        a.href = `#${id}`;
        a.textContent = h.textContent.replace(/^[0-9.]+\s*/, '');
        li.appendChild(a);
        tocList.appendChild(li);
      });
    }

    function loadDoc(idx) {
      currentDocIndex = idx;
      initNav();

      const doc = DOCS[idx];
      document.getElementById('breadcrumbPhase').textContent = doc.phase;
      document.getElementById('breadcrumbDoc').textContent = doc.file.replace('.md', '');

      const contentDiv = document.getElementById('docContent');
      contentDiv.innerHTML = renderMarkdownWithKaTeX(doc.content);

      // Re-run Prism highlight
      Prism.highlightAllUnder(contentDiv);

      // Re-run Mermaid
      mermaid.initialize({ startOnLoad: false, theme: 'dark' });
      document.querySelectorAll('.language-mermaid').forEach(el => {
        const parent = el.parentElement;
        const code = el.textContent;
        const div = document.createElement('div');
        div.className = 'mermaid';
        div.textContent = code;
        parent.replaceWith(div);
      });
      mermaid.run();

      // Generate TOC
      generateTOC();

      // Scroll top
      document.querySelector('.main-viewport').scrollTop = 0;
    }

    // Initialize
    window.addEventListener('DOMContentLoaded', () => {
      initNav();
      loadDoc(0);
    });
  </script>
</body>
</html>
"""

# Replace placeholder with serialized JSON data
rendered_index = INDEX_HTML.replace("__DOCS_DATA_PLACEHOLDER__", json.dumps(docs_data))

with open(os.path.join(ROOT_DIR, "index.html"), "w", encoding="utf-8") as f:
    f.write(rendered_index)

print("Generated master interactive portal: index.html")

# Also generate standalone HTML files in html/ folder
STANDALONE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <title>{{TITLE}}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500;600&family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css">
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10.8.0/dist/mermaid.min.js"></script>
  <style>
    body { font-family: 'Inter', sans-serif; background: #0a0d14; color: #f1f5f9; line-height: 1.7; padding: 40px 20px; }
    .container { max-width: 900px; margin: 0 auto; background: #111622; padding: 40px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.08); }
    h1 { font-family: 'Outfit', sans-serif; font-size: 2.2rem; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 12px; color: #fff; }
    h2 { font-family: 'Outfit', sans-serif; font-size: 1.5rem; margin-top: 36px; color: #e2e8f0; }
    h3 { font-size: 1.2rem; margin-top: 24px; color: #cbd5e1; }
    p, li { color: #cbd5e1; }
    code { font-family: 'Fira Code', monospace; background: rgba(255,255,255,0.08); color: #f472b6; padding: 2px 6px; border-radius: 4px; }
    pre { background: #0f141f !important; padding: 16px !important; border-radius: 8px; overflow-x: auto; }
    pre code { background: transparent !important; color: #e2e8f0; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; background: rgba(0,0,0,0.2); }
    th, td { border: 1px solid rgba(255,255,255,0.08); padding: 10px 14px; text-align: left; }
    th { background: rgba(255,255,255,0.05); }
    blockquote { border-left: 4px solid #6366f1; background: rgba(99,102,241,0.08); padding: 12px 18px; border-radius: 0 8px 8px 0; }
    .nav-back { display: inline-block; margin-bottom: 20px; color: #818cf8; text-decoration: none; font-size: 0.9rem; }
    .nav-back:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="container">
    <a href="../index.html" class="nav-back">← Back to Master Knowledge Hub</a>
    <div id="content"></div>
  </div>
  <script>
    const rawMarkdown = {{CONTENT_JSON}};
    function renderMath(text) {
      const blockMath = [];
      text = text.replace(/\$\$([\s\S]*?)\$\$/g, (m, math) => { blockMath.push(math); return `%%%BM_${blockMath.length-1}%%%`; });
      const inlineMath = [];
      text = text.replace(/\$([^\$\n]+?)\$/g, (m, math) => { inlineMath.push(math); return `%%%IM_${inlineMath.length-1}%%%`; });
      let html = marked.parse(text);
      html = html.replace(/%%%BM_(\d+)%%%/g, (m, id) => katex.renderToString(blockMath[parseInt(id)], { displayMode: true, throwOnError: false }));
      html = html.replace(/%%%IM_(\d+)%%%/g, (m, id) => katex.renderToString(inlineMath[parseInt(id)], { displayMode: false, throwOnError: false }));
      return html;
    }
    document.getElementById('content').innerHTML = renderMath(rawMarkdown);
    Prism.highlightAll();
    mermaid.initialize({ startOnLoad: true, theme: 'dark' });
  </script>
</body>
</html>
"""

for doc in docs_data:
    file_name = doc["file"].replace(".md", ".html")
    out_path = os.path.join(HTML_DIR, file_name)
    html_content = STANDALONE_TEMPLATE.replace("{{TITLE}}", doc["title"]).replace("{{CONTENT_JSON}}", json.dumps(doc["content"]))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)

print(f"Generated {len(docs_data)} standalone HTML files in {HTML_DIR}")
