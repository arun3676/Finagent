# FinAgent - Agent System Documentation

**Welcome to the FinAgent agent documentation hub.** This directory contains comprehensive guides for understanding and working with the multi-agent system that powers FinAgent's financial research capabilities.

## 📚 Documentation Structure

- **[README.md](README.md)** (this file) - Project overview, architecture, and error prevention
- **[backend-guide.md](backend-guide.md)** - Backend API implementation details
- **[agents-workflow.md](agents-workflow.md)** - LangGraph multi-agent workflow system
- **[frontend-guide.md](frontend-guide.md)** - Frontend UI and component guide

---

## 🎯 Project Identity

**FinAgent** is an enterprise-grade agentic RAG system for financial research that processes SEC filings and earnings calls to answer complex queries with compliance-grade citations.

**Core Value Proposition**: Solving the critical fintech problem of hallucinations and lack of trustworthiness in AI systems for regulated environments.

**Target**: Portfolio project demonstrating 2+ years of production AI engineering experience for $150K+ roles.

---

## 🏗️ Architecture Overview

```
User Query → FastAPI Backend → LangGraph Multi-Agent Workflow
                                    ↓
              Router → Planner → Retriever → Analyst → Synthesizer → Validator
                                    ↓
              Hybrid Search (BM25 + Dense Embeddings + Cohere Reranking)
                                    ↓
              Qdrant Vector DB with RBAC Filtering
                                    ↓
              Compliance-Grade Citations (every claim → exact source paragraph)
```

---

## 🌟 Key Differentiators

What makes this NOT a tutorial project:

1. **Document-Aware Chunking**: Preserves SEC 10-K section boundaries (Item 1A, 7, 8) - generic chunkers break these
2. **Hybrid Search with RRF**: BM25 catches exact terms ("EBITDA"), dense embeddings catch semantics ("operating profit")
3. **Self-Correcting Validator Agent**: Rejects unsupported claims, loops back for more evidence
4. **Compliance-Grade Citations**: Every number traces to exact source paragraph - required for financial compliance
5. **Multi-Hop Reasoning**: "Compare AAPL vs MSFT margins" requires planning, parallel retrieval, synthesis

---

## 💻 Technology Stack

| Layer | Technology | Why |
|-------|------------|-----|
| Backend | FastAPI (Python) | Async, type-safe, OpenAPI docs |
| Orchestration | LangGraph | State machines for agents, 2025 industry standard |
| Vector DB | Qdrant | Hybrid search, filtering, RBAC |
| Embeddings | OpenAI text-embedding-3-large | Best quality, 3072 dimensions |
| Reranking | Cohere rerank-v3 | 15-20% precision boost |
| LLM | GPT-4o / Claude 3.5 Sonnet | Best reasoning, function calling |
| Frontend | Next.js 14 + shadcn/ui | Professional, SSR |
| Data | SEC EDGAR API | Real SEC filings |

---

## 🎯 Success Metrics

| Metric | Target | Why It Matters |
|--------|--------|----------------|
| Recall@10 | >85% | Relevant chunks in top 10 |
| Faithfulness | >90% | Claims supported by context |
| Citation Coverage | >95% | Facts have sources |
| Simple Query Time | <8 seconds | Single company queries |
| Complex Query Time | <20 seconds | Multi-company comparisons |

---

## 🚨 CRITICAL: Error Prevention Context

### Known Issues to ALWAYS Avoid

#### 1. PowerShell `&&` Command Chaining Error
**Problem**: Using `&&` in subprocess calls fails on Windows PowerShell  
**Impact**: Scripts that work on Linux/Mac fail on Windows

**When generating code:**
- ❌ NEVER: `subprocess.run("cd x && python y")`
- ✅ ALWAYS: Use `pathlib.Path` + `subprocess.run([...], cwd=...)`

**Template for subprocess calls:**
```python
from pathlib import Path
import subprocess
import sys

script_path = Path(__file__).parent / "scripts" / "test.py"
subprocess.run([sys.executable, str(script_path)], cwd=Path(__file__).parent)
```

#### 2. Pydantic Settings v2 Configuration Error
**Problem**: Old-style `class Config:` doesn't load .env files in Pydantic v2  
**Impact**: Environment variables silently fail to load, causing "API key not found" errors

**When generating Settings classes:**
- ❌ NEVER: `class Config: env_file = ".env"`
- ✅ ALWAYS: Use `model_config = SettingsConfigDict(...)`

**Template for Settings classes:**
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    api_key: str
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
```

### Code Generation Guidelines

**When asked to create any script that:**
- Runs other Python scripts → Use subprocess with list args + cwd
- Needs environment variables → Use Pydantic v2 SettingsConfigDict
- References file paths → Use pathlib.Path throughout
- Chains commands → Use semicolon `;` or separate calls, not `&&`

**When reviewing existing code:**
- Flag any `&&` in subprocess calls
- Flag `class Config:` in Pydantic models
- Flag hardcoded path separators (`/` or `\\`)
- Flag bare `.env` strings without Path resolution

### Testing Reminders
Before suggesting code is "done":
1. Check: Will this work on both Windows and Linux?
2. Check: Are paths using pathlib.Path?
3. Check: Are Settings using v2 SettingsConfigDict?
4. Check: Are subprocess calls using list args?

---

## 📖 Quick Reference for Common Tasks

### Creating a New Settings Class
```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class YourSettings(BaseSettings):
    your_field: str
    
    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False
    )
```

### Running a Python Script from Another Script
```python
from pathlib import Path
import subprocess
import sys

script = Path(__file__).parent / "other_script.py"
result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
```

### Defining File Paths
```python
from pathlib import Path

# Get project root
project_root = Path(__file__).parent.parent

# Build paths
data_dir = project_root / "data" / "sec_filings"
config_file = project_root / "config" / "settings.yaml"
```

---

## 🔧 Context for AI Assistants

When helping with FinAgent:

### Priority 1: Prevent Known Errors
- **Always use Path() for file paths**
- **Always use SettingsConfigDict for Pydantic settings**
- **Never use `&&` in subprocess calls**
- **Test patterns work cross-platform**

### Priority 2: Follow Architecture
- Keep agent logic in `agents/` module
- Use LangGraph for orchestration
- Hybrid search for retrieval
- Citations for every claim

### Priority 3: Production Quality
- Type hints on all functions
- Async for I/O operations
- Proper error handling
- Comprehensive logging

### Code Review Checklist
When generating or reviewing code, verify:
- [ ] Imports use `pydantic_settings` not `pydantic` for Settings
- [ ] All paths use `Path()` not string concatenation
- [ ] No `&&` operators in any commands
- [ ] Settings classes use `model_config = SettingsConfigDict(...)`
- [ ] Type hints on all functions
- [ ] Async/await for I/O operations

---

## 🎤 Interview Pitch Context

"I built FinAgent, an agentic RAG system for financial research—essentially an open-source version of what Morgan Stanley deployed to 16,000 advisors with 98% adoption. The problem: Analysts spend 65% of their time on data gathering, not analysis. My solution addresses the critical 2026 fintech challenge of AI hallucinations in regulated environments through a self-correcting multi-agent pipeline with compliance-grade citations."

---

## 🗂️ File Organization

```
finagent/
├── backend/app/           # Python FastAPI
│   ├── agents/            # LangGraph workflow → See agents-workflow.md
│   ├── retrieval/         # Hybrid search
│   ├── citations/         # Citation engine
│   ├── ingestion/         # SEC EDGAR loader
│   ├── chunking/          # Document-aware chunking
│   └── evaluation/        # Metrics and test datasets
├── frontend/              # Next.js → See frontend-guide.md
│   ├── components/chat/   # Chat UI components
│   └── lib/api/           # Backend API clients
└── docs/                  # Documentation
    └── agents/            # This directory
```

---

## 🚀 Getting Started

1. **New to the project?** Start with this README to understand the architecture
2. **Backend developer?** Read [backend-guide.md](backend-guide.md) for API details
3. **Working on agents?** See [agents-workflow.md](agents-workflow.md) for LangGraph implementation
4. **Frontend developer?** Check [frontend-guide.md](frontend-guide.md) for UI components

---

## ⚠️ Error Response Templates

**If user reports `&&` error:**
"This is a known Windows PowerShell incompatibility. The fix is to use pathlib + subprocess with list args. See the error prevention section above. I'll update the code to use the correct pattern."

**If user reports Pydantic settings not loading:**
"This is a known Pydantic v2 migration issue. The fix is to use `SettingsConfigDict` instead of nested `Config` class. See the error prevention section above. I'll update to the v2 pattern."

**If user reports path errors:**
"This needs to use `pathlib.Path` for cross-platform compatibility. I'll update to use Path() throughout."

---

## 📊 Critical Problem This Solves

**Hallucinations in Fintech AI**: Up to 40% of agentic projects fail by 2027 from trust gaps. In finance, AI errors lead to multi-million-dollar fines. FinAgent solves this with:
- Validator agent enforcing self-correction loops
- Precise citations for auditability
- 90%+ faithfulness target through grounded generation

---

## 🎯 When Working on This Project

1. **Always verify real data**: Never use placeholder/mock SEC data
2. **Check type alignment**: Backend Pydantic ↔ Frontend TypeScript
3. **Test end-to-end**: Don't assume components work together
4. **Measure everything**: No claims without metrics
5. **Think production**: Error handling, logging, observability
