<div align="center">

# 🔬 Relational RAG for Jupyter Notebooks

### Stop reading notebooks. Start querying them.
 

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-1.2.8-1C3C3C?style=flat-square&logo=langchain&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-1.4.1-FF6B35?style=flat-square)](https://trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B-F55036?style=flat-square)](https://groq.com)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=flat-square)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

<br/>

> *"What model was trained in this notebook?"*
> *"What accuracy did it achieve?"*
> *"Walk me through the preprocessing steps."*

A local RAG system that goes beyond keyword search — it understands **what each cell does, why it exists, and what it produced.**

<br/>

**Demo — Query 1**

https://github.com/user-attachments/assets/7dc46a29-5116-43ab-b23d-beede80fd1ba


**Demo — Query 2**

https://github.com/user-attachments/assets/3ad5a410-17bd-4850-91ef-b2cc82b75e5c



<br/>

</div>

---

## ✨ What makes this different

Most RAG systems split documents into chunks and send those chunks directly to the LLM. This project uses a two-stage **Bait & Switch** retrieval pattern:

```
① Small chunk   →   semantic search   →   extract cell_id
                                                  ↓
③ Full cell     ←   instant lookup    ←   cell_id
```

The small chunk is only used to *find* the right cell. The **full uncut parent cell** — including raw code, cleaned output, LLM-generated explanation, intent, and statistical results — is what actually gets sent to the LLM.

This means the model always has complete context, not a fragmented slice.

---

## 🏗️ Architecture

![Architecture Diagram](assets/Jupyter_RAG_Architecture.png)

The system runs in three phases:

### Phase 1 — Ingestion & Analysis

Each notebook cell goes through a four-step enrichment pipeline:

| Step | File | What it does |
|------|------|--------------|
| Parse | `parser.py` | Extracts code cells, markdown cells, and outputs from `.ipynb` |
| Analyze | `analyzer.py` | AST parsing — identifies variables, libraries, function calls |
| Explain | `explainer.py` | Ollama / Llama 3.1 generates intent, explanation, statistical result |
| Build | `final_object_builder.py` | Assembles the enriched cell object |

All enriched cells are stored in `artifacts/custom_object.json` — the **Ground Truth Warehouse** — keyed by a unique `cell_id`.

### Phase 2 — Indexing

Two parallel embedding pipelines populate ChromaDB:

| Collection | Embedding Model | Status                     |
|------------|----------------|----------------------------|
| `notebook_text_rag` | `bge-m3` via Ollama | ✅ Active — queried at retrieval |
| `notebook_code_rag` | `CodeT5Embeddings` | 🔵 Built — but not queried |

Every chunk in both collections carries the `cell_id` of its parent cell as metadata. This is what enables the Bait & Switch lookup.

> **Why two embedding models?** Natural language and code occupy different semantic spaces. `bge-m3` understands explanations. `CodeT5` understands variable names, function signatures, and syntax — things `bge-m3` treats as noise.

### Phase 3 — Retrieval & Generation

```
1. User query
2. Semantic search → notebook_text_rag (ChromaDB)
3. Top-k chunks returned
4. cell_id extracted from chunk metadata
5. Full parent cell fetched from custom_object.json  ← Bait & Switch
6. Retrieved cells + conversation history → Chatbot Orchestrator
7. Structured prompt assembled
8. Groq / Llama 3.3 70B Versatile generates response
9. Response + question stored in instance memory
```

---

## 📁 Project Structure

```
jupyter-notebook-rag/
│
├── 📄 main.py                     ← Entry point, single-question demo
├── 📄 build_index.py              ← Run once to parse + embed notebook
├── 📄 query_engine.py             ← NotebookChatbot class + CLI loop
├── 📄 retrieval.py                ← RelationalRetriever, Bait & Switch logic
│
├── ── Ingestion Pipeline ──────────────────────────────
├── 📄 parser.py                   ← Notebook cell extraction (nbformat)
├── 📄 analyzer.py                 ← AST parsing, intent extraction
├── 📄 explainer.py                ← LLM-based cell explanation (Ollama)
├── 📄 final_object_builder.py     ← Assembles enriched cell object
│
├── ── Indexing Pipeline ───────────────────────────────
├── 📄 rag_document_builder.py     ← Builds LangChain Document objects
├── 📄 rag_text_splitter.py        ← Splits text and code with cell_id metadata
├── 📄 index_builder.py            ← Builds ChromaDB collections
├── 📄 codeEmbedder.py             ← CodeT5 custom embedding model
├── 📄 storage.py                  ← Save/load JSON artifacts
│
├── ── Artifacts (generated) ───────────────────────────
├── 📂 artifacts/
│   ├── custom_object.json         ← Ground Truth Warehouse
│   ├── code_docs.json
│   ├── text_docs.json
│   ├── split_code_docs.json
│   └── split_text_docs.json
│
├── 📂 chroma_store/               ← ChromaDB vector collections
├── 📂 assets/
│   ├── architecture.png           ← Architecture diagram
│   ├── JRAG_Q1.mp4                ← Demo video 1
│   └── JRAG_Q2.mp4                ← Demo video 2
│
├── 📄 requirements.txt            ← Full locked environment
├── 📄 requirements-core.txt       ← Direct dependencies only
└── 📄 .env                        ← API keys (not committed)
```

---

## ⚙️ Setup

### Prerequisites

- Python 3.10
- [Ollama](https://ollama.ai) installed and running locally
- [Groq API key](https://console.groq.com) (free tier available)

### 1. Clone the repository

```bash
git clone https://github.com/Priyanshu-Upadhyay-27/QnA_With_Jupyter.git
cd QnA_With_Jupyter
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
# Minimal — direct dependencies only
pip install -r requirements-core.txt

# Full — exact locked environment
pip install -r requirements.txt
```

### 4. Pull Ollama models

```bash
ollama pull llama3.1
ollama pull bge-m3
```

### 5. Set environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

---

## 🚀 Usage

### Step 1 — Build the index

Open `build_index.py` and set your notebook path:

```python
NOTEBOOK = "your_notebook.ipynb"
```

Then run:

```bash
python build_index.py
```

This will:
- Parse every code cell in the notebook
- Run LLM analysis on each cell (takes a few minutes for large notebooks)
- Save the Ground Truth Warehouse to `artifacts/custom_object.json`
- Build and populate both ChromaDB collections

> ⚠️ Run this once per notebook. Re-run only if the notebook changes.

### Step 2 — Chat with the notebook

```bash
python query_engine.py
```

```
Notebook chatbot ready. Type 'quit' to exit.

You: What model was trained in this notebook?
Bot: An SVM classifier was trained using scikit-learn's SVC with an RBF
     kernel. The model was trained on the Iris dataset after standard
     scaling was applied to all four features...

You: What accuracy did it achieve?
Bot: The model achieved 100% accuracy on the test set (30 samples).
     The training accuracy was also 100%, suggesting the dataset is
     linearly separable in the RBF kernel space...

You: Show me the training code
Bot: Here is the training cell:
     from sklearn.svm import SVC
     model = SVC(kernel='rbf', C=1.0)
     model.fit(X_train_scaled, y_train)
```

---

## 🧠 Models Used

| Component | Model | Provider | Runs |
|-----------|-------|----------|------|
| Cell analysis | Llama 3.1 | Ollama | Local |
| Text embeddings | bge-m3 | Ollama | Local |
| Code embeddings | CodeT5 | HuggingFace | Local |
| Chat & generation | Llama 3.3 70B Versatile | Groq API | Cloud |

All analysis happens locally. Only the final chat response uses the Groq API — keeping your notebook data private.

---

## 🔑 Key Technical Decisions

**Why store full cells separately from the vector index?**

ChromaDB chunks are optimized for similarity search, not for reading. A 50-word chunk loses the surrounding context that makes an answer meaningful. Storing full enriched cells in `custom_object.json` and fetching by `cell_id` gives the LLM complete information every time.

**Why two embedding models?**

`bge-m3` is trained on natural language — it understands semantic meaning in explanations. `CodeT5` is trained on source code — it understands variable names, function signatures, and syntax patterns. Using the right model for each modality improves retrieval precision for code-specific questions.

**Why Groq for generation but Ollama for analysis?**

Analysis runs once during indexing — speed does not matter. Chat responses are interactive — latency matters a lot. Groq's inference speed (700+ tokens/sec) makes the chatbot feel instant. Ollama handles the heavy lifting privately.

---

## 💡 Ideas to Make This More Robust

Two architectural gaps worth thinking about if this system were taken further:

---

### 🔀 Query Router

Currently, retrieval is hardcoded to query only `notebook_text_rag`.
A query router sitting before retrieval could classify intent and
route accordingly:

| Query type | Example | Routes to |
|---|---|---|
| Code question | "Show me the training code" | `notebook_code_rag` |
| Explanation question | "Why was StandardScaler used?" | `notebook_text_rag` |
| General question | "What is an SVM?" | Direct LLM — no retrieval |

The `notebook_code_rag` collection is already built and populated —
it just needs something to decide when to use it.

---

### 📚 Multi-Notebook Support

Right now the system indexes one notebook at a time. Extending it to
support multiple notebooks simultaneously — with `notebook_name` as
a metadata tag on every chunk — would enable cross-notebook queries:

> *"Which notebook had better accuracy — the SVM one or the Random Forest one?"*

> *"Show me all the preprocessing steps across my notebooks."*

This would require restructuring the Ground Truth Warehouse to key by
`notebook_name + cell_id` instead of `cell_id` alone, and adding
namespace separation between notebooks inside ChromaDB.

---

## 📦 Dependencies

Install core dependencies only:

```bash
pip install -r requirements-core.txt
```

Install full locked environment (recommended for exact reproducibility):

```bash
pip install -r requirements.txt
```

Core dependencies at a glance:

```
langchain-core              LangChain base abstractions
langchain-ollama            Ollama LLM + embeddings integration
langchain-groq              Groq LLM integration
langchain-chroma            ChromaDB vector store integration
langchain-text-splitters    RecursiveCharacterTextSplitter
chromadb                    Local vector database
transformers                CodeT5 model loading
torch                       PyTorch backend for CodeT5
nbformat                    Jupyter notebook parsing
python-dotenv               Environment variable management
```

---

## 🤝 Contributing

Contributions, ideas, and feedback are welcome.

If you have thoughts on the query router implementation, 
multi-notebook support, or any other improvement — feel free to:

- Open an **Issue** to discuss ideas or report bugs
- Open a **Pull Request** for any improvements
- Star the repo if you found it useful ⭐

This project is actively maintained and open to collaboration.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🔍 semantic search, 🧠 local LLMs, and the **Bait & Switch** retrieval pattern.

</div>
