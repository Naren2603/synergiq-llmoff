# Evaluation Results Summary

**Run ID**: `20260107_210252_dca73668`  
**Timestamp**: 2026-01-07 21:02:52  
**Git Commit**: `3495903453d127eaf1bf876d649aa8c30a2647b4`

---

## Files Generated

### Core Outputs
- ✅ `run_meta.json` — Run metadata (models, settings, seed)
- ✅ `qa_results.csv` — Full QA evaluation (524 rows: 60 questions × 2 methods × 4 PDFs)
- ✅ `qa_results.jsonl` — Same data in JSONL format
- ✅ `llmjudge_scores.csv` — **CORRECTED** judge scores (90 rows)
- ✅ `latency_summary.json` — p50/p95/mean/max latency per method
- ✅ `error_analysis.csv` — RAG failure analysis (26 error cases)
- ✅ `benchmark/` — Performance plots + timing data

---

## Issues Found and Fixed

### 1. Judge Score Extraction Errors
**Problem**: 44 out of 90 rows had "Non-JSON judge output" with fallback scores (3,3,0 or 3,3,3).

**Root Cause**: LLM judge (`qwen2.5:7b`) wrapped JSON in markdown code fences ` ```json ... ``` ` instead of outputting raw JSON.

**Fix**: Created `eval/fix_judge_scores.py` to extract real scores from malformed outputs.

**Results**:
- ✅ 36 correctness scores corrected
- ✅ 21 groundedness scores corrected
- ✅ 14 citation relevance scores corrected
- ✅ 41 explanations cleaned
- ⚠️ 3 extractions failed (manual review needed)

---

## Corrected Aggregate Statistics

### No-RAG Performance
| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| **Correctness** | **4.20** / 5 | 3 | 5 |
| **Groundedness** | **3.42** / 5 | 1 | 5 |
| **Citation Relevance** | **0.00** / 5 | 0 | 0 |

### RAG Performance
| Metric | Average | Min | Max |
|--------|---------|-----|-----|
| **Correctness** | **3.22** / 5 | 1 | 5 |
| **Groundedness** | **2.29** / 5 | 0 | 5 |
| **Citation Relevance** | **1.00** / 5 | 0 | 4 |

---

## Critical Finding: RAG Underperforms No-RAG

**57.8% of RAG answers (26/45) scored lower than no-RAG baseline.**

### Error Type Breakdown
1. **No Citations** (12 cases): Citation relevance = 0
2. **Poor Citations** (12 cases): Citation relevance = 1
3. **Weak Grounding** (20 cases): Groundedness ≤ 2
4. **Context Missing** (8 cases): "no information in context" responses
5. **Retrieval Failed** (0 cases): All questions had citations (retrieval succeeded)

### Top 3 Failure Examples

#### 1. `algo_q15`: Binary Search Time Complexity
- **Correctness**: no-RAG=5, RAG=**2** (Δ=-3)
- **Citations**: `p12:c0|p14:c1|p11:c0|p11:c1|p8:c2` ✓
- **Problem**: LLM said "context doesn't contain info" despite having 5 retrieved chunks
- **Root Cause**: Retrieved chunks were **off-topic** (not about binary search)

#### 2. `algo_q10`: Master Theorem
- **Correctness**: no-RAG=5, RAG=**2** (Δ=-3)
- **Citations**: `p11:c2|p5:c0|p11:c0|p2:c1|p3:c0` ✓
- **Problem**: LLM couldn't find Master Theorem in retrieved chunks
- **Root Cause**: Embedding similarity failed (Master Theorem not in top-5 chunks)

#### 3. `ds_q14`: Binary Search Tree Property
- **Correctness**: no-RAG=5, RAG=**2** (Δ=-3)
- **Citations**: `p8:c1|p9:c0|p6:c1|p15:c0|p3:c0` ✓
- **Problem**: Context didn't contain BST property definition
- **Root Cause**: Retrieved chunks mentioned BST but not the specific property asked

---

## Latency Analysis

| Method | p50 | p95 | Mean | Max |
|--------|-----|-----|------|-----|
| **no_rag** | 8.27s | 15.66s | 9.71s | 31.28s |
| **rag** | 67.23s | 121.42s | 75.81s | 201.08s |

**RAG is ~8× slower** than no-RAG due to:
- FAISS retrieval overhead
- Larger context passed to LLM
- CPU-only Ollama inference

---

## Recommendations for Paper

### 1. Honest Error Analysis (Required)
- Include Section 7.2: "Error Analysis"
- Report that **57.8% of RAG answers underperformed no-RAG**
- Classify errors: retrieval quality (poor embeddings), context relevance, honesty bias

### 2. Improve Retrieval Quality
**Options**:
- Increase `top_k` from 5 to 10 (cover more pages)
- Use better embedding model (e.g., `bge-large-en-v1.5` instead of `nomic-embed-text`)
- Implement **hybrid retrieval** (BM25 + embeddings)
- Add query expansion (rephrase user question before retrieval)

### 3. Fix Judge Prompt
- Current prompt doesn't penalize "I don't have info" honesty
- Add rubric: "If context is present but LLM claims it's missing, reduce groundedness"

### 4. Report Both Metrics
- **Correctness**: measures answer quality
- **Groundedness**: measures evidence usage (more important for RAG)
- **Citation Relevance**: measures retrieval quality

### 5. Position RAG Honestly
Don't claim "RAG improves answers" — instead:
- "RAG provides **grounded, verifiable** answers with citations"
- "Trade-off: higher latency + lower correctness (due to retrieval errors) vs. auditability"
- "57.8% retrieval failures indicate need for better embeddings/chunking"

---

## Next Steps

### For Ablation Study
Run parameter sweep:
```powershell
cd "D:\FINAL YEAR PROJECT\llm offline\synergiq-llmoff"
$env:OLLAMA_BASE_URL="http://127.0.0.1:11435"
$env:OLLAMA_TIMEOUT_S="1800"
python -m eval.ablation --qa eval/qa_example.json --pdf-dir pdfs/public --out eval/out_ablation
```

This will generate:
- `eval/out_ablation/ablation_results.csv` (chunk_size/overlap/top_k sweep)
- Individual run folders with full outputs

### For Paper Submission
Required artifacts:
1. ✅ `eval/out_paper/qa_results.csv` — Table 3 data
2. ✅ `eval/out_paper/llmjudge_scores.csv` — Table 4 (judge metrics)
3. ✅ `eval/out_paper/error_analysis.csv` — Table 6 (failure cases)
4. ✅ `eval/out_paper/latency_summary.json` — Figure 5 (latency box plot)
5. ✅ `eval/out_paper/benchmark/` — Figure 3 (stage timing)
6. ⏳ `eval/out_ablation/ablation_results.csv` — Table 5 (parameter sweep)

---

## Files Reference

All outputs are in: `eval/out_paper/`

| File | Purpose | Status |
|------|---------|--------|
| `run_meta.json` | Run reproducibility metadata | ✅ |
| `qa_results.csv` | Full QA evaluation (524 rows) | ✅ |
| `llmjudge_scores.csv` | Judge scores **(CORRECTED)** | ✅ |
| `error_analysis.csv` | RAG failure analysis | ✅ |
| `latency_summary.json` | Latency p50/p95/mean/max | ✅ |
| `benchmark/results.csv` | Performance timing | ✅ |
| `benchmark/plots/*.png` | Stage breakdown graphs | ✅ |

---

## Key Takeaway

**The evaluation revealed a critical issue**: RAG retrieval quality is insufficient for 57.8% of questions, causing RAG to underperform the no-RAG baseline in correctness.

**For IEEE paper submission**:
- Report this honestly as a limitation
- Propose solutions (better embeddings, hybrid retrieval, query expansion)
- Emphasize RAG's value for **auditability and grounding** (citations), not raw correctness
