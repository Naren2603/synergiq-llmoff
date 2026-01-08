# synergiq-llmoff

Offline LLM (Ollama) + embeddings (OllamaEmbeddings) PDF pipeline with FastAPI:
- Upload PDF
- Chat (RAG with FAISS)
- Summarize (map-reduce with LLM)
- Audio (edge-tts with gTTS fallback)
- Video (OpenCV/PIL rendered summary video with audio)

## Prerequisites

### Required
- Python 3.8+
- [Ollama](https://ollama.com/download) - for LLM inference
- Pull an Ollama model (recommended: `qwen2.5:7b` or `llama3.1`)
  ```bash
  ollama pull qwen2.5:7b
  ```

### Optional (for OCR of scanned PDFs)
- Tesseract OCR
  - **Linux**: `sudo apt-get install tesseract-ocr`
  - **macOS**: `brew install tesseract`
  - **Windows**: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### System Libraries (for video generation)
- **Linux**: 
  ```bash
  sudo apt-get install ffmpeg libsm6 libxext6 libxrender-dev
  ```
- **macOS**: 
  ```bash
  brew install ffmpeg
  ```
- **Windows**: Download [FFmpeg](https://ffmpeg.org/download.html) and add to PATH

## Setup

### 1. Clone and create virtual environment
```bash
git clone https://github.com/Naren2603/synergiq-llmoff.git
cd synergiq-llmoff
python -m venv .venv

# Activate virtual environment
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

### 3. Configure environment variables
Copy `.env.example` to `.env` and customize:
```bash
cp .env.example .env
```

Edit `.env`:
```env
# Ollama Configuration
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_BASE_URL=http://localhost:11434

# Embeddings for RAG indexing (falls back to OLLAMA_MODEL if unset)
# EMBED_MODEL=nomic-embed-text

# Summarization / Chunking
SUMMARY_MAP_CHARS=6000
CHUNK_SIZE=1200
CHUNK_OVERLAP=200

# RAG Configuration
TOP_K=5

# Optional: Tesseract (if using OCR)
# TESSERACT_CMD=/usr/bin/tesseract

# Optional: Data directory
# DATA_DIR=data
```

### 4. Start Ollama service
Ensure Ollama is running:
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it (typically automatic on install)
# or run: ollama serve
```

## Run

Start the FastAPI server:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Usage

### Complete Workflow (Recommended: `/media` endpoints)

#### 1. Upload PDF
Upload a PDF and trigger full processing pipeline (extraction, vectorization, summary, audio, video):

```bash
curl -X POST "http://localhost:8000/media/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_document.pdf"
```

**Response:**
```json
{
  "doc_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "filename": "your_document.pdf",
  "num_pages": 10,
  "status": "ready"
}
```

**Note**: Processing may take time depending on document length. Save the `doc_id` for subsequent requests.

#### 2. Check Processing Status
```bash
curl -X GET "http://localhost:8000/media/status/{doc_id}"
```

**Response (processing):**
```json
{
  "state": "tts",
  "step": "generating_audio"
}
```

**Response (ready):**
```json
{
  "state": "ready",
  "doc_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "num_pages": 10,
  "has_summary": true,
  "has_audio": true,
  "has_video": true
}
```

#### 3. Get Summary
```bash
# Detailed summary (default)
curl -X GET "http://localhost:8000/media/summary/{doc_id}"

# Brief summary
curl -X GET "http://localhost:8000/media/summary/{doc_id}?mode=brief"
```

**Response:**
```json
{
  "doc_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "mode": "detailed",
  "summary": "This document discusses...",
  "num_pages": 10
}
```

#### 4. Download Audio
```bash
curl -X GET "http://localhost:8000/media/audio/{doc_id}" \
  -o summary_audio.mp3
```

#### 5. Download Video
```bash
curl -X GET "http://localhost:8000/media/video/{doc_id}" \
  -o summary_video.mp4
```

#### 6. Chat with Document (RAG)
```bash
curl -X POST "http://localhost:8000/media/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "question": "What are the main findings?"
  }'
```

No-RAG baseline (for comparison):

```bash
curl -X POST "http://localhost:8000/media/chat?mode=no_rag" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "question": "What are the main findings?"
  }'
```

**Response:**
```json
{
  "answer": "Based on the document, the main findings are...",
  "sources": ["p3:c2", "p3:c5", "p7:c1"]
}
```

### Legacy Endpoints (In-Memory)

For backward compatibility, the original in-memory endpoints remain available:

#### Upload PDF (In-Memory)
```bash
curl -X POST "http://localhost:8000/upload_pdf" \
  -F "file=@your_document.pdf"
```

#### Chat (In-Memory)
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "your_doc_id_here",
    "question": "What is this document about?"
  }'
```

## Project Structure

```
synergiq-llmoff/
├── app/
│   ├── core/
│   │   ├── config.py          # Configuration management
│   │   ├── ocr.py             # OCR utilities (Tesseract)
│   │   ├── ollama_client.py   # Ollama LLM client
│   │   ├── pdf_loader.py      # PDF text extraction
│   │   ├── pdf_utils.py       # PDF utilities
│   │   ├── rag.py             # RAG (FAISS vectorstore, retrieval)
│   │   ├── storage.py         # Disk storage utilities
│   │   ├── summarize.py       # Text chunking utilities
│   │   ├── summarizer.py      # LLM-based summarization
│   │   ├── tts.py             # Text-to-speech (edge-tts/gTTS)
│   │   └── video.py           # Video generation (OpenCV)
│   ├── routes/
│   │   ├── chat.py            # In-memory chat endpoints
│   │   ├── media.py           # Complete /media pipeline
│   │   └── pdf.py             # In-memory PDF upload
│   ├── main.py                # FastAPI app
│   └── state.py               # In-memory state store
├── data/                      # Generated data (PDFs, indices, media)
│   ├── docs/                  # Per-document storage
│   └── indices/               # FAISS indices
├── .env.example               # Environment template
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Features

### 📄 PDF Processing
- Extract text from PDFs (pypdf)
- OCR for scanned pages (Tesseract, optional)
- Page-level text extraction
- Metadata tracking

### 🤖 RAG (Retrieval Augmented Generation)
- FAISS vector store for semantic search
- Ollama embeddings (offline)
- Evidence retrieval with page citations
- Context-aware answers using LLM

### 📝 Summarization
- Map-reduce approach for long documents
- Chunk-based processing to avoid context limits
- LLM-powered abstractive summaries
- Brief and detailed modes

### 🔊 Audio Generation
- **edge-tts** (Microsoft Edge TTS) - preferred, high quality
- **gTTS** (Google TTS) - fallback option
- **Piper** (offline neural TTS) - optional, fully offline
- MP3 output format
- Automatic voice selection

#### Offline human-like voice (Piper + casual narration)

For fully offline, more natural-sounding audio:

1) Install Piper TTS and download a Piper voice model (.onnx)
2) Set in `.env`:

```env
TTS_ENGINE=piper
SPOKEN_STYLE=casual
PIPER_MODEL=D:\path\to\voice.onnx
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

Notes:
- The pipeline will rewrite summaries into a casual spoken script using your local Ollama model, then synthesize audio via Piper.
- If `ffmpeg` is installed, Piper audio is converted to MP3; otherwise it will be served as WAV.

### 🎬 Video Generation
- Text-to-video rendering using OpenCV + PIL
- Multiple screens with text wrapping
- Audio muxing with moviepy
- MP4 output with H.264 codec
- Auto-looping to match audio duration

### 💾 Persistent Storage
- Disk-based document storage
- Restorable vector indices
- Status tracking
- Organized per-document directories

## Troubleshooting

### Ollama Connection Error
**Error**: `Connection refused to localhost:11434`

**Solution**: Ensure Ollama is running:
```bash
# Check if running
curl http://localhost:11434/api/tags

# Start Ollama service (if needed)
ollama serve
```

### Model Not Found
**Error**: `model 'qwen2.5:7b' not found`

**Solution**: Pull the model:
```bash
ollama pull qwen2.5:7b
# Or use another model and update .env
```

### OCR Not Working
**Error**: `TesseractNotFoundError`

**Solution**: Install Tesseract OCR (see Prerequisites) and optionally set `TESSERACT_CMD` in `.env`

### FFmpeg Error (Video Generation)
**Error**: `ffmpeg not found`

**Solution**: Install FFmpeg (see Prerequisites)

### Audio/Video Generation Fails
**Error**: Various audio/video generation errors

**Solution**: 
- Check that summary text is not empty
- Ensure FFmpeg is installed
- Check disk space in `data/` directory
- Review server logs for specific errors

### Import Errors
**Error**: `ModuleNotFoundError`

**Solution**: Reinstall dependencies:
```bash
pip install -r requirements.txt
```

## Performance Notes

- **Long PDFs**: Processing time increases with document length. Map-reduce summarization helps manage long documents.
- **Embeddings**: First-time embedding generation is slower; subsequent loads use cached FAISS indices.
- **Video Generation**: Can take 10-30 seconds depending on summary length.
- **Audio Generation**: edge-tts is fast; gTTS requires internet connection.

## Development

### Running in Development Mode
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Testing with Swagger UI
1. Navigate to http://localhost:8000/docs
2. Expand endpoints and click "Try it out"
3. Upload a PDF via `/media/upload`
4. Use the returned `doc_id` for other endpoints

## Benchmarking (IEEE graphs)

Generate CSV + PNG graphs (time vs pages, stage breakdown, index size vs pages) for a folder of PDFs:

```bash
pip install -r requirements.txt
python -m eval.benchmark "path/to/pdf_folder" --out "eval/out" --summary
```

Include TTS timing (optional):

```bash
python -m eval.benchmark "path/to/pdf_folder" --out "eval/out" --summary --tts
```

Notes:
- `--summary` includes summarization timing (requires Ollama model). Without it, the benchmark runs extraction/OCR + indexing only.
- Outputs:
  - `eval/out/results.csv`
  - `eval/out/plots/*.png`

## Quality Evaluation (IEEE: answer + summary quality)

This repository includes a minimal **quality evaluation** runner that:
- Executes **No-RAG** vs **RAG** answers for a fixed question set
- Saves outputs to CSV/JSONL for analysis
- Generates **human rating templates** (1–5) for correctness/groundedness/citation relevance and summary quality

1) Create your QA spec JSON (start from `eval/qa_example.json`) and place the referenced PDFs in a folder.

2) Run:

```bash
python -m eval.quality_eval --qa eval/qa_example.json --pdf-dir pdfs/public --out eval/out_quality --k 5 --summary
```

Outputs:
- `eval/out_quality/qa_results.csv`
- `eval/out_quality/qa_results.jsonl`
- `eval/out_quality/llmjudge_scores.csv` (if you run LLM-judge below)
- `eval/out_quality/qa_ratings_template.csv`
- `eval/out_quality/summary_ratings_template.csv`
- `eval/out_quality/latency_summary.json`
- `eval/out_quality/HUMAN_RUBRIC.md`

### Single-command paper run (recommended)

Run everything (QA outputs + LLM-judge + optional benchmark) with one command:

```bash
python -m eval.run_all --qa eval/qa_example.json --pdf-dir pdfs/public --out eval/out_paper --k 5 --seed 1337 --summary --benchmark
```

It produces:
- `eval/out_paper/run_meta.json` (run_id, timestamp, models, chunking settings)
- `eval/out_paper/qa_results.csv`
- `eval/out_paper/llmjudge_scores.csv`
- `eval/out_paper/qa_ratings_template.csv`
- `eval/out_paper/summary_ratings_template.csv`
- `eval/out_paper/latency_summary.json`
- `eval/out_paper/benchmark/results.csv` + `eval/out_paper/benchmark/plots/*.png`

If you want benchmark plots to include TTS time, add `--benchmark-tts`.

```bash
python -m eval.run_all --qa eval/qa_example.json --pdf-dir pdfs/public --out eval/out_paper --k 5 --seed 1337 --summary --benchmark --benchmark-tts
```

### Ablation sweep (chunking + top-k)

Run a parameter sweep over `CHUNK_SIZE`, `CHUNK_OVERLAP`, and `TOP_K` and write a single CSV for paper tables/plots:

```bash
python -m eval.ablation --qa eval/qa_example.json --pdf-dir pdfs/public --out eval/out_ablation
```

Outputs:
- `eval/out_ablation/ablation_results.csv`
- `eval/out_ablation/cs*_co*_k*/` (per-run folders with `run_meta.json`, `qa_results.csv`, `llmjudge_scores.csv`, etc.)

### Citation format

RAG citations are returned in the format `p{page}:c{chunk}` (example: `p3:c2|p3:c5|p7:c1`).
They are derived from the **top-k retrieved chunks actually provided to the LLM**, de-duplicated in retrieval order.

## License

MIT (or as per repository license)

## Notes

- For long PDFs, summarization uses chunked map-reduce to avoid context overflow.
- `/media/chat` answers are grounded in retrieved evidence with citations like `p{page}:c{chunk}`.
- All processing is done offline except for gTTS fallback (requires internet).
- Vector embeddings use Ollama's embedding model (same as chat model by default).
