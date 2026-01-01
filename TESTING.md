# Testing Guide for synergiq-llmoff

This document provides testing instructions and validation steps for the synergiq-llmoff API.

## Prerequisites for Full Testing

1. **Ollama Service Running**
   ```bash
   # Check if Ollama is running
   curl http://localhost:11434/api/tags
   
   # If not running, start it
   ollama serve
   
   # Pull the model (if not already pulled)
   ollama pull qwen2.5:7b
   ```

2. **Network Access** (for audio generation)
   - edge-tts requires internet connection to Microsoft TTS service
   - gTTS requires internet connection to Google TTS service
   - Video generation works offline

3. **System Libraries** (for video generation)
   - FFmpeg (for video encoding)
   - OpenCV dependencies (libsm6, libxext6, libxrender-dev on Linux)

## Quick Start Testing

### 1. Start the Server

```bash
# From the repository root
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server should start with output like:
```
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 2. Access Swagger UI

Open your browser to: http://localhost:8000/docs

You should see the interactive API documentation with all endpoints listed.

## Testing Workflow

### Test 1: Health Check

**Using curl:**
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "ok": true,
  "service": "offline-pdf-rag"
}
```

### Test 2: Upload PDF (Full Pipeline)

**Using curl:**
```bash
curl -X POST "http://localhost:8000/media/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

**Using Swagger UI:**
1. Navigate to `POST /media/upload`
2. Click "Try it out"
3. Click "Choose File" and select a PDF
4. Click "Execute"

**Expected Response:**
```json
{
  "doc_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "filename": "your_document.pdf",
  "num_pages": 10,
  "status": "ready"
}
```

**Save the `doc_id`** - you'll need it for subsequent tests!

**Note:** Processing may take 30 seconds to several minutes depending on:
- Document length
- LLM response time
- Audio/video generation time

### Test 3: Check Status

**Using curl:**
```bash
curl "http://localhost:8000/media/status/{doc_id}"
```

**Expected Response (during processing):**
```json
{
  "state": "tts",
  "step": "generating_audio"
}
```

**Expected Response (when complete):**
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

**Expected Response (on error):**
```json
{
  "state": "error",
  "error": "Error message here"
}
```

### Test 4: Get Summary

**Using curl:**
```bash
# Detailed summary (default)
curl "http://localhost:8000/media/summary/{doc_id}"

# Brief summary
curl "http://localhost:8000/media/summary/{doc_id}?mode=brief"
```

**Expected Response:**
```json
{
  "doc_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "mode": "detailed",
  "summary": "This document discusses... [full summary text]",
  "num_pages": 10
}
```

### Test 5: Download Audio

**Using curl:**
```bash
curl "http://localhost:8000/media/audio/{doc_id}" -o summary_audio.mp3
```

**Using Browser:**
Navigate to: `http://localhost:8000/media/audio/{doc_id}`

**Expected:** Download starts for an MP3 file

**Note:** If audio generation failed (no network), you'll get a 404 error:
```json
{
  "detail": "Audio not generated yet. Processing may still be in progress."
}
```

### Test 6: Download Video

**Using curl:**
```bash
curl "http://localhost:8000/media/video/{doc_id}" -o summary_video.mp4
```

**Using Browser:**
Navigate to: `http://localhost:8000/media/video/{doc_id}`

**Expected:** Download starts for an MP4 file

### Test 7: Chat with Document

**Using curl:**
```bash
curl -X POST "http://localhost:8000/media/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "question": "What are the main topics discussed in this document?"
  }'
```

**Using Swagger UI:**
1. Navigate to `POST /media/chat`
2. Click "Try it out"
3. Enter your `doc_id` and question
4. Click "Execute"

**Expected Response:**
```json
{
  "answer": "Based on the document, the main topics are... [detailed answer]",
  "sources": ["p1", "p3", "p5"]
}
```

## Testing Legacy Endpoints

### Upload PDF (In-Memory)

```bash
curl -X POST "http://localhost:8000/upload_pdf" \
  -F "file=@/path/to/your/document.pdf"
```

**Expected Response:**
```json
{
  "doc_id": "abc123def456",
  "pages": 10
}
```

### Chat (In-Memory)

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "doc_id": "abc123def456",
    "question": "What is this document about?"
  }'
```

## Troubleshooting Tests

### Issue: "Unknown doc_id" Error

**Cause:** The doc_id doesn't exist or was entered incorrectly

**Solution:** 
- Verify you're using the correct `doc_id` from the upload response
- Check that the document was successfully uploaded
- Use `GET /media/status/{doc_id}` to verify the document exists

### Issue: "Processing failed" Error

**Cause:** Error during one of the pipeline steps

**Solution:**
1. Check the error message in the response
2. Check the status endpoint for more details
3. Common causes:
   - Ollama not running → Start Ollama
   - Model not available → Pull the model with `ollama pull qwen2.5:7b`
   - PDF parsing error → Try a different PDF
   - OCR error → Ensure Tesseract is installed (if using scanned PDFs)

### Issue: Audio Generation Failed

**Cause:** No network access or TTS services unavailable

**Solution:**
- edge-tts requires internet connection to Microsoft services
- gTTS requires internet connection to Google services
- In offline environments, audio generation may fail (this is expected)
- The rest of the pipeline (summary, video, chat) will still work

### Issue: Video Generation Failed

**Cause:** Missing system dependencies

**Solution:**
```bash
# Linux
sudo apt-get install ffmpeg libsm6 libxext6 libxrender-dev

# macOS
brew install ffmpeg

# Windows
# Download FFmpeg from https://ffmpeg.org/download.html
# Add to PATH
```

### Issue: Ollama Connection Error

**Cause:** Ollama service not running or wrong URL

**Solution:**
```bash
# Check Ollama status
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Verify in .env:
OLLAMA_BASE_URL=http://localhost:11434
```

### Issue: Model Not Found

**Cause:** The specified model hasn't been pulled

**Solution:**
```bash
# Pull the model
ollama pull qwen2.5:7b

# Or update .env to use a different model:
OLLAMA_MODEL=llama3.1
```

## Performance Benchmarks

Typical processing times (on a modern laptop with Ollama running):

- **PDF Upload & Extraction**: 2-5 seconds (for 10-page PDF)
- **Vector Store Building**: 5-15 seconds
- **Summary Generation**: 20-60 seconds (depends on document length and LLM speed)
- **Audio Generation**: 5-15 seconds
- **Video Generation**: 10-30 seconds
- **Chat Query**: 5-15 seconds per question

**Total pipeline time:** 1-3 minutes for a typical 10-page document

## Automated Testing Script

Save this as `test_api.sh`:

```bash
#!/bin/bash

# Configuration
BASE_URL="http://localhost:8000"
PDF_FILE="test_document.pdf"

echo "=== Testing synergiq-llmoff API ==="

# Test 1: Health check
echo -e "\n1. Health Check..."
curl -s "$BASE_URL/health" | jq .

# Test 2: Upload PDF
echo -e "\n2. Uploading PDF..."
RESPONSE=$(curl -s -X POST "$BASE_URL/media/upload" \
  -F "file=@$PDF_FILE")
echo "$RESPONSE" | jq .

DOC_ID=$(echo "$RESPONSE" | jq -r .doc_id)
echo "Doc ID: $DOC_ID"

# Test 3: Check status
echo -e "\n3. Checking status..."
curl -s "$BASE_URL/media/status/$DOC_ID" | jq .

# Wait for processing
echo -e "\nWaiting for processing to complete..."
sleep 30

# Test 4: Get summary
echo -e "\n4. Getting summary..."
curl -s "$BASE_URL/media/summary/$DOC_ID" | jq .

# Test 5: Chat
echo -e "\n5. Chatting with document..."
curl -s -X POST "$BASE_URL/media/chat" \
  -H "Content-Type: application/json" \
  -d "{\"doc_id\": \"$DOC_ID\", \"question\": \"What is this document about?\"}" | jq .

echo -e "\n=== Tests Complete ==="
```

Run with:
```bash
chmod +x test_api.sh
./test_api.sh
```

## Success Criteria

All tests pass if:
- ✅ Health check returns `{"ok": true}`
- ✅ PDF upload returns a valid `doc_id`
- ✅ Status eventually shows `"state": "ready"`
- ✅ Summary returns coherent text
- ✅ Video file is generated and downloadable
- ✅ Chat returns relevant answers with sources
- ✅ Audio file is generated (if network available)

## Notes

- First request to any endpoint may be slower due to model loading
- Long documents take proportionally longer to process
- Audio generation requires network access in current implementation
- All processing is done on the server; client just needs to poll status
- Generated files are stored in `data/docs/{doc_id}/` directory
