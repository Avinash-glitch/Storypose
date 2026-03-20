import os
import io
import asyncio
import base64
import tempfile
import json
import random
import requests
from pathlib import Path
from dotenv import load_dotenv

import httpx
import anthropic
import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FRONTEND_DIR   = Path(__file__).parent / "frontend"
OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.2")
FAL_KEY        = os.getenv("FAL_KEY")
ANTHROPIC_KEY  = os.getenv("ANTHROPIC_API_KEY")

# ---------------------------------------------------------------------------
# Model loading (done once at startup)
# ---------------------------------------------------------------------------

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper ready.")

if not FAL_KEY:
    print("WARNING: FAL_KEY not set in .env — image generation will fail.")
else:
    print("fal.ai ready.")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="StoryPose API")

# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class GenerateImageRequest(BaseModel):
    prompt: str

class DetectGapsRequest(BaseModel):
    text: str
    previous_pages: list = []

# ---------------------------------------------------------------------------
# Gap detection — Ollama (primary) with heuristic fallback
# ---------------------------------------------------------------------------

_FALLBACK_QUESTIONS = {
    "character": [
        "Ooh, who's in your story? Tell me about the main character!",
        "Who is the hero of this adventure?",
    ],
    "setting": [
        "Where does this happen? Is it in a magical forest, a castle, or somewhere else?",
        "Tell me more — where are they right now?",
    ],
    "action": [
        "What happens next? What are they doing?",
        "What exciting thing is happening in the story?",
    ],
    "emotion": [
        "How does the character feel about all of this?",
        "Is the character happy, scared, or excited?",
    ],
}

_SYSTEM_PROMPT = """You are a friendly, encouraging children's storytelling assistant helping a child (ages 4-10) create an illustrated storybook.

Analyze the child's narrated story segment and decide if it is rich enough to generate a vivid storybook illustration.

A good segment should have:
- A CHARACTER (who is in the scene?)
- A SETTING (where are they?)
- AN ACTION (what are they doing?)
- EMOTION or MOOD (optional but great)

If any critical elements are missing, generate ONE short, excited, child-friendly follow-up question.
Keep questions very simple — max 12 words. Always start with an exclamation like "Ooh!", "Wow!", "Amazing!".
Never say "I didn't understand". Instead say things like "Tell me more!"

Respond ONLY with valid JSON, no extra text:
{
  "isComplete": true or false,
  "missingElement": "character" or "setting" or "action" or "emotion" or "none",
  "followUpQuestion": "the question to ask, or null if complete",
  "enrichedContext": "a 1-sentence description of what the scene contains"
}"""


def _heuristic_fallback(text: str) -> dict:
    """Simple keyword-based fallback when Ollama is unavailable."""
    words = set(text.lower().split())
    char_words = {"i","me","my","he","she","they","boy","girl","dragon","princess",
                  "knight","wizard","cat","dog","hero","monster","robot","fairy"}
    setting_words = {"forest","castle","house","garden","ocean","sky","mountain",
                     "school","park","jungle","cave","space","island","beach","town"}
    action_words = {"run","jump","fly","swim","fight","dance","find","save","escape",
                    "went","ran","flew","climbed","discovered","explored","played"}

    if len(text.split()) < 6:
        return {
            "isComplete": False,
            "missingElement": "action",
            "followUpQuestion": "Ooh, tell me more! What's happening in your story?",
            "enrichedContext": text,
        }
    if not (words & char_words):
        return {"isComplete": False, "missingElement": "character",
                "followUpQuestion": random.choice(_FALLBACK_QUESTIONS["character"]),
                "enrichedContext": text}
    if not any(w in text.lower() for w in setting_words):
        return {"isComplete": False, "missingElement": "setting",
                "followUpQuestion": random.choice(_FALLBACK_QUESTIONS["setting"]),
                "enrichedContext": text}
    if not (words & action_words):
        return {"isComplete": False, "missingElement": "action",
                "followUpQuestion": random.choice(_FALLBACK_QUESTIONS["action"]),
                "enrichedContext": text}
    return {"isComplete": True, "missingElement": "none",
            "followUpQuestion": None, "enrichedContext": text}


async def detect_story_gaps(text: str, previous_pages: list) -> dict:
    """Call Claude Haiku for story gap analysis; fall back to heuristic on any error."""
    context_snippet = ""
    if previous_pages:
        last = previous_pages[-1] if isinstance(previous_pages[-1], str) else str(previous_pages[-1])
        context_snippet = f"\n\nPrevious story so far: {last[:300]}"

    user_msg = f"Child's narration: \"{text}\"{context_snippet}"

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        message = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        content = message.content[0].text.strip()
        print(f"[detect_story_gaps] Claude raw response: {content[:200]}")
        result = json.loads(content)

        result.setdefault("missingElement", "none")
        result.setdefault("followUpQuestion", None)
        result.setdefault("enrichedContext", text)
        return result

    except Exception as exc:
        print(f"[detect_story_gaps] Claude unavailable ({exc}), using heuristic fallback.")
        return _heuristic_fallback(text)

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Receive audio blob from browser, transcribe with local Whisper."""
    audio_bytes = await audio.read()
    print(f"[transcribe] Received {len(audio_bytes)} bytes, content_type={audio.content_type}")

    suffix = ".webm" if audio.content_type and "webm" in audio.content_type else ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        result = whisper_model.transcribe(tmp_path)
        text = result["text"].strip()
        print(f"[transcribe] Result: {repr(text)}")
    except Exception as exc:
        print(f"[transcribe] ERROR: {exc}")
        # If webm decode failed (no ffmpeg), tell the browser clearly
        raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return JSONResponse({"text": text})


@app.post("/generate-image")
async def generate_image(request: GenerateImageRequest):
    """Generate a storybook illustration using fal.ai."""
    if not request.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    if not FAL_KEY:
        raise HTTPException(status_code=500, detail="FAL_KEY not configured")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://queue.fal.run/fal-ai/fast-sdxl",
            headers={"Authorization": f"Key {FAL_KEY}", "Content-Type": "application/json"},
            json={"prompt": request.prompt, "image_size": "square", "num_inference_steps": 25},
        )
        resp.raise_for_status()
        request_id = resp.json()["request_id"]

        # Poll for result
        while True:
            poll = await client.get(
                f"https://queue.fal.run/fal-ai/fast-sdxl/requests/{request_id}/status",
                headers={"Authorization": f"Key {FAL_KEY}"},
            )
            status = poll.json().get("status")
            if status == "COMPLETED":
                break
            elif status == "FAILED":
                raise HTTPException(status_code=500, detail="fal.ai generation failed")
            await asyncio.sleep(1)

        result = await client.get(
            f"https://queue.fal.run/fal-ai/fast-sdxl/requests/{request_id}",
            headers={"Authorization": f"Key {FAL_KEY}"},
        )
        image_url = result.json()["images"][0]["url"]

    # Download image and return as base64
    img_resp = requests.get(image_url)
    b64 = base64.b64encode(img_resp.content).decode("utf-8")
    return JSONResponse({"image": b64})


@app.post("/detect-gaps")
async def detect_gaps_endpoint(request: DetectGapsRequest):
    """Analyse story text and return gap-filling follow-up question."""
    result = await detect_story_gaps(request.text, request.previous_pages)
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Static frontend serving
# ---------------------------------------------------------------------------

@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")

# Mount after the explicit routes so they take priority
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
