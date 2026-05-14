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
from storybook_pipeline import (
    StorybookPipelineError,
    create_storybook_html,
    create_storybook_pdf,
    estimate_claude_cost,
    generate_images_for_pages,
    generate_story_pages,
)

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


class StorybookRequest(BaseModel):
    transcript: str
    story_index: int = 0
    claude_model: str | None = None
    max_pages: int = 1
    draft_only: bool = False


class StorybookManualPageRequest(BaseModel):
    title: str
    story_text: str
    image_description: str
    page_number: int = 1

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


@app.post("/storybook-from-text")
async def storybook_from_text(request: StorybookRequest):
    """
    Build a full storybook from transcript text:
    1) Claude page/story planning
    2) fal.ai image generation
    3) PDF assembly
    """
    transcript = (request.transcript or "").strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="Transcript is required")
    if request.story_index < 0 or request.story_index > 4:
        raise HTTPException(status_code=400, detail="story_index must be between 0 and 4")
    if request.max_pages < 1 or request.max_pages > 10:
        raise HTTPException(status_code=400, detail="max_pages must be between 1 and 10")

    try:
        if request.claude_model:
            story = await asyncio.to_thread(
                generate_story_pages, transcript, request.claude_model
            )
        else:
            story = await asyncio.to_thread(generate_story_pages, transcript)
        stories = story.get("stories", [])
        if stories:
            selected_story = stories[request.story_index]
        else:
            selected_story = {"title": story["title"], "pages": story["pages"], "story_number": 1}

        selected_story = {
            **selected_story,
            "pages": selected_story["pages"][: request.max_pages],
        }

        if request.draft_only:
            cost = estimate_claude_cost(story.get("token_counts", []), pages=len(selected_story["pages"]))
            return JSONResponse(
                {
                    "title": selected_story["title"],
                    "story_number": selected_story.get("story_number", request.story_index + 1),
                    "story_index": request.story_index,
                    "model_used": story.get("model_used"),
                    "models_tried": story.get("models_tried", []),
                    "token_counts": story.get("token_counts", []),
                    "cost": cost,
                    "stories": stories if stories else [selected_story],
                    "pages": selected_story["pages"],
                    "draft_only": True,
                }
            )

        pages_with_images = await asyncio.to_thread(
            generate_images_for_pages, selected_story["pages"]
        )
        pdf_path = None
        pdf_error = None
        try:
            pdf_path = await asyncio.to_thread(
                create_storybook_pdf, selected_story["title"], pages_with_images
            )
        except StorybookPipelineError as exc:
            pdf_error = str(exc)
        html_path = await asyncio.to_thread(
            create_storybook_html, selected_story["title"], pages_with_images
        )
    except StorybookPipelineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Storybook pipeline failed: {exc}"
        ) from exc

    filename = Path(pdf_path).name if pdf_path else None
    cost = estimate_claude_cost(story.get("token_counts", []), pages=len(selected_story["pages"]))
    return JSONResponse(
        {
            "title": selected_story["title"],
            "story_number": selected_story.get("story_number", request.story_index + 1),
            "story_index": request.story_index,
            "model_used": story.get("model_used"),
            "models_tried": story.get("models_tried", []),
            "token_counts": story.get("token_counts", []),
            "cost": cost,
            "stories": stories if stories else [selected_story],
            "pages": pages_with_images,
            "draft_only": False,
            "pdf_path": pdf_path,
            "html_path": html_path,
            "pdf_download_url": f"/storybook-pdf/{filename}" if pdf_path else None,
            "pdf_error": pdf_error,
        }
    )


@app.post("/storybook-from-manual-page")
async def storybook_from_manual_page(request: StorybookManualPageRequest):
    """Generate exactly one page from provided story_text + image_description (no Claude call)."""
    title = (request.title or "").strip()
    story_text = (request.story_text or "").strip()
    image_description = (request.image_description or "").strip()
    page_number = int(request.page_number or 1)

    if not title:
        raise HTTPException(status_code=400, detail="title is required")
    if not story_text:
        raise HTTPException(status_code=400, detail="story_text is required")
    if not image_description:
        raise HTTPException(status_code=400, detail="image_description is required")
    if page_number < 1:
        raise HTTPException(status_code=400, detail="page_number must be >= 1")

    page = {
        "page_number": page_number,
        "story_text": story_text,
        "image_description": image_description,
    }

    try:
        pages_with_images = await asyncio.to_thread(generate_images_for_pages, [page])
        pdf_path = None
        pdf_error = None
        try:
            pdf_path = await asyncio.to_thread(create_storybook_pdf, title, pages_with_images)
        except StorybookPipelineError as exc:
            pdf_error = str(exc)
        html_path = await asyncio.to_thread(create_storybook_html, title, pages_with_images)
    except StorybookPipelineError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Manual page pipeline failed: {exc}") from exc

    filename = Path(pdf_path).name if pdf_path else None
    return JSONResponse(
        {
            "title": title,
            "pages": pages_with_images,
            "draft_only": False,
            "claude_used": False,
            "cost": {
                "input_tokens": 0,
                "output_tokens": 0,
                "session_cost_usd": 0,
                "per_page_cost_usd": 0,
            },
            "pdf_path": pdf_path,
            "html_path": html_path,
            "pdf_download_url": f"/storybook-pdf/{filename}" if pdf_path else None,
            "pdf_error": pdf_error,
        }
    )


@app.get("/storybook-pdf/{filename}")
async def get_storybook_pdf(filename: str):
    """Download a generated storybook PDF by filename from STORYBOOK_OUTPUT_DIR."""
    output_dir = Path(os.getenv("STORYBOOK_OUTPUT_DIR", os.getcwd()))
    candidate = (output_dir / filename).resolve()

    if candidate.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")

    # Prevent path escape outside configured output dir.
    if output_dir.resolve() not in candidate.parents and candidate != output_dir.resolve():
        raise HTTPException(status_code=400, detail="Invalid file path")

    return FileResponse(str(candidate), media_type="application/pdf", filename=filename)


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
