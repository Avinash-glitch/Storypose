# -*- coding: utf-8 -*-
"""StoryPose FLUX + Claude one-cell Colab runner.

Paste this whole file into one Colab cell, or upload/run it as a script.

What it does:
1. Installs the runtime packages.
2. Mounts Google Drive.
3. Accepts a Hugging Face token for FLUX.1-dev.
4. Optionally accepts an Anthropic key for Claude story/page planning.
5. Lets you either upload spoken-story audio or paste a transcript.
6. Uses Claude to create a page plan and consistent image prompts.
7. Generates images with local FLUX.1-dev on the Colab GPU.
8. Saves a Lovable-style HTML flipbook and a PDF with internal next/previous links.
"""

import base64
import gc
import html
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Any


def pip_install() -> None:
    packages = [
        "pillow==11.3.0",
        "diffusers==0.35.1",
        "transformers==4.55.4",
        "accelerate>=1.10.0",
        "sentencepiece",
        "protobuf",
        "safetensors",
        "huggingface_hub>=0.26.0",
        "anthropic>=0.50.0",
        "requests>=2.32.0",
        "reportlab>=4.2.0",
        "openai-whisper>=20250625",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", *packages], check=True)


pip_install()

import requests
import torch
from anthropic import Anthropic
from diffusers import FluxPipeline
from google.colab import drive, files
from huggingface_hub import login as hf_login
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.utils import ImageReader


# -------------------------
# Runtime config
# -------------------------
MODEL_ID = os.environ.get("STORYPOSE_FLUX_MODEL", "black-forest-labs/FLUX.1-dev")
CLAUDE_MODEL = os.environ.get("STORYPOSE_CLAUDE_MODEL", "claude-sonnet-4-5-20250929")
WIDTH = int(os.environ.get("STORYPOSE_WIDTH", "1024"))
HEIGHT = int(os.environ.get("STORYPOSE_HEIGHT", "1024"))
STEPS = int(os.environ.get("STORYPOSE_STEPS", "28"))
GUIDANCE = float(os.environ.get("STORYPOSE_GUIDANCE", "3.5"))
SEED = int(os.environ.get("STORYPOSE_SEED", "20260515"))
MAX_PAGES = int(os.environ.get("STORYPOSE_MAX_PAGES", "6"))
TEST_PAGES = int(os.environ.get("STORYPOSE_TEST_PAGES", "1"))
WHISPER_MODEL = os.environ.get("STORYPOSE_WHISPER_MODEL", "medium")
USE_PAGE1_VISUAL_MEMORY = os.environ.get("STORYPOSE_USE_PAGE1_VISUAL_MEMORY", "1").strip() != "0"

DRIVE_ROOT = Path("/content/drive/MyDrive/storypose-colab")
OUTPUT_ROOT = DRIVE_ROOT / "outputs"
IMAGE_ROOT = OUTPUT_ROOT / "images"
HTML_ROOT = OUTPUT_ROOT / "html"
PDF_ROOT = OUTPUT_ROOT / "pdf"


SYSTEM_PROMPT = """
You are StoryPose: a professional children's picture-book author and visual art director.
Return only valid JSON. Do not wrap JSON in markdown.
Create a cohesive storybook from the child's spoken story.
You must preserve the child's core idea while making it readable, gentle, and age-appropriate.
For visual consistency, create a reusable character_bible and style_bible.
Every page image_description must repeat the exact visual traits needed for recurring characters.
Every page image_description must primarily describe what is happening in the scene: action, setting,
character placement, emotion, props, weather/light, and story stakes.
Avoid static character portraits or character lineups unless the page story explicitly requires them.
Images must contain no written words, captions, speech bubbles, logos, or watermarks.
""".strip()


def user_prompt(transcript: str, max_pages: int) -> str:
    return f"""
Child transcript:
{transcript}

Create a picture-book plan.

Requirements:
- Break into 3-{max_pages} pages unless the story is very short.
- For testing, the caller may use only the first page, but still make the full plan coherent.
- Each story_text must be under 100 words.
- Each image_description must be detailed enough for FLUX.1-dev.
- Each image_description must show a clear story moment, not just characters posing.
- Start each image_description with the main action happening in that page.
- Include: who is doing what, where they are, what has changed, emotional mood, key objects, and camera/composition.
- Prefer dynamic verbs like flying, catching, hiding, searching, building, lifting, chasing, rescuing, whispering, discovering.
- Avoid generic prompts like "the characters standing together" unless the page is actually about standing together.
- Use the page design tone of a premium printed storybook: warm, soft, whimsical, emotionally clear.
- The image style should be consistent across pages.
- Do not include text inside images.

Return JSON exactly:
{{
  "title": "string",
  "subtitle": "string",
  "character_bible": "string",
  "style_bible": "string",
  "pages": [
    {{
      "page_number": 1,
      "story_text": "string",
      "image_description": "string"
    }}
  ]
}}
""".strip()


@dataclass
class TokenCost:
    input_tokens: int | None
    output_tokens: int | None
    estimated_input_cost_usd: float | None
    estimated_output_cost_usd: float | None
    estimated_total_cost_usd: float | None


MODEL_PRICES_PER_MTOK = {
    # Update these if Anthropic changes pricing.
    "claude-sonnet-4-5-20250929": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
}


def ensure_dirs() -> None:
    for folder in [OUTPUT_ROOT, IMAGE_ROOT, HTML_ROOT, PDF_ROOT]:
        folder.mkdir(parents=True, exist_ok=True)


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug[:70] or "storybook"


def require_tokens() -> None:
    hf_token = os.environ.get("HF_TOKEN") or getpass("Paste your Hugging Face token for FLUX.1-dev (hf_...): ").strip()
    os.environ["HF_TOKEN"] = hf_token
    hf_login(token=hf_token)

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or getpass(
        "Paste your Anthropic key for Claude (press Enter to skip Claude test): "
    ).strip()
    if anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_key


def count_claude_tokens(model: str, system: str, prompt: str) -> int | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    response = requests.post(
        "https://api.anthropic.com/v1/messages/count_tokens",
        headers={
            "content-type": "application/json",
            "anthropic-version": "2023-06-01",
            "x-api-key": api_key,
        },
        json={
            "model": model,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    response.raise_for_status()
    return int(response.json()["input_tokens"])


def estimate_cost(model: str, input_tokens: int | None, output_tokens: int | None) -> TokenCost:
    prices = MODEL_PRICES_PER_MTOK.get(model)
    if not prices or input_tokens is None or output_tokens is None:
        return TokenCost(input_tokens, output_tokens, None, None, None)
    input_cost = input_tokens / 1_000_000 * prices["input"]
    output_cost = output_tokens / 1_000_000 * prices["output"]
    return TokenCost(input_tokens, output_tokens, input_cost, output_cost, input_cost + output_cost)


def test_claude_model(model: str = CLAUDE_MODEL) -> str:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "Claude skipped: ANTHROPIC_API_KEY is not set."
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    result = client.messages.create(
        model=model,
        max_tokens=20,
        temperature=0,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
    )
    return result.content[0].text.strip()


def create_visual_memory_from_image(image_path: str, story: dict[str, Any], model: str = CLAUDE_MODEL) -> str:
    """Create text-only visual memory from page 1. No image vectors are stored."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return ""

    raw = Path(image_path).read_bytes()
    image_b64 = base64.b64encode(raw).decode("ascii")
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = f"""
You are creating a reusable visual continuity memory for a children's storybook.
Look at this generated page 1 image and describe only stable visual details that should stay consistent.

Story title: {story.get("title", "")}
Existing character bible:
{story.get("character_bible", "")}

Return compact plain text, no markdown. Include:
- recurring character appearance
- outfit colors
- face/hair/body traits
- illustration style
- color palette
- any stable recurring objects/settings

Do not describe page-specific action unless it affects continuity.
Keep under 70 words. Plain comma-separated phrases are best.
""".strip()
    message = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return compact_prompt_text(
        "".join(block.text for block in message.content if getattr(block, "type", None) == "text"),
        max_words=80,
    )


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            text = match.group(1)
    return json.loads(text)


def generate_story_pages_with_claude(transcript: str, max_pages: int = MAX_PAGES, model: str = CLAUDE_MODEL) -> tuple[dict[str, Any], TokenCost]:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY is missing, so Claude cannot create pages.")

    prompt = user_prompt(transcript, max_pages=max_pages)
    input_tokens = count_claude_tokens(model, SYSTEM_PROMPT, prompt)
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    last_error: Exception | None = None
    for attempt in range(2):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=5000,
                temperature=0.55,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
            data = extract_json(text)
            if not data.get("title") or not isinstance(data.get("pages"), list):
                raise ValueError("Claude JSON missing title/pages.")
            output_tokens = getattr(message.usage, "output_tokens", None)
            return data, estimate_cost(model, input_tokens, output_tokens)
        except Exception as exc:
            last_error = exc
            prompt += "\n\nRetry: return only syntactically valid JSON matching the schema exactly."

    raise RuntimeError(f"Claude failed to produce valid story JSON: {last_error}")


def upload_and_transcribe_story() -> str:
    print("Upload an audio file of the child speaking the story.")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No audio file uploaded.")
    audio_path = Path("/content") / next(iter(uploaded.keys()))
    prepared_audio_path = prepare_audio_for_whisper(audio_path)

    import whisper

    print(f"Loading Whisper model: {WHISPER_MODEL}")
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(
        str(prepared_audio_path),
        language="en",
        task="transcribe",
        fp16=torch.cuda.is_available(),
        temperature=0,
        beam_size=5,
        best_of=5,
        condition_on_previous_text=True,
        initial_prompt=(
            "This is a children's fantasy story. Preserve the opening sentences, character names, "
            "storybook language, and unusual proper nouns as accurately as possible."
        ),
    )
    transcript = result["text"].strip()
    print("\nTranscript:\n", transcript)
    return transcript


def prepare_audio_for_whisper(audio_path: Path) -> Path:
    """Convert uploaded audio to 16 kHz mono WAV with leading silence.

    The short silence helps Whisper avoid clipping/missing speech that starts
    immediately at timestamp zero.
    """
    output_path = Path("/content") / f"storypose_prepared_{uuid.uuid4().hex[:8]}.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-t",
        "1.5",
        "-i",
        "anullsrc=channel_layout=mono:sample_rate=16000",
        "-i",
        str(audio_path),
        "-filter_complex",
        "[1:a]aresample=16000,aformat=channel_layouts=mono,volume=1.4[a1];[0:a][a1]concat=n=2:v=0:a=1[out]",
        "-map",
        "[out]",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def load_flux_pipeline() -> FluxPipeline:
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU available. In Colab use Runtime > Change runtime type > GPU.")

    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    pipe = FluxPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype, token=os.environ["HF_TOKEN"])
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    return pipe


def compact_prompt_text(text: str, max_words: int = 140) -> str:
    text = re.sub(r"[*_#`>-]+", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]).rsplit(",", 1)[0].strip()


def enrich_image_prompt(page: dict[str, Any], story: dict[str, Any], visual_memory: str = "") -> str:
    """Build a short FLUX prompt.

    FLUX uses CLIP and T5 text encoders. CLIP is very short and T5 is capped,
    so put the current scene first and keep continuity text compact.
    """
    scene = compact_prompt_text(page["image_description"], max_words=70)
    character_lock = compact_prompt_text(story.get("character_bible", ""), max_words=55)
    style_lock = compact_prompt_text(story.get("style_bible", ""), max_words=35)
    memory_lock = compact_prompt_text(visual_memory, max_words=45)

    parts = [
        f"Page {page['page_number']} scene: {scene}",
        f"Character continuity: {character_lock}" if character_lock else "",
        f"Approved page 1 visual memory: {memory_lock}" if memory_lock else "",
        f"Style: {style_lock}" if style_lock else "",
        (
            "Show the story action clearly, not a portrait. Premium children's storybook illustration, "
            "soft watercolor and gouache texture, warm emotional lighting, rounded friendly shapes, "
            "consistent character design, no words, no letters, no captions, no watermark."
        ),
    ]
    return compact_prompt_text(" ".join(part for part in parts if part), max_words=210)


def build_flux_prompts(page: dict[str, Any], story: dict[str, Any], visual_memory: str = "") -> tuple[str, str]:
    """Return separate prompts for FLUX CLIP and T5 encoders.

    `prompt` feeds CLIP, which is limited to ~77 tokens. `prompt_2` feeds T5,
    which can carry the fuller scene/continuity instruction.
    """
    scene = compact_prompt_text(page["image_description"], max_words=32)
    memory_lock = compact_prompt_text(visual_memory, max_words=18)
    style = "soft watercolor storybook, dreamy pastels, warm magical light, no text"
    clip_parts = [
        scene,
        memory_lock,
        style,
    ]
    clip_prompt = compact_prompt_text(" ".join(part for part in clip_parts if part), max_words=58)
    t5_prompt = enrich_image_prompt(page, story, visual_memory=visual_memory)
    return clip_prompt, t5_prompt


def generate_images(story: dict[str, Any], pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pipe = load_flux_pipeline()
    slug = safe_slug(story["title"])
    complete_pages: list[dict[str, Any]] = []
    visual_memory = ""

    for page in pages:
        page_number = int(page["page_number"])
        clip_prompt, t5_prompt = build_flux_prompts(page, story, visual_memory=visual_memory)
        generator = torch.Generator("cuda").manual_seed(SEED + page_number)
        print(f"Generating page {page_number}: {page['story_text'][:70]}...")
        image = pipe(
            prompt=clip_prompt,
            prompt_2=t5_prompt,
            width=WIDTH,
            height=HEIGHT,
            num_inference_steps=STEPS,
            guidance_scale=GUIDANCE,
            generator=generator,
            max_sequence_length=512,
        ).images[0]

        image_path = IMAGE_ROOT / f"{slug}-page-{page_number:02d}-{uuid.uuid4().hex[:8]}.png"
        image.save(image_path)

        complete_pages.append(
            {
                **page,
                "image_path": str(image_path),
                "clip_prompt": clip_prompt,
                "t5_prompt": t5_prompt,
                "final_prompt": t5_prompt,
                "visual_memory_used": visual_memory,
            }
        )

        if page_number == 1 and USE_PAGE1_VISUAL_MEMORY and len(pages) > 1:
            print("Creating text-only visual memory from page 1 for later pages...")
            try:
                visual_memory = create_visual_memory_from_image(str(image_path), story)
                (OUTPUT_ROOT / f"{slug}-visual-memory.txt").write_text(visual_memory, encoding="utf-8")
                print("Visual memory:", visual_memory[:500])
            except Exception as exc:
                print(f"Visual memory skipped: {type(exc).__name__}: {exc}")
                visual_memory = ""

        gc.collect()
        torch.cuda.empty_cache()

    return complete_pages


def image_data_uri(path: str) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def split_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n+|(?<=[.!?])\s+(?=[A-Z\"“])", text) if part.strip()]
    return parts or [text]


def save_flipbook_html(title: str, pages: list[dict[str, Any]], subtitle: str = "") -> str:
    slug = safe_slug(title)
    path = HTML_ROOT / f"{slug}-{uuid.uuid4().hex[:8]}.html"
    page_blocks = []

    for idx, page in enumerate(pages):
        page_id = f"page-{idx + 1}"
        next_id = f"page-{idx + 2}" if idx + 1 < len(pages) else "page-1"
        prev_id = f"page-{idx}" if idx > 0 else f"page-{len(pages)}"
        paragraphs = split_paragraphs(str(page["story_text"]))
        first = paragraphs[0]
        rest = paragraphs[1:]
        drop = html.escape(first[0])
        first_tail = html.escape(first[1:])

        page_blocks.append(
            f"""
            <article class="spread" id="{page_id}">
              <header class="top-row">
                <div class="brand"><span class="book-icon">□</span> STORYPOSE · READ ALOUD</div>
                <div class="counter">{idx + 1:02d} / {len(pages):02d}</div>
              </header>
              <div class="book-wrap">
                <div class="dot pink"></div>
                <div class="dot aqua"></div>
                <div class="dot sun"></div>
                <div class="book-card">
                  <div class="image-panel">
                    <img src="{image_data_uri(page["image_path"])}" alt="Story illustration page {page["page_number"]}">
                    <div class="chapter-pill">Chapter One · {html.escape(title[:28])}</div>
                  </div>
                  <section class="text-panel">
                    <div>
                      <p class="eyebrow">A Storybook Classic</p>
                      <h1>{html.escape(title)}</h1>
                      {f'<p class="subtitle">{html.escape(subtitle)}</p>' if subtitle else ''}
                      <div class="story-copy">
                        <p class="drop"><span>{drop}</span>{first_tail}</p>
                        {''.join(f'<p>{html.escape(p)}</p>' for p in rest)}
                      </div>
                    </div>
                    <footer>
                      <span>Page {page["page_number"]} of {len(pages)}</span>
                      <nav>
                        <a class="ghost-button" href="#{prev_id}">Back</a>
                        <a class="turn-button" href="#{next_id}">Turn the page</a>
                      </nav>
                    </footer>
                  </section>
                </div>
              </div>
              <div class="pager-dots">
                {''.join(f'<a href="#page-{i + 1}" class="pager-dot {("active" if i == idx else "")}" aria-label="Page {i + 1}"></a>' for i in range(len(pages)))}
              </div>
            </article>
            """
        )

    css = """
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,700;9..144,900&family=Inter:wght@400;600;700;800;900&display=swap');
    :root {
      --ink: #3b1711;
      --muted: #756c64;
      --sun: #ffd94d;
      --accent: #ce7a55;
      --paper: #fffdf6;
      --aqua: #56c7c8;
      --pink: #f06e93;
      --soft: rgba(65, 36, 22, .14);
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 10% 14%, rgba(255, 222, 116, .62), transparent 32%),
        radial-gradient(circle at 90% 84%, rgba(87, 211, 214, .45), transparent 30%),
        linear-gradient(135deg, #ffe28f 0%, #ffd0c6 47%, #c9f4ef 100%);
    }
    main {
      min-height: 100vh;
      width: 100%;
      display: block;
      padding: 48px 32px;
    }
    .spread {
      display: none;
      position: relative;
      width: min(1150px, 100%);
      margin: 0 auto;
    }
    .spread:target { display: block; }
    .spread:first-child { display: block; }
    main:has(.spread:target) .spread:first-child { display: none; }
    .top-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin: 0 6px 28px;
      color: var(--ink);
      font-size: 12px;
      font-weight: 900;
      letter-spacing: .36em;
    }
    .brand { text-transform: uppercase; }
    .book-icon { display: inline-block; margin-right: 8px; color: var(--accent); }
    .counter { color: #4f4038; letter-spacing: .18em; }
    .book-wrap { position: relative; }
    .book-card {
      display: grid;
      grid-template-columns: 1.05fr 1fr;
      min-height: 640px;
      background: var(--paper);
      border: 1px solid rgba(85, 55, 35, .12);
      border-radius: 28px;
      overflow: hidden;
      box-shadow: 0 34px 90px rgba(83, 46, 24, .14);
    }
    .image-panel {
      position: relative;
      min-height: 640px;
      overflow: hidden;
      background: #f7ead3;
    }
    .image-panel img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .chapter-pill {
      position: absolute;
      left: 24px;
      top: 22px;
      background: var(--sun);
      color: #2f2116;
      border-radius: 999px;
      padding: 8px 18px;
      font-size: 14px;
      font-weight: 800;
    }
    .text-panel {
      position: relative;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 36px;
      min-height: 640px;
      padding: 56px 56px 42px;
    }
    .eyebrow {
      margin: 0 0 20px;
      color: var(--accent);
      font-size: 11px;
      font-weight: 900;
      letter-spacing: .32em;
      text-transform: uppercase;
    }
    h1 {
      margin: 0 0 18px;
      font-family: Fraunces, Georgia, Cambria, "Times New Roman", serif;
      font-size: clamp(34px, 4.1vw, 56px);
      line-height: 1.06;
      font-weight: 900;
      letter-spacing: 0;
    }
    .subtitle {
      margin: -6px 0 28px;
      color: var(--muted);
      font-size: 18px;
      font-style: italic;
    }
    .story-copy {
      max-width: 540px;
      color: rgba(59, 23, 17, .92);
      font-size: 18px;
      line-height: 1.8;
    }
    .story-copy p { margin: 0 0 22px; }
    .story-copy .drop span {
      float: left;
      padding: 8px 9px 0 0;
      color: var(--ink);
      font-family: Fraunces, Georgia, Cambria, "Times New Roman", serif;
      font-size: 66px;
      line-height: .76;
      font-weight: 900;
    }
    footer {
      border-top: 1px solid var(--soft);
      padding-top: 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      color: #59666b;
      font-size: 14px;
    }
    nav { display: flex; gap: 12px; flex-wrap: wrap; justify-content: flex-end; }
    .turn-button, .ghost-button {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-height: 44px;
      padding: 0 24px;
      border-radius: 999px;
      text-decoration: none;
      font-weight: 900;
      transition: transform .15s ease;
    }
    .turn-button { background: var(--sun); color: #2f2116; }
    .ghost-button { background: #f3f0e8; color: #5c5146; }
    .turn-button:hover, .ghost-button:hover { transform: translateY(-1px) scale(1.02); }
    .dot {
      position: absolute;
      width: 34px;
      height: 34px;
      border-radius: 999px;
      z-index: 3;
    }
    .dot.aqua { left: -14px; top: 96px; background: var(--aqua); }
    .dot.pink { right: 28px; top: -16px; background: var(--pink); }
    .dot.sun { left: 33%; bottom: -14px; width: 24px; height: 24px; background: var(--sun); }
    .pager-dots {
      margin-top: 24px;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
    }
    .pager-dot {
      display: block;
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: rgba(59, 23, 17, .2);
      transition: width .18s ease, background .18s ease;
    }
    .pager-dot.active { width: 32px; background: var(--accent); }
    @media (max-width: 820px) {
      main { padding: 22px; }
      .book-card { grid-template-columns: 1fr; min-height: auto; }
      .image-panel { min-height: 380px; }
      .text-panel { padding: 36px 28px 30px; }
      h1 { font-size: 40px; }
      footer { align-items: flex-start; flex-direction: column; }
      nav { justify-content: flex-start; }
    }
    """

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} - StoryPose</title>
  <style>{css}</style>
</head>
<body>
  <main>
    {''.join(page_blocks)}
  </main>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")
    return str(path)


def draw_button(c: canvas.Canvas, x: float, y: float, w: float, h: float, label: str, dest: str) -> None:
    c.setFillColor(colors.HexColor("#ffd94d"))
    c.roundRect(x, y, w, h, h / 2, fill=1, stroke=0)
    c.setFillColor(colors.HexColor("#2f2116"))
    c.setFont("Helvetica-Bold", 10)
    c.drawCentredString(x + w / 2, y + h / 2 - 3, label)
    c.linkRect("", dest, (x, y, x + w, y + h), relative=0, thickness=0)


def save_linked_pdf(title: str, pages: list[dict[str, Any]], subtitle: str = "") -> str:
    slug = safe_slug(title)
    path = PDF_ROOT / f"{slug}-{uuid.uuid4().hex[:8]}.pdf"
    c = canvas.Canvas(str(path), pagesize=landscape(letter))
    page_w, page_h = landscape(letter)

    text_style = ParagraphStyle(
        name="StoryText",
        fontName="Helvetica",
        fontSize=15,
        leading=24,
        textColor=colors.HexColor("#253238"),
    )

    for idx, page in enumerate(pages):
        bookmark = f"page_{idx + 1}"
        c.bookmarkPage(bookmark)

        c.setFillColor(colors.HexColor("#ffe9b0"))
        c.rect(0, 0, page_w, page_h, fill=1, stroke=0)
        margin = 0.52 * inch
        panel_w = page_w - 2 * margin
        panel_h = page_h - 2 * margin
        c.setFillColor(colors.HexColor("#fffdf8"))
        c.roundRect(margin, margin, panel_w, panel_h, 20, fill=1, stroke=0)

        left_w = panel_w * 0.52
        right_w = panel_w - left_w
        left_x = margin
        right_x = margin + left_w
        panel_y = margin

        img = ImageReader(page["image_path"])
        img_w, img_h = img.getSize()
        scale = max(left_w / img_w, panel_h / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        c.saveState()
        clip = c.beginPath()
        clip.rect(left_x, panel_y, left_w, panel_h)
        c.clipPath(clip, stroke=0, fill=0)
        c.drawImage(
            img,
            left_x + (left_w - draw_w) / 2,
            panel_y + (panel_h - draw_h) / 2,
            draw_w,
            draw_h,
            preserveAspectRatio=True,
            mask="auto",
        )
        c.restoreState()

        # Keep the text page clean even if a PDF renderer handles clipping loosely.
        c.setFillColor(colors.HexColor("#fffdf8"))
        c.rect(right_x, panel_y, right_w, panel_h, fill=1, stroke=0)

        c.setFillColor(colors.HexColor("#ffd94d"))
        c.roundRect(left_x + 18, page_h - margin - 34, 96, 22, 11, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#2f2116"))
        c.setFont("Helvetica-Bold", 9)
        c.drawCentredString(left_x + 66, page_h - margin - 28, "Chapter One")

        text_x = right_x + 42
        top = page_h - margin - 54
        c.setFillColor(colors.HexColor("#d88659"))
        c.setFont("Helvetica-Bold", 8)
        c.drawString(text_x, top, "O N C E   U P O N   A   T I M E")

        c.setFillColor(colors.HexColor("#3a120f"))
        c.setFont("Times-Bold", 32)
        title_lines = title.split(" ", 3)
        if len(title_lines) > 3:
            title_draw = f"{' '.join(title_lines[:3])}\n{title_lines[3]}"
        else:
            title_draw = title
        y = top - 42
        for line in title_draw.splitlines():
            c.drawString(text_x, y, line)
            y -= 36

        if subtitle:
            c.setFillColor(colors.HexColor("#5f6f76"))
            c.setFont("Helvetica-Oblique", 11)
            c.drawString(text_x, y + 8, subtitle[:80])
            y -= 18

        paragraphs = split_paragraphs(str(page["story_text"]))
        flow = Paragraph("<br/><br/>".join(html.escape(p) for p in paragraphs), text_style)
        flow.wrapOn(c, right_w - 78, y - margin - 78)
        flow.drawOn(c, text_x, max(margin + 86, y - flow.height - 10))

        c.setFillColor(colors.HexColor("#59666b"))
        c.setFont("Helvetica", 10)
        c.line(text_x, margin + 64, page_w - margin - 42, margin + 64)
        c.drawString(text_x, margin + 38, f"Page {page['page_number']}")

        next_dest = f"page_{idx + 2}" if idx + 1 < len(pages) else "page_1"
        prev_dest = f"page_{idx}" if idx > 0 else f"page_{len(pages)}"
        draw_button(c, page_w - margin - 128, margin + 28, 104, 26, "Turn page ->", next_dest)
        if len(pages) > 1:
            draw_button(c, page_w - margin - 238, margin + 28, 92, 26, "<- Back", prev_dest)

        c.showPage()

    c.save()
    return str(path)


def run_storypose_from_text(transcript: str, test_pages: int = TEST_PAGES) -> dict[str, Any]:
    ensure_dirs()
    story, cost = generate_story_pages_with_claude(transcript, max_pages=MAX_PAGES, model=CLAUDE_MODEL)
    pages = story["pages"][:test_pages] if test_pages else story["pages"]
    pages = generate_images(story, pages)
    html_path = save_flipbook_html(story["title"], pages, story.get("subtitle", ""))
    pdf_path = save_linked_pdf(story["title"], pages, story.get("subtitle", ""))
    return {
        "title": story["title"],
        "subtitle": story.get("subtitle", ""),
        "pages": pages,
        "token_cost": cost.__dict__,
        "html_path": html_path,
        "pdf_path": pdf_path,
    }


def run_storypose_from_audio(test_pages: int = TEST_PAGES) -> dict[str, Any]:
    transcript = upload_and_transcribe_story()
    return run_storypose_from_text(transcript, test_pages=test_pages)


# -------------------------
# One-cell execution
# -------------------------
drive.mount("/content/drive")
ensure_dirs()
require_tokens()

print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
print("Claude model test:", test_claude_model(CLAUDE_MODEL))

mode = input("Type 'audio' to upload spoken story, or press Enter to paste/type a transcript: ").strip().lower()
if mode == "audio":
    result = run_storypose_from_audio(test_pages=TEST_PAGES)
else:
    transcript = input("Paste/type the child's story transcript: ").strip()
    if not transcript:
        transcript = "A little turtle is afraid to cross the pond, but he helps a lost duckling find her way home."
        print("Using demo transcript:", transcript)
    result = run_storypose_from_text(transcript, test_pages=TEST_PAGES)

print("\nDONE")
print("Title:", result["title"])
print("Estimated Claude token/cost:", result["token_cost"])
print("HTML:", result["html_path"])
print("PDF:", result["pdf_path"])
for page in result["pages"]:
    print(f"Page {page['page_number']} image: {page['image_path']}")

result
