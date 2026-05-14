import json
import os
import re
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import anthropic
import requests
from dotenv import load_dotenv

try:
    import fal_client
except ModuleNotFoundError:
    fal_client = None

try:
    from fpdf import FPDF
except ModuleNotFoundError:
    FPDF = None

load_dotenv()

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-sonnet-20241022")
ANTHROPIC_API_BASE = os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com")
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
CLAUDE_INPUT_COST_PER_MTOK = float(os.getenv("CLAUDE_INPUT_COST_PER_MTOK", "0"))
CLAUDE_OUTPUT_COST_PER_MTOK = float(os.getenv("CLAUDE_OUTPUT_COST_PER_MTOK", "0"))
FAL_MODEL_ID = os.getenv("FAL_MODEL_ID", "fal-ai/flux/schnell")
IMAGE_BACKEND = os.getenv("IMAGE_BACKEND", "fal").strip().lower()
COMFY_API_URL = os.getenv("COMFY_API_URL", "http://127.0.0.1:8188").rstrip("/")
COMFY_WORKFLOW_PATH = os.getenv(
    "COMFY_WORKFLOW_PATH",
    "/Users/avinashkannan/ComfyUI/user/default/workflows/storypose.json",
)
STORYBOOK_SYSTEM_PROMPT = (
    "You are a professional children's book author. Generate exactly 5 different candidate stories "
    "from the user's spoken transcript. For each candidate story: create a title and break it into "
    "3-10 pages. For each page provide: story_text (max 100 words) and image_description (detailed "
    "visual prompt for FLUX). Keep visual continuity across pages in each story.\n"
    "Return JSON only with this structure:\n"
    "{\n"
    "  \"stories\": [\n"
    "    {\n"
    "      \"story_number\": 1,\n"
    "      \"title\": \"string\",\n"
    "      \"pages\": [\n"
    "        {\n"
    "          \"page_number\": 1,\n"
    "          \"story_text\": \"string\",\n"
    "          \"image_description\": \"string\"\n"
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n"
    "No markdown. No prose. JSON only."
)


class StorybookPipelineError(RuntimeError):
    pass


def _parse_node_ids(value: str, default: list[int]) -> list[int]:
    raw = (value or "").strip()
    if not raw:
        return default
    try:
        return [int(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise StorybookPipelineError(
            f"Invalid node id list '{value}'. Use comma-separated integers."
        ) from exc


COMFY_POSITIVE_NODE_IDS = _parse_node_ids(os.getenv("COMFY_POSITIVE_NODE_ID", ""), [5])
COMFY_NEGATIVE_NODE_IDS = _parse_node_ids(os.getenv("COMFY_NEGATIVE_NODE_ID", ""), [9])
COMFY_SAVE_NODE_IDS = _parse_node_ids(os.getenv("COMFY_SAVE_NODE_ID", ""), [7])
COMFY_POSITIVE_NODE_ID = COMFY_POSITIVE_NODE_IDS[0]
COMFY_NEGATIVE_NODE_ID = COMFY_NEGATIVE_NODE_IDS[0]
COMFY_SAVE_NODE_ID = COMFY_SAVE_NODE_IDS[0]


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise StorybookPipelineError(f"Missing required environment variable: {name}")
    return value


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise StorybookPipelineError("Claude response did not include JSON.")
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise StorybookPipelineError("Claude returned malformed JSON.") from exc


def _validate_story_payload(payload: dict[str, Any]) -> dict[str, Any]:
    title = payload.get("title")
    pages = payload.get("pages")
    if not isinstance(title, str) or not title.strip():
        raise StorybookPipelineError("Claude JSON missing valid 'title'.")
    if not isinstance(pages, list) or not (3 <= len(pages) <= 10):
        raise StorybookPipelineError("Claude JSON must include 3-10 pages.")

    validated_pages = []
    for idx, page in enumerate(pages, start=1):
        if not isinstance(page, dict):
            raise StorybookPipelineError(f"Page {idx} is not an object.")
        story_text = page.get("story_text")
        image_description = page.get("image_description")
        page_number = page.get("page_number", idx)
        if not isinstance(story_text, str) or not story_text.strip():
            raise StorybookPipelineError(f"Page {idx} missing 'story_text'.")
        if not isinstance(image_description, str) or not image_description.strip():
            raise StorybookPipelineError(f"Page {idx} missing 'image_description'.")
        validated_pages.append(
            {
                "page_number": int(page_number),
                "story_text": story_text.strip(),
                "image_description": image_description.strip(),
            }
        )

    return {"title": title.strip(), "pages": validated_pages}


def _validate_story_options_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    stories = payload.get("stories")
    if not isinstance(stories, list) or len(stories) != 5:
        raise StorybookPipelineError("Claude JSON must include exactly 5 stories.")

    validated = []
    for idx, story in enumerate(stories, start=1):
        if not isinstance(story, dict):
            raise StorybookPipelineError(f"Story {idx} is not an object.")
        single = _validate_story_payload(story)
        validated.append(
            {
                "story_number": int(story.get("story_number", idx)),
                "title": single["title"],
                "pages": single["pages"],
            }
        )
    return validated


def _count_tokens_for_messages(
    *,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "system": [{"type": "text", "text": system_prompt}],
        "messages": [{"role": "user", "content": user_prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION,
        "x-api-key": api_key,
    }
    try:
        resp = requests.post(
            f"{ANTHROPIC_API_BASE.rstrip('/')}/v1/messages/count_tokens",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        body = resp.json()
        return {
            "ok": True,
            "model": model,
            "input_tokens": body.get("input_tokens"),
            "output_tokens": body.get("output_tokens"),
            "raw": body,
        }
    except Exception as exc:
        return {
            "ok": False,
            "model": model,
            "error": str(exc),
        }


def estimate_claude_cost(token_counts: list[dict[str, Any]], pages: int = 1) -> dict[str, Any]:
    input_tokens = 0
    output_tokens = 0
    for item in token_counts:
        if not item.get("ok"):
            continue
        input_tokens += int(item.get("input_tokens") or 0)
        output_tokens += int(item.get("output_tokens") or 0)

    session_cost = (
        (input_tokens / 1_000_000.0) * CLAUDE_INPUT_COST_PER_MTOK
        + (output_tokens / 1_000_000.0) * CLAUDE_OUTPUT_COST_PER_MTOK
    )
    per_page_cost = session_cost / pages if pages > 0 else session_cost
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "session_cost_usd": round(session_cost, 6),
        "per_page_cost_usd": round(per_page_cost, 6),
        "rate_input_per_mtok_usd": CLAUDE_INPUT_COST_PER_MTOK,
        "rate_output_per_mtok_usd": CLAUDE_OUTPUT_COST_PER_MTOK,
    }


def generate_story_pages(transcript: str, model_override: str | None = None) -> dict[str, Any]:
    """Generate 5 candidate stories and return the first as default + all options."""
    if not transcript or not transcript.strip():
        raise StorybookPipelineError("Transcript is empty.")

    api_key = _require_env("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    user_prompt = (
        "Create 5 children's storybook candidates from this transcript.\n\n"
        f"Transcript:\n{transcript.strip()}\n\n"
        "Return JSON with key: stories."
    )

    model_candidates = []
    for m in [
        model_override.strip() if model_override else None,
        CLAUDE_MODEL,
        "claude-3-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-haiku-20240307",
    ]:
        if m and m not in model_candidates:
            model_candidates.append(m)

    last_error: Exception | None = None
    token_count_log: list[dict[str, Any]] = []
    tried_models: list[str] = []
    for model_name in model_candidates:
        tried_models.append(model_name)
        for attempt in (1, 2):
            try:
                token_count_log.append(
                    {
                        "attempt": attempt,
                        "stage": "count_tokens",
                        **_count_tokens_for_messages(
                            api_key=api_key,
                            model=model_name,
                            system_prompt=STORYBOOK_SYSTEM_PROMPT,
                            user_prompt=user_prompt,
                        ),
                    }
                )
                response = client.messages.create(
                    model=model_name,
                    max_tokens=4000,
                    system=STORYBOOK_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                raw = response.content[0].text if response.content else ""
                payload = _extract_json(raw)
                stories = _validate_story_options_payload(payload)
                selected = stories[0]
                return {
                    "title": selected["title"],
                    "pages": selected["pages"],
                    "stories": stories,
                    "model_used": model_name,
                    "models_tried": tried_models,
                    "token_counts": token_count_log,
                }
            except Exception as exc:
                last_error = exc
                message = str(exc).lower()
                # If model doesn't exist for the account, try next model candidate.
                if "not_found" in message or "model:" in message:
                    break
                if attempt == 2:
                    # Non-model error after retry: try next candidate.
                    break
    raise StorybookPipelineError(
        "Could not generate valid story pages from Claude after retry. "
        f"Models tried: {', '.join(tried_models)}"
    ) from last_error


def _run_fal_image(prompt: str, image_size: str = "square_hd") -> str:
    if fal_client is None:
        raise StorybookPipelineError(
            "fal-client is not installed. Install it or set IMAGE_BACKEND=comfy."
        )
    result = fal_client.run(
        FAL_MODEL_ID,
        arguments={"prompt": prompt, "image_size": image_size},
    )
    images = result.get("images", [])
    if not images or "url" not in images[0]:
        raise StorybookPipelineError("fal.ai response missing image URL.")
    return images[0]["url"]


def _load_workflow_template() -> dict[str, Any]:
    path = Path(COMFY_WORKFLOW_PATH)
    if not path.exists():
        raise StorybookPipelineError(f"Comfy workflow not found: {path}")
    return json.loads(path.read_text())


def _set_node_text(workflow: dict[str, Any], node_id: int, text: str) -> None:
    for node in workflow.get("nodes", []):
        if node.get("id") == node_id:
            values = node.get("widgets_values", [])
            if not values:
                node["widgets_values"] = [text]
            else:
                values[0] = text
            return
    raise StorybookPipelineError(f"Node id {node_id} not found in Comfy workflow.")


def _set_save_prefix(workflow: dict[str, Any], node_id: int, prefix: str) -> None:
    for node in workflow.get("nodes", []):
        if node.get("id") == node_id:
            values = node.get("widgets_values", [])
            if not values:
                node["widgets_values"] = [prefix]
            else:
                values[0] = prefix
            return
    raise StorybookPipelineError(f"Save node id {node_id} not found in Comfy workflow.")


def _submit_comfy_prompt(workflow: dict[str, Any]) -> str:
    client_id = str(uuid.uuid4())
    payload = {"prompt": workflow, "client_id": client_id}
    resp = requests.post(f"{COMFY_API_URL}/prompt", json=payload, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    prompt_id = body.get("prompt_id")
    if not prompt_id:
        raise StorybookPipelineError("Comfy did not return prompt_id.")
    return prompt_id


def _find_nodes_by_type(workflow: dict[str, Any], node_type: str) -> list[dict[str, Any]]:
    nodes = [n for n in workflow.get("nodes", []) if n.get("type") == node_type]
    nodes.sort(key=lambda n: int(n.get("id", 0)))
    return nodes


def _extract_save_image_outputs(entry: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    outputs = entry.get("outputs", {})
    found = []
    for node_id, node_output in outputs.items():
        images = node_output.get("images", [])
        for img in images:
            found.append({"node_id": int(node_id), **img})
    if len(found) < expected_count:
        raise StorybookPipelineError(
            f"Comfy finished but returned {len(found)} image(s); expected at least {expected_count}."
        )
    found.sort(key=lambda x: x["node_id"])
    return found


def _wait_comfy_image(prompt_id: str, save_node_id: int, timeout_s: int = 900) -> str:
    start = time.time()
    history_url = f"{COMFY_API_URL}/history/{prompt_id}"
    while time.time() - start < timeout_s:
        resp = requests.get(history_url, timeout=30)
        resp.raise_for_status()
        history = resp.json()
        entry = history.get(prompt_id)
        if entry:
            outputs = entry.get("outputs", {})
            node_output = outputs.get(str(save_node_id), {})
            images = node_output.get("images", [])
            if images:
                img = images[0]
                filename = img.get("filename")
                subfolder = img.get("subfolder", "")
                image_type = img.get("type", "output")
                if not filename:
                    break
                view_params = {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": image_type,
                }
                view_resp = requests.get(f"{COMFY_API_URL}/view", params=view_params, timeout=60)
                view_resp.raise_for_status()
                out_dir = Path(tempfile.mkdtemp(prefix="storypose_comfy_"))
                out_path = out_dir / filename
                out_path.write_bytes(view_resp.content)
                return str(out_path)
            # If entry exists but save output is missing, treat as failure.
            raise StorybookPipelineError("Comfy finished but no image was saved by SaveImage node.")
        time.sleep(1.0)
    raise StorybookPipelineError(f"Timed out waiting for Comfy prompt {prompt_id}.")


def _run_comfy_image(prompt: str, save_prefix: str, negative_prompt: str = "") -> str:
    workflow = _load_workflow_template()
    _set_node_text(workflow, COMFY_POSITIVE_NODE_ID, prompt)
    _set_node_text(workflow, COMFY_NEGATIVE_NODE_ID, negative_prompt)
    _set_save_prefix(workflow, COMFY_SAVE_NODE_ID, save_prefix)
    prompt_id = _submit_comfy_prompt(workflow)
    return _wait_comfy_image(prompt_id, COMFY_SAVE_NODE_ID)


def _run_comfy_multi_page_images(pages: list[dict[str, Any]]) -> list[str]:
    workflow = _load_workflow_template()
    clip_nodes = _find_nodes_by_type(workflow, "CLIPTextEncode")
    save_nodes = _find_nodes_by_type(workflow, "SaveImage")

    if len(clip_nodes) < 2 or len(save_nodes) < 1:
        raise StorybookPipelineError(
            "Workflow must include CLIPTextEncode and SaveImage nodes."
        )

    # For 4-page storybook workflow this maps:
    # positives: [20, 30, 40, 50], negatives: [21, 31, 41, 51]
    # by convention sorted CLIP nodes alternate positive/negative.
    if len(COMFY_POSITIVE_NODE_IDS) > 1:
        positive_nodes = [n for n in clip_nodes if int(n["id"]) in COMFY_POSITIVE_NODE_IDS]
        positive_nodes.sort(key=lambda n: COMFY_POSITIVE_NODE_IDS.index(int(n["id"])))
    else:
        positive_nodes = clip_nodes[0::2]

    if len(COMFY_NEGATIVE_NODE_IDS) > 1:
        negative_nodes = [n for n in clip_nodes if int(n["id"]) in COMFY_NEGATIVE_NODE_IDS]
        negative_nodes.sort(key=lambda n: COMFY_NEGATIVE_NODE_IDS.index(int(n["id"])))
    else:
        negative_nodes = clip_nodes[1::2]

    if len(COMFY_SAVE_NODE_IDS) > 1:
        save_nodes = [n for n in save_nodes if int(n["id"]) in COMFY_SAVE_NODE_IDS]
        save_nodes.sort(key=lambda n: COMFY_SAVE_NODE_IDS.index(int(n["id"])))
    page_slots = min(len(pages), len(positive_nodes), len(save_nodes))
    if page_slots == 0:
        raise StorybookPipelineError("No available page slots in Comfy workflow.")

    for i in range(page_slots):
        _set_node_text(workflow, int(positive_nodes[i]["id"]), pages[i]["image_description"])
        if i < len(negative_nodes):
            _set_node_text(workflow, int(negative_nodes[i]["id"]), "")
        _set_save_prefix(workflow, int(save_nodes[i]["id"]), f"StoryPose/page_{i+1:02d}")

    prompt_id = _submit_comfy_prompt(workflow)

    start = time.time()
    history_url = f"{COMFY_API_URL}/history/{prompt_id}"
    while time.time() - start < 900:
        resp = requests.get(history_url, timeout=30)
        resp.raise_for_status()
        history = resp.json()
        entry = history.get(prompt_id)
        if entry:
            output_images = _extract_save_image_outputs(entry, page_slots)
            local_paths = []
            for i, img in enumerate(output_images[:page_slots], start=1):
                filename = img.get("filename")
                subfolder = img.get("subfolder", "")
                image_type = img.get("type", "output")
                if not filename:
                    continue
                view_params = {
                    "filename": filename,
                    "subfolder": subfolder,
                    "type": image_type,
                }
                view_resp = requests.get(f"{COMFY_API_URL}/view", params=view_params, timeout=60)
                view_resp.raise_for_status()
                out_dir = Path(tempfile.mkdtemp(prefix="storypose_comfy_multi_"))
                out_path = out_dir / f"page_{i:02d}.png"
                out_path.write_bytes(view_resp.content)
                local_paths.append(str(out_path))
            if len(local_paths) < page_slots:
                raise StorybookPipelineError(
                    f"Comfy returned only {len(local_paths)} downloadable image(s); expected {page_slots}."
                )
            return local_paths
        time.sleep(1.0)

    raise StorybookPipelineError(f"Timed out waiting for Comfy prompt {prompt_id}.")


def _simplify_prompt(prompt: str) -> str:
    simplified = re.sub(r"\s+", " ", prompt).strip()
    if len(simplified) > 220:
        simplified = simplified[:220].rsplit(" ", 1)[0]
    return (
        simplified
        + ", children's book illustration, clean composition, single scene, high quality"
    )


def _download_image(url: str, target_path: Path) -> Path:
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    target_path.write_bytes(resp.content)
    return target_path


def generate_images_for_pages(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate images concurrently for each page and return local image paths."""
    if IMAGE_BACKEND == "fal":
        _require_env("FAL_KEY")
    output_dir = Path(tempfile.mkdtemp(prefix="storypose_pages_"))

    if IMAGE_BACKEND == "comfy":
        try:
            comfy_paths = _run_comfy_multi_page_images(pages)
            results = []
            for page, src_path in zip(sorted(pages, key=lambda p: int(p["page_number"])), comfy_paths):
                page_number = int(page["page_number"])
                dst = output_dir / f"page_{page_number:02d}.png"
                Path(src_path).replace(dst)
                results.append(
                    {
                        **page,
                        "image_url": None,
                        "image_path": str(dst),
                        "image_error": None,
                    }
                )
            return results
        except Exception:
            # Fall back to single-page Comfy calls if workflow isn't multi-page compatible.
            pass

    def generate_one(page: dict[str, Any]) -> dict[str, Any]:
        page_number = int(page["page_number"])
        prompt = page["image_description"]
        filename = output_dir / f"page_{page_number:02d}.png"

        try:
            if IMAGE_BACKEND == "comfy":
                image_path = _run_comfy_image(prompt, save_prefix=f"StoryPose/page_{page_number:02d}")
                Path(image_path).replace(filename)
                return {
                    **page,
                    "image_url": None,
                    "image_path": str(filename),
                    "image_error": None,
                }
            url = _run_fal_image(prompt)
        except Exception:
            retry_prompt = _simplify_prompt(prompt)
            try:
                if IMAGE_BACKEND == "comfy":
                    image_path = _run_comfy_image(
                        retry_prompt, save_prefix=f"StoryPose/page_{page_number:02d}", negative_prompt=""
                    )
                    Path(image_path).replace(filename)
                    return {
                        **page,
                        "image_url": None,
                        "image_path": str(filename),
                        "image_error": None,
                    }
                url = _run_fal_image(retry_prompt)
            except Exception:
                return {
                    **page,
                    "image_url": None,
                    "image_path": None,
                    "image_error": "Image generation failed after retry.",
                }

        _download_image(url, filename)
        return {
            **page,
            "image_url": url,
            "image_path": str(filename),
            "image_error": None,
        }

    if IMAGE_BACKEND == "comfy":
        # Comfy queue is global; run sequentially to keep ordering stable.
        results = [generate_one(page) for page in pages]
    else:
        with ThreadPoolExecutor(max_workers=min(8, max(1, len(pages)))) as executor:
            results = list(executor.map(generate_one, pages))

    # Keep results ordered by page number
    results.sort(key=lambda p: int(p["page_number"]))
    return results


if FPDF is not None:
    class StorybookPDF(FPDF):
        def header(self) -> None:
            # Intentionally blank for clean spreads
            return


def create_storybook_pdf(title: str, pages_with_images: list[dict[str, Any]]) -> str:
    """Create landscape two-column PDF: left text, right image."""
    if FPDF is None:
        raise StorybookPipelineError(
            "fpdf2 is not installed. Install fpdf2 to enable PDF output."
        )
    output_dir = Path(os.getenv("STORYBOOK_OUTPUT_DIR", os.getcwd()))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "storybook.pdf"

    pdf = StorybookPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=False)
    margin = 15
    gap = 8
    left_w = (297 - 2 * margin - gap) / 2
    right_w = left_w
    usable_h = 210 - 2 * margin

    # Cover page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 28)
    pdf.set_xy(margin, 70)
    pdf.multi_cell(297 - 2 * margin, 16, title, align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.multi_cell(297 - 2 * margin, 10, "A StoryPose Storybook", align="C")

    for page in pages_with_images:
        pdf.add_page()
        left_x = margin
        right_x = margin + left_w + gap
        top_y = margin

        pdf.set_xy(left_x, top_y)
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(left_w, 8, f"Page {page['page_number']}")
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(left_w, 7, page["story_text"])

        image_path = page.get("image_path")
        if image_path and Path(image_path).exists():
            pdf.image(image_path, x=right_x, y=top_y, w=right_w, h=usable_h, keep_aspect_ratio=True)
        else:
            pdf.set_xy(right_x, top_y)
            pdf.set_font("Helvetica", "I", 12)
            pdf.multi_cell(
                right_w,
                8,
                "Image unavailable for this page.\n"
                + (page.get("image_error") or "Unknown image generation error."),
            )

    pdf.output(str(output_path))
    return str(output_path)


def create_storybook_html(title: str, pages_with_images: list[dict[str, Any]]) -> str:
    """Create HTML storybook with left image and right text spreads."""
    output_dir = Path(os.getenv("STORYBOOK_OUTPUT_DIR", os.getcwd()))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "my-amazing-story.html"

    safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    parts = [
        "<!DOCTYPE html>",
        "<html>",
        "<head>",
        "  <meta charset=\"UTF-8\">",
        f"  <title>{safe_title}</title>",
        "  <style>",
        "    body { font-family: 'Patrick Hand', cursive; background: #fdf6e3; margin: 0; padding: 20px; }",
        "    .page-spread { display: flex; margin-bottom: 40px; border: 2px solid #8B7355; border-radius: 6px; overflow: hidden; page-break-after: always; }",
        "    .page { flex: 1; padding: 24px; background: #fdf6e3; }",
        "    .page-left { border-right: 2px solid #8B7355; display: flex; align-items: center; justify-content: center; }",
        "    img { max-width: 100%; border-radius: 8px; }",
        "    p { font-size: 1.1rem; line-height: 1.9; color: #3d2b1f; }",
        "    h1 { font-size: 2rem; color: #e07b39; text-align: center; margin-bottom: 40px; }",
        "    .pose-tag { font-size: 0.75rem; color: #999; font-style: italic; }",
        "    @media print { .page-spread { page-break-after: always; } }",
        "  </style>",
        "</head>",
        "<body>",
        f"  <h1>📖 {safe_title}</h1>",
    ]

    for page in pages_with_images:
        text = page.get("story_text", "")
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        image_path = page.get("image_path")
        img_src = ""
        if image_path and Path(image_path).exists():
            img_src = Path(image_path).resolve().as_uri()
        else:
            img_src = ""

        parts.extend(
            [
                "  <div class=\"page-spread\">",
                "    <div class=\"page page-left\">",
                f"      <img src=\"{img_src}\" alt=\"Page {page.get('page_number', '')} illustration\" />",
                "    </div>",
                "    <div class=\"page page-right\">",
                f"      <p>{safe_text}</p>",
                "    </div>",
                "  </div>",
            ]
        )

    parts.extend(["</body>", "</html>"])
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return str(output_path)


def main_storybook_pipeline(transcript: str) -> str:
    """Orchestrate full pipeline from transcript to storybook PDF path."""
    print("[1/3] Generating story pages with Claude...")
    story = generate_story_pages(transcript)
    print(f"  Title: {story['title']} | Pages: {len(story['pages'])}")

    print("[2/3] Generating page images with fal.ai...")
    pages_with_images = generate_images_for_pages(story["pages"])
    failures = [p for p in pages_with_images if p.get("image_error")]
    print(f"  Images generated: {len(pages_with_images) - len(failures)}/{len(pages_with_images)}")

    print("[3/3] Building storybook PDF...")
    pdf_path = create_storybook_pdf(story["title"], pages_with_images)
    print(f"  Storybook saved: {pdf_path}")
    return pdf_path
