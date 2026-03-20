/**
 * app.js — Main session orchestrator for StoryPose.
 *
 * State machine:
 *   IDLE → LISTENING → CLARIFYING → GENERATING → REVIEWING → LISTENING → ...
 *   Any state → DONE (on "the end" keyword)
 */

// ============================================================
// State
// ============================================================

const State = {
  IDLE:       "idle",
  LISTENING:  "listening",
  CLARIFYING: "clarifying",
  GENERATING: "generating",
  REVIEWING:  "reviewing",
  DONE:       "done",
};

let currentState      = State.IDLE;
let pageNumber        = 0;        // pages generated so far
let characterDesc     = null;     // extracted from first page for consistency
let storyEnded        = false;
let storyTitle        = "My Amazing Story";
let storyPages        = [];       // all page texts so far (for image context)

// DOM refs
const btnStart     = document.getElementById("btn-start");
const btnEnd       = document.getElementById("btn-end");
const btnDownload  = document.getElementById("btn-download");
const btnRestart   = document.getElementById("btn-restart");
const statusPill   = document.getElementById("status-pill");
const statusText   = document.getElementById("status-text");
const micBarWrapper   = document.getElementById("mic-bar-wrapper");
const micBar          = document.getElementById("mic-bar");
const liveTranscript  = document.getElementById("live-transcript");
const genOverlay    = document.getElementById("generating-overlay");
const genText       = document.getElementById("generating-text");
const aiBubble      = document.getElementById("ai-bubble");
const aiBubbleText  = document.getElementById("ai-bubble-text");

// ============================================================
// UI helpers
// ============================================================

function setStatus(state, label) {
  currentState = state;
  statusPill.className = `status-pill ${state}`;
  statusText.textContent = label;
}

function showAIBubble(text) {
  aiBubbleText.textContent = text;
  aiBubble.style.display = "block";
}

function hideAIBubble() {
  aiBubble.style.display = "none";
}

function showGenerating(msg) {
  genText.textContent = msg || "Drawing your illustration...";
  genOverlay.style.display = "flex";
}

function hideGenerating() {
  genOverlay.style.display = "none";
}

function setMicVolume(v) {
  if (micBar) micBar.style.width = `${Math.round(v * 100)}%`;
}

// ============================================================
// API calls
// ============================================================

async function apiDetectGaps(text, previousPages) {
  const res = await fetch("/detect-gaps", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, previous_pages: previousPages }),
  });
  if (!res.ok) throw new Error("Gap detection failed");
  return res.json();
}

async function apiGenerateImage(prompt) {
  const res = await fetch("/generate-image", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  if (!res.ok) throw new Error("Image generation failed");
  const data = await res.json();
  return data.image; // base64 PNG
}

// ============================================================
// Prompt builder
// ============================================================

function buildImagePrompt(storyText, poseDescriptor) {
  const consistency = characterDesc
    ? `MAINTAIN CHARACTER CONSISTENCY: ${characterDesc}`
    : "";

  const storyContext = storyPages.length > 0
    ? `STORY SO FAR (for context and character/setting consistency):\n${storyPages.map((t, i) => `Page ${i + 1}: ${t}`).join("\n")}\n`
    : "";

  return `Children's storybook illustration, watercolor style, warm and magical lighting.

${storyContext}
CURRENT SCENE TO ILLUSTRATE: ${storyText}

CHARACTER POSE: A child character who is ${poseDescriptor}.

STYLE REQUIREMENTS:
- Soft watercolor or gouache illustration style
- Bright, warm, inviting colors
- Large expressive eyes on characters
- Whimsical and dreamlike background
- Safe and joyful atmosphere
- No text or letters in the image
- Full scene composition, not portrait

${consistency}

The illustration should feel like it belongs in a classic children's picture book.`.trim();
}

// ============================================================
// Split long transcription into page-sized sentence groups
// ============================================================

function splitIntoPageSegments(text) {
  // Split on sentence boundaries
  const raw = text.match(/[^.!?]+[.!?]*/g) || [text];
  const sentences = raw.map(s => s.trim()).filter(s => s.length > 3);

  // Group into segments of ~2 sentences each (so pages aren't too thin)
  const segments = [];
  for (let i = 0; i < sentences.length; i += 2) {
    const group = sentences.slice(i, i + 2).join(" ").trim();
    if (group) segments.push(group);
  }
  return segments.length > 0 ? segments : [text];
}

// ============================================================
// End-of-story keyword detection
// ============================================================

function isEndOfStory(text) {
  const lower = text.toLowerCase();
  return (
    lower.includes("the end") ||
    lower.includes("that's all") ||
    lower.includes("thats all") ||
    lower.includes("finish") ||
    lower.includes("i'm done") ||
    lower.includes("im done") ||
    lower.includes("story over") ||
    lower.includes("happily ever after")
  );
}

// ============================================================
// Extract character description from enriched context (page 1)
// ============================================================

function extractCharacterDescription(enrichedContext, storyText) {
  // Simple heuristic: take the first sentence as character desc
  const combined = enrichedContext || storyText;
  const sentences = combined.split(/[.!?]/);
  return sentences[0]?.trim() || combined.slice(0, 120);
}

// ============================================================
// Core story loop
// ============================================================

async function handleStorySegment(transcribedText) {
  const poseAtMoment = PoseTracker.getPoseSnapshot();
  PoseTracker.setActive(false); // freeze pose snapshot while processing

  const storySegments = [transcribedText];
  const previousPageTexts = []; // could track page texts here if needed

  // --- Conversation loop: max 2 follow-up questions ---
  setStatus(State.CLARIFYING, "Thinking...");
  let followUps = 0;
  const MAX_FOLLOW_UPS = 2;

  while (followUps < MAX_FOLLOW_UPS) {
    const combinedText = storySegments.join(" ");
    let gapResult;

    try {
      gapResult = await apiDetectGaps(combinedText, previousPageTexts);
    } catch (err) {
      console.warn("[app] Gap detection error:", err);
      break;
    }

    if (gapResult.isComplete) break;

    const question = gapResult.followUpQuestion;
    if (!question) break;

    // Stop listening before speaking so we don't pick up TTS as child speech
    VoiceRecorder.stopListening();
    micBarWrapper.style.display = "none";
    liveTranscript.style.display = "none";
    liveTranscript.textContent = "";

    // Show question in bubble and speak it
    showAIBubble(question);
    setStatus(State.CLARIFYING, "Asking...");
    await TTS.speakToChild(question);
    // Small pause so child is ready before mic opens
    await new Promise(r => setTimeout(r, 400));

    // Listen for child's response
    setStatus(State.LISTENING, "Listening...");
    micBarWrapper.style.display = "block";

    const childResponse = await VoiceRecorder.startListening(setMicVolume, (text) => {
      liveTranscript.style.display = "block";
      liveTranscript.textContent = text;
    });
    micBarWrapper.style.display = "none";
    liveTranscript.style.display = "none";
    liveTranscript.textContent = "";
    setMicVolume(0);

    if (childResponse) storySegments.push(childResponse);
    hideAIBubble();
    followUps++;
  }

  hideAIBubble();

  // --- Generate image ---
  pageNumber++;
  const fullText    = storySegments.join(" ");
  const imagePrompt = buildImagePrompt(fullText, poseAtMoment);

  showGenerating(`Drawing page ${pageNumber}... (this takes ~20 seconds)`);
  setStatus(State.GENERATING, "Drawing...");

  let imageBase64 = null;
  try {
    imageBase64 = await apiGenerateImage(imagePrompt);
  } catch (err) {
    console.error("[app] Image generation error:", err);
    // Continue with placeholder — don't block the story
  }

  hideGenerating();

  // After page 1, lock in character description
  if (pageNumber === 1) {
    const gapResult = await apiDetectGaps(fullText, []).catch(() => ({ enrichedContext: fullText }));
    characterDesc = extractCharacterDescription(gapResult.enrichedContext, fullText);
    // Use first meaningful words as story title
    storyTitle = fullText.split(" ").slice(0, 5).join(" ") + "...";
  }

  // Track story pages for future image context
  storyPages.push(fullText);

  // Render the page
  Storybook.addPage(imageBase64, fullText, poseAtMoment, pageNumber);

  PoseTracker.setActive(true);
  setStatus(State.REVIEWING, "Page ready!");

  return fullText;
}

// ============================================================
// Main story session
// ============================================================

async function runStorybookSession() {
  storyEnded = false;
  pageNumber  = 0;
  characterDesc = null;
  storyPages  = [];

  Storybook.hideWelcome();

  setStatus(State.LISTENING, "Starting...");
  await TTS.speakToChild("Hi! I'm ready to make your story. What happens first?");

  while (!storyEnded) {
    setStatus(State.LISTENING, "Listening...");
    micBarWrapper.style.display = "block";

    let transcribed = "";
    try {
      transcribed = await VoiceRecorder.startListening(setMicVolume, (text) => {
      liveTranscript.style.display = "block";
      liveTranscript.textContent = text;
    });
    } catch (err) {
      console.error("[app] Recording error:", err);
    }

    micBarWrapper.style.display = "none";
    liveTranscript.style.display = "none";
    liveTranscript.textContent = "";
    setMicVolume(0);

    if (!transcribed) {
      // No speech detected — prompt again
      await TTS.speakToChild("I didn't quite catch that. What happens in your story?");
      continue;
    }

    // Check for story-end keywords
    if (isEndOfStory(transcribed)) {
      storyEnded = true;
      break;
    }

    // Split into page-sized segments and process each
    const segments = splitIntoPageSegments(transcribed);
    for (const segment of segments) {
      if (storyEnded) break;
      if (isEndOfStory(segment)) { storyEnded = true; break; }
      try {
        await handleStorySegment(segment);
      } catch (err) {
        console.error("[app] Segment error:", err);
        setStatus(State.LISTENING, "Hmm...");
      }
    }

    if (!storyEnded) {
      const prompts = [
        "That's amazing! What happens next?",
        "Wow, I love it! What happens after that?",
        "Ooh! Keep going, what comes next in the adventure?",
        "Incredible! And then what happened?",
      ];
      const pick = prompts[Math.floor(Math.random() * prompts.length)];
      showAIBubble(pick);
      setStatus(State.REVIEWING, "Your turn!");
      await TTS.speakToChild(pick);
      hideAIBubble();
    }
  }

  // --- Story finished ---
  await finishStory();
}

async function finishStory() {
  setStatus(State.DONE, "Story complete!");
  TTS.stopSpeaking();

  const endMsg = "What an amazing story! Let me make your book cover now!";
  showAIBubble(endMsg);
  await TTS.speakToChild(endMsg);
  hideAIBubble();

  Storybook.renderCover(storyTitle, pageNumber);

  btnEnd.disabled           = true;
  btnDownload.disabled      = false;
  btnDownload.style.display = "block";
  btnRestart.style.display  = "block";

  await TTS.speakToChild("Your storybook is ready! You can download it or start a new story!");
}

// ============================================================
// Button handlers
// ============================================================

btnStart.addEventListener("click", async () => {
  btnStart.disabled = true;
  btnEnd.disabled   = false;

  try {
    await runStorybookSession();
  } catch (err) {
    console.error("[app] Session error:", err);
    setStatus(State.IDLE, "Error — try refreshing");
  }
});

btnEnd.addEventListener("click", async () => {
  storyEnded = true;
  VoiceRecorder.stopListening();
  TTS.stopSpeaking();
  hideAIBubble();
  hideGenerating();
  await finishStory();
});

btnDownload.addEventListener("click", () => {
  Storybook.downloadStory();
});

btnRestart.addEventListener("click", () => {
  // Reset UI
  btnStart.disabled         = false;
  btnEnd.disabled           = true;
  btnDownload.disabled      = true;
  btnDownload.style.display = "none";
  btnRestart.style.display  = "none";
  // Clear storybook pages and show welcome screen again
  Storybook.reset();
  setStatus(State.IDLE, "Ready");
});

// ============================================================
// Initialise on page load
// ============================================================

window.addEventListener("DOMContentLoaded", async () => {
  Storybook.showWelcome();

  const videoEl  = document.getElementById("webcam");
  const canvasEl = document.getElementById("pose-canvas");
  const labelEl  = document.getElementById("pose-label");

  try {
    await PoseTracker.init(videoEl, canvasEl, labelEl);
    setStatus(State.IDLE, "Ready");
    btnStart.disabled = false;
    console.log("[app] Pose tracker initialised.");
  } catch (err) {
    console.warn("[app] Pose init failed:", err);
    labelEl.textContent = "Pose unavailable";
    // App still works — pose will default to "standing naturally"
    setStatus(State.IDLE, "Ready (no pose)");
    btnStart.disabled = false;
  }
});
