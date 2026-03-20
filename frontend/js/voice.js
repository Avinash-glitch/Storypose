/**
 * voice.js — Live transcription using Web Speech API (browser built-in, free).
 *
 * Exposes:
 *   VoiceRecorder.startListening(onVolumeUpdate, onInterimResult) → Promise<string>
 *     Starts live recognition. Resolves with final transcribed text once silence detected.
 *     `onVolumeUpdate(0..1)` called each frame with mic volume level.
 *     `onInterimResult(text)` called with live partial results as child speaks.
 *
 *   VoiceRecorder.stopListening() → void
 *     Force-stops current recording early.
 */

const VoiceRecorder = (() => {
  const SILENCE_DURATION    = 3000;  // ms of silence after last word before segment ends
  const MAX_RECORDING_TIME  = 30000; // hard stop after 30s

  let recognition     = null;
  let audioCtx        = null;
  let analyser        = null;
  let stream          = null;
  let animFrameId     = null;
  let silenceTimer    = null;
  let maxTimer        = null;
  let resolveCallback = null;
  let finalTranscript = "";
  let forceStopped    = false;

  // -----------------------------------------------------------------------
  // Internal helpers
  // -----------------------------------------------------------------------

  function getRMS(dataArray) {
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const val = (dataArray[i] - 128) / 128;
      sum += val * val;
    }
    return Math.sqrt(sum / dataArray.length);
  }

  function cleanup() {
    forceStopped = true;
    if (animFrameId)  { cancelAnimationFrame(animFrameId); animFrameId = null; }
    if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
    if (maxTimer)     { clearTimeout(maxTimer);     maxTimer     = null; }
    if (recognition)  { try { recognition.stop(); } catch(e) {} recognition = null; }
    if (stream)       { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (audioCtx && audioCtx.state !== "closed") { audioCtx.close(); audioCtx = null; }
    analyser        = null;
    finalTranscript = "";
    forceStopped    = false;
    resolveCallback = null;
  }

  // -----------------------------------------------------------------------
  // startListening
  // -----------------------------------------------------------------------
  async function startListening(onVolumeUpdate, onInterimResult) {
    cleanup();

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      throw new Error("Web Speech API not supported. Please use Chrome.");
    }

    // Set up mic volume meter (just for the visual indicator)
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      audioCtx = new AudioContext();
      const src = audioCtx.createMediaStreamSource(stream);
      analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      src.connect(analyser);
      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      function monitorVolume() {
        if (!analyser) return;
        analyser.getByteTimeDomainData(dataArray);
        const rms = getRMS(dataArray);
        if (onVolumeUpdate) onVolumeUpdate(Math.min(rms * 8, 1));
        animFrameId = requestAnimationFrame(monitorVolume);
      }
      animFrameId = requestAnimationFrame(monitorVolume);
    } catch(e) {
      console.warn("[VoiceRecorder] Volume monitor unavailable:", e);
    }

    // Set up live speech recognition
    recognition = new SpeechRecognition();
    recognition.continuous      = true;
    recognition.interimResults  = true;
    recognition.lang            = "en-US";
    recognition.maxAlternatives = 1;

    finalTranscript = "";

    return new Promise((resolve, reject) => {
      resolveCallback = resolve;
      forceStopped    = false;

      function finish() {
        const text = finalTranscript.trim();
        cleanup();
        resolve(text);
      }

      recognition.onresult = (event) => {
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += transcript + " ";
          } else {
            interim = transcript;
          }
        }

        // Show live text to user
        if (onInterimResult) {
          onInterimResult((finalTranscript + interim).trim());
        }

        // Reset silence timer on every new word
        if (silenceTimer) { clearTimeout(silenceTimer); silenceTimer = null; }
        silenceTimer = setTimeout(() => {
          if (finalTranscript.trim()) finish();
        }, SILENCE_DURATION);
      };

      recognition.onerror = (event) => {
        if (event.error === "no-speech" || event.error === "aborted") {
          finish();
        } else {
          console.error("[VoiceRecorder] Speech error:", event.error);
          finish();
        }
      };

      recognition.onend = () => {
        // Auto-restarts unless we force-stopped or have a silence timer pending
        if (!forceStopped && !silenceTimer) {
          try { recognition.start(); } catch(e) {}
        }
      };

      // Hard stop after MAX_RECORDING_TIME
      maxTimer = setTimeout(() => {
        console.warn("[VoiceRecorder] Max time reached — stopping");
        finish();
      }, MAX_RECORDING_TIME);

      recognition.start();
      console.log("[VoiceRecorder] Live recognition started");
    });
  }

  // -----------------------------------------------------------------------
  // stopListening — force-stop
  // -----------------------------------------------------------------------
  function stopListening() {
    const text = finalTranscript.trim();
    forceStopped = true;
    cleanup();
    if (resolveCallback) resolveCallback(text);
  }

  return { startListening, stopListening };
})();
