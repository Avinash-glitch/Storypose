/**
 * tts.js — Text-to-Speech wrapper for speaking to the child.
 *
 * Exposes:
 *   speakToChild(text)  → Promise<void>  (resolves when utterance ends)
 *   stopSpeaking()      → void
 */

const TTS = (() => {
  const synth = window.speechSynthesis;

  /** Cancel any current speech immediately. */
  function stopSpeaking() {
    synth.cancel();
  }

  /**
   * Speak a string to the child in a warm, friendly voice.
   * Returns a Promise that resolves when speech finishes.
   */
  function speakToChild(text) {
    return new Promise((resolve) => {
      stopSpeaking();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate   = 0.85;  // slightly slower — easy for kids
      utterance.pitch  = 1.2;   // a little higher — friendly tone
      utterance.volume = 1.0;

      // Prefer a female/child-friendly English voice when available
      const voices = synth.getVoices();
      const preferred = voices.find(
        (v) =>
          v.lang.startsWith("en") &&
          (v.name.toLowerCase().includes("female") ||
           v.name.toLowerCase().includes("girl") ||
           v.name.toLowerCase().includes("zira") ||  // Windows
           v.name.toLowerCase().includes("samantha")) // macOS
      );
      if (preferred) utterance.voice = preferred;

      utterance.onend   = () => resolve();
      utterance.onerror = () => resolve(); // don't block app on TTS error

      synth.speak(utterance);
    });
  }

  // Voices may load asynchronously; trigger a warm-up
  if (typeof speechSynthesis !== "undefined") {
    speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
    speechSynthesis.getVoices();
  }

  return { speakToChild, stopSpeaking };
})();
