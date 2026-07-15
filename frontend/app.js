import * as strudel from "https://cdn.jsdelivr.net/npm/@strudel/web@1.3.0/dist/index.mjs";

const DOUGH = "https://raw.githubusercontent.com/felixroos/dough-samples/main";
const UZU = "https://raw.githubusercontent.com/tidalcycles/uzu-drumkit/main";

const worker = new Worker("worker.js", { type: "module" });

const generateBtn = document.getElementById("generate");
const playBtn = document.getElementById("play");
const stopBtn = document.getElementById("stop");
const statusEl = document.getElementById("status");
const codeEl = document.getElementById("code");

const strudelReady = strudel.initStrudel({
  prebake: async () => {
    await Promise.all([
      strudel.samples(`${UZU}/strudel.json`),
      strudel.samples(`${DOUGH}/Dirt-Samples.json`),
      strudel.samples(`${DOUGH}/tidal-drum-machines.json`),
      strudel.samples(`${DOUGH}/piano.json`),
    ]);
  },
});
let generating = false;
let currentCode = "";

const setStatus = (text) => { statusEl.textContent = text; };

worker.onmessage = async ({ data }) => {
  if (data.type === "status") {
    const ready = data.status === "ready";
    generateBtn.disabled = !ready || generating;
    setStatus(ready ? "Ready" : (data.detail ?? "Loading model…"));
  } else if (data.type === "result") {
    currentCode = data.code;
    codeEl.textContent = currentCode;
    playBtn.disabled = false;
    generating = false;
    generateBtn.disabled = false;
    setStatus("Ready to play");
  } else if (data.type === "error") {
    setStatus("Error: " + data.message);
    generating = false;
    generateBtn.disabled = false;
  }
};

generateBtn.addEventListener("click", () => {
  if (generating) return;
  generating = true;
  generateBtn.disabled = true;
  setStatus("Generating…");
  worker.postMessage({ type: "generate", prompt: "" });
});

playBtn.addEventListener("click", async () => {
  if (!currentCode) return;
  playBtn.disabled = true;
  setStatus("Loading audio…");
  try {
    await strudelReady;
    strudel.getAudioContext?.()?.resume?.();
    await strudel.evaluate(currentCode);
    setStatus("Playing");
  } catch (err) {
    setStatus("Playback error: " + err.message);
  } finally {
    playBtn.disabled = false;
  }
});

stopBtn.addEventListener("click", () => {
  strudel.hush();
  setStatus("Stopped");
});
