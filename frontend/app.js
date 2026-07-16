const worker = new Worker("worker.js", { type: "module" });

const generateBtn = document.getElementById("generate");
const playBtn = document.getElementById("play");
const stopBtn = document.getElementById("stop");
const statusEl = document.getElementById("status");
const editor = document.getElementById("editor");

let generating = false;

const setStatus = (text) => { statusEl.textContent = text; };

const setEditorCode = (code) => {
  editor.code = code;
  editor.editor?.setCode(code);
};

const whenEditorReady = (cb) => {
  if (editor.editor) return cb();
  setTimeout(() => whenEditorReady(cb), 50);
};

whenEditorReady(() => {
  playBtn.disabled = false;
  stopBtn.disabled = false;
});

worker.onmessage = ({ data }) => {
  if (data.type === "status") {
    const ready = data.status === "ready";
    generateBtn.disabled = !ready || generating;
    setStatus(ready ? "Ready" : (data.detail ?? "Loading model…"));
  } else if (data.type === "result") {
    setEditorCode(data.code);
    generating = false;
    generateBtn.disabled = false;
    setStatus("Generated — press Play");
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

playBtn.addEventListener("click", () => {
  if (!editor.editor) return;
  editor.editor.evaluate();
  setStatus("Playing");
});

stopBtn.addEventListener("click", () => {
  if (!editor.editor) return;
  editor.editor.stop();
  setStatus("Stopped");
});
