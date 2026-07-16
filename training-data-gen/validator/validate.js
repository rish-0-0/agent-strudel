// Browser-environment shims so the browser-only @strudel/web bundle imports in Node.
// Goal: detect whether a snippet would throw in the browser. No real audio is produced.

if (!globalThis.window) globalThis.window = globalThis;
if (!globalThis.addEventListener) globalThis.addEventListener = () => {};
if (!globalThis.removeEventListener) globalThis.removeEventListener = () => {};
if (!globalThis.navigator) globalThis.navigator = { userAgent: "node" };
if (!globalThis.requestAnimationFrame) globalThis.requestAnimationFrame = (cb) => setTimeout(() => cb(Date.now()), 0);
if (!globalThis.cancelAnimationFrame) globalThis.cancelAnimationFrame = (id) => clearTimeout(id);
if (!globalThis.URL.createObjectURL) globalThis.URL.createObjectURL = () => "blob:stub";

if (!globalThis.document) {
  const el = () => ({ style: {}, getContext: () => ({}), appendChild() {}, removeChild() {}, addEventListener() {}, removeEventListener() {}, setAttribute() {}, querySelector: () => null, querySelectorAll: () => [] });
  globalThis.document = {
    createElement: el,
    createTextNode: () => ({}),
    addEventListener() {},
    removeEventListener() {},
    dispatchEvent() {},
    querySelector: () => null,
    querySelectorAll: () => [],
    body: el(),
    documentElement: el(),
    head: el(),
  };
}

// Permissive AudioContext: any node method returns a chainable no-op node.
const node = new Proxy(function () {}, {
  get: (t, p) => {
    if (p === Symbol.toPrimitive) return () => 0;
    if (p === "then") return undefined;
    return () => node;
  },
});
const ctx = new Proxy({}, {
  get: (t, p) => {
    if (p === "currentTime") return 0;
    if (p === "state") return "running";
    if (p === "sampleRate") return 44100;
    if (p === "destination") return node;
    if (p === "listener") return node;
    if (p === "audioWorklet") return { addModule: () => Promise.resolve() };
    if (p === "resume" || p === "close" || p === "suspend") return () => Promise.resolve();
    if (p === "decodeAudioData") return () => Promise.resolve({ getChannelData: () => new Float32Array(1), duration: 0, numberOfChannels: 1, length: 1, sampleRate: 44100 });
    return () => node;
  },
});
if (!globalThis.BaseAudioContext) globalThis.BaseAudioContext = class BaseAudioContext {};
if (!globalThis.AudioContext) {
  globalThis.AudioContext = class AudioContext extends globalThis.BaseAudioContext { constructor() { super(); return ctx; } };
  globalThis.OfflineAudioContext = globalThis.AudioContext;
}
for (const name of ["AudioWorkletNode","AudioBufferSourceNode","GainNode","OscillatorNode","BiquadFilterNode","StereoPannerNode","WaveShaperNode","ConstantSourceNode","ChannelMergerNode","ChannelSplitterNode","PannerNode","AnalyserNode","DelayNode","ConvolverNode","DynamicsCompressorNode","IIRFilterNode","AudioDestinationNode","AudioListener","AudioParam","AudioParamMap","PeriodicWave","AudioBuffer","AudioNode","MediaStreamAudioDestinationNode"]) {
  if (!globalThis[name]) globalThis[name] = class { constructor() {} };
}

// Swallow strudel's stdout/stderr noise; capture the "[eval] error:" signal it logs on failure.
let evalError = null;
const origStdoutWrite = process.stdout.write.bind(process.stdout);
const origStderrWrite = process.stderr.write.bind(process.stderr);
const swallow = (chunk) => {
  const s = typeof chunk === "string" ? chunk : chunk.toString();
  const m = s.match(/\[eval\]\s+error:\s*(.*)/i);
  if (m) evalError = (m[1] || s).trim();
  return true;
};
process.stdout.write = swallow;
process.stderr.write = swallow;

const { initStrudel, evaluate } = await import("@strudel/web/dist/index.mjs");

let ready;
const ensureInit = () => {
  if (!ready) ready = initStrudel();
  return ready;
};

let code = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (c) => (code += c));
process.stdin.on("end", async () => {
  try {
    await ensureInit();
    evalError = null;
    await evaluate(code.trim(), false);
  } catch (e) {
    evalError = e && e.message ? e.message : String(e);
  }
  process.stdout.write = origStdoutWrite;
  process.stderr.write = origStderrWrite;
  origStdoutWrite(evalError ? "ERROR: " + evalError + "\n" : "OK\n");
  process.exit(0);
});
