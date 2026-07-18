import { pipeline, env } from "https://esm.sh/@huggingface/transformers";

env.allowLocalModels = true;

const MODEL_ID = "smollm2-strudel";
const SYSTEM_PROMPT =
  "You are a Strudel live-coding assistant. Given a description of a groove, " +
  "write complete, valid, balanced Strudel (strudel.cc) code that produces it. " +
  "Output ONLY raw Strudel code — no markdown, no code fences, no explanation. " +
  "Ensure every parenthesis, bracket, brace, and string quote is matched and " +
  "the snippet is syntactically complete and runnable as-is.";

let generator = null;

export async function init(onStatus) {
  if (generator) return;
  onStatus?.("Loading model…");
  generator = await pipeline("text-generation", MODEL_ID, {
    dtype: "q8",
    device: "wasm",
    progress_callback: (p) => {
      if (p.status === "progress" && p.file && p.total) {
        const pct = Math.round((p.loaded / p.total) * 100);
        onStatus?.(`Loading ${p.file.split("/").pop()}… ${pct}%`);
      }
    },
  });
  onStatus?.("ready");
}

// Strip markdown fences and close any delimiters left open by truncation,
// so the snippet is always syntactically complete enough to evaluate.
function cleanCode(raw) {
  let code = raw.trim();
  const fence = code.match(/^```[a-zA-Z]*\s*\n([\s\S]*)$/);
  if (fence) code = fence[1];
  code = code.replace(/\n?```\s*$/, "");

  const stack = [];
  let inString = false;
  let escape = false;
  for (let i = 0; i < code.length; i++) {
    const c = code[i];
    if (escape) { escape = false; continue; }
    if (inString) {
      if (c === "\\") { escape = true; continue; }
      if (c === '"') inString = false;
      continue;
    }
    if (c === '"') { inString = true; continue; }
    if (c === "/" && code[i + 1] === "/") {
      while (i < code.length && code[i] !== "\n") i++;
      continue;
    }
    if (c === "(" || c === "[" || c === "{") stack.push(c);
    else if (c === ")" && stack[stack.length - 1] === "(") stack.pop();
    else if (c === "]" && stack[stack.length - 1] === "[") stack.pop();
    else if (c === "}" && stack[stack.length - 1] === "{") stack.pop();
  }
  let suffix = "";
  if (inString) suffix += '"';
  const close = { "(": ")", "[": "]", "{": "}" };
  for (let i = stack.length - 1; i >= 0; i--) suffix += close[stack[i]];
  return suffix ? code + suffix : code;
}

export async function generate(prompt = "") {
  if (!generator) throw new Error("model not loaded");
  const userMsg = prompt.trim()
    ? `Write Strudel code for: ${prompt}`
    : "Write a Strudel groove";
  const messages = [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: userMsg },
  ];
  const out = await generator(messages, {
    max_new_tokens: 512,
    do_sample: true,
    temperature: 0.8,
    top_p: 0.95,
  });
  const text = out[0].generated_text;
  const last = Array.isArray(text) ? text.at(-1) : { content: text };
  return cleanCode(last.content);
}
