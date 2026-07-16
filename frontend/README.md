# Agent DJ — Frontend

Click **Generate** to produce Strudel live-coding code, then press **Play** inside the editor to hear it. Playback is always user-initiated (no autoplay).

The editor is the full Strudel REPL (`@strudel/repl`'s `<strudel-editor>` web component), so it loads the complete default sample/synth bank set on its own — no manual sample loading, and AI-generated patterns that use any standard sound will just work.

For now the "model" is a stub that picks a random pattern (see `model.js`). The rest of the pipeline is real and ready for the trained model.

## Run it

Strudel + Web Workers need a real HTTP origin, so don't open the file directly. From this directory:

    python -m http.server 8000

then open http://localhost:8000

(Or, with Node installed: `npx serve`)

## Files

- `index.html` — the UI: a Generate button, a status line, and the `<strudel-editor>` (code + Play/Stop/eval).
- `app.js` — main thread. Talks to the worker, feeds the generated code to the editor. No audio/sample code — the editor handles all of that.
- `worker.js` — runs the model off the main thread so generation never blocks the UI. Receives `generate` requests, replies with code.
- `model.js` — THE SWAP POINT. Exports `generate(prompt) -> Promise<string>`. Today it returns a random hardcoded pattern. Replace its body with a Transformers.js call to use the trained model.
- `styles.css` — minimal dark styling.

## How it flows

    [Generate click]
        |
    app.js  --{generate}-->  worker.js  -->  model.js (stub / future model)
        |
    app.js  <--{result, code}--  worker.js
        |
    app.js  -->  editor.code = code / editor.editor.setCode(code)

    [Play click (in the editor)]
        |
    <strudel-editor>  -->  evaluate(code)  -->  WebAudio  -->  speakers

The editor loads `@strudel/repl` from a CDN; there is no build step.

## Plugging in the real model

1. In `worker.js`, import Transformers.js from a CDN and load your ONNX model at startup, posting `status` messages as it loads:

       import { pipeline } from "https://esm.sh/@huggingface/transformers";

2. Replace `generate()` in `model.js` with a call that runs the model and returns the decoded Strudel string.
3. AI-generated code will sometimes be invalid. A validation/regeneration loop (try-transpile before setting code; retry on failure) should be added at that point.

Nothing else changes — the worker / main-thread boundary already isolates inference.

## Why `@strudel/repl` (and not `@strudel/web`)

`@strudel/repl`'s `<strudel-editor>` runs the full Strudel prebake on connect, which loads every default sample bank (uzu-drumkit, Dirt-Samples, tidal-drum-machines, piano, vcsl, mridangam), soundfonts, and synth sounds. That breadth matters once the model emits arbitrary code — `@strudel/web` registers synths only and would need a hand-maintained bank list that misses sounds. The tradeoff is the editor UI, which we want anyway to show the generated code.

## Notes

- Audio resumes on the editor's Play click (an explicit user gesture), so autoplay restrictions don't apply.
- `@strudel/repl` is loaded via `<script defer src="https://unpkg.com/@strudel/repl@1.3.0">`, which auto-registers the `<strudel-editor>` custom element. If audio is silent, fall back to the jsDelivr prebuilt ESM: `import "https://cdn.jsdelivr.net/npm/@strudel/repl@1.3.0/dist/index.mjs"` (same worklet-inlining fix that `@strudel/web` needed).
- The `@strudel/*` packages are AGPL-3.0; fine for local use, but review it if you ever distribute or host the app.
