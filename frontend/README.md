# Agent DJ — Frontend

Click **Generate** to produce Strudel live-coding code, then click **Play** to hear it. Playback is always user-initiated (no autoplay), so there are no browser audio restrictions to fight.

For now the "model" is a stub that picks a random pattern (see `model.js`). The rest of the pipeline is real and ready for the trained model.

## Run it

Strudel + Web Workers need a real HTTP origin, so don't open the file directly. From this directory:

    python -m http.server 8000

then open http://localhost:8000

(Or, with Node installed: `npx serve`)

## Files

- `index.html` — the UI: Generate, Play, and Stop buttons, a status line, and a panel showing the generated code.
- `app.js` — main thread. Initialises Strudel, talks to the worker, shows the result, and plays it on demand.
- `worker.js` — runs the model off the main thread so generation never blocks audio/UI. Receives `generate` requests, replies with code.
- `model.js` — THE SWAP POINT. Exports `generate(prompt) -> Promise<string>`. Today it returns a random hardcoded pattern. Replace its body with a Transformers.js call to use the trained model.
- `styles.css` — minimal dark styling.

## How it flows

    [Generate click]
        |
    app.js  --{generate}-->  worker.js  -->  model.js (stub / future model)
        |
    app.js  <--{result, code}--  worker.js      (code shown; nothing plays yet)

    [Play click]
        |
    app.js  -->  resume AudioContext + evaluate(code)  -->  Strudel WebAudio  -->  speakers

Strudel itself is loaded from a CDN; there is no build step.

## Plugging in the real model

1. In `worker.js`, import Transformers.js from a CDN and load your ONNX model at startup, posting `status` messages as it loads:

       import { pipeline } from "https://esm.sh/@huggingface/transformers";

2. Replace `generate()` in `model.js` with a call that runs the model and returns the decoded Strudel string.
3. Optionally validate the output (the `try/catch` around `evaluate` in `app.js` already surfaces broken code) and regenerate on failure.

Nothing else changes — the worker / main-thread boundary already isolates inference.

## Why `@strudel/web` and not `@strudel/repl`

`@strudel/web` is the lean "evaluate-and-play" bundle — exactly what a few-button app needs. `@strudel/repl` adds the full CodeMirror editor + visuals as a `<strudel-editor>` web component; it's the upgrade path if you later want users to see/edit the code inline. To switch: load `https://unpkg.com/@strudel/repl@1.3.0`, drop in `<strudel-editor code="...">`, and drive it via `.editor.setCode()` / `.editor.evaluate()`.

## Notes

- Audio resumes on the Play click (an explicit user gesture), so autoplay restrictions don't apply.
- Drum samples (`bd`, `sd`, `hh`, `cp`, …) aren't bundled with `@strudel/web` — its default prebake registers synths only. `app.js` passes a `prebake` to `initStrudel()` that loads sample banks from GitHub (same sources `@strudel/repl` uses): the classic drum hits come from `tidalcycles/uzu-drumkit` (`strudel.json`); `felixroos/dough-samples` adds Dirt-Samples, drum machines, and piano. First Play waits for these to finish loading.
- Strudel is loaded from jsDelivr's **prebuilt** `dist/index.mjs`, which inlines the AudioWorklet as a Blob URL. Do **not** swap this for `esm.sh/@strudel/web` — esm.sh rebuilds from source and drops the worklet, leaving you with no sound. If jsDelivr misbehaves, the unpkg global build (`<script src="https://unpkg.com/@strudel/web@1.3.0">`, then `initStrudel` / `evaluate` / `hush` as globals) is the fallback.
- The `@strudel/*` packages are AGPL-3.0; fine for local use, but review it if you ever distribute or host the app.
