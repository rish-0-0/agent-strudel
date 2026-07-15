import { generate } from "./model.js";

postMessage({ type: "status", status: "ready" });

onmessage = async ({ data }) => {
  if (data.type !== "generate") return;
  try {
    const code = await generate(data.prompt ?? "");
    postMessage({ type: "result", code });
  } catch (err) {
    postMessage({ type: "error", message: err.message });
  }
};
