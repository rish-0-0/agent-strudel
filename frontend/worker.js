import { init, generate } from "./model.js";

init((status) => {
  postMessage({
    type: "status",
    status: status === "ready" ? "ready" : "loading",
    detail: status,
  });
});

onmessage = async ({ data }) => {
  if (data.type !== "generate") return;
  try {
    const code = await generate(data.prompt ?? "");
    postMessage({ type: "result", code });
  } catch (err) {
    postMessage({ type: "error", message: err.message });
  }
};
