import { apiError, guardSubmission, proxyModal, readJsonBody } from "@/lib/modal-api";
import { DEFAULT_SEED, MAX_TEXT_LENGTH } from "@/lib/speech";

export async function POST(request: Request) {
  const rejection = await guardSubmission(request);
  if (rejection) return rejection;
  let payload: unknown;
  try {
    payload = await readJsonBody(request);
  } catch (cause) {
    return cause instanceof Response ? cause : apiError("Enter valid speech text.", 400);
  }
  if (!payload || typeof payload !== "object" || !("text" in payload)
    || Array.isArray(payload) || Object.keys(payload).some((key) => key !== "text" && key !== "seed")
    || typeof payload.text !== "string" || !payload.text.trim() || Array.from(payload.text.trim()).length > MAX_TEXT_LENGTH) {
    return apiError("Enter between 1 and 400 characters, with only text and an optional seed.", 422);
  }
  const seed = "seed" in payload ? payload.seed : DEFAULT_SEED;
  if (typeof seed !== "number" || !Number.isInteger(seed) || seed < 0 || seed > 2147483647) {
    return apiError("Enter a valid integer seed.", 422);
  }
  return proxyModal("/api/synthesize", {
    method: "POST", headers: { "content-type": "application/json" },
    body: JSON.stringify({ text: payload.text.trim(), seed }),
    signal: request.signal,
  });
}
