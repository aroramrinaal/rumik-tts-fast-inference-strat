import { BASE_PATH } from "@/lib/config";
import { CALL_ID_PATTERN, DEFAULT_SEED, isJobResult } from "@/lib/speech";

export class ApiError extends Error {
  constructor(message: string, public status: number) {
    super(message);
  }
}

async function readResponse(response: Response): Promise<Record<string, unknown>> {
  let data: unknown;
  try {
    data = await response.json();
  } catch {
    throw new ApiError("The speech service returned an unreadable response. Please try again.", response.ok ? 502 : response.status);
  }
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    throw new ApiError("The speech service returned an unexpected response.", 502);
  }
  const record = data as Record<string, unknown>;
  if (!response.ok) {
    throw new ApiError(typeof record.message === "string" ? record.message
      : typeof record.detail === "string" ? record.detail : "The request failed. Please try again.", response.status);
  }
  return record;
}

function request(path: string, signal: AbortSignal, init?: RequestInit) {
  return fetch(`${BASE_PATH}/api/${path}`, {
    ...init, cache: "no-store", signal: AbortSignal.any([signal, AbortSignal.timeout(40_000)]),
  });
}

export async function submitJob(kind: "warmup" | "synthesize", text: string, signal: AbortSignal) {
  const response = await request(kind, signal, {
    method: "POST",
    headers: { "content-type": "application/json" },
    ...(kind === "synthesize" ? { body: JSON.stringify({ text: text.trim(), seed: DEFAULT_SEED }) } : {}),
  });
  const data = await readResponse(response);
  if (response.status !== 202 || data.status !== "accepted" || typeof data.call_id !== "string" || !CALL_ID_PATTERN.test(data.call_id)) {
    throw new ApiError("The speech service did not return a valid job. Please try again later.", 502);
  }
  return data.call_id;
}

function pause(ms: number, signal: AbortSignal) {
  signal.throwIfAborted();
  return new Promise<void>((resolve, reject) => {
    const onAbort = () => { clearTimeout(timer); reject(signal.reason); };
    const timer = setTimeout(() => { signal.removeEventListener("abort", onAbort); resolve(); }, ms);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export async function pollJob(callId: string, signal: AbortSignal, onProgress: (message: string) => void) {
  let failures = 0;
  while (true) {
    signal.throwIfAborted();
    let delay = 1500;
    try {
      const response = await request(`jobs/${encodeURIComponent(callId)}`, signal);
      const data = await readResponse(response);
      if (response.status === 200 && isJobResult(data)) return data;
      if (response.status !== 202 || data.status !== "running") {
        throw new ApiError("The speech service returned an unexpected job result.", 502);
      }
      if (typeof data.message === "string") onProgress(data.message);
      const retryAfter = Number(response.headers.get("retry-after"));
      delay = Math.min(10_000, Math.max(1500, retryAfter * 1000 || 1500));
      failures = 0;
    } catch (cause) {
      signal.throwIfAborted();
      // Only repeat reads. Retrying a submission could start a second GPU job.
      if (cause instanceof ApiError && ![429, 502, 503, 504].includes(cause.status)) throw cause;
      failures += 1;
      if (failures > 4) throw cause;
      onProgress("Reconnecting to your speech request…");
      delay = Math.min(10_000, 1500 * 2 ** failures);
    }
    await pause(delay, signal);
  }
}
