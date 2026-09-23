import "server-only";
import { getCloudflareContext } from "@opennextjs/cloudflare";

export function apiError(message: string, status: number, headers?: HeadersInit) {
  const responseHeaders = new Headers(headers);
  responseHeaders.set("cache-control", "no-store");
  return Response.json({ message }, { status, headers: responseHeaders });
}

export async function readJsonBody(request: Request): Promise<unknown> {
  if (request.headers.get("content-type")?.split(";")[0].trim().toLowerCase() !== "application/json") {
    throw apiError("Send a JSON request.", 415);
  }
  const limit = 4096;
  if (Number(request.headers.get("content-length")) > limit) throw apiError("Request too large.", 413);
  const reader = request.body?.getReader();
  if (!reader) throw apiError("Enter valid speech text.", 400);
  let size = 0;
  let body = "";
  const decoder = new TextDecoder("utf-8", { fatal: true });
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > limit) {
        await reader.cancel();
        throw apiError("Request too large.", 413);
      }
      body += decoder.decode(value, { stream: true });
    }
    return JSON.parse(body + decoder.decode());
  } catch (cause) {
    if (cause instanceof Response) throw cause;
    throw apiError("Enter valid speech text.", 400);
  } finally {
    reader.releaseLock();
  }
}

export async function proxyModal(path: string, init?: RequestInit) {
  try {
    const { env } = await getCloudflareContext({ async: true });
    const endpoint = env.MODAL_ENDPOINT || process.env.MODAL_ENDPOINT;
    const tokenId = env.MODAL_PROXY_TOKEN_ID || process.env.MODAL_PROXY_TOKEN_ID;
    const tokenSecret = env.MODAL_PROXY_TOKEN_SECRET || process.env.MODAL_PROXY_TOKEN_SECRET;
    if (!endpoint || !tokenId || !tokenSecret) {
      return apiError("The speech backend has not been configured yet.", 503);
    }
    const headers = new Headers(init?.headers);
    headers.set("Modal-Key", tokenId);
    headers.set("Modal-Secret", tokenSecret);
    const upstream = await fetch(`${endpoint.replace(/\/$/, "")}${path}`, {
      // Workers rejects redirect: "error" before sending the request.
      // Handle redirects explicitly so Modal credentials stay on this host.
      ...init, headers, cache: "no-store", redirect: "manual",
      signal: AbortSignal.any([AbortSignal.timeout(30_000), ...(init?.signal ? [init.signal] : [])]),
    });
    if (upstream.status >= 300 && upstream.status < 400) {
      await upstream.body?.cancel();
      return apiError("The speech backend returned an unexpected redirect. Check the configured endpoint.", 503);
    }
    if (!upstream.headers.get("content-type")?.includes("application/json")) {
      await upstream.body?.cancel();
      return apiError("The speech backend is unavailable. Please try again shortly.", 503);
    }
    const responseHeaders = new Headers({ "content-type": "application/json", "cache-control": "no-store" });
    const retryAfter = upstream.headers.get("retry-after");
    if (retryAfter) responseHeaders.set("retry-after", retryAfter);
    return new Response(upstream.body, {
      status: upstream.status,
      headers: responseHeaders,
    });
  } catch (cause) {
    if (cause instanceof Error && cause.name === "TimeoutError") {
      return apiError("The speech backend took too long to respond.", 504);
    }
    return apiError("Unable to reach the speech backend.", 503);
  }
}

export async function guardSubmission(request: Request) {
  const origin = request.headers.get("origin");
  if (request.headers.get("sec-fetch-site") === "cross-site" || (origin && origin !== new URL(request.url).origin)) {
    return apiError("Cross-origin requests are not supported.", 403);
  }
  try {
    const { env } = await getCloudflareContext({ async: true });
    const { success } = await env.TTS_RATE_LIMITER.limit({
      key: request.headers.get("cf-connecting-ip") || "local-preview",
    });
    if (!success) {
      return apiError("Please wait a minute before starting another GPU job.", 429, { "retry-after": "60" });
    }
  } catch {
    return apiError("GPU request limiting is temporarily unavailable.", 503);
  }
  return null;
}
