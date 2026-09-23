import { guardSubmission, proxyModal } from "@/lib/modal-api";

export async function POST(request: Request) {
  const rejection = await guardSubmission(request);
  if (rejection) return rejection;
  return proxyModal("/api/warmup", { method: "POST", signal: request.signal });
}
