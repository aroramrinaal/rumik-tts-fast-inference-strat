import { proxyModal } from "@/lib/modal-api";

export async function GET(request: Request) {
  return proxyModal("/api/status", { signal: request.signal });
}
