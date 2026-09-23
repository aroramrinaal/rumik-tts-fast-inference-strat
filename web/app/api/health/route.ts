import { proxyModal } from "@/lib/modal-api";

export async function GET(request: Request) {
  return proxyModal("/api/health", { signal: request.signal });
}
