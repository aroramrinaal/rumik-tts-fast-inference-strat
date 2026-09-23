import { apiError, proxyModal } from "@/lib/modal-api";
import { CALL_ID_PATTERN } from "@/lib/speech";

export async function GET(request: Request, { params }: { params: Promise<{ callId: string }> }) {
  const { callId } = await params;
  if (!CALL_ID_PATTERN.test(callId)) {
    return apiError("Invalid job ID.", 422);
  }
  return proxyModal(`/api/jobs/${encodeURIComponent(callId)}`, { signal: request.signal });
}
