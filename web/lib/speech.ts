export const MAX_TEXT_LENGTH = 400;
export const DEFAULT_SEED = 42;
export const CALL_ID_PATTERN = /^fc-[A-Za-z0-9_-]{3,128}$/;

export type Warmup = {
  status: "complete";
  kind: "warmup";
  load_seconds: number;
  warmup_seconds: number;
};

export type Speech = {
  status: "complete";
  kind: "speech";
  audio_base64: string;
  audio_seconds: number;
  inference_seconds: number;
  rtf: number;
  realtime_factor: number;
};

export function isJobResult(value: Record<string, unknown>): value is Record<string, unknown> & (Warmup | Speech) {
  if (value.status !== "complete") return false;
  const finite = (key: string) => typeof value[key] === "number" && Number.isFinite(value[key]) && value[key] >= 0;
  if (value.kind === "warmup") return finite("load_seconds") && finite("warmup_seconds");
  return value.kind === "speech" && typeof value.audio_base64 === "string" && value.audio_base64.length > 0
    && finite("audio_seconds") && finite("inference_seconds") && finite("rtf") && finite("realtime_factor");
}
