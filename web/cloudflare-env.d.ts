interface CloudflareEnv {
  MODAL_ENDPOINT: string;
  MODAL_PROXY_TOKEN_ID: string;
  MODAL_PROXY_TOKEN_SECRET: string;
  TTS_RATE_LIMITER: { limit(options: { key: string }): Promise<{ success: boolean }> };
}
