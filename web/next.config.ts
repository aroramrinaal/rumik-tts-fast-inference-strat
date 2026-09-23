import type { NextConfig } from "next";
import { initOpenNextCloudflareForDev } from "@opennextjs/cloudflare";
import { BASE_PATH } from "./lib/config";

const nextConfig: NextConfig = { basePath: BASE_PATH, reactStrictMode: true };
export default nextConfig;
initOpenNextCloudflareForDev();
