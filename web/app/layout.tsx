import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Rumik — Fast text to speech",
  description: "Generate speech on an H100 and measure the complete inference pipeline.",
  authors: [{ name: "Mrinaal Arora", url: "https://x.com/arora_mrinaal" }],
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
