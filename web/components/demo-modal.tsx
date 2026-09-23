"use client";

import { useEffect, useId, useRef } from "react";
import { WRITE_UP_URL } from "@/lib/config";

type DemoModalProps = { onDismiss: () => void };

// Adapted from Before It Codes' analysis-modal and mobile-analysis-modal.
export function DemoModal({ onDismiss }: DemoModalProps) {
  const titleId = useId();
  const descriptionId = useId();
  const card = useRef<HTMLElement>(null);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    document.body.style.overflow = "hidden";
    card.current?.focus();
    return () => {
      document.body.style.overflow = previousOverflow;
      const target = previousFocus && previousFocus !== document.body && previousFocus.isConnected
        ? previousFocus : document.getElementById("speech-text");
      target?.focus({ preventScroll: true });
    };
  }, []);

  return (
    <div className="demo-modal-backdrop" onKeyDown={(event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onDismiss();
      }
      if (event.key !== "Tab") return;
      const buttons = card.current?.querySelectorAll<HTMLElement>("button:not(:disabled), a[href]");
      if (!buttons?.length) return;
      const first = buttons[0];
      const last = buttons[buttons.length - 1];
      if (event.shiftKey && (document.activeElement === first || document.activeElement === card.current)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }}>
      <section ref={card} className="demo-modal-card" role="dialog" aria-modal="true"
        aria-labelledby={titleId} aria-describedby={descriptionId} tabIndex={-1}>
        <header className="demo-modal-header">
          <h2 id={titleId}>About this demo</h2>
          <button type="button" className="demo-modal-close" aria-label="Close introduction" onClick={onDismiss}>×</button>
        </header>
        <p id={descriptionId} className="demo-modal-description">Pushing text-to-speech inference on an NVIDIA H100.</p>
        <div className="demo-modal-stack">
          <h3>Inference stack</h3>
          <p>INT8 weight-only decode, fused Triton kernels, Hopper programmatic dependent launch, and a GPU-controlled CUDA WHILE graph. A BF16 phase-aware audio head feeds the Mimi waveform decoder. Served one request at a time on Modal, behind a Next.js / Cloudflare Worker frontend.</p>
        </div>
        <div className="demo-modal-notes">
          <div>
            <h3>GPU startup</h3>
            <p>Generate speech directly, or warm up the GPU first. The first request can take a few minutes while the H100 loads the model. It scales down after 2 minutes idle.</p>
          </div>
          <div>
            <h3>Inference timing</h3>
            <p>Measures text to the completed CPU waveform. Model loading, warmup, WAV encoding, queueing, and network time are excluded.</p>
          </div>
        </div>
        <div className="demo-modal-actions">
          <a href={WRITE_UP_URL}>Read write-up</a>
          <button type="button" onClick={onDismiss}>Open demo</button>
        </div>
      </section>
    </div>
  );
}
