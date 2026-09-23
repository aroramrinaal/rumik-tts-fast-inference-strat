"use client";

import { useEffect, useRef, useState } from "react";
import { ApiError, pollJob, submitJob } from "@/lib/speech-client";
import { MAX_TEXT_LENGTH, type Speech } from "@/lib/speech";

import { AudioPlayer } from "@/components/audio-player";
import { DemoModal } from "@/components/demo-modal";

function RefLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  );
}

const SAMPLE_TEXT = "Hi, I am Rumik, a text-to-speech model built for India. I turn your words into speech across Indian languages and English.";

type PendingJob = { id: string; kind: "warmup" | "synthesize"; started: number };

export default function Home() {
  const [introOpen, setIntroOpen] = useState(true);
  const [text, setText] = useState(SAMPLE_TEXT);
  const [working, setWorking] = useState<"warmup" | "synthesize" | null>(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [speech, setSpeech] = useState<Speech | null>(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [wallSeconds, setWallSeconds] = useState<number | null>(null);
  const [pending, setPending] = useState<PendingJob | null>(null);
  const controller = useRef<AbortController | null>(null);
  const audioRef = useRef("");
  const active = useRef(false);
  const busy = working !== null;
  const textLength = Array.from(text.trim()).length;
  const tooLong = textLength > MAX_TEXT_LENGTH;

  useEffect(() => () => {
    controller.current?.abort();
    if (audioRef.current) URL.revokeObjectURL(audioRef.current);
  }, []);

  async function run(kind: "warmup" | "synthesize", existing?: PendingJob) {
    if (active.current || (!existing && kind === "synthesize" && (!textLength || tooLong))) return;
    active.current = true;
    const abort = new AbortController();
    controller.current = abort;
    const signal = AbortSignal.any([abort.signal, AbortSignal.timeout(20 * 60_000)]);
    const started = existing?.started ?? performance.now();
    setWorking(kind);
    setError("");
    setStatus(existing ? "Checking your existing request…" : kind === "warmup"
      ? "Starting the H100 and preparing the voice…" : "Sending your text to the H100…");
    try {
      const id = existing?.id ?? await submitJob(kind, text, signal);
      setPending({ id, kind, started });
      const result = await pollJob(id, signal, setStatus);
      if ((kind === "warmup") !== (result.kind === "warmup")) throw new Error("The service returned the wrong kind of result.");
      if (result.kind === "warmup") {
        setStatus(`GPU ready. Model setup took ${result.load_seconds.toFixed(1)}s and warmup ${result.warmup_seconds.toFixed(1)}s. It scales down after 2 minutes idle.`);
      } else {
        const bytes = Uint8Array.from(atob(result.audio_base64), (character) => character.charCodeAt(0));
        const nextUrl = URL.createObjectURL(new Blob([bytes], { type: "audio/wav" }));
        if (audioRef.current) URL.revokeObjectURL(audioRef.current);
        audioRef.current = nextUrl;
        setAudioUrl(nextUrl);
        setSpeech(result);
        setWallSeconds((performance.now() - started) / 1000);
        setStatus("Your speech is ready to play.");
      }
      setPending(null);
    } catch (cause) {
      if (!abort.signal.aborted) {
        setError(signal.aborted ? "This is taking longer than expected. Check the existing request again in a moment."
          : cause instanceof Error ? cause.message : "The request failed.");
        setStatus("Unable to retrieve your result.");
        if (cause instanceof ApiError && [400, 404, 410, 422, 500].includes(cause.status)) setPending(null);
      }
    } finally {
      active.current = false;
      if (!abort.signal.aborted) setWorking(null);
    }
  }

  return (
    <>
    <main className="page" inert={introOpen}>
      <header className="intro">
        <h1>Fast text<br className="mobile-break" /> to <span>speech.</span></h1>
        <p className="tagline">Rumik OSS 1 text-to-speech inference on an NVIDIA H100.</p>
        <p className="stack-summary">INT8 weight-only decode, fused Triton kernels, and GPU-controlled CUDA graphs.</p>
        <button type="button" className="demo-info" onClick={() => setIntroOpen(true)}>About this demo</button>
      </header>

      <section className="demo" aria-label="Text to speech">
        <form onSubmit={(event) => { event.preventDefault(); if (!pending) void run("synthesize"); }} aria-busy={busy}>
          <div className="input-heading">
            <label htmlFor="speech-text">Text to speak</label>
            <span id="text-count" className={tooLong ? "count is-invalid" : "count"}>{textLength} / {MAX_TEXT_LENGTH}</span>
          </div>
          <textarea id="speech-text" value={text} onChange={(event) => setText(event.target.value)}
            rows={6} required readOnly={busy || !!pending} aria-describedby="text-count"
            aria-invalid={tooLong} spellCheck placeholder={SAMPLE_TEXT} />
          <div className="actions">
            <button type="submit" className="primary" disabled={busy || !!pending || !textLength || tooLong}>
              {working === "synthesize" ? "Generating speech…" : "Generate speech"}<span aria-hidden="true">↗</span>
            </button>
            <button type="button" className="secondary" disabled={busy || !!pending} onClick={() => void run("warmup")}>
              {working === "warmup" ? "Warming up…" : "Warm up GPU"}
            </button>
          </div>
        </form>
        <div className={`request-status${busy ? " is-working" : ""}`} hidden={!status && !error && !pending}>
          {busy && <div className="activity" aria-hidden="true"><span /></div>}
          <p role="status" aria-live="polite">{status}</p>
          {error && <p className="error" role="alert">{error}</p>}
          {pending && !busy && <div className="recovery-actions">
            <button className="secondary" onClick={() => void run(pending.kind, pending)}>Check result again</button>
            <button className="text-button" onClick={() => { setPending(null); setError(""); setStatus("You can start a new request. The previous GPU job may still finish."); }}>Start over</button>
          </div>}
        </div>
      </section>

      {speech && <section className="result" aria-labelledby="result-title">
        <div className="section-heading"><h2 id="result-title">Your audio</h2><a className="download" href={audioUrl} download="rumik-speech.wav">Download WAV <span aria-hidden="true">↓</span></a></div>
        <AudioPlayer src={audioUrl} duration={speech.audio_seconds} />
        <dl className="metrics">
          <div><dt>Inference</dt><dd>{speech.inference_seconds.toFixed(3)} <span>s</span></dd></div>
          <div><dt>Audio length</dt><dd>{speech.audio_seconds.toFixed(2)} <span>s</span></dd></div>
          <div><dt>Real-time factor</dt><dd>{speech.rtf.toFixed(3)}</dd></div>
          <div><dt>Playback / inference</dt><dd>{speech.realtime_factor.toFixed(2)}<span>×</span></dd></div>
        </dl>
        <p className="measurement-note">Total browser wait: {wallSeconds?.toFixed(1)} s.</p>
      </section>}

      <section className="references" aria-labelledby="references-title">
        <h2 id="references-title">References</h2>
        <ul>
          <li>
            Original model: <RefLink href="https://huggingface.co/rumik-ai/rumik-oss-1">rumik-ai/rumik-oss-1</RefLink> on Hugging Face.
          </li>
          <li>
            Rumik research: <RefLink href="https://rumik.ai/research/rumik-oss">Introducing Rumik OSS 1</RefLink>.
          </li>
          <li>
            Base language model: <RefLink href="https://huggingface.co/CohereLabs/tiny-aya-fire">CohereLabs/tiny-aya-fire</RefLink>.
          </li>
          <li>
            Audio codec: <RefLink href="https://huggingface.co/kyutai/mimi">kyutai/mimi</RefLink>.
          </li>
        </ul>
      </section>
      <footer><a href="https://x.com/arora_mrinaal" target="_blank" rel="noopener noreferrer">by <span>@arora_mrinaal</span></a></footer>
    </main>
    {introOpen && <DemoModal onDismiss={() => setIntroOpen(false)} />}
    </>
  );
}
