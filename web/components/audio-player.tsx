"use client";

import { useRef, useState, type CSSProperties } from "react";
import styles from "./audio-player.module.css";

type AudioPlayerProps = { src: string; duration: number };

function timestamp(seconds: number) {
  const whole = Math.floor(Number.isFinite(seconds) ? Math.max(0, seconds) : 0);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

// Remount for a new recording so playback state never leaks between results.
export function AudioPlayer(props: AudioPlayerProps) {
  return <RecordingPlayer key={props.src} {...props} />;
}

function RecordingPlayer({ src, duration: initialDuration }: AudioPlayerProps) {
  const audio = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [starting, setStarting] = useState(false);
  const [ready, setReady] = useState(false);
  const [muted, setMuted] = useState(false);
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(Number.isFinite(initialDuration) ? Math.max(0, initialDuration) : 0);
  const [error, setError] = useState("");
  const progress = duration > 0 ? Math.min(100, (position / duration) * 100) : 0;

  async function togglePlayback() {
    const element = audio.current;
    if (!element || starting) return;
    if (!element.paused) {
      element.pause();
      return;
    }
    setError("");
    setStarting(true);
    try {
      if (element.ended) element.currentTime = 0;
      await element.play();
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === "AbortError")) {
        setError("Unable to play this recording. Try again or download the WAV.");
      }
    } finally {
      setStarting(false);
    }
  }

  return (
    <div className={styles.player} role="group" aria-label="Generated speech playback">
      <audio ref={audio} src={src} preload="metadata" hidden
        onLoadedMetadata={(event) => {
          const length = event.currentTarget.duration;
          if (Number.isFinite(length) && length > 0) {
            setDuration(length);
            setReady(true);
          }
        }}
        onTimeUpdate={(event) => setPosition(event.currentTarget.currentTime)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={(event) => { setPlaying(false); setPosition(event.currentTarget.currentTime); }}
        onVolumeChange={(event) => setMuted(event.currentTarget.muted)}
        onError={() => { setPlaying(false); setReady(false); setStarting(false); setError("Unable to load this recording. Download the WAV to listen."); }}
      />
      <div className={styles.controls}>
        <button type="button" className={styles.play} onClick={() => void togglePlayback()}
          disabled={starting} aria-label={playing ? "Pause audio" : "Play audio"} aria-busy={starting}>
          <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
            {playing ? <path d="M6 4h4v16H6zM14 4h4v16h-4z" /> : <path d="m8 4 12 8-12 8z" />}
          </svg>
        </button>
        <span className={styles.time} aria-hidden="true">{timestamp(position)}</span>
        <input className={styles.seek} type="range" min={0} max={duration || 1} step={0.01}
          value={Math.min(position, duration)} disabled={!ready}
          aria-label="Playback position" aria-valuetext={`${timestamp(position)} of ${timestamp(duration)}`}
          style={{ "--playback-progress": `${progress}%` } as CSSProperties}
          onChange={(event) => {
            if (!audio.current || !ready) return;
            const nextPosition = Number(event.target.value);
            audio.current.currentTime = nextPosition;
            setPosition(nextPosition);
          }} />
        <span className={styles.time} aria-hidden="true">{timestamp(duration)}</span>
        <button type="button" className={styles.mute} aria-label={muted ? "Unmute audio" : "Mute audio"}
          onClick={() => { if (audio.current) audio.current.muted = !audio.current.muted; }}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M11 5 6 9H3v6h3l5 4V5Z" />
            {muted ? <path d="m16 9 6 6m0-6-6 6" /> : <><path d="M15 8a6 6 0 0 1 0 8" /><path d="M18 5a10 10 0 0 1 0 14" /></>}
          </svg>
        </button>
      </div>
      {error && <p className={styles.error} role="alert">{error}</p>}
    </div>
  );
}
