"""Small, credential-free progress records for the speech UI."""

from datetime import datetime, timezone


PHASE_MESSAGES = {
    "accepted": "Your request is waiting for the H100.",
    "warming": "Starting the H100 and preparing the voice. A cold start can take a few minutes.",
    "queued": "Waiting for the H100 to finish another request.",
    "generating": "Turning your text into speech.",
    "encoding": "Preparing your WAV file.",
    "processing": "Your request is still processing.",
    "complete": "Your result is ready.",
}


def progress_record(phase):
    return {
        "phase": phase,
        "message": PHASE_MESSAGES.get(phase, PHASE_MESSAGES["processing"]),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def pending_progress(progress, snapshot):
    # The result can take a moment to become readable after the worker finishes.
    phase = progress.get("phase") if isinstance(progress, dict) else None
    if phase == "complete":
        return {**progress, "phase": "processing", "message": "Retrieving your result."}
    if phase in {"generating", "encoding"}:
        return progress
    if snapshot.get("gpu") == "warming":
        return progress_record("warming")
    if snapshot.get("backlog", 0) > 0:
        return progress_record("queued")
    return progress_record("processing")
