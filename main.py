import os
import re
import shutil
import tempfile
import subprocess
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UnclearSegment(BaseModel):
    start: float
    end: float
    label: str = "Low volume / unclear"
    confidence: Optional[float] = None


class UnclearResponse(BaseModel):
    duration: float
    segments: List[UnclearSegment]


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }

    try:
        from database import db  # type: ignore

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:  # pragma: no cover - best effort
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"

    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:  # pragma: no cover - best effort
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"

    return response


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )


def ffprobe_duration(path: str) -> float:
    """Return media duration in seconds using ffprobe."""
    proc = _run([
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ])
    if proc.returncode != 0:
        # Try without select_streams in case only audio
        proc = _run([
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ])
    try:
        return float(proc.stdout.strip())
    except Exception:
        return 0.0


def detect_low_volume(path: str, noise_db: str = "-30dB", min_d: float = 0.4) -> List[tuple[float, float]]:
    """Use ffmpeg silencedetect to find low-volume regions considered 'unclear'."""
    args = [
        "ffmpeg",
        "-hide_banner",
        "-i",
        path,
        "-af",
        f"silencedetect=noise={noise_db}:d={min_d}",
        "-f",
        "null",
        "-",
    ]
    proc = _run(args)
    text = proc.stderr
    starts: List[float] = []
    ends: List[float] = []
    for line in text.splitlines():
        if "silence_start" in line:
            m = re.search(r"silence_start: ([-\d\.]+)", line)
            if m:
                starts.append(float(m.group(1)))
        elif "silence_end" in line:
            m = re.search(r"silence_end: ([-\d\.]+) \| silence_duration: ([-\d\.]+)", line)
            if m:
                ends.append(float(m.group(1)))
    # Pair starts and ends; if trailing start unmatched, close at duration later
    pairs: List[tuple[float, float]] = []
    while starts and ends:
        s = starts.pop(0)
        # find the first end after start
        e_candidates = [e for e in ends if e >= s]
        if not e_candidates:
            break
        e = e_candidates[0]
        # remove consumed ends
        idx = ends.index(e)
        ends = ends[idx + 1 :]
        pairs.append((s, e))
    return pairs


@app.post("/detect-unclear", response_model=UnclearResponse)
async def detect_unclear(file: UploadFile = File(...), noise_db: str = "-30dB", min_duration: float = 0.4):
    """Identify low-volume/unclear portions of the audio track.

    It leverages ffmpeg silencedetect with a configurable threshold. Returns segments
    where the audio falls below the threshold, which are usually the parts needing
    subtitles or volume boost.
    """
    # Ensure ffmpeg exists
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise HTTPException(status_code=500, detail="ffmpeg/ffprobe not available")

    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, file.filename or "input")
        content = await file.read()
        with open(dst, "wb") as f:
            f.write(content)
        duration = ffprobe_duration(dst)
        pairs = detect_low_volume(dst, noise_db=noise_db, min_d=min_duration)

    segments: List[UnclearSegment] = []
    for s, e in pairs:
        # clamp to duration and ignore absurd segments
        start = max(0.0, float(s))
        end = min(float(e), duration or float(e))
        if end - start <= 0.05:
            continue
        confidence = 1.0  # from silencedetect; higher certainty of low volume
        segments.append(UnclearSegment(start=start, end=end, confidence=confidence))

    return UnclearResponse(duration=duration, segments=segments)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
