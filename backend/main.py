import os
import tempfile
import subprocess
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    import imageio_ffmpeg as ioff
except Exception:  # pragma: no cover
    ioff = None

app = FastAPI(title="AI Video Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Highlight(BaseModel):
    start: float
    end: float
    label: str
    score: Optional[float] = None


class ExtractResponse(BaseModel):
    duration: float
    highlights: List[Highlight]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-highlights", response_model=ExtractResponse)
async def extract_highlights(
    file: UploadFile = File(...),
    desired_segments: int = Form(6),
):
    if ioff is None:
        raise HTTPException(status_code=500, detail="ffmpeg not available")

    # Save incoming file to a temp path
    suffix = os.path.splitext(file.filename or "video.mp4")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        duration = probe_duration(tmp_path)
        if duration is None:
            raise HTTPException(status_code=400, detail="Unable to probe duration")

        # Detect silences to infer active/high-energy regions
        silences = detect_silences(tmp_path)
        active = build_active_regions(duration, silences)
        # Score by length; clamp to reasonable clip size and spread
        segments = pick_segments(active, desired_segments, min_len=8.0, max_len=20.0, total_duration=duration)
        highlights = [
            Highlight(
                start=s,
                end=e,
                label="High energy segment",
                score=(e - s),
            )
            for (s, e) in segments
        ]
        return ExtractResponse(duration=duration, highlights=highlights)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def ffmpeg_paths():
    ffmpeg = ioff.get_ffmpeg_exe()
    ffprobe = ioff.get_ffprobe_exe()
    return ffmpeg, ffprobe


def probe_duration(path: str) -> Optional[float]:
    _, ffprobe = ffmpeg_paths()
    try:
        # Get duration in seconds
        out = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stderr=subprocess.STDOUT,
        )
        dur = float(out.decode().strip())
        if dur and dur > 0:
            return dur
        # Fallback: stream duration
        out = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            stderr=subprocess.STDOUT,
        )
        return float(out.decode().strip())
    except Exception:
        return None


def detect_silences(path: str):
    ffmpeg, _ = ffmpeg_paths()
    # Use silencedetect to find silence periods (threshold -30dB, min duration 0.5s)
    # Output is printed to stderr; parse lines with "silence_start" and "silence_end"
    cmd = [
        ffmpeg,
        "-i",
        path,
        "-af",
        "silencedetect=n=-30dB:d=0.5",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        text = proc.stderr.decode(errors="ignore")
    except subprocess.CalledProcessError as e:
        # even when finishing OK, ffmpeg returns non-zero for -f null sometimes; still parse stderr
        text = (e.stderr or b"").decode(errors="ignore")
    silences = []
    start = None
    for line in text.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                start = float(line.split("silence_start:")[-1].strip())
            except Exception:
                start = None
        elif "silence_end:" in line and start is not None:
            try:
                parts = line.split("silence_end:")[-1].strip().split("|")
                end = float(parts[0].strip())
                silences.append((start, end))
            except Exception:
                pass
            start = None
    return merge_intervals(silences)


def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = merged[-1]
        if s <= le + 0.1:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def build_active_regions(duration: float, silences):
    if not silences:
        return [(0.0, duration)]
    regions = []
    cursor = 0.0
    for s, e in silences:
        if s > cursor:
            regions.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration:
        regions.append((cursor, duration))
    # filter very short noise
    return [(s, e) for s, e in regions if (e - s) >= 1.5]


def pick_segments(active, k: int, min_len: float, max_len: float, total_duration: float):
    if not active:
        return []
    # Score by length
    scored = sorted(active, key=lambda ab: ab[1] - ab[0], reverse=True)
    chosen = []
    for seg in scored:
        if len(chosen) >= k:
            break
        s, e = seg
        length = e - s
        # clamp
        if length > max_len:
            # center crop
            mid = (s + e) / 2
            s = max(0.0, mid - max_len / 2)
            e = min(total_duration, s + max_len)
        elif length < min_len:
            # expand where possible
            pad = (min_len - length) / 2
            s = max(0.0, s - pad)
            e = min(total_duration, e + pad)
        # ensure ordering and within bounds
        s = max(0.0, min(s, total_duration))
        e = max(0.0, min(e, total_duration))
        if e - s >= 1.0:
            chosen.append((s, e))
    # If not enough, sample evenly across duration
    while len(chosen) < k:
        idx = len(chosen) + 1
        center = total_duration * idx / (k + 1)
        s = max(0.0, center - max_len / 2)
        e = min(total_duration, s + max_len)
        chosen.append((s, e))
    # sort by start
    chosen = sorted(chosen, key=lambda x: x[0])
    return chosen[:k]


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}


@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": [],
    }
    try:
        from database import db

        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, "name") else "✅ Connected"
            response["connection_status"] = "Connected"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"

    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    return response


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
