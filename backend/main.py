from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import io

app = FastAPI(title="Video AI Backend")

# Allow CORS for the frontend dev server
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


class HighlightsResponse(BaseModel):
    duration: float
    moments: List[Highlight]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract-highlights", response_model=HighlightsResponse)
async def extract_highlights(
    file: UploadFile = File(...),
    desired_segments: int = Form(6),
):
    # Read file to ensure upload succeeds
    data = await file.read()
    size_mb = len(data) / (1024 * 1024)

    # Mock duration based on file size (avoid heavy deps like ffprobe in this template)
    # Assume ~ 5 minutes per 50MB as a rough placeholder
    duration = max(60.0, min(45 * 60.0, (size_mb / 50.0) * 5 * 60.0))

    # Generate evenly spaced moments
    n = max(1, min(12, int(desired_segments)))
    segment = duration / (n + 1)
    moments: List[Highlight] = []
    for i in range(1, n + 1):
        start = max(0.0, segment * i - 5.0)
        end = min(duration, start + 12.0)
        label = [
            "Peak action",
            "Clutch play",
            "Combo chain",
            "Boss phase",
            "Crowd hype",
            "Insane aim",
        ][(i - 1) % 6]
        moments.append(Highlight(start=start, end=end, label=label))

    return HighlightsResponse(duration=duration, moments=moments)
