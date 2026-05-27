"""FastAPI server for SkillOpt Web Console and REST API."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from skillopt.library.catalog import SkillLibrary
from skillopt.runner import (
    load_run_log,
    load_run_summary,
    result_to_dict,
    run_ab_compare,
    run_evaluation,
    run_optimization,
    run_transfer,
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web" / "static"
DATA_DIR = Path("data")
RUNS_DIR = DATA_DIR / "runs"
LIBRARY_DIR = DATA_DIR / "skill_library"

app = FastAPI(title="SkillOpt API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

RUNS_DIR.mkdir(parents=True, exist_ok=True)
LIBRARY_DIR.mkdir(parents=True, exist_ok=True)

_jobs: dict[str, dict[str, Any]] = {}
_library = SkillLibrary(LIBRARY_DIR)


class OptimizeRequest(BaseModel):
    config_path: str = "examples/demo_qa/config.yaml"


class EvaluateRequest(BaseModel):
    skill_path: str
    dataset_path: str
    harness: str = "direct_chat"
    model: str = "mock"


class TransferRequest(BaseModel):
    skill_path: str
    dataset_path: str
    harness: str = "direct_chat"
    model: str = "mock"
    baseline_skill_path: str | None = None


class CompareRequest(BaseModel):
    skill_a_path: str
    skill_b_path: str
    dataset_path: str
    harness: str = "direct_chat"
    model: str = "mock"


class LibraryAddRequest(BaseModel):
    skill_path: str
    name: str
    domain: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    benchmark: str = ""
    harness: str = "direct_chat"
    model: str = "mock"
    score: float = 0.0


class ReviewRequest(BaseModel):
    status: str = "reviewed"
    reviewer: str = ""


def _run_job(job_id: str, config_path: Path) -> None:
    try:
        _jobs[job_id]["status"] = "running"
        result = run_optimization(config_path)
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = result_to_dict(result)
        _jobs[job_id]["artifacts_dir"] = str(Path(result.best_skill_path).parent)
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}


@app.post("/api/optimize")
def start_optimize(req: OptimizeRequest, background: BackgroundTasks) -> dict:
    config_path = Path(req.config_path)
    if not config_path.exists():
        raise HTTPException(404, f"Config not found: {config_path}")

    job_id = str(uuid.uuid4())[:8]
    _jobs[job_id] = {"id": job_id, "status": "queued", "config": req.config_path}

    def task():
        _run_job(job_id, config_path)

    background.add_task(task)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return list(_jobs.values())


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")
    job = dict(_jobs[job_id])
    artifacts = job.get("artifacts_dir")
    if artifacts:
        job["summary"] = load_run_summary(Path(artifacts))
        job["log"] = load_run_log(Path(artifacts))
    return job


@app.post("/api/evaluate")
def evaluate(req: EvaluateRequest) -> dict:
    return run_evaluation(
        Path(req.skill_path),
        Path(req.dataset_path),
        req.harness,
        req.model,
    )


@app.post("/api/transfer")
def transfer(req: TransferRequest) -> dict:
    return run_transfer(
        Path(req.skill_path),
        Path(req.dataset_path),
        req.harness,
        req.model,
        Path(req.baseline_skill_path) if req.baseline_skill_path else None,
    )


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict:
    return run_ab_compare(
        Path(req.skill_a_path),
        Path(req.skill_b_path),
        Path(req.dataset_path),
        req.harness,
        req.model,
    )


@app.get("/api/library")
def library_list(domain: str | None = None, status: str | None = None) -> list[dict]:
    return [e.to_dict() for e in _library.list(domain=domain, status=status)]


@app.post("/api/library")
def library_add(req: LibraryAddRequest) -> dict:
    entry = _library.add(
        Path(req.skill_path),
        name=req.name,
        domain=req.domain,
        description=req.description,
        tags=req.tags,
        benchmark=req.benchmark,
        harness=req.harness,
        model=req.model,
        score=req.score,
    )
    return entry.to_dict()


@app.post("/api/library/{skill_id}/review")
def library_review(skill_id: str, req: ReviewRequest) -> dict:
    try:
        entry = _library.review(skill_id, req.status, req.reviewer)
        return entry.to_dict()
    except KeyError:
        raise HTTPException(404, "Skill not found")


@app.get("/api/library/{skill_id}/export")
def library_export(skill_id: str) -> FileResponse:
    dest = RUNS_DIR / f"{skill_id}.md"
    try:
        _library.export(skill_id, dest)
        return FileResponse(dest, filename=f"{skill_id}.md")
    except KeyError:
        raise HTTPException(404, "Skill not found")


@app.get("/api/benchmarks")
def list_benchmarks() -> list[dict]:
    benchmarks_dir = Path("benchmarks")
    if not benchmarks_dir.exists():
        return []
    result = []
    for preset_dir in sorted(benchmarks_dir.iterdir()):
        if preset_dir.is_dir() and (preset_dir / "config.yaml").exists():
            result.append({
                "id": preset_dir.name,
                "config": str(preset_dir / "config.yaml"),
                "description": (preset_dir / "README.md").read_text(encoding="utf-8")[:200]
                if (preset_dir / "README.md").exists()
                else "",
            })
    return result


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")


def serve(host: str = "0.0.0.0", port: int = 8080) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)
