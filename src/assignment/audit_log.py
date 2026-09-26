"""
Assignment 11 — Audit Log starter (TODO).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store a pending request and return its stable request id."""
        rid = request_id or uuid4().hex
        self._open[rid] = {
            "request_id": rid,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "started_perf": perf_counter(),
        }
        return rid

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Finish a pending request and append a serializable audit entry."""
        rid = request_id
        if rid is None:
            # Prefer the most recent pending request for this user.
            rid = next(
                (
                    key for key, value in reversed(list(self._open.items()))
                    if value.get("user_id") == user_id
                ),
                None,
            )

        pending = self._open.pop(rid, None) if rid else None
        ended_perf = perf_counter()
        entry = {
            "request_id": rid or uuid4().hex,
            "user_id": user_id,
            "input": pending.get("input", "") if pending else "",
            "output": text,
            "blocked": bool(blocked),
            "layer": layer,
            "started_at": pending.get("started_at") if pending else None,
            "completed_at": utc_now_iso(),
            "latency_ms": (
                round((ended_perf - pending["started_perf"]) * 1000, 3)
                if pending else 0.0
            ),
        }
        self.logs.append(entry)
        return entry

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.logs, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
