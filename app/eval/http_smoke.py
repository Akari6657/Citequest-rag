"""Real HTTP acceptance smoke for the local CiteQuest demo service."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from app.core.config import DEFAULT_RAG_TOP_K
from app.eval.runtime_info import collect_accelerator_info


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEARCH_MODES = ("lexical", "vector", "hybrid")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _git_value(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return {}
    return body if isinstance(body, dict) else {}


def _tail(path: Path, lines: int = 40) -> list[str]:
    if not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return []


def _base_report(
    *,
    status: str,
    db_path: Path,
    index_dir: Path,
    log_path: Path,
    base_url: str,
    query: str,
    top_k: int,
    errors: list[str],
    endpoints: dict[str, Any] | None = None,
    startup_ms: float | None = None,
    server_stopped: bool = True,
    server_exit_code: int | None = None,
) -> dict[str, Any]:
    git_status = _git_value("status", "--porcelain")
    return {
        "schema_version": 1,
        "report_type": "demo_http_smoke",
        "status": status,
        "official_benchmark": False,
        "quality_metrics_computed": False,
        "created_at": _utc_now(),
        "configuration": {
            "database_file": _display_path(db_path),
            "index_dir": _display_path(index_dir),
            "base_url": base_url,
            "query": query,
            "top_k": top_k,
            "server_log": _display_path(log_path),
            "llm_mode": "forced_mock",
            "external_llm_called": False,
        },
        "provenance": {
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_dirty": bool(git_status),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "accelerator": collect_accelerator_info(),
        },
        "server": {
            "startup_ms": round(startup_ms, 3) if startup_ms is not None else None,
            "stopped": server_stopped,
            "exit_code_after_shutdown": server_exit_code,
        },
        "endpoints": endpoints or {},
        "errors": errors,
        "server_log_tail": _tail(log_path) if errors else [],
    }


def run_http_smoke(
    *,
    db_path: str | Path,
    index_dir: str | Path,
    log_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    query: str = "machine learning",
    top_k: int = 5,
    alpha: float = 0.5,
    startup_timeout: float = 60.0,
    request_timeout: float = 180.0,
) -> dict[str, Any]:
    """Start Uvicorn, exercise real HTTP routes, then stop it cleanly."""
    db_path = Path(db_path).resolve()
    index_dir = Path(index_dir).resolve()
    log_path = Path(log_path).resolve()
    base_url = f"http://{host}:{port}"
    errors: list[str] = []
    endpoints: dict[str, Any] = {}

    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    required = (
        db_path,
        index_dir / "index.faiss",
        index_dir / "id_map.json",
        index_dir / "build_meta.json",
    )
    missing = [_display_path(path) for path in required if not path.is_file()]
    if missing:
        errors.append(f"Missing required artifacts: {', '.join(missing)}")
        return _base_report(
            status="failed",
            db_path=db_path,
            index_dir=index_dir,
            log_path=log_path,
            base_url=base_url,
            query=query,
            top_k=top_k,
            errors=errors,
        )
    if not _port_available(host, port):
        errors.append(f"Port is already in use: {host}:{port}")
        return _base_report(
            status="failed",
            db_path=db_path,
            index_dir=index_dir,
            log_path=log_path,
            base_url=base_url,
            query=query,
            top_k=top_k,
            errors=errors,
        )

    env = os.environ.copy()
    env.update(
        {
            "CITEQUEST_DB_PATH": str(db_path),
            "CITEQUEST_FAISS_DIR": str(index_dir),
            "CITEQUEST_HYBRID_ALPHA": str(alpha),
            "LLM_API_KEY": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        "info",
    ]
    process: subprocess.Popen | None = None
    client: httpx.Client | None = None
    startup_ms: float | None = None
    server_stopped = False
    server_exit_code: int | None = None

    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "w", encoding="utf-8") as server_log:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        timeout = httpx.Timeout(request_timeout, connect=5.0)
        client = httpx.Client(base_url=base_url, timeout=timeout)
        started = time.perf_counter()
        deadline = started + startup_timeout
        health_response: httpx.Response | None = None
        while time.perf_counter() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Uvicorn exited during startup with code {process.returncode}"
                )
            try:
                candidate = client.get("/health")
                # Any non-server-error response proves Uvicorn is accepting
                # requests. Route correctness is asserted immediately below.
                if candidate.status_code < 500:
                    health_response = candidate
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        if health_response is None:
            raise TimeoutError(f"Uvicorn did not become ready within {startup_timeout}s")
        startup_ms = (time.perf_counter() - started) * 1000

        def request(method: str, path: str, **kwargs: Any) -> tuple[httpx.Response, float]:
            request_started = time.perf_counter()
            response = client.request(method, path, **kwargs)
            return response, (time.perf_counter() - request_started) * 1000

        health_body = _response_json(health_response)
        health_passed = (
            health_body.get("status") == "healthy"
            and health_body.get("capabilities", {}).get("hybrid_search") is True
        )
        endpoints["health"] = {
            "passed": health_passed,
            "status_code": health_response.status_code,
            "service_status": health_body.get("status"),
            "capabilities": health_body.get("capabilities"),
        }
        if not health_passed:
            errors.append("GET /health did not report healthy hybrid retrieval")

        response, wall_ms = request("GET", "/openapi.json")
        openapi_body = _response_json(response)
        expected_paths = {"/health", "/search", "/ask"}
        observed_paths = set(openapi_body.get("paths", {}))
        openapi_passed = response.status_code == 200 and expected_paths <= observed_paths
        endpoints["openapi"] = {
            "passed": openapi_passed,
            "status_code": response.status_code,
            "wall_latency_ms": round(wall_ms, 3),
            "expected_paths_present": expected_paths <= observed_paths,
        }
        if not openapi_passed:
            errors.append("GET /openapi.json is missing required routes")

        response, wall_ms = request("GET", "/")
        frontend_passed = response.status_code == 200 and "CiteQuest" in response.text
        endpoints["frontend"] = {
            "passed": frontend_passed,
            "status_code": response.status_code,
            "wall_latency_ms": round(wall_ms, 3),
        }
        if not frontend_passed:
            errors.append("GET / did not serve the CiteQuest frontend")

        hybrid_results: list[dict[str, Any]] = []
        for mode in SEARCH_MODES:
            response, wall_ms = request(
                "POST",
                "/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "mode": mode,
                    "include_overview": False,
                },
            )
            body = _response_json(response)
            results = body.get("results", [])
            result_count = len(results) if isinstance(results, list) else 0
            expected_alpha = alpha if mode == "hybrid" else None
            passed = (
                response.status_code == 200
                and body.get("mode") == mode
                and body.get("effective_alpha") == expected_alpha
                and result_count > 0
            )
            endpoints[f"search_{mode}"] = {
                "passed": passed,
                "status_code": response.status_code,
                "result_count": result_count,
                "effective_alpha": body.get("effective_alpha"),
                "server_latency_ms": body.get("latency_ms"),
                "wall_latency_ms": round(wall_ms, 3),
            }
            if not passed:
                errors.append(f"POST /search mode={mode} failed or returned no results")
            if mode == "hybrid" and isinstance(results, list):
                hybrid_results = results

        if hybrid_results:
            rag_top_k = min(top_k, len(hybrid_results), DEFAULT_RAG_TOP_K)
            response, wall_ms = request(
                "POST",
                "/ask",
                json={
                    "question": "What do the retrieved papers study?",
                    "top_k": rag_top_k,
                    "retrieval_mode": "hybrid",
                    "pre_retrieved": hybrid_results[:rag_top_k],
                },
            )
            body = _response_json(response)
            citations = body.get("citations", [])
            citation_count = len(citations) if isinstance(citations, list) else 0
            passed = (
                response.status_code == 200
                and bool(body.get("answer"))
                and citation_count > 0
                and body.get("citation_valid") is True
                and body.get("effective_alpha") == alpha
            )
            endpoints["ask_mock_rag"] = {
                "passed": passed,
                "status_code": response.status_code,
                "citation_count": citation_count,
                "citation_valid": body.get("citation_valid"),
                "effective_alpha": body.get("effective_alpha"),
                "server_latency_ms": body.get("latency_ms"),
                "wall_latency_ms": round(wall_ms, 3),
                "external_llm_called": False,
            }
            if not passed:
                errors.append("POST /ask did not return a valid mock citation")
        else:
            endpoints["ask_mock_rag"] = {
                "passed": False,
                "skipped": True,
                "external_llm_called": False,
            }
            errors.append("POST /ask skipped because hybrid search returned no evidence")
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        if client is not None:
            client.close()
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            server_exit_code = process.returncode
            server_stopped = process.poll() is not None
            if not server_stopped:
                errors.append("Uvicorn process did not stop cleanly")

    status = "passed" if not errors and all(
        endpoint.get("passed") for endpoint in endpoints.values()
    ) else "failed"
    return _base_report(
        status=status,
        db_path=db_path,
        index_dir=index_dir,
        log_path=log_path,
        base_url=base_url,
        query=query,
        top_k=top_k,
        errors=errors,
        endpoints=endpoints,
        startup_ms=startup_ms,
        server_stopped=server_stopped,
        server_exit_code=server_exit_code,
    )


def render_http_smoke_markdown(report: dict[str, Any]) -> str:
    accelerator = report.get("provenance", {}).get("accelerator", {})
    device = accelerator.get("device_name") or "CPU"
    lines = [
        "# CiteQuest 10k Real HTTP Smoke Report",
        "",
        "> Operational HTTP smoke only; this is not Retrieval Benchmark v1.",
        "",
        f"**Status:** `{report['status']}`  ",
        f"**Created:** `{report['created_at']}`  ",
        f"**Base URL:** `{report['configuration']['base_url']}`  ",
        f"**Server stopped:** `{report['server']['stopped']}`  ",
        f"**Accelerator:** `{device}`",
        "",
        "| HTTP check | Result | Status | Wall latency (ms) |",
        "|---|---|---:|---:|",
    ]
    for name, endpoint in report.get("endpoints", {}).items():
        lines.append(
            f"| `{name}` | {'PASS' if endpoint.get('passed') else 'FAIL'} | "
            f"{endpoint.get('status_code', 'n/a')} | "
            f"{endpoint.get('wall_latency_ms', 'n/a')} |"
        )
    if report.get("errors"):
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Uvicorn was started as a real subprocess and every check used HTTP over localhost.",
            "- The LLM API key was forced empty, so `/ask` used `MockLLMProvider` and made no external request.",
            "- Search latency in this report includes cold model/index loading where applicable and is not a quality metric.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_http_smoke_outputs(
    report: dict[str, Any],
    *,
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    _atomic_write_text(
        Path(json_path),
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    )
    _atomic_write_text(Path(markdown_path), render_http_smoke_markdown(report))
