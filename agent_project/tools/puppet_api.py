"""HTTP tools for an external image-generation service.

Talks to the service over HTTP only and never imports its implementation.
A tool never fabricates success: any non-done state is surfaced as an error
or an explicit pending status.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx

from .. import config

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


class PuppetAPIError(RuntimeError):
    """Generation-service failure with a user-facing Chinese message."""


# ---------------------------------------------------------------------------
# Error translation for the public service contract
# ---------------------------------------------------------------------------

def _translate_http_error(exc: httpx.HTTPStatusError) -> PuppetAPIError:
    status = exc.response.status_code
    detail = ""
    try:
        body = exc.response.json()
        if isinstance(body.get("detail"), str):
            detail = body["detail"]
    except Exception:
        pass
    if status == 429:
        msg = "请求过于频繁：生成接口每分钟限 10 次，请稍候再试。"
    elif status == 503:
        msg = "皮影生成服务的模型尚未加载或服务不可用，请稍后再试。"
    elif status == 404:
        msg = "未找到对应的任务或资源。"
    elif status == 400:
        msg = f"请求参数有误：{detail}" if detail else "请求参数有误，请检查输入。"
    else:
        msg = f"生成服务异常（HTTP {status}），请稍后重试。"
    return PuppetAPIError(msg)


def _request(method: str, path: str, *, timeout: float, **kwargs) -> dict[str, Any]:
    url = f"{config.PUPPET_API_BASE}{path}"
    try:
        resp = httpx.request(method, url, timeout=timeout, **kwargs)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise _translate_http_error(exc) from exc
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise PuppetAPIError(
            f"无法连接皮影生成服务（{config.PUPPET_API_BASE}），请确认后端已启动。"
        ) from exc
    return resp.json()


def _absolute_url(maybe_relative: str | None) -> str | None:
    if maybe_relative and maybe_relative.startswith("/"):
        return f"{config.PUPPET_API_BASE}{maybe_relative}"
    return maybe_relative


def _absolutize_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if not result:
        return result
    # Keep the public Agent response limited to product-level result fields.
    # Service-specific previews and score breakdowns stay server-side.
    allowed = {"selected_url", "final_url", "candidate_urls", "selected_index", "elapsed_seconds"}
    out = {key: value for key, value in result.items() if key in allowed}
    for key in ("selected_url", "final_url"):
        out[key] = _absolute_url(out.get(key))
    out["candidate_urls"] = [_absolute_url(u) for u in out.get("candidate_urls", [])]
    return out


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def check_health() -> dict[str, Any]:
    """GET /api/health — model_loaded, gpu_name, queue_length, running_task_id."""
    return _request("GET", "/api/health", timeout=10)


def generate_shadow_puppet(
    prompt: str,
    lineart_path: str,
    negative_prompt: str | None = None,
    k: int = config.DEFAULT_K,
    steps: int = config.DEFAULT_STEPS,
    guidance_scale: float = config.DEFAULT_GUIDANCE_SCALE,
    conditioning_scale: float = config.DEFAULT_CONDITIONING_SCALE,
    seed: int = config.DEFAULT_SEED,
    postprocess: bool = config.DEFAULT_POSTPROCESS,
) -> dict[str, Any]:
    """POST /api/generate. Returns {task_id, position}.

    Validates the lineart locally first (mirrors backend limits) so the agent
    can fail fast with an actionable message instead of a remote 400.
    """
    if not prompt or not prompt.strip():
        raise PuppetAPIError("提示词不能为空。")
    if not 1 <= k <= 8:
        raise PuppetAPIError("候选数量 k 需在 1～8 之间。")
    if not 1 <= steps <= 150:
        raise PuppetAPIError("采样步数需在 1～150 之间。")

    lineart = Path(lineart_path)
    if not lineart.is_file():
        raise PuppetAPIError(f"线稿文件不存在：{lineart_path}。请先上传线稿。")
    if lineart.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise PuppetAPIError("线稿格式不支持，请使用 PNG / JPEG / WEBP / BMP。")
    if lineart.stat().st_size > MAX_UPLOAD_BYTES:
        raise PuppetAPIError("线稿文件超过 5MB 限制，请压缩后再上传。")

    files: dict[str, Any] = {"lineart": (lineart.name, lineart.read_bytes())}
    data = {
        "prompt": prompt.strip(),
        "k": k,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "conditioning_scale": conditioning_scale,
        "seed": seed,
        "postprocess": postprocess,
    }
    if negative_prompt and negative_prompt.strip():
        data["negative_prompt"] = negative_prompt.strip()

    return _request(
        "POST",
        "/api/generate",
        timeout=config.PUPPET_SUBMIT_TIMEOUT,
        files=files,
        data=data,
    )


def get_generation_status(task_id: str) -> dict[str, Any]:
    """GET /api/tasks/{task_id}; image URLs converted to absolute URLs."""
    if not task_id or not task_id.strip():
        raise PuppetAPIError("task_id 不能为空。")
    payload = _request("GET", f"/api/tasks/{task_id.strip()}", timeout=10)
    payload["result"] = _absolutize_result(payload.get("result"))
    return payload


def get_generation_history(limit: int = 10, offset: int = 0) -> dict[str, Any]:
    """GET /api/history; thumbnail URLs converted to absolute URLs."""
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    payload = _request("GET", "/api/history", timeout=10, params={"limit": limit, "offset": offset})
    for item in payload.get("items", []):
        item["thumbnail_url"] = _absolute_url(item.get("thumbnail_url"))
    return payload
