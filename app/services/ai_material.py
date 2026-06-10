import base64
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import requests
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip

from app.config import config
from app.models.schema import VideoAspect
from app.services import llm
from app.services.material import _get_tls_verify
from app.utils import utils


def _api_setting(name: str, fallback: str = "") -> str:
    value = config.app.get(name, fallback)
    return str(value).strip() if value is not None else ""


def _api_key() -> str:
    api_key = _api_setting("ai_media_api_key") or _api_setting("openai_api_key")
    if not api_key:
        raise ValueError(
            "ai_media_api_key is not set; configure it or openai_api_key in config.toml"
        )
    return api_key


def _base_url() -> str:
    return (
        _api_setting("ai_media_base_url")
        or _api_setting("openai_base_url")
        or "https://api.openai.com/v1"
    ).rstrip("/")


def _endpoint(path_setting: str, default_path: str, **values) -> str:
    path = _api_setting(path_setting, default_path).format(**values)
    return f"{_base_url()}/{path.lstrip('/')}"


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {_api_key()}"}


def _response_json(response: requests.Response) -> dict[str, Any]:
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("AI media provider returned a non-object JSON response")
    return payload


def _extract_json_array(response: str) -> list[dict[str, str]]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        start = response.find("[")
        end = response.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("scene planner did not return a JSON array")
        value = json.loads(response[start : end + 1])

    if not isinstance(value, list):
        raise ValueError("scene planner response is not a JSON array")

    scenes = []
    for item in value:
        if not isinstance(item, dict):
            continue
        image_prompt = str(item.get("image_prompt", "")).strip()
        motion_prompt = str(item.get("motion_prompt", "")).strip()
        if image_prompt:
            scenes.append(
                {
                    "image_prompt": image_prompt,
                    "motion_prompt": motion_prompt
                    or "Subtle natural cinematic camera movement.",
                }
            )
    if not scenes:
        raise ValueError("scene planner returned no usable scenes")
    return scenes


def generate_scene_plan(
    video_subject: str,
    video_script: str,
    scene_count: int,
    video_aspect: VideoAspect,
) -> list[dict[str, str]]:
    prompt = f"""
# Role: Storyboard planner for an image-to-video production

Create exactly {scene_count} chronological scenes that visually follow the voiceover.
Every scene must be strictly relevant to a concrete part of the voiceover.
Keep recurring people, locations, era, color palette, lighting, and visual style coherent
across all scenes. Repeat the same detailed continuity description in every image_prompt
so each image can be generated independently without losing visual consistency.
Do not include text, captions, logos, watermarks, or split screens.
The requested aspect ratio is {video_aspect}.

Return only a JSON array. Each item must contain:
- "image_prompt": a detailed standalone prompt for one coherent still image
- "motion_prompt": concise natural subject and camera motion for animating that image

Video subject:
{video_subject}

Complete voiceover:
{video_script}
""".strip()
    return _extract_json_array(llm._generate_response(prompt))


def _image_size(video_aspect: VideoAspect) -> str:
    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.landscape:
        return _api_setting("ai_image_size_landscape", "1536x1024")
    if aspect == VideoAspect.square:
        return _api_setting("ai_image_size_square", "1024x1024")
    return _api_setting("ai_image_size_portrait", "1024x1536")


def generate_image(prompt: str, output_path: str, video_aspect: VideoAspect) -> str:
    payload = {
        "model": _api_setting("ai_image_model_name", "gpt-image-1"),
        "prompt": prompt,
        "n": 1,
        "size": _image_size(video_aspect),
    }
    response = requests.post(
        _endpoint("ai_image_generation_path", "/images/generations"),
        headers={**_headers(), "Content-Type": "application/json"},
        json=payload,
        proxies=config.proxy,
        verify=_get_tls_verify(),
        timeout=(30, int(config.app.get("ai_media_request_timeout", 600))),
    )
    data = _response_json(response).get("data") or []
    if not data or not isinstance(data[0], dict):
        raise ValueError("image provider returned no image")

    image = data[0]
    if image.get("b64_json"):
        content = base64.b64decode(image["b64_json"])
    elif image.get("url"):
        download = requests.get(
            image["url"],
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, int(config.app.get("ai_media_request_timeout", 600))),
        )
        download.raise_for_status()
        content = download.content
    else:
        raise ValueError("image provider returned neither b64_json nor url")

    Path(output_path).write_bytes(content)
    if not os.path.getsize(output_path):
        raise ValueError("image provider returned an empty image")
    return output_path


def _video_size(video_aspect: VideoAspect) -> str:
    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.landscape:
        return _api_setting("ai_video_size_landscape", "1280x720")
    if aspect == VideoAspect.square:
        return _api_setting("ai_video_size_square", "720x1280")
    return _api_setting("ai_video_size_portrait", "720x1280")


def _video_duration(requested_duration: int) -> int:
    supported = config.app.get("ai_video_supported_durations", [4, 8, 12])
    if not isinstance(supported, list) or not supported:
        return requested_duration

    durations = sorted(
        {
            int(duration)
            for duration in supported
            if str(duration).strip().isdigit() and int(duration) > 0
        }
    )
    if not durations:
        return requested_duration
    return next(
        (duration for duration in durations if duration >= requested_duration),
        durations[-1],
    )


def _video_id(payload: dict[str, Any]) -> str:
    return str(payload.get("id") or payload.get("video_id") or "").strip()


def _video_status(payload: dict[str, Any]) -> str:
    return str(payload.get("status") or payload.get("state") or "").lower().strip()


def _download_video(video_id: str, output_path: str) -> str:
    response = requests.get(
        _endpoint("ai_video_content_path", "/videos/{video_id}/content", video_id=video_id),
        headers=_headers(),
        proxies=config.proxy,
        verify=_get_tls_verify(),
        timeout=(30, int(config.app.get("ai_media_request_timeout", 600))),
    )
    response.raise_for_status()
    Path(output_path).write_bytes(response.content)

    clip = None
    try:
        clip = VideoFileClip(output_path)
        if clip.duration <= 0 or clip.fps <= 0:
            raise ValueError("generated video has invalid duration or frame rate")
    finally:
        if clip is not None:
            clip.close()
    return output_path


def generate_video(
    image_path: str,
    motion_prompt: str,
    output_path: str,
    duration: int,
    video_aspect: VideoAspect,
) -> str:
    with open(image_path, "rb") as image_file:
        response = requests.post(
            _endpoint("ai_video_generation_path", "/videos"),
            headers=_headers(),
            data={
                "model": _api_setting("ai_video_model_name", "sora-2"),
                "prompt": motion_prompt,
                "seconds": str(_video_duration(duration)),
                "size": _video_size(video_aspect),
            },
            files={"input_reference": (os.path.basename(image_path), image_file, "image/png")},
            proxies=config.proxy,
            verify=_get_tls_verify(),
            timeout=(30, int(config.app.get("ai_media_request_timeout", 600))),
        )
    video_id = _video_id(_response_json(response))
    if not video_id:
        raise ValueError("video provider returned no video id")

    poll_interval = max(1, int(config.app.get("ai_video_poll_interval", 10)))
    poll_timeout = max(poll_interval, int(config.app.get("ai_video_poll_timeout", 1800)))
    deadline = time.monotonic() + poll_timeout
    while time.monotonic() < deadline:
        status_payload = _response_json(
            requests.get(
                _endpoint("ai_video_status_path", "/videos/{video_id}", video_id=video_id),
                headers=_headers(),
                proxies=config.proxy,
                verify=_get_tls_verify(),
                timeout=(30, int(config.app.get("ai_media_request_timeout", 600))),
            )
        )
        status = _video_status(status_payload)
        if status in {"completed", "succeeded", "success", "ready"}:
            return _download_video(video_id, output_path)
        if status in {"failed", "cancelled", "canceled", "expired"}:
            error = status_payload.get("error") or status_payload.get("message") or status
            raise ValueError(f"video generation failed: {error}")
        time.sleep(poll_interval)

    raise TimeoutError(f"video generation timed out after {poll_timeout} seconds")


def generate_videos(
    task_id: str,
    video_subject: str,
    video_script: str,
    audio_duration: float,
    clip_duration: int,
    video_aspect: VideoAspect,
) -> list[str]:
    max_scenes = max(1, int(config.app.get("ai_media_max_scenes", 20)))
    scene_count = min(max_scenes, max(1, math.ceil(audio_duration / clip_duration)))
    scenes = generate_scene_plan(
        video_subject=video_subject,
        video_script=video_script,
        scene_count=scene_count,
        video_aspect=video_aspect,
    )

    output_dir = Path(utils.task_dir(task_id)) / "ai-materials"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scene-plan.json").write_text(
        utils.to_json(scenes), encoding="utf-8"
    )

    videos = []
    for index, scene in enumerate(scenes, start=1):
        logger.info(f"generating AI image {index}/{len(scenes)}")
        image_path = generate_image(
            scene["image_prompt"], str(output_dir / f"scene-{index:03d}.png"), video_aspect
        )
        logger.info(f"generating AI video {index}/{len(scenes)}")
        videos.append(
            generate_video(
                image_path=image_path,
                motion_prompt=scene["motion_prompt"],
                output_path=str(output_dir / f"scene-{index:03d}.mp4"),
                duration=clip_duration,
                video_aspect=video_aspect,
            )
        )
    return videos
