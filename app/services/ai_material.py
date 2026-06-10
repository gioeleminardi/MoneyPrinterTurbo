import json
import math
import os
import time
from pathlib import Path

import requests
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip

from app.config import config
from app.models.schema import VideoAspect
from app.services import llm
from app.utils import utils

OPENROUTER_VIDEOS_URL = "https://openrouter.ai/api/v1/videos"
OPENROUTER_VIDEO_MODEL = "x-ai/grok-imagine-video"


def _setting(name: str, fallback=""):
    value = config.app.get(name, fallback)
    return value.strip() if isinstance(value, str) else value


def _api_key() -> str:
    api_key = _setting("openrouter_api_key")
    if not api_key:
        raise ValueError("openrouter_api_key is not set")
    return api_key


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }


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
        prompt = str(item.get("prompt", "")).strip()
        if prompt:
            scenes.append({"prompt": prompt})
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
# Role: Storyboard planner for text-to-video production

Create exactly {scene_count} chronological scenes that visually follow the voiceover.
Every scene must be strictly relevant to a concrete part of the voiceover.
Keep recurring people, locations, era, color palette, lighting, and visual style coherent.
Each prompt must be standalone and describe both the visual scene and natural motion.
Do not include text, captions, logos, watermarks, audio, or split screens.
The requested aspect ratio is {video_aspect}.

Return only a JSON array. Each item must contain one "prompt" field.

Video subject:
{video_subject}

Complete voiceover:
{video_script}
""".strip()
    return _extract_json_array(llm._generate_response(prompt))


def _video_size(video_aspect: VideoAspect) -> str:
    configured = _setting("openrouter_video_size")
    if configured:
        return configured
    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.landscape:
        return "1080x720"
    if aspect == VideoAspect.square:
        return "720x720"
    return "720x1080"


def _validate_video(output_path: str) -> str:
    clip = None
    try:
        clip = VideoFileClip(output_path)
        if clip.duration <= 0 or clip.fps <= 0:
            raise ValueError("generated video has invalid duration or frame rate")
    finally:
        if clip is not None:
            clip.close()
    return output_path


def _download_video(video_url: str, output_path: str) -> str:
    timeout = int(_setting("openrouter_request_timeout", 600))
    response = requests.get(
        video_url,
        headers={"Authorization": f"Bearer {_api_key()}"},
        proxies=config.proxy,
        timeout=(30, timeout),
    )
    response.raise_for_status()
    Path(output_path).write_bytes(response.content)
    if not os.path.getsize(output_path):
        raise ValueError("OpenRouter returned an empty video")
    return _validate_video(output_path)


def generate_video(
    prompt: str,
    output_path: str,
    duration: int,
    video_aspect: VideoAspect,
) -> str:
    timeout = int(_setting("openrouter_request_timeout", 600))
    response = requests.post(
        _setting("openrouter_videos_url", OPENROUTER_VIDEOS_URL),
        headers=_headers(),
        json={
            "model": OPENROUTER_VIDEO_MODEL,
            "prompt": prompt,
            "duration": max(1, int(duration)),
            "size": _video_size(video_aspect),
        },
        proxies=config.proxy,
        timeout=(30, timeout),
    )
    response.raise_for_status()
    result = response.json()
    job_id = str(result.get("id", "")).strip()
    polling_url = str(result.get("polling_url", "")).strip()
    if not job_id or not polling_url:
        raise ValueError("OpenRouter returned no video job id or polling URL")

    poll_interval = max(1, int(_setting("openrouter_poll_interval", 5)))
    poll_timeout = max(poll_interval, int(_setting("openrouter_poll_timeout", 1800)))
    deadline = time.monotonic() + poll_timeout
    poll_headers = {"Authorization": f"Bearer {_api_key()}"}
    while time.monotonic() < deadline:
        poll_response = requests.get(
            polling_url,
            headers=poll_headers,
            proxies=config.proxy,
            timeout=(30, timeout),
        )
        poll_response.raise_for_status()
        status_data = poll_response.json()
        status = str(status_data.get("status", "")).lower()
        logger.info(f"OpenRouter video {job_id}: status={status}")
        if status == "completed":
            urls = status_data.get("unsigned_urls") or []
            if not urls:
                raise ValueError("OpenRouter completed the job without a video URL")
            return _download_video(urls[0], output_path)
        if status == "failed":
            raise ValueError(
                f"OpenRouter video generation failed: {status_data.get('error', 'Unknown error')}"
            )
        time.sleep(poll_interval)
    raise TimeoutError(f"OpenRouter video generation timed out after {poll_timeout} seconds")


def generate_videos(
    task_id: str,
    video_subject: str,
    video_script: str,
    audio_duration: float,
    clip_duration: int,
    video_aspect: VideoAspect,
) -> list[str]:
    max_scenes = max(1, int(_setting("openrouter_max_scenes", 20)))
    scene_count = min(max_scenes, max(1, math.ceil(audio_duration / clip_duration)))
    scenes = generate_scene_plan(
        video_subject=video_subject,
        video_script=video_script,
        scene_count=scene_count,
        video_aspect=video_aspect,
    )
    output_dir = Path(utils.task_dir(task_id)) / "openrouter-materials"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scene-plan.json").write_text(
        utils.to_json(scenes), encoding="utf-8"
    )

    videos = []
    for index, scene in enumerate(scenes, start=1):
        logger.info(f"generating OpenRouter video {index}/{len(scenes)}")
        videos.append(
            generate_video(
                prompt=scene["prompt"],
                output_path=str(output_dir / f"scene-{index:03d}.mp4"),
                duration=clip_duration,
                video_aspect=video_aspect,
            )
        )
    return videos
