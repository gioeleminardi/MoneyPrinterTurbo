import base64
import json
import math
import os
import time
from pathlib import Path
from typing import Any
from openai import OpenAI

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


def _api_bool(name: str, fallback: bool = False) -> bool:
    value = config.app.get(name, fallback)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _fallback_media_provider() -> str:
    llm_provider = _api_setting("llm_provider").lower()
    if llm_provider == "aihubmix" and _api_setting("aihubmix_api_key"):
        return "aihubmix"
    if llm_provider == "openai" and _api_setting("openai_api_key"):
        return "openai"
    if _api_setting("openai_api_key"):
        return "openai"
    if _api_setting("aihubmix_api_key"):
        return "aihubmix"
    return "openai"


def _api_key() -> str:
    api_key = _api_setting("ai_media_api_key")
    if not api_key:
        api_key = _api_setting(f"{_fallback_media_provider()}_api_key")
    if not api_key:
        raise ValueError(
            "ai_media_api_key is not set; configure it or an OpenAI/AiHubMix API key"
        )
    return api_key


def _base_url() -> str:
    base_url = _api_setting("ai_media_base_url")
    if not base_url:
        provider = _fallback_media_provider()
        default = (
            "https://aihubmix.com/v1"
            if provider == "aihubmix"
            else "https://api.openai.com/v1"
        )
        base_url = _api_setting(f"{provider}_base_url", default) or default
    return base_url.rstrip("/")


def _response_field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _extract_json_array(response: str) -> list[dict[str, str]]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError:
        start = response.find("[")
        end = response.rfind("]")
        if start < 0 or end <= start:
            raise ValueError("scene planner did not return a JSON array")
        value = json.loads(response[start: end + 1])

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
    configured_size = _api_setting("ai_image_size")
    if configured_size:
        return configured_size

    aspect = VideoAspect(video_aspect)
    if aspect == VideoAspect.landscape:
        return _api_setting("ai_image_size_landscape", "1536x1024")
    if aspect == VideoAspect.square:
        return _api_setting("ai_image_size_square", "1024x1024")
    return _api_setting("ai_image_size_portrait", "1024x1536")


def generate_image(prompt: str, output_path: str, video_aspect: VideoAspect) -> str:
    generate_args = {
        "model": _api_setting("ai_image_model_name", "gpt-image-1"),
        "prompt": prompt,
        "n": 1,
        "size": _image_size(video_aspect),
    }
    quality = _api_setting("ai_image_quality", "auto")
    if quality:
        generate_args["quality"] = quality

    client = OpenAI(
        api_key=_api_key(),
        base_url=_base_url(),
        timeout=float(config.app.get("ai_media_request_timeout", 600)),
    )
    try:
        response = client.images.generate(**generate_args)
    finally:
        client.close()

    data = _response_field(response, "data") or []
    if not data:
        raise ValueError("image provider returned no image")

    image = data[0]
    b64_json = _response_field(image, "b64_json")
    image_url = _response_field(image, "url")
    if b64_json:
        content = base64.b64decode(b64_json)
    elif image_url:
        download = requests.get(
            image_url,
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


def _video_model_name() -> str:
    return _api_setting("ai_video_model_name", "sora-2")


def _video_optional_parameter(name: str) -> bool:
    if _video_model_name().lower().startswith("veo-"):
        return False
    if name in config.app:
        return _api_bool(name)
    return True


def _video_id(payload: Any) -> str:
    return str(
        _response_field(payload, "id") or _response_field(payload, "video_id") or ""
    ).strip()


def _video_status(payload: Any) -> str:
    return str(
        _response_field(payload, "status") or _response_field(payload, "state") or ""
    ).lower().strip()


def _video_error(payload: Any) -> str:
    error = _response_field(payload, "error")
    return str(
        _response_field(error, "message")
        or _response_field(error, "code")
        or error
        or _response_field(payload, "message")
        or _video_status(payload)
    )


def _is_video_parameter_compatibility_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).lower()
    parameter_terms = ("duration", "seconds", "resolution", "1080p", "720p", "size")
    return status_code == 400 and any(term in message for term in parameter_terms)


def _create_video_with_compatibility_fallback(client: OpenAI, create_args: dict) -> Any:
    try:
        return client.videos.create(**create_args)
    except Exception as exc:
        has_optional_parameters = "seconds" in create_args or "size" in create_args
        if not has_optional_parameters or not _is_video_parameter_compatibility_error(exc):
            raise

        fallback_args = {
            key: value
            for key, value in create_args.items()
            if key not in {"seconds", "size"}
        }
        logger.warning(
            "video provider rejected the requested duration/size combination; "
            "retrying without seconds and size"
        )
        return client.videos.create(**fallback_args)


def _download_video(client: OpenAI, video_id: str, output_path: str) -> str:
    content = client.videos.download_content(video_id)
    content.write_to_file(output_path)

    clip = None
    try:
        clip = VideoFileClip(output_path)
        if clip.duration <= 0 or clip.fps <= 0:
            raise ValueError("generated video has invalid duration or frame rate")
    finally:
        if clip is not None:
            clip.close()
    return output_path


def _video_prompt(image_prompt: str, motion_prompt: str, include_reference: bool) -> str:
    if include_reference:
        return motion_prompt
    return (
        "Visual scene:\n"
        f"{image_prompt}\n\n"
        "Motion and camera direction:\n"
        f"{motion_prompt}"
    )


def generate_video(
        image_path: str | None,
        image_prompt: str,
        motion_prompt: str,
        output_path: str,
        duration: int,
        video_aspect: VideoAspect,
) -> str:
    include_reference = _api_bool("ai_video_include_input_reference", True)
    create_args = {
        "model": _video_model_name(),
        "prompt": _video_prompt(image_prompt, motion_prompt, include_reference),
    }
    if _video_optional_parameter("ai_video_include_seconds"):
        create_args["seconds"] = str(_video_duration(duration))
    if _video_optional_parameter("ai_video_include_size"):
        create_args["size"] = _video_size(video_aspect)
    if include_reference:
        if not image_path:
            raise ValueError("image_path is required when input_reference is enabled")
        create_args["input_reference"] = Path(image_path)

    client = OpenAI(
        api_key=_api_key(),
        base_url=_base_url(),
        timeout=float(config.app.get("ai_media_request_timeout", 600)),
    )
    try:
        video = _create_video_with_compatibility_fallback(client, create_args)
        video_id = _video_id(video)
        if not video_id:
            raise ValueError("video provider returned no video id")

        poll_interval = max(1, int(config.app.get("ai_video_poll_interval", 10)))
        poll_timeout = max(
            poll_interval, int(config.app.get("ai_video_poll_timeout", 1800))
        )
        deadline = time.monotonic() + poll_timeout
        while time.monotonic() < deadline:
            status = _video_status(video)
            progress = _response_field(video, "progress") or 0
            logger.info(f"AI video {video_id}: status={status}, progress={progress}%")
            if status in {"completed", "succeeded", "success", "ready"}:
                return _download_video(client, video_id, output_path)
            if status in {"failed", "cancelled", "canceled", "expired"}:
                raise ValueError(f"video generation failed: {_video_error(video)}")

            try:
                video = client.videos.retrieve(video_id)
            except Exception as exc:
                logger.warning(f"failed to retrieve video {video_id} status: {str(exc)}")
            time.sleep(poll_interval)

        raise TimeoutError(f"video generation timed out after {poll_timeout} seconds")
    finally:
        client.close()


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
    include_reference = _api_bool("ai_video_include_input_reference", True)
    for index, scene in enumerate(scenes, start=1):
        image_path = None
        if include_reference:
            logger.info(f"generating AI image {index}/{len(scenes)}")
            image_path = generate_image(
                scene["image_prompt"],
                str(output_dir / f"scene-{index:03d}.png"),
                video_aspect,
            )
        else:
            logger.info(
                f"skipping AI image {index}/{len(scenes)}; using direct text-to-video"
            )
        logger.info(f"generating AI video {index}/{len(scenes)}")
        videos.append(
            generate_video(
                image_path=image_path,
                image_prompt=scene["image_prompt"],
                motion_prompt=scene["motion_prompt"],
                output_path=str(output_dir / f"scene-{index:03d}.mp4"),
                duration=clip_duration,
                video_aspect=video_aspect,
            )
        )
    return videos
