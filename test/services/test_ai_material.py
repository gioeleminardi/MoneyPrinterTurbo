import base64
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from app.config import config
from app.models.schema import VideoAspect, VideoParams
from app.models.schema import VideoConcatMode
from app.services import ai_material
from app.services import task as task_service


class TestAiMaterialService(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app.update(
            {
                "ai_media_api_key": "test-key",
                "ai_media_base_url": "https://media.example/v1",
                "ai_image_model_name": "gpt-image-1",
                "ai_image_size": "",
                "ai_image_quality": "auto",
                "ai_video_model_name": "sora-2",
                "ai_video_include_seconds": True,
                "ai_video_include_size": True,
                "ai_video_include_input_reference": True,
                "ai_video_poll_interval": 1,
            }
        )

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_scene_plan_extracts_json_array_from_wrapped_response(self):
        response = (
            'Storyboard:\n[{"image_prompt":"same hero in a library",'
            '"motion_prompt":"slow dolly in"}]'
        )
        with patch.object(ai_material.llm, "_generate_response", return_value=response):
            scenes = ai_material.generate_scene_plan(
                "Books", "Books preserve knowledge.", 1, VideoAspect.landscape
            )

        self.assertEqual(
            scenes,
            [
                {
                    "image_prompt": "same hero in a library",
                    "motion_prompt": "slow dolly in",
                }
            ],
        )

    def test_generate_image_accepts_openai_base64_response(self):
        response = types.SimpleNamespace(
            data=[
                types.SimpleNamespace(
                    b64_json=base64.b64encode(b"image-bytes").decode(), url=None
                )
            ]
        )
        client = Mock()
        client.images.generate.return_value = response

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "scene.png")
            with patch.object(ai_material, "OpenAI", return_value=client) as openai:
                result = ai_material.generate_image(
                    "cinematic library", output, VideoAspect.landscape
                )

            self.assertEqual(result, output)
            with open(output, "rb") as image_file:
                self.assertEqual(image_file.read(), b"image-bytes")
            openai.assert_called_once_with(
                api_key="test-key",
                base_url="https://media.example/v1",
                timeout=600.0,
            )
            client.images.generate.assert_called_once_with(
                model="gpt-image-1",
                prompt="cinematic library",
                n=1,
                size="1536x1024",
                quality="auto",
            )
            client.close.assert_called_once()

    def test_generate_image_supports_aihubmix_auto_size_and_quality(self):
        config.app.update(
            {
                "ai_media_base_url": "https://aihubmix.com/v1",
                "ai_image_model_name": "gpt-image-2-free",
                "ai_image_size": "auto",
                "ai_image_quality": "high",
            }
        )
        response = types.SimpleNamespace(
            data=[
                types.SimpleNamespace(
                    b64_json=base64.b64encode(b"image-bytes").decode(), url=None
                )
            ]
        )
        client = Mock()
        client.images.generate.return_value = response

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "scene.png")
            with patch.object(ai_material, "OpenAI", return_value=client) as openai:
                ai_material.generate_image("prompt", output, VideoAspect.portrait)

        openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://aihubmix.com/v1",
            timeout=600.0,
        )
        client.images.generate.assert_called_once_with(
            model="gpt-image-2-free",
            prompt="prompt",
            n=1,
            size="auto",
            quality="high",
        )
        client.close.assert_called_once()

    def test_aihubmix_llm_credentials_are_reused_for_media(self):
        config.app.update(
            {
                "ai_media_api_key": "",
                "ai_media_base_url": "",
                "llm_provider": "aihubmix",
                "aihubmix_api_key": "aihubmix-key",
                "aihubmix_base_url": "https://aihubmix.com/v1",
            }
        )

        self.assertEqual(ai_material._api_key(), "aihubmix-key")
        self.assertEqual(ai_material._base_url(), "https://aihubmix.com/v1")

    def test_generate_videos_plans_enough_scenes_for_audio_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(ai_material.utils, "task_dir", return_value=directory),
                patch.object(
                    ai_material,
                    "generate_scene_plan",
                    return_value=[
                        {"image_prompt": "one", "motion_prompt": "move one"},
                        {"image_prompt": "two", "motion_prompt": "move two"},
                        {"image_prompt": "three", "motion_prompt": "move three"},
                    ],
                ) as plan,
                patch.object(
                    ai_material, "generate_image", side_effect=lambda _, path, __: path
                ),
                patch.object(
                    ai_material, "generate_video", side_effect=lambda **kwargs: kwargs["output_path"]
                ),
            ):
                videos = ai_material.generate_videos(
                    task_id="task",
                    video_subject="subject",
                    video_script="script",
                    audio_duration=10,
                    clip_duration=4,
                    video_aspect=VideoAspect.portrait,
                )

        self.assertEqual(len(videos), 3)
        self.assertEqual(plan.call_args.kwargs["scene_count"], 3)

    def test_video_duration_uses_nearest_supported_duration(self):
        self.assertEqual(ai_material._video_duration(2), 4)
        self.assertEqual(ai_material._video_duration(5), 8)
        self.assertEqual(ai_material._video_duration(10), 12)

    def test_generate_video_uses_openai_create_poll_and_content_contract(self):
        queued_video = types.SimpleNamespace(
            id="video-123", status="queued", progress=0, error=None
        )
        completed_video = types.SimpleNamespace(
            id="video-123", status="completed", progress=100, error=None
        )

        def write_video(path):
            with open(path, "wb") as video_file:
                video_file.write(b"video-bytes")

        content = Mock()
        content.write_to_file.side_effect = write_video
        client = Mock()
        client.videos.create.return_value = queued_video
        client.videos.retrieve.return_value = completed_video
        client.videos.download_content.return_value = content

        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "scene.png")
            output_path = os.path.join(directory, "scene.mp4")
            with open(image_path, "wb") as image_file:
                image_file.write(b"image")

            clip = Mock(duration=4, fps=30)
            with (
                patch.object(ai_material, "OpenAI", return_value=client) as openai,
                patch.object(ai_material, "VideoFileClip", return_value=clip),
                patch.object(ai_material.time, "sleep"),
            ):
                result = ai_material.generate_video(
                    image_path=image_path,
                    image_prompt="a cinematic library",
                    motion_prompt="slow dolly in",
                    output_path=output_path,
                    duration=5,
                    video_aspect=VideoAspect.portrait,
                )

        self.assertEqual(result, output_path)
        openai.assert_called_once_with(
            api_key="test-key",
            base_url="https://media.example/v1",
            timeout=600.0,
        )
        client.videos.create.assert_called_once_with(
            model="sora-2",
            prompt="slow dolly in",
            seconds="8",
            size="720x1280",
            input_reference=Path(image_path),
        )
        client.videos.retrieve.assert_called_once_with("video-123")
        client.videos.download_content.assert_called_once_with("video-123")
        content.write_to_file.assert_called_once_with(output_path)
        client.close.assert_called_once()
        clip.close.assert_called_once()

    def test_generate_video_supports_aihubmix_veo_request_shape(self):
        config.app.update(
            {
                "ai_video_model_name": "veo-3.1-fast-generate-preview",
                "ai_video_include_input_reference": True,
            }
        )
        config.app.pop("ai_video_include_seconds", None)
        config.app.pop("ai_video_include_size", None)
        queued_video = types.SimpleNamespace(
            id="veo-123", status="queued", progress=0, error=None
        )
        completed_video = types.SimpleNamespace(
            id="veo-123", status="completed", progress=100, error=None
        )
        client = Mock()
        client.videos.create.return_value = queued_video
        client.videos.retrieve.return_value = completed_video

        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "scene.png")
            output_path = os.path.join(directory, "scene.mp4")
            with open(image_path, "wb") as image_file:
                image_file.write(b"image")

            with (
                patch.object(ai_material, "OpenAI", return_value=client),
                patch.object(
                    ai_material, "_download_video", return_value=output_path
                ) as download,
                patch.object(ai_material.time, "sleep"),
            ):
                result = ai_material.generate_video(
                    image_path=image_path,
                    image_prompt="a dog riding a motorcycle",
                    motion_prompt="animate this image",
                    output_path=output_path,
                    duration=5,
                    video_aspect=VideoAspect.portrait,
                )

        self.assertEqual(result, output_path)
        client.videos.create.assert_called_once_with(
            model="veo-3.1-fast-generate-preview",
            prompt="animate this image",
            input_reference=Path(image_path),
        )
        client.videos.retrieve.assert_called_once_with("veo-123")
        download.assert_called_once_with(client, "veo-123", output_path)
        client.close.assert_called_once()

    def test_generate_videos_skips_images_for_direct_text_to_video(self):
        config.app["ai_video_include_input_reference"] = False
        scene = {
            "image_prompt": "A red sailboat crossing a stormy ocean.",
            "motion_prompt": "Waves surge as the camera tracks beside the boat.",
        }

        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(ai_material.utils, "task_dir", return_value=directory),
                patch.object(ai_material, "generate_scene_plan", return_value=[scene]),
                patch.object(ai_material, "generate_image") as generate_image,
                patch.object(
                    ai_material,
                    "generate_video",
                    return_value=os.path.join(directory, "scene-001.mp4"),
                ) as generate_video,
            ):
                videos = ai_material.generate_videos(
                    task_id="task",
                    video_subject="Sailing",
                    video_script="A boat crosses the ocean.",
                    audio_duration=4,
                    clip_duration=4,
                    video_aspect=VideoAspect.landscape,
                )

        generate_image.assert_not_called()
        expected_video = os.path.join(directory, "ai-materials", "scene-001.mp4")
        generate_video.assert_called_once_with(
            image_path=None,
            image_prompt=scene["image_prompt"],
            motion_prompt=scene["motion_prompt"],
            output_path=expected_video,
            duration=4,
            video_aspect=VideoAspect.landscape,
        )
        self.assertEqual(videos, [os.path.join(directory, "scene-001.mp4")])

    def test_generate_video_combines_prompts_without_input_reference(self):
        config.app.update(
            {
                "ai_video_include_input_reference": False,
                "ai_video_include_seconds": False,
                "ai_video_include_size": False,
            }
        )
        completed_video = types.SimpleNamespace(
            id="video-123", status="completed", progress=100, error=None
        )
        client = Mock()
        client.videos.create.return_value = completed_video

        with (
            patch.object(ai_material, "OpenAI", return_value=client),
            patch.object(ai_material, "_download_video", return_value="output.mp4"),
        ):
            result = ai_material.generate_video(
                image_path=None,
                image_prompt="A red sailboat crossing a stormy ocean.",
                motion_prompt="Waves surge as the camera tracks beside the boat.",
                output_path="output.mp4",
                duration=5,
                video_aspect=VideoAspect.landscape,
            )

        self.assertEqual(result, "output.mp4")
        client.videos.create.assert_called_once_with(
            model="sora-2",
            prompt=(
                "Visual scene:\n"
                "A red sailboat crossing a stormy ocean.\n\n"
                "Motion and camera direction:\n"
                "Waves surge as the camera tracks beside the boat."
            ),
        )

    def test_generate_video_retries_without_incompatible_duration_and_size(self):
        config.app.update(
            {
                "ai_video_model_name": "custom-video-model",
                "ai_video_include_input_reference": False,
                "ai_video_include_seconds": True,
                "ai_video_include_size": True,
            }
        )

        compatibility_error = Exception(
            "1080p is not supported for a duration of 4 seconds."
        )
        compatibility_error.status_code = 400
        completed_video = types.SimpleNamespace(
            id="video-123", status="completed", progress=100, error=None
        )
        client = Mock()
        client.videos.create.side_effect = [compatibility_error, completed_video]

        with (
            patch.object(ai_material, "OpenAI", return_value=client),
            patch.object(ai_material, "_download_video", return_value="output.mp4"),
        ):
            result = ai_material.generate_video(
                image_path=None,
                image_prompt="A red sailboat crossing a stormy ocean.",
                motion_prompt="Track beside the boat.",
                output_path="output.mp4",
                duration=4,
                video_aspect=VideoAspect.portrait,
            )

        self.assertEqual(result, "output.mp4")
        self.assertEqual(client.videos.create.call_count, 2)
        first_call = client.videos.create.call_args_list[0].kwargs
        second_call = client.videos.create.call_args_list[1].kwargs
        self.assertEqual(first_call["seconds"], "4")
        self.assertEqual(first_call["size"], "720x1280")
        self.assertNotIn("seconds", second_call)
        self.assertNotIn("size", second_call)
        self.assertEqual(second_call["model"], "custom-video-model")
        self.assertEqual(second_call["prompt"], first_call["prompt"])

    def test_veo_models_omit_duration_and_size_despite_stale_enabled_settings(self):
        config.app.update(
            {
                "ai_video_model_name": "veo-3.1-lite-generate-preview",
                "ai_video_include_seconds": True,
                "ai_video_include_size": True,
            }
        )

        self.assertFalse(
            ai_material._video_optional_parameter("ai_video_include_seconds")
        )
        self.assertFalse(ai_material._video_optional_parameter("ai_video_include_size"))

    def test_video_error_extracts_nested_message(self):
        self.assertEqual(
            ai_material._video_error(
                {"status": "failed", "error": {"message": "generation rejected"}}
            ),
            "generation rejected",
        )


class TestAiMaterialTaskIntegration(unittest.TestCase):
    def test_ai_source_dispatches_generated_clips_without_stock_terms(self):
        params = VideoParams(
            video_subject="Ocean",
            video_script="The ocean moves.",
            video_source="ai",
            video_clip_duration=5,
        )
        with patch.object(
            task_service.ai_material, "generate_videos", return_value=["generated.mp4"]
        ) as generate:
            result = task_service.get_video_materials(
                "task-id", params, params.video_script, "", 12
            )

        self.assertEqual(result, ["generated.mp4"])
        generate.assert_called_once_with(
            task_id="task-id",
            video_subject="Ocean",
            video_script="The ocean moves.",
            audio_duration=12,
            clip_duration=5,
            video_aspect=params.video_aspect,
        )

    def test_ai_source_keeps_storyboard_order_when_combining(self):
        params = VideoParams(
            video_subject="Ocean",
            video_script="The ocean moves.",
            video_source="ai",
            video_count=2,
        )
        with (
            patch.object(task_service.utils, "task_dir", return_value="/tmp/task"),
            patch.object(task_service.video, "combine_videos") as combine,
            patch.object(task_service.video, "generate_video"),
            patch.object(task_service.sm.state, "update_task"),
        ):
            task_service.generate_final_videos(
                "task-id", params, ["one.mp4", "two.mp4"], "audio.mp3", ""
            )

        self.assertEqual(combine.call_count, 2)
        for call in combine.call_args_list:
            self.assertEqual(
                call.kwargs["video_concat_mode"], VideoConcatMode.sequential
            )
