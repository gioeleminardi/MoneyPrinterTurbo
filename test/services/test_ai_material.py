import base64
import os
import tempfile
import unittest
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
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [{"b64_json": base64.b64encode(b"image-bytes").decode()}]
        }

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "scene.png")
            with patch.object(ai_material.requests, "post", return_value=response) as post:
                result = ai_material.generate_image(
                    "cinematic library", output, VideoAspect.landscape
                )

            self.assertEqual(result, output)
            with open(output, "rb") as image_file:
                self.assertEqual(image_file.read(), b"image-bytes")
            self.assertEqual(
                post.call_args.args[0], "https://media.example/v1/images/generations"
            )
            self.assertEqual(post.call_args.kwargs["json"]["size"], "1536x1024")
            self.assertEqual(post.call_args.kwargs["json"]["quality"], "auto")

    def test_generate_image_supports_aihubmix_auto_size_and_quality(self):
        config.app.update(
            {
                "ai_media_base_url": "https://aihubmix.com/v1",
                "ai_image_model_name": "gpt-image-2",
                "ai_image_size": "auto",
                "ai_image_quality": "high",
            }
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [{"b64_json": base64.b64encode(b"image-bytes").decode()}]
        }

        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, "scene.png")
            with patch.object(ai_material.requests, "post", return_value=response) as post:
                ai_material.generate_image("prompt", output, VideoAspect.portrait)

        self.assertEqual(post.call_args.args[0], "https://aihubmix.com/v1/images/generations")
        self.assertEqual(
            post.call_args.kwargs["json"],
            {
                "model": "gpt-image-2",
                "prompt": "prompt",
                "n": 1,
                "size": "auto",
                "quality": "high",
            },
        )

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
        create_response = Mock()
        create_response.raise_for_status.return_value = None
        create_response.json.return_value = {"id": "video-123", "status": "queued"}
        status_response = Mock()
        status_response.raise_for_status.return_value = None
        status_response.json.return_value = {"id": "video-123", "status": "completed"}
        content_response = Mock()
        content_response.raise_for_status.return_value = None
        content_response.content = b"video-bytes"

        with tempfile.TemporaryDirectory() as directory:
            image_path = os.path.join(directory, "scene.png")
            output_path = os.path.join(directory, "scene.mp4")
            with open(image_path, "wb") as image_file:
                image_file.write(b"image")

            clip = Mock(duration=4, fps=30)
            with (
                patch.object(ai_material.requests, "post", return_value=create_response) as post,
                patch.object(
                    ai_material.requests,
                    "get",
                    side_effect=[status_response, content_response],
                ) as get,
                patch.object(ai_material, "VideoFileClip", return_value=clip),
            ):
                result = ai_material.generate_video(
                    image_path=image_path,
                    motion_prompt="slow dolly in",
                    output_path=output_path,
                    duration=5,
                    video_aspect=VideoAspect.portrait,
                )

        self.assertEqual(result, output_path)
        self.assertEqual(post.call_args.args[0], "https://media.example/v1/videos")
        self.assertEqual(post.call_args.kwargs["data"]["seconds"], "8")
        self.assertEqual(
            get.call_args_list[0].args[0], "https://media.example/v1/videos/video-123"
        )
        self.assertEqual(
            get.call_args_list[1].args[0],
            "https://media.example/v1/videos/video-123/content",
        )
        clip.close.assert_called_once()


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
