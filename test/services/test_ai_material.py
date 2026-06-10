import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.config import config
from app.models.schema import VideoAspect
from app.services import ai_material


class TestOpenRouterVideoService(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app.update(
            {
                "openrouter_api_key": "test-key",
                "openrouter_poll_interval": 1,
                "openrouter_poll_timeout": 30,
                "openrouter_request_timeout": 60,
            }
        )

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_scene_plan_extracts_openrouter_prompts(self):
        with patch.object(
            ai_material.llm,
            "_generate_response",
            return_value='[{"prompt":"A cinematic library, slow dolly in"}]',
        ):
            scenes = ai_material.generate_scene_plan(
                "Books", "Books preserve knowledge.", 1, VideoAspect.landscape
            )
        self.assertEqual(scenes, [{"prompt": "A cinematic library, slow dolly in"}])

    def test_generate_video_submits_polls_and_downloads(self):
        submitted = Mock()
        submitted.json.return_value = {
            "id": "job-123",
            "polling_url": "https://openrouter.ai/poll/job-123",
        }
        submitted.raise_for_status = Mock()
        completed = Mock()
        completed.json.return_value = {
            "status": "completed",
            "unsigned_urls": ["https://cdn.example/video.mp4"],
        }
        completed.raise_for_status = Mock()
        download = Mock(content=b"video-bytes")
        download.raise_for_status = Mock()

        with tempfile.TemporaryDirectory() as directory:
            output_path = os.path.join(directory, "video.mp4")
            with (
                patch.object(ai_material.requests, "post", return_value=submitted) as post,
                patch.object(
                    ai_material.requests, "get", side_effect=[completed, download]
                ) as get,
                patch.object(
                    ai_material,
                    "VideoFileClip",
                    return_value=SimpleNamespace(duration=4, fps=30, close=lambda: None),
                ),
            ):
                result = ai_material.generate_video(
                    prompt="A serene mountain landscape",
                    output_path=output_path,
                    duration=4,
                    video_aspect=VideoAspect.portrait,
                )

        self.assertEqual(result, output_path)
        self.assertEqual(
            post.call_args.kwargs["json"],
            {
                "model": "x-ai/grok-imagine-video",
                "prompt": "A serene mountain landscape",
                "duration": 4,
                "size": "720x1080",
            },
        )
        self.assertEqual(get.call_args_list[0].args[0], "https://openrouter.ai/poll/job-123")
        self.assertEqual(get.call_args_list[1].args[0], "https://cdn.example/video.mp4")
        self.assertEqual(
            get.call_args_list[1].kwargs["headers"],
            {"Authorization": "Bearer test-key"},
        )

    def test_generate_video_reports_failed_job(self):
        submitted = Mock()
        submitted.json.return_value = {"id": "job-123", "polling_url": "poll-url"}
        failed = Mock()
        failed.json.return_value = {"status": "failed", "error": "generation rejected"}
        with (
            patch.object(ai_material.requests, "post", return_value=submitted),
            patch.object(ai_material.requests, "get", return_value=failed),
        ):
            with self.assertRaisesRegex(ValueError, "generation rejected"):
                ai_material.generate_video(
                    prompt="prompt",
                    output_path="video.mp4",
                    duration=1,
                    video_aspect=VideoAspect.landscape,
                )


if __name__ == "__main__":
    unittest.main()
