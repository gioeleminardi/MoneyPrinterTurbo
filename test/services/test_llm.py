import types
import unittest
from unittest.mock import Mock, patch

from app.config import config
from app.services import llm


class TestAIHubMixProvider(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        config.app.update(
            {
                "llm_provider": "aihubmix",
                "aihubmix_api_key": "test-key",
                "aihubmix_base_url": "https://aihubmix.com/v1",
                "aihubmix_model_name": "gpt-5.4-mini",
            }
        )

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)

    def test_generate_response_only_uses_aihubmix(self):
        client = Mock()
        client.chat.completions.create.return_value = types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="hello\nworld")
                )
            ]
        )
        with patch.object(llm, "OpenAI", return_value=client) as openai:
            result = llm._generate_response("Say hello")

        self.assertEqual(result, "helloworld")
        openai.assert_called_once_with(
            api_key="test-key", base_url="https://aihubmix.com/v1"
        )
        client.chat.completions.create.assert_called_once_with(
            model="gpt-5.4-mini",
            messages=[{"role": "user", "content": "Say hello"}],
        )
        client.close.assert_called_once()

    def test_generate_response_requires_aihubmix_key(self):
        config.app["aihubmix_api_key"] = ""
        self.assertIn("aihubmix_api_key is not set", llm._generate_response("test"))

    def test_build_script_prompt_appends_advanced_requirements(self):
        prompt = llm.build_script_prompt(
            video_subject="Coffee",
            language="en",
            paragraph_number=2,
            video_script_prompt="Keep it concise",
        )
        self.assertIn("- video subject: Coffee", prompt)
        self.assertIn("- number of paragraphs: 2", prompt)
        self.assertIn("Keep it concise", prompt)


if __name__ == "__main__":
    unittest.main()
