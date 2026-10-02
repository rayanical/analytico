"""Luna requests stay explicit and reasoning experiments remain bounded."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import config
from modules.column_interpretation import _luna_request, _model_name, interpret_column


class LunaConfigurationTests(unittest.TestCase):
    def test_interpreter_default_is_gpt_6_luna(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_model_name("luna"), "gpt-6-luna")

    def test_luna_chat_options_use_supported_budget_and_no_storage(self):
        with patch.object(config, "OPENAI_MODEL", "gpt-6-luna"):
            options = config.chat_completion_options(800, 0)
        self.assertEqual(options, dict(max_completion_tokens=800, temperature=0,
                                       store=False, reasoning_effort="none"))

    def test_explicit_legacy_chat_override_does_not_receive_reasoning_parameter(self):
        with patch.object(config, "OPENAI_MODEL", "gpt-4o-mini"):
            self.assertNotIn("reasoning_effort", config.chat_completion_options(100, 0.3))

    def test_reasoning_experiment_budget_is_explicit(self):
        state = dict(column_name="field", values=["1"], context={})
        for effort, budget in [("none", 300), ("low", 1200)]:
            with self.subTest(effort=effort):
                request = _luna_request(state, "gpt-6-luna", effort)
                self.assertEqual(request["reasoning"], {"effort": effort})
                self.assertEqual(request["max_output_tokens"], budget)
                self.assertFalse(request["store"])
                self.assertEqual(request["model"], "gpt-6-luna")
                self.assertTrue(request["text"]["format"]["strict"])

    def test_invalid_reasoning_is_rejected_before_network(self):
        with patch("modules.column_interpretation.httpx.Client", side_effect=AssertionError("offline")):
            result = interpret_column(dict(column_name="a", values=["1"], context={}),
                                      provider="luna", reasoning_effort="unbounded")
        self.assertEqual(result.error_code, "invalid_reasoning_effort")
        self.assertIsNone(result.decision)


if __name__ == "__main__":
    unittest.main()
