"""Provider contract tests for the opt-in column interpretation adapter."""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from typing import get_args
from unittest.mock import patch

import httpx


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from modules import column_interpretation as interpretation
from evals.evaluate_interpretation import ALLOWED_VALUES


def _decision(**overrides):
    value = {
        "role": "metric",
        "unit": "USD",
        "parsing_policy": "parse_currency_decimal",
        "recommended_aggregation": "sum",
        "needs_clarification": False,
    }
    value.update(overrides)
    return value


def _jev_response(*, overrides=None, probabilities_overrides=None, clarification=0.1,
                  include_confidence=True, routing=None, usage=None,
                  typesafe_confidence=None, cost=None):
    overrides = overrides or {}
    probabilities_overrides = probabilities_overrides or {}
    answers = {}
    for field_name, choices in interpretation._DECISION_CHOICES.items():
        selected = overrides.get(field_name, _decision()[field_name])
        probabilities = {choice: 0.0 for choice in choices}
        probabilities[selected] = 1.0
        probabilities.update(probabilities_overrides.get(field_name, {}))
        answer = {"type": "choice", "choice": selected, "probabilities": probabilities}
        if include_confidence:
            answer["confidence"] = 0.83
        answers[field_name] = answer
    answers["needs_clarification"] = {"type": "boolean", "probability": clarification}
    response = {
        "model": "typesafe-ai/jev",
        "answers": answers,
        "usage": usage or {"inputTokens": 47, "outputTokens": 12},
    }
    if routing is not None or cost is not None or typesafe_confidence is not None:
        gateway_metadata = {}
        if routing is not None:
            gateway_metadata["routing"] = routing
        if cost is not None:
            gateway_metadata["cost"] = cost
        metadata = {"gateway": gateway_metadata}
        if typesafe_confidence is not None:
            metadata["typesafe"] = {"confidence": typesafe_confidence}
        response["providerMetadata"] = metadata
    return response


def _luna_response(decision=None, *, status="completed", usage=None):
    return {
        "id": "resp_example",
        "status": status,
        "model": "gpt-6-luna-2026-09-15",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": json.dumps(decision or _decision())}
                ],
            }
        ],
        "usage": usage or {"input_tokens": 35, "output_tokens": 17, "total_tokens": 52},
    }


def _transport(response, captured=None):
    def handle(request):
        if captured is not None:
            captured.append(request)
        return response

    return httpx.MockTransport(handle)


def _input(**overrides):
    value = {
        "column_name": "net_revenue",
        "values": ["$1,250.00", "$80.00", None],
        "context": {
            "purpose": "Analyze uploaded table columns",
            "row_count": 3,
            "missing_count": 1,
            "sample_unique_count": 2,
            "other_column_names": ["account_id", "invoice_date"],
            "profile": {"dtype": "object", "percent_numeric": 0.67},
        },
    }
    value.update(overrides)
    return value


class ColumnInterpretationTests(unittest.TestCase):
    def test_decision_literals_match_the_evaluator_contract(self):
        self.assertEqual(set(interpretation._ROLE_CHOICES), ALLOWED_VALUES["role"])
        self.assertEqual(set(interpretation._UNIT_CHOICES), ALLOWED_VALUES["unit"])
        self.assertEqual(set(interpretation._PARSING_CHOICES), ALLOWED_VALUES["parsing_policy"])
        self.assertEqual(
            set(interpretation._AGGREGATION_CHOICES), ALLOWED_VALUES["recommended_aggregation"]
        )
        self.assertEqual(set(get_args(interpretation.Role)), ALLOWED_VALUES["role"])
        self.assertEqual(set(get_args(interpretation.Unit)), ALLOWED_VALUES["unit"])
        self.assertEqual(set(get_args(interpretation.ParsingPolicy)), ALLOWED_VALUES["parsing_policy"])
        self.assertEqual(
            set(get_args(interpretation.RecommendedAggregation)),
            ALLOWED_VALUES["recommended_aggregation"],
        )

    def test_interpreter_is_disabled_by_default_without_request(self):
        with patch.dict(os.environ, {}, clear=True):
            result = interpretation.interpret_column(
                _input(), transport=_transport(httpx.Response(200))
            )

        self.assertEqual(result.status, "disabled")
        self.assertIsNone(result.decision)
        self.assertIsNone(result.error_code)

    def test_jev_request_uses_named_questions_pinned_provider_and_native_outputs(self):
        requests = []
        response = httpx.Response(200, json=_jev_response(
            routing={"resolvedProvider": "typesafe-ai", "finalProvider": "typesafe-ai"},
            typesafe_confidence={"role": 0.91, "unit": 0.88},
            cost="0.00001155",
        ))
        with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "gateway-secret"}, clear=True):
            result = interpretation.interpret_column(
                _input(), provider="jev", transport=_transport(response, requests)
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.provider, "typesafe-ai")
        self.assertEqual(result.model, "typesafe-ai/jev")
        self.assertEqual(
            result.usage,
            {"input_tokens": 47, "output_tokens": 12, "cost": 0.00001155, "cost_currency": "USD"},
        )
        self.assertEqual(
            result.confidence["typesafe"]["confidence"], {"role": 0.91, "unit": 0.88}
        )
        self.assertEqual(result.confidence["choices"]["role"]["confidence"], 0.83)
        self.assertEqual(result.confidence["choices"]["role"]["probabilities"]["metric"], 1.0)
        self.assertNotIn("selected_probability", result.confidence["choices"]["role"])
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(str(request.url), "https://ai-gateway.vercel.sh/v1/evaluate")
        self.assertEqual(request.headers["Authorization"], "Bearer gateway-secret")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "typesafe-ai/jev")
        self.assertEqual(body["providerOptions"], {"gateway": {"only": ["typesafe-ai"]}})
        self.assertEqual(
            set(body["questions"]), set(interpretation._DECISION_CHOICES) | {"needs_clarification"}
        )
        self.assertEqual(body["questions"]["role"]["type"], "choice")
        self.assertEqual(body["questions"]["needs_clarification"]["type"], "boolean")
        self.assertEqual(body["state"]["context"]["profile"]["dtype"], "object")

    def test_jev_accepts_choice_probabilities_without_native_confidence(self):
        response = httpx.Response(200, json=_jev_response(include_confidence=False))
        with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "key"}, clear=True):
            result = interpretation.interpret_column(
                _input(), provider="jev", transport=_transport(response)
            )

        self.assertEqual(result.status, "ok")
        self.assertNotIn("confidence", result.confidence["choices"]["role"])
        self.assertIn("probabilities", result.confidence["choices"]["role"])

    def test_jev_rejects_provider_routing_to_another_provider(self):
        response = httpx.Response(200, json=_jev_response(
            routing={"resolvedProvider": "openai", "finalProvider": "openai"}
        ))
        with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "key"}, clear=True):
            result = interpretation.interpret_column(
                _input(), provider="jev", transport=_transport(response)
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "invalid_response")

    def test_invalid_provider_confidence_is_rejected_and_invalid_cost_is_omitted(self):
        invalid_confidence = httpx.Response(
            200,
            json=_jev_response(
                routing={"resolvedProvider": "typesafe-ai"},
                typesafe_confidence={"role": True},
            ),
        )
        with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "key"}, clear=True):
            rejected = interpretation.interpret_column(
                _input(), provider="jev", transport=_transport(invalid_confidence)
            )
        self.assertEqual(rejected.error_code, "invalid_response")

        response = httpx.Response(
            200,
            json=_jev_response(
                routing={"resolvedProvider": "typesafe-ai"},
                cost="not-a-number",
            ),
        )
        with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "key"}, clear=True):
            result = interpretation.interpret_column(
                _input(), provider="jev", transport=_transport(response)
            )
        self.assertEqual(result.status, "ok")
        self.assertNotIn("cost", result.usage)
        self.assertNotIn("cost_currency", result.usage)

    def test_missing_provider_key_is_unavailable_without_network_activity(self):
        requests = []
        with patch.dict(os.environ, {}, clear=True):
            result = interpretation.interpret_column(
                _input(), provider="jev",
                transport=_transport(httpx.Response(200, json=_jev_response()), requests),
            )

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.error_code, "missing_api_key")
        self.assertIsNone(result.decision)
        self.assertEqual(requests, [])

    def test_whitespace_provider_key_is_treated_as_missing(self):
        requests = []
        with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "   "}, clear=True):
            result = interpretation.interpret_column(
                _input(), provider="jev",
                transport=_transport(httpx.Response(200, json=_jev_response()), requests),
            )

        self.assertEqual(result.error_code, "missing_api_key")
        self.assertEqual(requests, [])

    def test_luna_request_uses_strict_responses_schema_without_tools_or_storage(self):
        requests = []
        response = httpx.Response(200, json=_luna_response())
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "openai-secret", "COLUMN_INTERPRETER_MODEL": "gpt-6-luna"},
            clear=True,
        ):
            result = interpretation.interpret_column(
                _input(), provider="luna", transport=_transport(response, requests)
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.provider, "openai")
        self.assertEqual(result.model, "gpt-6-luna-2026-09-15")
        self.assertIsNone(result.confidence)
        self.assertEqual(result.usage, {"input_tokens": 35, "output_tokens": 17, "total_tokens": 52})
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(str(request.url), "https://api.openai.com/v1/responses")
        self.assertEqual(request.headers["Authorization"], "Bearer openai-secret")
        body = json.loads(request.content)
        self.assertEqual(body["model"], "gpt-6-luna")
        self.assertEqual(body["reasoning"], {"effort": "none"})
        self.assertFalse(body["store"])
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertTrue(body["text"]["format"]["strict"])
        schema = body["text"]["format"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            set(schema["required"]),
            {"role", "unit", "parsing_policy", "recommended_aggregation", "needs_clarification"},
        )
        self.assertNotIn("tools", body)
        self.assertIn("untrusted data", body["input"][0]["content"])

    def test_jev_rejects_unknown_choices_and_invalid_probability_shapes(self):
        mutations = [
            lambda response: response["answers"]["role"].update(choice="not-an-option"),
            lambda response: response["answers"]["role"]["probabilities"].update(identifier=float("nan")),
            lambda response: response["answers"]["role"]["probabilities"].update(metric=0.4),
            lambda response: response["answers"]["role"].update(confidence=True),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                response = _jev_response()
                mutate(response)
                provider_response = httpx.Response(200, text=json.dumps(response, allow_nan=True))
                with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "key"}, clear=True):
                    result = interpretation.interpret_column(
                        _input(), provider="jev", transport=_transport(provider_response)
                    )

                self.assertEqual(result.status, "error")
                self.assertEqual(result.error_code, "invalid_response")
                self.assertIsNone(result.decision)

    def test_jev_valid_unknown_decision_is_reported_as_uncertain(self):
        response = _jev_response(
            overrides={"role": "unknown", "unit": "unknown", "parsing_policy": "unknown"},
            clarification=0.99,
        )
        with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "key"}, clear=True):
            result = interpretation.interpret_column(
                _input(), provider="jev", transport=_transport(httpx.Response(200, json=response))
            )

        self.assertEqual(result.status, "uncertain")
        self.assertTrue(result.decision.needs_clarification)
        self.assertEqual(result.confidence["needs_clarification_probability"], 0.99)

    def test_jev_cannot_claim_confident_automation_with_unknown_role(self):
        response = _jev_response(overrides={"role": "unknown"}, clarification=0.1)
        with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "key"}, clear=True):
            result = interpretation.interpret_column(
                _input(), provider="jev", transport=_transport(httpx.Response(200, json=response))
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "invalid_response")

    def test_luna_refusal_is_sanitized(self):
        refusal_response = {
            "status": "completed",
            "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "private refusal text"}]}],
        }
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}, clear=True):
            result = interpretation.interpret_column(
                _input(), provider="luna", transport=_transport(httpx.Response(200, json=refusal_response))
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "provider_refusal")
        self.assertNotIn("private refusal text", repr(result))

    def test_luna_rejects_nonfinite_unknown_and_non_boolean_decisions(self):
        decision_texts = [
            '{"role":"metric","unit":"USD","parsing_policy":"parse_currency_decimal","recommended_aggregation":"sum","needs_clarification":NaN}',
            '{"role":"metric","unit":"USD","parsing_policy":"parse_currency_decimal","recommended_aggregation":"sum","needs_clarification":1}',
            '{"role":"maybe","unit":"USD","parsing_policy":"parse_currency_decimal","recommended_aggregation":"sum","needs_clarification":false}',
        ]
        for decision_text in decision_texts:
            with self.subTest(decision_text=decision_text):
                response = {
                    "status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": decision_text}]}],
                }
                with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}, clear=True):
                    result = interpretation.interpret_column(
                        _input(), provider="luna", transport=_transport(httpx.Response(200, json=response))
                    )

                self.assertEqual(result.status, "error")
                self.assertEqual(result.error_code, "invalid_response")
                self.assertIsNone(result.decision)

    def test_incomplete_luna_response_is_not_used_as_a_decision(self):
        response = _luna_response(status="incomplete")
        response["incomplete_details"] = {"reason": "max_output_tokens"}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}, clear=True):
            result = interpretation.interpret_column(
                _input(), provider="luna", transport=_transport(httpx.Response(200, json=response))
            )

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "incomplete_response")

    def test_invalid_input_is_rejected_before_provider_call(self):
        requests = []
        bad_input = _input(extra="must not leave the process")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}, clear=True):
            result = interpretation.interpret_column(
                bad_input, provider="luna",
                transport=_transport(httpx.Response(200, json=_luna_response()), requests),
            )

        self.assertEqual(result.error_code, "invalid_input")
        self.assertIsNone(result.decision)
        self.assertEqual(requests, [])

    def test_too_many_samples_are_rejected_before_provider_call(self):
        requests = []
        bad_input = _input(values=[str(value) for value in range(13)])
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}, clear=True):
            result = interpretation.interpret_column(
                bad_input, provider="luna",
                transport=_transport(httpx.Response(200, json=_luna_response()), requests),
            )

        self.assertEqual(result.error_code, "invalid_input")
        self.assertEqual(requests, [])

    def test_provider_error_body_and_request_values_do_not_escape_result(self):
        requests = []
        error = httpx.Response(503, text="credential=provider-secret; value=confidential-name")
        with patch.dict(os.environ, {"AI_GATEWAY_API_KEY": "gateway-secret"}, clear=True):
            result = interpretation.interpret_column(
                _input(), provider="jev", transport=_transport(error, requests)
            )

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.error_code, "provider_unavailable")
        self.assertNotIn("provider-secret", repr(result))
        self.assertNotIn("confidential-name", repr(result))
        self.assertNotIn("gateway-secret", repr(result))


if __name__ == "__main__":
    unittest.main()
