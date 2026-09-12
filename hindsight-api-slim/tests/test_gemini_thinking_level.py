"""The native Gemini lane honours ``reasoning_effort`` as ``thinking_level``.

Before this, ``GeminiLLM`` accepted the effort into its constructor and never
looked at it again, so ``HINDSIGHT_API_*_REASONING_EFFORT`` was a no-op on the
``gemini`` and ``vertexai`` providers (a loud one since #3449, a silent one
before). Gemini 3 exposes the same idea as a discrete ``thinking_level``, so a
configured effort now becomes that level on every request — the plain call, the
tool-calling call, and their cached variants — while the pre-3 generations, which
have no level parameter, keep the startup warning.
"""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("google.genai")

from google.genai import types as genai_types  # noqa: E402

from hindsight_api.engine.llm_interface import LLM_TOOL_CHOICE_AUTO  # noqa: E402
from hindsight_api.engine.providers.gemini_llm import GeminiLLM, _thinking_level_for  # noqa: E402


def _make_provider(model: str = "gemini-3.7-flash", reasoning_effort: str | None = None, **kwargs) -> GeminiLLM:
    with patch("google.genai.Client") as mock_client_cls:
        mock_client_cls.return_value = MagicMock()
        provider = GeminiLLM(
            provider="gemini",
            api_key="fake-api-key",
            base_url="",
            model=model,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )
        provider._client = MagicMock()
        return provider


def _text_response(text: str = '{"ok": true}') -> MagicMock:
    response = MagicMock()
    response.text = text
    response.candidates = [MagicMock(content=MagicMock(parts=[MagicMock(text=text, function_call=None)]))]
    response.usage_metadata = MagicMock(
        prompt_token_count=1, candidates_token_count=1, total_token_count=2, cached_content_token_count=0
    )
    return response


def _sent_config(generate: AsyncMock) -> genai_types.GenerateContentConfig:
    return generate.call_args.kwargs["config"]


class TestMapping:
    @pytest.mark.parametrize(
        ("effort", "level"),
        [("none", "MINIMAL"), ("minimal", "MINIMAL"), ("low", "LOW"), ("medium", "MEDIUM"), ("high", "HIGH")],
    )
    def test_shared_vocabulary_maps_to_a_gemini_level(self, effort, level):
        # Intention: one env value steers every provider, so the words the OpenAI lane
        # accepts must mean the same thing here.
        assert _thinking_level_for("gemini-3.7-flash", effort) == level

    def test_xhigh_clamps_to_high(self):
        # Gemini 3 tops out at HIGH; an operator asking for more gets the most it offers.
        assert _thinking_level_for("gemini-3.7-flash", "xhigh") == "HIGH"

    def test_case_insensitive(self):
        assert _thinking_level_for("gemini-3.7-flash", "High") == "HIGH"

    def test_unconfigured_sends_nothing(self):
        # Unset means unset: the model runs at its own default level.
        assert _thinking_level_for("gemini-3.7-flash", None) is None

    @pytest.mark.parametrize(
        "model", ["gemini-2.5-flash", "gemini-2.0-flash", "google/gemini-2.5-pro", "gemini-1.5-pro"]
    )
    def test_pre_3_generations_take_no_level(self, model):
        # thinking_level is a Gemini 3 parameter; sending it to 2.x is a 400 on every call.
        assert _thinking_level_for(model, "high") is None

    @pytest.mark.parametrize("model", ["gemini-3.7-flash", "gemini-3.1-pro-preview", "google/gemini-3.1-flash-lite"])
    def test_gemini_3_and_vertex_ids_take_a_level(self, model):
        assert _thinking_level_for(model, "low") == "LOW"

    def test_unknown_effort_fails_at_startup(self):
        # A typo must surface once, at construction, not as a rejected request per call.
        with pytest.raises(ValueError, match="not a Gemini thinking level"):
            _thinking_level_for("gemini-3.7-flash", "maximum")


class TestStartup:
    def test_gemini_3_no_longer_warns(self, caplog):
        with caplog.at_level(logging.WARNING):
            provider = _make_provider("gemini-3.7-flash", "high")
        assert provider._thinking_level == "HIGH"
        assert not [r for r in caplog.records if "reasoning_effort" in r.message]

    def test_pre_3_still_warns_that_the_value_is_ignored(self, caplog):
        # The #3449 contract stands where the parameter genuinely cannot be sent.
        with caplog.at_level(logging.WARNING):
            provider = _make_provider("gemini-2.5-flash", "high")
        assert provider._thinking_level is None
        assert any("reasoning_effort" in r.message and "ignored" in r.message for r in caplog.records)

    def test_startup_line_reports_the_level_in_force(self, caplog):
        with caplog.at_level(logging.INFO):
            _make_provider("gemini-3.7-flash", "low")
        assert any("thinking_level=LOW" in r.message for r in caplog.records)


@pytest.mark.asyncio
class TestRequests:
    async def test_plain_call_sends_thinking_level(self):
        provider = _make_provider("gemini-3.7-flash", "high")
        generate = AsyncMock(return_value=_text_response())
        provider._client.aio.models.generate_content = generate

        await provider.call(messages=[{"role": "user", "content": "hi"}], scope="retain", max_retries=0)

        config = _sent_config(generate)
        assert config.thinking_config is not None
        assert config.thinking_config.thinking_level == genai_types.ThinkingLevel.HIGH

    async def test_plain_call_sends_nothing_when_unconfigured(self):
        provider = _make_provider("gemini-3.7-flash", None)
        generate = AsyncMock(return_value=_text_response())
        provider._client.aio.models.generate_content = generate

        await provider.call(messages=[{"role": "user", "content": "hi"}], scope="retain", max_retries=0)

        config = _sent_config(generate)
        assert config is None or config.thinking_config is None

    async def test_plain_call_sends_nothing_on_pre_3_models(self):
        provider = _make_provider("gemini-2.5-flash", "high")
        generate = AsyncMock(return_value=_text_response())
        provider._client.aio.models.generate_content = generate

        await provider.call(messages=[{"role": "user", "content": "hi"}], scope="retain", max_retries=0)

        config = _sent_config(generate)
        assert config is None or config.thinking_config is None

    async def test_tool_call_sends_thinking_level(self):
        # Reflect is a tool-calling loop; the level must ride on that path too.
        provider = _make_provider("gemini-3.7-flash", "low")
        generate = AsyncMock(return_value=_text_response("done"))
        provider._client.aio.models.generate_content = generate
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "description": "does nothing",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

        await provider.call_with_tools(
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice=LLM_TOOL_CHOICE_AUTO,
            scope="reflect",
            max_retries=0,
        )

        config = _sent_config(generate)
        assert config.thinking_config.thinking_level == genai_types.ThinkingLevel.LOW

    async def test_explicit_extra_body_thinking_config_wins(self):
        # extra_body is the operator's escape hatch for raw GenerateContentConfig
        # fields; a hand-written thinking_config there is not overridden.
        provider = _make_provider("gemini-3.7-flash", "high", extra_body={"thinking_config": {"thinking_level": "LOW"}})
        generate = AsyncMock(return_value=_text_response())
        provider._client.aio.models.generate_content = generate

        await provider.call(messages=[{"role": "user", "content": "hi"}], scope="retain", max_retries=0)

        assert _sent_config(generate).thinking_config.thinking_level == genai_types.ThinkingLevel.LOW
