"""OpenAI reasoning models are recognised by family, not a fixed list of names.

Both OpenAI providers used to decide "reasoning model" with
`any(x in model for x in ["gpt-5", "o1", "o3"])`. A later generation such as
gpt-6-luna matched none of them, so the Responses provider silently dropped the
configured reasoning effort, and both providers sent temperature and skipped the
16000-token output floor, the request shape a reasoning model needs. o4-mini was
missed the same way.
"""

import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hindsight_api.engine.providers.openai_compatible_llm import OpenAICompatibleLLM, _is_openai_reasoning_model
from hindsight_api.engine.providers.openai_responses_llm import OpenAIResponsesLLM


@pytest.mark.parametrize(
    "model",
    [
        "gpt-6-luna",
        "openai/gpt-6-luna",
        "GPT-6-Luna",
        "gpt-10",
        "o4-mini",
        "o4-mini-2025-04-16",
    ],
)
def test_later_generations_are_recognised(model):
    # Intention: the reported bug. Names the old list could not match.
    # Expected: recognised by family, with no per-model change.
    assert _is_openai_reasoning_model(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "gpt-5",
        "gpt-5-mini",
        "gpt-5.4-nano",
        "gpt-5.6-terra",
        "o1",
        "o1-mini",
        "o3",
        "o3-pro",
        "openai/o3-mini",
        "ft:o4-mini-2025-04-16:acme::abc123",
        "ft:gpt-5-mini-2025-08-07:acme::abc123",
        # Azure: the model field is the deployment name the operator chose.
        "prod-gpt-5-mini",
    ],
)
def test_names_the_old_list_matched_still_match(model):
    # Intention: widening the match must not narrow it. Provider prefixes, fine-tune
    # ids and Azure deployment names all matched the substring check.
    # Expected: every one is still a reasoning model.
    assert _is_openai_reasoning_model(model) is True


@pytest.mark.parametrize(
    "model",
    [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4o-mini",
        "ft:gpt-4o-mini-2024-07-18:acme:custom:abc123",
        "gpt-oss-120b",
        "openai/gpt-oss-20b",
        "claude-sonnet-5",
        "anthropic/claude-opus-5-5",
        "gemini-3-pro",
        "gemini-2.5-flash",
        "omni-moderation-latest",
    ],
)
def test_non_reasoning_models_are_not(model):
    # Intention: the generation number decides, so gpt-4.x stays out, and "o" must
    # be followed by a digit to be an o-series name.
    # Expected: none of these get the reasoning request shape.
    assert _is_openai_reasoning_model(model) is False


@pytest.mark.asyncio
async def test_responses_provider_applies_reasoning_shape_to_gpt6():
    # Intention: the reported bug end to end, gpt-6-luna with a configured effort.
    # Expected: the effort reaches the request, temperature is suppressed and the
    # output floor applies.
    llm = OpenAIResponsesLLM(
        provider="openai-responses", api_key="sk-test", base_url="", model="gpt-6-luna", reasoning_effort="high"
    )
    usage = types.SimpleNamespace(
        input_tokens=10,
        output_tokens=5,
        input_tokens_details=types.SimpleNamespace(cached_tokens=0),
        output_tokens_details=types.SimpleNamespace(reasoning_tokens=0),
    )
    create = AsyncMock(return_value=types.SimpleNamespace(output_text="ok", output=[], usage=usage, status="completed"))
    llm._client.responses.create = create
    with patch("hindsight_api.engine.providers.openai_responses_llm.get_metrics_collector"):
        await llm.call(messages=[{"role": "user", "content": "hi"}], temperature=0.5, max_completion_tokens=500)

    kwargs = create.call_args.kwargs
    assert kwargs["reasoning"] == {"effort": "high"}
    assert "temperature" not in kwargs
    assert kwargs["max_output_tokens"] == 16000


@pytest.mark.asyncio
async def test_compatible_provider_applies_reasoning_shape_to_gpt6():
    # Intention: the same gap on chat/completions, where the effort was already
    # sent (#3449) but the request shape still keyed off the name list.
    # Expected: the output floor applies and temperature is suppressed.
    llm = OpenAICompatibleLLM(provider="openai", api_key="test", base_url="", model="gpt-6-luna")
    response = MagicMock()
    response.error = None
    response.model_dump.return_value = {}
    response.usage.prompt_tokens = 10
    response.usage.completion_tokens = 5
    response.usage.total_tokens = 15
    response.usage.completion_tokens_details = None
    response.choices[0].finish_reason = "stop"
    response.choices[0].message.content = "ok"
    response.choices[0].message.tool_calls = None
    with patch.object(llm._client.chat.completions, "create", new_callable=AsyncMock) as create:
        create.return_value = response
        await llm.call(
            messages=[{"role": "user", "content": "hi"}],
            max_completion_tokens=2000,
            temperature=0.3,
            max_retries=0,
        )

    params = create.await_args.kwargs
    assert params["max_completion_tokens"] == 16000
    assert "temperature" not in params
