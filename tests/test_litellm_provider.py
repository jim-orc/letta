import pytest

from letta.llm_api.openai_client import _convert_responses_payload_to_chat_payload, use_responses_api
from letta.schemas.llm_config import LLMConfig
from letta.schemas.providers.litellm import LiteLLMProvider


@pytest.mark.asyncio
async def test_litellm_provider_uses_stable_litellm_handles():
    provider = LiteLLMProvider(name="litellm", base_url="http://litellm:4000/v1")

    models = await provider._list_llm_models([
        {"id": "gpt-5.4"},
    ])

    assert len(models) == 1
    assert models[0].handle == "litellm/gpt-5.4"
    assert models[0].model_endpoint_type == "litellm"


def test_litellm_prefers_responses_api_for_tools_even_when_model_is_not_reasoning():
    llm_config = LLMConfig(
        model="gpt-4.1",
        model_endpoint_type="litellm",
        model_endpoint="http://litellm:4000/v1",
        context_window=128000,
        provider_name="litellm",
    )

    assert use_responses_api(llm_config, tools=[{"name": "send_message"}]) is True


def test_litellm_does_not_force_responses_without_reasoning_or_tools():
    llm_config = LLMConfig(
        model="gpt-4.1",
        model_endpoint_type="litellm",
        model_endpoint="http://litellm:4000/v1",
        context_window=128000,
        provider_name="litellm",
    )

    assert use_responses_api(llm_config, tools=None) is False


def test_convert_responses_payload_to_chat_payload_for_litellm_fallback():
    responses_payload = {
        "model": "gpt-4.1",
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "hello"},
                    {"type": "input_text", "text": "world"},
                ],
            }
        ],
        "tools": [{"type": "function", "name": "send_message", "parameters": {"type": "object"}, "strict": True}],
        "tool_choice": {"type": "function", "name": "send_message"},
        "store": False,
        "include": ["reasoning.encrypted_content"],
    }

    chat_payload = _convert_responses_payload_to_chat_payload(responses_payload)

    assert "input" not in chat_payload
    assert chat_payload["messages"][0]["role"] == "user"
    assert chat_payload["messages"][0]["content"] == "hello\nworld"
    assert chat_payload["tools"][0]["type"] == "function"
    assert chat_payload["tools"][0]["function"]["name"] == "send_message"
    assert chat_payload["tool_choice"] == {"type": "function", "function": {"name": "send_message"}}
