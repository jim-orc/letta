from typing import Literal

from pydantic import Field

from letta.schemas.enums import ProviderCategory, ProviderType
from letta.schemas.llm_config import LLMConfig
from letta.schemas.providers.openai import OpenAIProvider


class LiteLLMProvider(OpenAIProvider):
    """First-class LiteLLM provider built on top of the OpenAI-compatible provider stack."""

    provider_type: Literal[ProviderType.litellm] = Field(ProviderType.litellm, description="The type of the provider.")
    provider_category: ProviderCategory = Field(ProviderCategory.byok, description="The category of the provider (base or byok)")
    base_url: str = Field("http://localhost:4000/v1", description="Base URL for the LiteLLM API.")

    async def _list_llm_models(self, data: list[dict]) -> list[LLMConfig]:
        """List models from LiteLLM using stable `litellm/*` handles.

        LiteLLM normalizes model/provider metadata, so we intentionally avoid
        OpenAI-specific allow/deny filtering and defer capability behavior to
        runtime request routing.
        """
        configs: list[LLMConfig] = []
        for model in data:
            model_name_and_context = await self._do_model_checks_for_name_and_context_size_async(model)
            if model_name_and_context is None:
                continue
            model_name, context_window_size = model_name_and_context

            config = LLMConfig(
                model=model_name,
                model_endpoint_type="litellm",
                model_endpoint=self.base_url,
                context_window=context_window_size,
                handle=self.get_handle(model_name, base_name=self.name or "litellm"),
                max_tokens=await self.get_default_max_output_tokens_async(model_name),
                provider_name=self.name,
                provider_category=self.provider_category,
            )
            config = self._set_model_parameter_tuned_defaults(model_name, config)
            configs.append(config)

        configs.sort(key=lambda x: x.model)
        return configs
