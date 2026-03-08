# Letta native LiteLLM provider spike (findings + implementation plan)

## Findings: current provider architecture

1. **Providers are first-class persisted entities** (`Provider`, `ProviderModel`) and are typed by `ProviderType`.
   - Runtime provider instances are created via `Provider.cast_to_subtype()`.
2. **Provider type enumeration/validation** is centralized in `ProviderType` and in `LLMConfig`/`Model` endpoint literals.
3. **Model sync path** is centralized in `ProviderManager._sync_default_models_for_provider()`, where each typed provider is asked to list LLM and embedding models.
4. **Runtime request behavior** is delegated to `LLMClient.create(provider_type=...)` and provider-specific client classes (OpenAI, Anthropic, Gemini, etc.).
5. **OpenAI semantics assumptions** currently sit in the OpenAI client adapter (`build_request_data`), including Chat-vs-Responses selection logic.

## Recommendation: Option A wins (minimal in-tree provider)

Use **Option A** (minimal in-tree `litellm` provider) now.

Why:
- The provider seam is already explicit and can absorb a new provider with a small, local diff.
- Existing OpenAI-compatible request adapters can be reused.
- This keeps fork maintenance cost low and avoids broad architectural movement before proving the LiteLLM path end-to-end.

Plugin architecture should be deferred until there is evidence that additional non-core providers are frequent and costly to carry in-tree.

## Minimal implementation shape

### New provider type
- Add `ProviderType.litellm`.
- Add `LiteLLMProvider` class as an OpenAI-compatible provider with stable `litellm/*` model handles.

### Model sync + selection
- Ensure LiteLLM models sync with `model_endpoint_type="litellm"`.
- Keep model listing via LiteLLM `/v1/models` and store provider metadata in Letta as usual.

### Runtime request-mode routing
- Keep OpenAI client stack.
- Add provider-aware request mode selection:
  - use `/v1/responses` for reasoning models (existing behavior), and
  - for `litellm`, prefer `/v1/responses` when reasoning/tool settings are present.

### Base-provider bootstrapping
- Add optional env config for first-class base LiteLLM provider:
  - `LITELLM_API_BASE`
  - `LITELLM_API_KEY`
  - `LITELLM_HANDLE_BASE` (optional)

## Estimated scope

- Files touched: ~10-15
- Diff size: small/medium; mostly enum/plumbing + one new provider class + request mode tweak
- No broad provider-architecture refactor required

## Setup notes

Example env:

```bash
export LITELLM_API_BASE="http://litellm:4000"
export LITELLM_API_KEY="sk-your-litellm-key"
export LITELLM_HANDLE_BASE="litellm"
```

Letta will register LiteLLM as a base provider when `LITELLM_API_BASE` and `LITELLM_API_KEY` are set.

## Concise test plan

1. Provider registration + model sync
   - Start Letta with LiteLLM env vars.
   - Verify provider appears as `provider_type=litellm` and models sync with `litellm/*` handles.
2. Agent CRUD
   - Create/update/load agent on a LiteLLM-backed model handle.
3. Request mode routing
   - Use a GPT-5.x model through LiteLLM with tools + reasoning effort.
   - Confirm Letta emits Responses-format payloads (not chat-completions payloads).
4. Non-regression
   - Smoke test existing OpenAI and Anthropic flows.

## Deferred follow-ups

- Capability cache/manifest per model from LiteLLM metadata.
- Optional explicit per-model override for forced chat vs responses mode.
- Broader provider registry/pluginization once multiple out-of-tree providers justify it.
