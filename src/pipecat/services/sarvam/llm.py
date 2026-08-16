#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Sarvam LLM service implementation using OpenAI-compatible interface."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from loguru import logger
from openai import NOT_GIVEN

from pipecat.adapters.services.open_ai_adapter import OpenAILLMInvocationParams
from pipecat.adapters.services.open_ai_adapter import is_given as openai_is_given
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.sarvam._sdk import sdk_headers
from pipecat.services.settings import NOT_GIVEN as _NOT_GIVEN
from pipecat.services.settings import _NotGiven, is_given


@dataclass
class SarvamLLMSettings(OpenAILLMSettings):
    """Settings for SarvamLLMService.

    Parameters:
        wiki_grounding: Sarvam wiki grounding toggle.
        reasoning_effort: Reasoning effort level (low, medium, high).
    """

    wiki_grounding: bool | None | _NotGiven = field(default_factory=lambda: _NOT_GIVEN)
    reasoning_effort: Literal["low", "medium", "high"] | None | _NotGiven = field(
        default_factory=lambda: _NOT_GIVEN
    )


class SarvamLLMService(OpenAILLMService):
    """A service for interacting with Sarvam's API using the OpenAI-compatible interface.

    This service extends OpenAILLMService to connect to Sarvam's API endpoint while
    maintaining full compatibility with OpenAI's interface and functionality.
    """

    # Sarvam doesn't support the "developer" message role.
    # This value is used by BaseOpenAILLMService when calling the adapter.
    supports_developer_role = False

    _SUPPORTED_MODELS = frozenset(
        {"sarvam-30b", "sarvam-30b-16k", "sarvam-105b", "sarvam-105b-32k"}
    )
    Settings = SarvamLLMSettings
    _settings: Settings

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.sarvam.ai/v1",
        settings: Settings | None = None,
        default_headers: Mapping[str, str] | None = None,
        **kwargs,
    ):
        """Initialize Sarvam LLM service.

        Args:
            api_key: Sarvam API key used for both OpenAI auth and Sarvam subscription header.
            base_url: Sarvam OpenAI-compatible base URL.
            settings: Runtime-updatable settings.
            default_headers: Additional HTTP headers to include in requests.
            **kwargs: Additional keyword arguments passed to ``OpenAILLMService``.
        """
        # Initialize only Sarvam-specific defaults; inherited defaults are
        # provided by the OpenAI base service initialization.
        default_settings = self.Settings(
            model="sarvam-30b",
            wiki_grounding=None,
            reasoning_effort=None,
        )

        # Apply settings delta (canonical API, always wins)
        if settings is not None:
            default_settings.apply_update(settings)

        model = default_settings.model
        if not isinstance(model, str):
            raise ValueError("Sarvam LLM requires a non-empty model string.")
        self._validate_model(model)

        super().__init__(
            api_key=api_key,
            base_url=base_url,
            settings=default_settings,
            default_headers=default_headers,
            **kwargs,
        )

    def create_client(
        self,
        api_key=None,
        base_url=None,
        organization=None,
        project=None,
        default_headers=None,
        **kwargs,
    ):
        """Create OpenAI-compatible client for Sarvam API endpoint.

        Ensures Sarvam auth and SDK identification headers are always attached.
        """
        merged_headers = dict(default_headers or {})
        # sdk_headers() carries Pipecat User-Agent and should override caller-provided value.
        merged_headers.update(sdk_headers())
        if api_key:
            merged_headers["api-subscription-key"] = api_key

        logger.debug(f"Creating Sarvam client with API {base_url}")
        return super().create_client(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            project=project,
            default_headers=merged_headers,
            **kwargs,
        )

    def build_chat_completion_params(self, params_from_context: OpenAILLMInvocationParams) -> dict:
        """Build parameters for Sarvam chat completion request.

        Starts from OpenAI-compatible defaults, then removes unsupported
        request fields and applies Sarvam-specific options.

        Sarvam-specific fixes:
        - Sets ``tool_choice="auto"`` when tools are present but no explicit
          tool_choice was set. Sarvam-105b/30b sometimes fail to invoke
          transition functions without this hint, causing workflow nodes to
          stall. With ``tool_choice="auto"`` the model is explicitly told it
          MAY call tools, which significantly improves function-calling
          reliability.
        """
        import time as _time
        print(f"[LLM-TRACE] build_chat_completion_params called at {_time.time():.3f}", flush=True)
        self._validate_tool_parameters(params_from_context)

        params = super().build_chat_completion_params(params_from_context)
        params.pop("stream_options", None)
        params.pop("max_completion_tokens", None)
        params.pop("service_tier", None)

        # Sarvam function-calling reliability fix: if tools are present but
        # tool_choice is NOT_GIVEN, explicitly set "auto" so the model knows
        # it is allowed (and expected) to call functions. Without this,
        # sarvam-105b often generates text instead of calling the transition
        # function, causing the workflow to stall on the current node.
        tools = params.get("tools")
        tool_choice = params.get("tool_choice")
        if tools and (tool_choice is None or tool_choice == NOT_GIVEN):
            params["tool_choice"] = "auto"

        if is_given(self._settings.wiki_grounding) and self._settings.wiki_grounding is not None:
            params["wiki_grounding"] = self._settings.wiki_grounding
        if (
            is_given(self._settings.reasoning_effort)
            and self._settings.reasoning_effort is not None
        ):
            params["reasoning_effort"] = self._settings.reasoning_effort

        print(f"[LLM-TRACE] build_chat_completion_params done, model={params.get('model')}", flush=True)
        return params

    async def get_chat_completions(self, context):
        """Override to wrap the stream and fix Sarvam's non-standard tool arguments.

        Sarvam sometimes sends tool call arguments as multiple ``{}`` chunks
        during streaming. When concatenated by the parent's _process_context,
        these produce invalid JSON like ``{}{}`` which causes
        ``json.loads()`` to fail and the function call to be silently
        dropped — stalling the workflow.

        This override wraps the returned async stream and normalizes each
        chunk's ``tool_call.function.arguments`` so that:
        - Empty ``{}`` chunks are replaced with ``""`` (no contribution)
        - The first real argument JSON is preserved
        - Trailing ``{}`` artifacts are stripped

        This way, when the parent accumulates the arguments string, it
        produces valid JSON instead of ``{}{}``.
        """
        stream = await super().get_chat_completions(context)

        async def _fixed_stream():
            async for chunk in stream:
                try:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.tool_calls:
                        for tc in chunk.choices[0].delta.tool_calls:
                            if tc.function and tc.function.arguments:
                                args = tc.function.arguments
                                # Sarvam sends "{}" as a no-op chunk.
                                # Replace with empty string so it doesn't
                                # corrupt the accumulated arguments.
                                if args.strip() == "{}":
                                    tc.function.arguments = ""
                except Exception:
                    pass  # don't let stream wrapping break the pipeline
                yield chunk

        return _fixed_stream()

    def _validate_model(self, model: str):
        if model not in self._SUPPORTED_MODELS:
            allowed = ", ".join(sorted(self._SUPPORTED_MODELS))
            raise ValueError(f"Unsupported Sarvam LLM model '{model}'. Allowed values: {allowed}.")

    def _validate_tool_parameters(self, params_from_context: OpenAILLMInvocationParams):
        tools = params_from_context.get("tools", NOT_GIVEN)
        tool_choice = params_from_context.get("tool_choice", NOT_GIVEN)

        has_tools = (
            openai_is_given(tools)
            and tools is not None
            and (not isinstance(tools, list) or len(tools) > 0)
        )
        has_tool_choice = openai_is_given(tool_choice) and tool_choice is not None

        if has_tool_choice and not has_tools:
            raise ValueError("Sarvam requires non-empty `tools` when `tool_choice` is provided.")
