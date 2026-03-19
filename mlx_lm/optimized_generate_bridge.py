from __future__ import annotations

from typing import Any, Dict, Generator, Optional

from .tokenizer_utils import TokenizerWrapper


_SUPPORTED_COMMON_KWARGS = {
    "prompt_cache",
    "prefill_step_size",
    "sampler",
    "logits_processors",
    "prompt_progress_callback",
}

_UNSUPPORTED_KWARGS = {
    "draft_model",
    "input_embeddings",
    "kv_bits",
    "kv_group_size",
    "quantized_kv_start",
    "max_kv_size",
    "num_draft_tokens",
}


def _reject_unsupported_kwargs(
    kwargs: Dict[str, Any],
    *,
    allow_verbose: bool = False,
    extra_supported: Optional[set[str]] = None,
) -> None:
    extra_supported = extra_supported or set()
    unsupported = [key for key in kwargs if key in _UNSUPPORTED_KWARGS]
    if unsupported:
        key = unsupported[0]
        raise ValueError(
            f"[optimized_generate_bridge] `{key}` is not supported by the "
            "optimized mlx.nn generation path."
        )

    unknown = [
        key
        for key in kwargs
        if key not in _SUPPORTED_COMMON_KWARGS
        and key not in extra_supported
        and key != "verbose"
    ]
    if unknown:
        unexpected = ", ".join(sorted(unknown))
        raise ValueError(
            f"[optimized_generate_bridge] Unsupported keyword arguments: {unexpected}"
        )

    if not allow_verbose and kwargs.get("verbose", False):
        raise ValueError(
            "[optimized_generate_bridge] `verbose` is only supported by "
            "optimized_generate_bridge(...)."
        )


def to_generation_response(response: Any) -> Any:
    """Convert an ``mlx.nn`` streamed response into ``mlx_lm`` generation form."""

    from .generate import GenerationResponse

    return GenerationResponse(
        text=response.text,
        token=response.token,
        logprobs=response.logprobs,
        from_draft=getattr(response, "from_draft", False),
        prompt_tokens=response.prompt_tokens,
        prompt_tps=response.prompt_tps,
        generation_tokens=response.generation_tokens,
        generation_tps=response.generation_tps,
        peak_memory=getattr(response, "peak_memory", 0.0),
        finish_reason=getattr(response, "finish_reason", None),
    )


def _normalize_tokenizer(tokenizer: Any) -> Any:
    if isinstance(tokenizer, TokenizerWrapper):
        return tokenizer
    return TokenizerWrapper(tokenizer)


def optimized_stream_generate_bridge(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    max_tokens: int = 256,
    draft_model: Optional[Any] = None,
    **kwargs,
) -> Generator[Any, None, None]:
    """Stream generation through the optimized ``mlx.nn`` path."""

    if draft_model is not None:
        raise ValueError(
            "[optimized_stream_generate_bridge] `draft_model` is not supported "
            "by the optimized mlx.nn generation path."
        )

    tokenizer = _normalize_tokenizer(tokenizer)
    _reject_unsupported_kwargs(kwargs)
    prompt_cache = kwargs.pop("prompt_cache", None)
    prefill_step_size = kwargs.pop("prefill_step_size", 2048)
    sampler = kwargs.pop("sampler", None)
    logits_processors = kwargs.pop("logits_processors", None)
    prompt_progress_callback = kwargs.pop("prompt_progress_callback", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise ValueError(
            f"[optimized_stream_generate_bridge] Unsupported keyword arguments: {unexpected}"
        )

    import mlx.nn as nn

    for response in nn.optimized_stream_generate(
        model,
        tokenizer,
        prompt,
        prompt_cache=prompt_cache,
        max_tokens=max_tokens,
        prefill_step_size=prefill_step_size,
        sampler=sampler,
        logits_processors=logits_processors,
        prompt_progress_callback=prompt_progress_callback,
    ):
        yield to_generation_response(response)


def optimized_generate_bridge(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    verbose: bool = False,
    draft_model: Optional[Any] = None,
    **kwargs,
) -> str:
    """Generate text through the optimized ``mlx.nn`` path."""

    if draft_model is not None:
        raise ValueError(
            "[optimized_generate_bridge] `draft_model` is not supported by the "
            "optimized mlx.nn generation path."
        )

    tokenizer = _normalize_tokenizer(tokenizer)
    _reject_unsupported_kwargs(
        kwargs, allow_verbose=True, extra_supported={"max_tokens"}
    )
    max_tokens = kwargs.pop("max_tokens", 256)
    prompt_cache = kwargs.pop("prompt_cache", None)
    prefill_step_size = kwargs.pop("prefill_step_size", 2048)
    sampler = kwargs.pop("sampler", None)
    logits_processors = kwargs.pop("logits_processors", None)
    prompt_progress_callback = kwargs.pop("prompt_progress_callback", None)
    kwargs.pop("verbose", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs.keys()))
        raise ValueError(
            f"[optimized_generate_bridge] Unsupported keyword arguments: {unexpected}"
        )

    import mlx.nn as nn

    return nn.optimized_generate(
        model,
        tokenizer,
        prompt,
        verbose=verbose,
        max_tokens=max_tokens,
        prompt_cache=prompt_cache,
        prefill_step_size=prefill_step_size,
        sampler=sampler,
        logits_processors=logits_processors,
        prompt_progress_callback=prompt_progress_callback,
    )


__all__ = [
    "optimized_generate_bridge",
    "optimized_stream_generate_bridge",
    "to_generation_response",
]
