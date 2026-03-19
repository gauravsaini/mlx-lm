# Copyright © 2026 Apple Inc.

import contextlib
import inspect
import io
import unittest
from unittest import mock

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx_lm import generate, stream_generate
from mlx_lm.generate import GenerationResponse


class DummyTokenizer:
    bos_token = None
    chat_template = None
    eos_token_id = 0
    eos_token_ids = [0]
    clean_up_tokenization_spaces = False

    def get_vocab(self):
        return {"<eos>": 0, "a": 1, "b": 2, "c": 3, "d": 4}

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        if isinstance(text, str):
            return [1, 2]
        return list(text)

    def decode(self, tokens):
        return "".join(chr(ord("a") + int(token)) for token in tokens)


class DummyModel(nn.Module):
    def make_cache(self):
        return [type("DummyCache", (), {"state": mx.array([0])})()]

    def __call__(self, tokens, cache=None):
        del cache
        token_list = mx.contiguous(tokens[0]).tolist()
        next_token = (token_list[-1] + 1) % 5
        logits = mx.full((1, len(token_list), 5), -100.0)
        logits[:, -1, next_token] = 2.0
        return logits


class TestGenerateOptimized(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.set_default_device(mx.cpu)
        cls.model = DummyModel()
        cls.tokenizer = DummyTokenizer()

    def _supports_optimized_kwarg(self):
        return "use_mlx_nn_optimized" in inspect.signature(stream_generate).parameters

    def test_stock_stream_generate_unchanged(self):
        with mock.patch(
            "mlx_lm.generate._optimized_generate_bridge",
            side_effect=AssertionError("optimized bridge should not be used"),
        ):
            responses = list(
                stream_generate(
                    self.model,
                    self.tokenizer,
                    [1, 2],
                    max_tokens=2,
                    sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
                    prompt_progress_callback=lambda *_: None,
                )
            )

        self.assertEqual(len([r.text for r in responses if r.text]), 2)
        self.assertTrue(all(isinstance(r, GenerationResponse) for r in responses))
        self.assertEqual(responses[-1].finish_reason, "length")

    def test_stock_generate_unchanged(self):
        with mock.patch(
            "mlx_lm.generate._optimized_generate_bridge",
            side_effect=AssertionError("optimized bridge should not be used"),
        ):
            text = generate(
                self.model,
                self.tokenizer,
                [1, 2],
                max_tokens=2,
                sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
                verbose=False,
            )

        self.assertTrue(isinstance(text, str))
        self.assertGreater(len(text), 0)

    def test_opt_in_stream_generate_contract(self):
        if not self._supports_optimized_kwarg():
            self.skipTest("optimized mlx_lm generation path is not implemented yet")

        events = []
        responses = list(
            stream_generate(
                self.model,
                self.tokenizer,
                [1, 2, 3, 4, 5],
                max_tokens=2,
                sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
                prefill_step_size=2,
                prompt_progress_callback=lambda processed, total: events.append(
                    (processed, total)
                ),
                use_mlx_nn_optimized=True,
            )
        )

        self.assertTrue(all(isinstance(r, GenerationResponse) for r in responses))
        self.assertEqual(len([r.text for r in responses if r.text]), 2)
        self.assertEqual(responses[-1].finish_reason, "length")
        self.assertEqual(events[0], (0, 5))
        self.assertEqual(events[-1], (5, 5))
        self.assertTrue(all(total == 5 for _, total in events))
        self.assertTrue(all(processed <= 5 for processed, _ in events))

    def test_opt_in_generate_verbose_and_plain_tokenizer(self):
        if not self._supports_optimized_kwarg():
            self.skipTest("optimized mlx_lm generation path is not implemented yet")

        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            text = generate(
                self.model,
                self.tokenizer,
                [1, 2],
                max_tokens=2,
                sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
                verbose=True,
                use_mlx_nn_optimized=True,
            )

        output = stream.getvalue()
        self.assertTrue(isinstance(text, str))
        self.assertGreater(len(text), 0)
        self.assertIn("==========", output)
        self.assertIn(text, output)
        self.assertIn("Prompt:", output)
        self.assertIn("Generation:", output)
        self.assertIn("Peak memory:", output)

    def test_opt_in_rejects_unsupported_kwargs(self):
        if not self._supports_optimized_kwarg():
            self.skipTest("optimized mlx_lm generation path is not implemented yet")

        unsupported_kwargs = (
            {"draft_model": self.model},
            {"num_draft_tokens": 2},
            {"input_embeddings": mx.array([1, 2])},
            {"kv_bits": 4},
            {"kv_group_size": 8},
            {"quantized_kv_start": 1},
            {"max_kv_size": 16},
        )

        for extra_kwargs in unsupported_kwargs:
            with self.subTest(kwargs=extra_kwargs):
                with self.assertRaises(ValueError):
                    list(
                        stream_generate(
                            self.model,
                            self.tokenizer,
                            [1, 2],
                            max_tokens=2,
                            sampler=lambda logprobs: mx.argmax(
                                logprobs, axis=-1
                            ),
                            use_mlx_nn_optimized=True,
                            **extra_kwargs,
                        )
                    )

        for extra_kwargs in unsupported_kwargs:
            with self.subTest(generate_kwargs=extra_kwargs):
                with self.assertRaises(ValueError):
                    generate(
                        self.model,
                        self.tokenizer,
                        [1, 2],
                        max_tokens=2,
                        sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
                        use_mlx_nn_optimized=True,
                        **extra_kwargs,
                    )

    def test_opt_in_tokenizer_without_detokenizer(self):
        if not self._supports_optimized_kwarg():
            self.skipTest("optimized mlx_lm generation path is not implemented yet")

        class PlainTokenizer:
            bos_token = None
            chat_template = None
            eos_token_id = 0
            eos_token_ids = [0]
            clean_up_tokenization_spaces = False

            def get_vocab(self):
                return {"<eos>": 0, "a": 1, "b": 2, "c": 3, "d": 4}

            def encode(self, text, add_special_tokens=True):
                del add_special_tokens
                if isinstance(text, str):
                    return [1, 2]
                return list(text)

            def decode(self, tokens):
                return "".join(chr(ord("a") + int(token)) for token in tokens)

        responses = list(
            stream_generate(
                self.model,
                PlainTokenizer(),
                "prompt text",
                max_tokens=2,
                sampler=lambda logprobs: mx.argmax(logprobs, axis=-1),
                use_mlx_nn_optimized=True,
            )
        )

        self.assertEqual(len([r.text for r in responses if r.text]), 2)
        self.assertEqual(responses[-1].finish_reason, "length")


if __name__ == "__main__":
    unittest.main()
