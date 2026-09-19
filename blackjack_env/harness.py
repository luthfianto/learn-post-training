# SPDX-License-Identifier: BSD-3-Clause

"""OpenEnv harness integration for the Blackjack environment.

This module provides three things:

1. ``BlackjackSession`` / ``BlackjackSessionFactory`` — an in-process
   ``ResourceSession`` implementation that drives ``BlackjackEnvironment``
   directly (no HTTP round-trips), exposing the game as a single MCP tool
   named ``play``.
2. ``build_scripted_model_step`` — a deterministic basic-strategy "teacher"
   used by ``collect_sft.py`` and the evaluation baseline.
3. ``build_blackjack_rollout_func`` — a TRL ``GRPOTrainer`` custom
   ``rollout_func``. Unlike ``openenv.core.harness.build_harness_rollout_func``
   (which concatenates per-turn prompt ids across a multi-turn episode), this
   builds a single coherent token stream: the initial prompt plus every model
   token and environment-observation token, with an ``env_mask`` marking which
   tokens the policy actually generated. That is exactly the contract TRL's
   tool-calling path expects.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from openenv.core.env_server.mcp_types import Tool
from openenv.core.harness import (
    ModelStepResult,
    ResourceSession,
    ResourceSessionFactory,
    ToolResult,
    VerifyResult,
)
from openenv.core.llm_client import LLMResponse, ToolCall

try:  # pragma: no cover - import path depends on launch mode
    from .models import BlackjackAction
    from .strategy import basic_strategy_action
except ImportError:  # pragma: no cover
    from models import BlackjackAction
    from strategy import basic_strategy_action


SYSTEM_PROMPT = (
    "You are playing blackjack with a 6-deck shoe. The dealer stands on all 17. "
    "On every turn you must choose exactly one action: 'hit' or 'stand'. "
    "Reply with a single JSON object and nothing else, for example "
    '{"action": "hit"} or {"action": "stand"}. '
    "Payouts: player blackjack +1.5, win +1, push 0, loss -1. "
    "Play to maximise expected reward."
)

PLAY_TOOL = Tool(
    name="play",
    description="Take a blackjack action: hit (draw a card) or stand (end your turn).",
    input_schema={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["hit", "stand"],
                "description": "The action to take.",
            }
        },
        "required": ["action"],
    },
)


# --------------------------------------------------------------------------- #
# Text helpers
# --------------------------------------------------------------------------- #
def parse_action_from_text(text: str) -> str:
    """Extract ``hit``/``stand`` from a model completion, defaulting to stand."""
    match = re.search(r'"action"\s*:\s*"?(hit|stand)"?', text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    match = re.search(r"\b(hit|stand)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return "stand"


def extract_seed(conversation: Any) -> Optional[int]:
    """Pull a ``Seed: N`` marker out of a conversational prompt, if present."""
    if isinstance(conversation, int):
        return conversation
    if isinstance(conversation, dict) and "seed" in conversation:
        return int(conversation["seed"])
    text = (
        json.dumps(conversation) if not isinstance(conversation, str) else conversation
    )
    match = re.search(r"seed\s*[:=]\s*(\d+)", text, re.IGNORECASE)
    return int(match.group(1)) if match else None


def _state_dict(observation: Any) -> dict[str, Any]:
    return {
        "player_hand": list(observation.player_hand),
        "dealer_upcard": observation.dealer_upcard,
        "player_total": observation.player_total,
        "legal_actions": list(observation.legal_actions),
        "done": observation.done,
    }


def extract_state_from_messages(
    messages: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Find the most recent machine-readable state embedded in a transcript."""
    for message in reversed(messages):
        content = message.get("content") or ""
        if not isinstance(content, str):
            continue
        match = re.search(r'\{[^{}]*"player_hand"[^{}]*\}', content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
    return None


# --------------------------------------------------------------------------- #
# Resource session
# --------------------------------------------------------------------------- #
class BlackjackSession(ResourceSession):
    """Drive a ``BlackjackEnvironment`` instance for a single rollout."""

    def __init__(
        self,
        env: Any,
        *,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        system_prompt: str = SYSTEM_PROMPT,
    ):
        self._env = env
        self._system_prompt = system_prompt
        self._step_count = 0
        self._last_reward: float = 0.0
        self._done = False
        self._observation = self._env.reset(seed=seed, episode_id=episode_id)

    def initial_messages(self) -> list[dict[str, Any]]:
        state = _state_dict(self._observation)
        return [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "user",
                "content": (
                    f"{self._observation.message}\n\n"
                    f"State: {json.dumps(state)}\n"
                    "Reply with one JSON action."
                ),
            },
        ]

    def list_tools(self) -> list[Tool]:
        return [PLAY_TOOL]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        if name != PLAY_TOOL.name:
            raise KeyError(f"Unknown tool: {name}")
        action = arguments.get("action")
        if action not in ("hit", "stand"):
            raise ValueError(f"Invalid action {action!r}; expected 'hit' or 'stand'.")

        observation = self._env.step(BlackjackAction(action=action))
        self._step_count += 1
        self._last_reward = float(observation.reward or 0.0)
        self._done = bool(observation.done)
        self._observation = observation

        data = {
            "message": observation.message,
            "player_hand": list(observation.player_hand),
            "dealer_upcard": observation.dealer_upcard,
            "player_total": observation.player_total,
            "legal_actions": list(observation.legal_actions),
            "reward": observation.reward,
            "done": observation.done,
        }
        return ToolResult(
            data=data,
            done=self._done,
            metadata={"reward": observation.reward, "action": action},
        )

    def verify(
        self,
        transcript: list[dict[str, Any]],
        final_state: Any | None = None,
    ) -> VerifyResult:
        return VerifyResult(
            env_reward=self._last_reward,
            done=self._done,
            metrics={
                "steps": self._step_count,
                "terminal_reason": self._observation.terminal_reason,
            },
            artifacts={"final_message": self._observation.message},
        )

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if callable(close):
            close()


class BlackjackSessionFactory(ResourceSessionFactory[BlackjackSession]):
    """Create isolated ``BlackjackSession`` instances for each rollout."""

    def __init__(self, num_decks: int = 6):
        self._num_decks = num_decks

    def create(
        self,
        task: Any = None,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> BlackjackSession:
        try:
            from .server.blackjack_env_environment import BlackjackEnvironment
        except ImportError:  # pragma: no cover
            from server.blackjack_env_environment import BlackjackEnvironment

        if seed is None:
            seed = extract_seed(task)
        env = BlackjackEnvironment(num_decks=self._num_decks)
        return BlackjackSession(env, seed=seed, episode_id=episode_id)


# --------------------------------------------------------------------------- #
# Scripted teacher
# --------------------------------------------------------------------------- #
def build_scripted_model_step():
    """A ``ModelStep`` that plays exact basic strategy (no LLM required)."""

    def model_step(messages, tools, sampling):
        state = extract_state_from_messages(messages)
        if state and state.get("player_hand"):
            action = basic_strategy_action(
                state["player_hand"], int(state["dealer_upcard"])
            )
        else:
            action = "stand"
        return ModelStepResult(
            response=LLMResponse(
                content=json.dumps({"action": action}),
                tool_calls=[
                    ToolCall(
                        id=f"scripted-{action}",
                        name=PLAY_TOOL.name,
                        args={"action": action},
                    )
                ],
            )
        )

    return model_step


# --------------------------------------------------------------------------- #
# TRL GRPO rollout function
# --------------------------------------------------------------------------- #
def _format_prompt(tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template) and getattr(tokenizer, "chat_template", None):
        try:
            return apply_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:  # pragma: no cover - fall back to manual formatting
            pass
    parts = [f"{m['role']}: {m['content']}" for m in messages]
    parts.append("assistant:")
    return "\n".join(parts)


def _generate_turn(model, tokenizer, input_ids, *, max_new_tokens, temperature, top_p):
    import torch

    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], device=device)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    do_sample = temperature is not None and temperature > 0
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=pad_token_id,
        )

    generated = output.sequences[0, input_tensor.shape[1] :].tolist()
    logprobs: list[float] = []
    for step, scores in enumerate(output.scores):
        log_softmax = torch.log_softmax(scores[0].float(), dim=-1)
        logprobs.append(float(log_softmax[generated[step]].item()))
    return generated, logprobs


def build_blackjack_rollout_func(
    session_factory: BlackjackSessionFactory,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    max_turns: int = 8,
    max_new_tokens: int = 16,
):
    """Build a TRL ``rollout_func`` for ``GRPOTrainer``.

    Returns a callable ``(prompts, trainer) -> dict`` whose keys match the
    experimental custom-rollout contract: ``prompt_ids``, ``completion_ids``,
    ``logprobs``, plus ``env_mask`` (1 = policy token, 0 = environment token),
    ``env_reward`` and ``verify_metrics``.
    """

    def rollout_func(prompts, trainer):
        model = trainer.model
        if getattr(trainer, "accelerator", None) is not None:
            model = trainer.accelerator.unwrap_model(model)
        tokenizer = trainer.processing_class

        args = getattr(trainer, "args", None)
        temperature = getattr(args, "temperature", 1.0)
        top_p = getattr(args, "top_p", 1.0)
        completion_cap = getattr(args, "max_completion_length", None)
        turn_budget = max_turns
        if completion_cap:
            turn_budget = max(
                1, min(max_turns, completion_cap // max(1, max_new_tokens))
            )

        was_training = model.training
        model.eval()

        batch = {
            "prompt_ids": [],
            "completion_ids": [],
            "logprobs": [],
            "env_mask": [],
            "env_reward": [],
            "verify_metrics": [],
        }

        try:
            for conversation in prompts:
                session = session_factory.create(task=conversation)
                try:
                    messages = session.initial_messages()
                    prompt_text = _format_prompt(tokenizer, messages)
                    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)[
                        "input_ids"
                    ]
                    input_ids = list(prompt_ids)

                    completion_ids: list[int] = []
                    logprobs: list[float] = []
                    env_mask: list[int] = []
                    reward = 0.0

                    for _ in range(turn_budget):
                        gen_ids, gen_logprobs = _generate_turn(
                            model,
                            tokenizer,
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                        if not gen_ids:
                            break
                        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                        completion_ids.extend(gen_ids)
                        logprobs.extend(gen_logprobs)
                        env_mask.extend([1] * len(gen_ids))
                        input_ids.extend(gen_ids)

                        action = parse_action_from_text(text)
                        tool_result = session.call_tool("play", {"action": action})
                        reward = float(tool_result.metadata.get("reward") or 0.0)
                        if tool_result.done:
                            break

                        observation_text = (
                            "\n" + str(tool_result.data.get("message", "")) + "\n"
                        )
                        obs_ids = tokenizer(observation_text, add_special_tokens=False)[
                            "input_ids"
                        ]
                        completion_ids.extend(obs_ids)
                        logprobs.extend([0.0] * len(obs_ids))
                        env_mask.extend([0] * len(obs_ids))
                        input_ids.extend(obs_ids)

                    verify = session.verify(transcript=[])
                    batch["prompt_ids"].append(prompt_ids)
                    batch["completion_ids"].append(completion_ids)
                    batch["logprobs"].append(logprobs)
                    batch["env_mask"].append(env_mask)
                    batch["env_reward"].append(reward)
                    batch["verify_metrics"].append(dict(verify.metrics))
                finally:
                    session.close()
        finally:
            if was_training:
                model.train()

        return batch

    return rollout_func


def env_reward_func(completions, env_reward=None, **kwargs):
    """Reward function forwarding the environment reward to GRPO."""
    if env_reward is None:
        return [0.0] * len(completions)
    return [float(value) for value in env_reward]


__all__ = [
    "BlackjackSession",
    "BlackjackSessionFactory",
    "PLAY_TOOL",
    "SYSTEM_PROMPT",
    "build_blackjack_rollout_func",
    "build_scripted_model_step",
    "env_reward_func",
    "extract_seed",
    "extract_state_from_messages",
    "parse_action_from_text",
]
