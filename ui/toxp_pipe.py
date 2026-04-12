"""
title: TOXP - Team of Experts
description: Parallel multi-agent reasoning via AWS Bedrock. Spawns N independent reasoning agents and synthesizes their outputs through a coordinator.
author: Team of Experts Contributors
version: 0.5.0
license: MIT
"""

import asyncio
import time
from typing import AsyncGenerator, Callable, Awaitable, Optional

from pydantic import BaseModel, Field


class Pipe:
    """Open WebUI Pipe for TOXP multi-agent reasoning.

    Progress appears in the status bar (updates in-place).
    Configure per-chat: Controls panel (top-right) > expand Valves.
    Configure admin: Workspace > Functions > TOXP (gear icon).
    """

    class Valves(BaseModel):
        """Admin settings: Workspace > Functions > TOXP (gear icon)."""

        aws_profile: str = Field(
            default="toxp",
            description="AWS profile name from ~/.aws/config (empty = use toxp config / env vars)",
        )
        region: str = Field(
            default="",
            description="AWS region (empty = use toxp config)",
        )
        model: str = Field(
            default="",
            description="Bedrock model ID (empty = use toxp config)",
        )
        context_1m: Optional[bool] = Field(
            default=None,
            description="Enable 1M token context (None = use toxp config)",
        )

    class UserValves(BaseModel):
        """Per-chat settings: Controls panel (top-right) > Valves."""

        num_agents: int = Field(
            default=15,
            ge=2,
            le=32,
            description="Number of parallel reasoning agents (2-32)",
        )
        max_concurrency: int = Field(
            default=5,
            ge=1,
            le=20,
            description="Max concurrent Bedrock API calls",
        )
        temperature: float = Field(
            default=0.9,
            ge=0.0,
            le=1.0,
            description="Agent sampling temperature (higher = more diverse)",
        )
        coordinator_temperature: float = Field(
            default=0.7,
            ge=0.0,
            le=1.0,
            description="Coordinator synthesis temperature",
        )
        max_tokens: int = Field(
            default=8192,
            ge=256,
            le=65536,
            description="Max tokens per response",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

    def _build_config_overrides(self, user_valves: Optional["Pipe.UserValves"] = None) -> dict:
        uv = user_valves or self.user_valves
        overrides: dict = {
            "num_agents": uv.num_agents,
            "max_concurrency": uv.max_concurrency,
            "temperature": uv.temperature,
            "coordinator_temperature": uv.coordinator_temperature,
            "max_tokens": uv.max_tokens,
        }
        # Always pass aws_profile to override TOXP's config default.
        # Empty string → BedrockProvider uses profile_name=None (env vars).
        # Named profile → used directly (credential_process must be accessible).
        overrides["aws_profile"] = self.valves.aws_profile
        if self.valves.region:
            overrides["region"] = self.valves.region
        if self.valves.model:
            overrides["model"] = self.valves.model
        if self.valves.context_1m is not None:
            overrides["context_1m"] = self.valves.context_1m
        return overrides

    async def pipe(
        self,
        body: dict,
        __user__: dict,
        __event_emitter__: Optional[Callable[[dict], Awaitable[None]]] = None,
    ) -> AsyncGenerator[str, None]:

        # --- Guard: skip non-streaming re-invocations (title gen, save) ---
        if body.get("stream") is False:
            return

        try:
            from toxp import run_query, Message, ToxpError
        except ImportError:
            yield "**Error**: `toxp` package is not installed.\n\nInstall: `pip install toxp`"
            return

        messages = body.get("messages", [])
        if not messages:
            yield "No messages provided."
            return

        current_query = messages[-1].get("content", "")
        if not current_query.strip():
            yield "Empty message."
            return

        conversation_history: list[Message] | None = None
        if len(messages) > 1:
            conversation_history = [
                {"role": m["role"], "content": m["content"]}
                for m in messages[:-1]
                if m.get("role") in ("user", "assistant") and m.get("content")
            ]
            if not conversation_history:
                conversation_history = None

        # UserValves come via __user__["valves"], not __user_valves__ param
        user_valves = __user__.get("valves") if isinstance(__user__, dict) else None
        config_overrides = self._build_config_overrides(user_valves)
        num_agents = config_overrides["num_agents"]
        concurrency = config_overrides["max_concurrency"]

        # --- Status helper (updates in-place in UI chrome) ---
        async def emit_status(description: str, done: bool = False) -> None:
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "status", "data": {"description": description, "done": done}}
                )

        # --- Event queue ---
        event_queue: asyncio.Queue[tuple[str, ...]] = asyncio.Queue()

        class StreamCallbacks:
            def on_agent_start(self, agent_id: int) -> None:
                event_queue.put_nowait(("agent_start", agent_id))

            def on_agent_complete(
                self, agent_id: int, success: bool, error: str | None = None
            ) -> None:
                event_queue.put_nowait(("agent_done", agent_id, success, error))

            def on_agent_token(self, agent_id: int, token: str) -> None:
                pass

            def on_agents_done(self) -> None:
                event_queue.put_nowait(("agents_done",))

            def on_coordinator_token(self, token: str) -> None:
                event_queue.put_nowait(("coord_token", token))

        async def _run_toxp():
            try:
                result = await run_query(
                    current_query,
                    config_overrides=config_overrides,
                    callbacks=StreamCallbacks(),
                    conversation_history=conversation_history,
                )
                event_queue.put_nowait(("finished",))
                return result
            except Exception as e:
                event_queue.put_nowait(("error", f"{type(e).__name__}: {e}"))
                event_queue.put_nowait(("finished",))
                raise

        task = asyncio.create_task(_run_toxp())
        start_time = time.time()

        await emit_status(f"Reasoning: 0/{num_agents} agents done (concurrency {concurrency})")

        # --- Phase 1: Agent progress (status bar only, no response text) ---
        succeeded = 0
        failed = 0
        running = 0
        agents_phase_done = False

        while not agents_phase_done:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=300)
            except asyncio.TimeoutError:
                await emit_status("Timed out", done=True)
                yield "**Error**: Timed out after 5 minutes."
                task.cancel()
                return

            etype = event[0]

            if etype == "agent_start":
                running += 1

            elif etype == "agent_done":
                running -= 1
                if event[2]:
                    succeeded += 1
                else:
                    failed += 1
                done_count = succeeded + failed
                elapsed = time.time() - start_time
                await emit_status(
                    f"Reasoning: {done_count}/{num_agents} agents done, "
                    f"{running} active ({elapsed:.0f}s)"
                )

            elif etype == "agents_done":
                agents_phase_done = True

            elif etype == "error":
                await emit_status("Error", done=True)
                yield f"**Error**: {event[1]}"
                return

            elif etype == "finished":
                agents_phase_done = True

        # --- Summary line in response ---
        elapsed = time.time() - start_time
        fail_text = f", {failed} failed" if failed else ""
        yield (
            f"> {succeeded}/{succeeded + failed} agents succeeded{fail_text} "
            f"in {elapsed:.1f}s "
            f"({num_agents} agents, concurrency {concurrency}, "
            f"temp {config_overrides['temperature']})\n\n"
        )

        # --- Phase 2: Coordinator synthesis ---
        await emit_status("Synthesizing...")

        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=300)
            except asyncio.TimeoutError:
                await emit_status("Timed out", done=True)
                yield "\n\n**Error**: Coordinator timed out."
                task.cancel()
                return

            etype = event[0]
            if etype == "coord_token":
                yield event[1]
            elif etype == "error":
                await emit_status("Error", done=True)
                yield f"\n\n**Error**: {event[1]}"
                return
            elif etype == "finished":
                break

        # --- Finalize ---
        try:
            result = await task
        except Exception:
            await emit_status("Error", done=True)
            return

        total_time = time.time() - start_time

        if result and result.confidence:
            yield (
                f"\n\n---\n*Confidence: {result.confidence} "
                f"| {total_time:.1f}s "
                f"| ~{result.total_tokens:,} tokens*"
            )

        # done=True is the absolute last action
        await emit_status(
            f"Done in {total_time:.1f}s | {result.confidence} confidence",
            done=True,
        )
