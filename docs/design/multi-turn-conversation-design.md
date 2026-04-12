# TOXP Multi-Turn Conversation Support

**Status**: DRAFT  
**Date**: 2026-04-12  
**Target version**: TBD (no version bump yet)  
**Scope**: Backward-compatible — CLI behavior unchanged, multi-turn capability added to library API  

---

## 1. Motivation

TOXP is currently single-shot: one query in, one synthesized answer out. To support UI frontends like Open WebUI, we need multi-turn conversation where follow-up questions carry context from prior exchanges.

**Example**: A user asks "What causes inflation?", gets a coordinator synthesis, then asks "Elaborate on the demand-pull factors." The second query must include the first exchange as context — otherwise "demand-pull factors" has no referent.

### Goals

- Multi-turn conversation through the library API (`run_query`)
- Native Bedrock Converse API multi-turn messages (not string concatenation)
- Reasoning agents see conversation history for context
- Coordinator maintains continuity across turns
- CLI remains single-shot and completely unchanged
- All existing tests pass without modification

### Non-Goals

- Server/HTTP mode for TOXP (that's the Open WebUI Pipe's job)
- Conversation persistence or session management (that's the UI layer's job)
- Automatic history summarization or truncation (future enhancement)

---

## 2. Architecture Overview

```
                    Open WebUI (or any UI)
                           │
                    messages[] array
                    (user/assistant turns)
                           │
                           ▼
                ┌─────────────────────┐
                │   run_query()       │
                │   query: str        │
                │   conversation_     │
                │   history: [Message] │  ◄── NEW optional param
                └─────────┬───────────┘
                          │
                    Orchestrator
                    (unchanged logic)
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
     Agent 0          Agent 1        Agent N-1
     ┌──────────┐    ┌──────────┐   ┌──────────┐
     │ history + │    │ history + │   │ history + │
     │ current Q │    │ current Q │   │ current Q │
     │ as native │    │ as native │   │ as native │
     │ multi-turn│    │ multi-turn│   │ multi-turn│
     └─────┬─────┘    └─────┬─────┘   └─────┬─────┘
           │               │               │
           └───────────────┼───────────────┘
                           ▼
                    Coordinator
                    ┌──────────────────┐
                    │ conversation     │
                    │ context in       │
                    │ system prompt    │
                    │ + agent outputs  │
                    │ + current query  │
                    └──────────────────┘
                           │
                           ▼
                      QueryResult
```

---

## 3. The Message Type

A shared type across all layers, intentionally matching the OpenAI/Open WebUI message format:

```python
class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str
```

Lives in `toxp/models/conversation.py`. Exported from `toxp.__init__`.

---

## 4. Layer-by-Layer Changes

### 4.1 Provider Layer (bottom)

**Files**: `providers/base.py`, `providers/bedrock.py`

Both `invoke_model` and `invoke_model_stream` gain an optional `messages` parameter:

- When `messages` is provided → pass directly to Bedrock Converse API as multi-turn
- When `None` → wrap `user_message` in a single-turn list (current behavior)

Bedrock Converse API already supports alternating `user`/`assistant` turns natively. This change just stops hardcoding a single turn and allows the caller to provide a full conversation.

**Backward compatibility**: Default is `None`, so every existing caller works identically.

### 4.2 Reasoning Agent

**File**: `agents/reasoning.py`

`reason()` gains an optional `conversation_history` parameter. When provided, it builds a native multi-turn messages list: `history + [current query]`.

**Why agents need history**: Without it, follow-up references like "point 3" or "the second approach" are meaningless.

**Why native multi-turn (not concatenation)**: Claude models process `user`/`assistant` turns with architectural awareness of conversation structure. A concatenated string is treated as a single utterance containing a transcript — weaker context grounding.

**Diversity is preserved**: Agents use temperature 0.9. They're answering the *follow-up* question independently, not re-answering the original. The history is shared context, not an answer to parrot.

### 4.3 Coordinator Agent

**Files**: `agents/coordinator.py`, `agents/prompts.py`

A new coordinator prompt template for multi-turn adds a "Conversation so far" section before the current question and agent outputs. The coordinator sees full history (including its own prior syntheses) so it can maintain coherent, continuous answers.

`format_coordinator_prompt` gains an optional `conversation_history` parameter. When `None`, uses the existing single-turn template.

### 4.4 Orchestrator (pass-through)

**File**: `orchestrator.py`

`process_query` gains an optional `conversation_history` parameter and threads it to:
- Each reasoning agent's `reason()` call
- The coordinator's `format_coordinator_prompt()` call

No new orchestration logic — the spawn/validate/synthesize pipeline is unchanged.

### 4.5 Public API (top)

**File**: `api.py`

`run_query` gains an optional `conversation_history` parameter:

```python
async def run_query(
    query: str, *,
    config_overrides=None, callbacks=None, cancel_token=None,
    conversation_history: list[Message] | None = None,   # NEW
) -> QueryResult
```

The CLI passes only `query` (unchanged). UI consumers pass both.

---

## 5. Data Flow Example — Turn 2

User's second question: "Elaborate on the demand-pull factors"

**What Open WebUI sends to the Pipe**:
```
messages = [
  {role: "user",      content: "What causes inflation?"},
  {role: "assistant", content: "<coordinator synthesis from turn 1>"},
  {role: "user",      content: "Elaborate on the demand-pull factors"},
]
```

**What the Pipe calls**:
```
run_query(
  query="Elaborate on the demand-pull factors",
  conversation_history=[
    {role: "user",      content: "What causes inflation?"},
    {role: "assistant", content: "<coordinator synthesis from turn 1>"},
  ]
)
```

**What each reasoning agent sends to Bedrock Converse API**:
```
system: <REASONING_AGENT_SYSTEM_PROMPT>    (unchanged)
messages: [
  {role: "user",      content: "What causes inflation?"},
  {role: "assistant", content: "<coordinator synthesis from turn 1>"},
  {role: "user",      content: "Elaborate on the demand-pull factors"},
]
```

**What the coordinator sends to Bedrock Converse API**:
```
system: <COORDINATOR_SYSTEM_PROMPT_MULTITURN>
  includes: conversation context + current question + all agent outputs
messages: [
  {role: "user", content: "Please provide your synthesis..."}
]
```

**Turn 1 is byte-for-byte identical to current behavior** — `conversation_history=None` at every layer.

---

## 6. Files Changed

| File | Change | Breaking? |
|------|--------|-----------|
| `models/conversation.py` | NEW — `Message` TypedDict | No |
| `providers/base.py` | Add optional `messages` param | No |
| `providers/bedrock.py` | Build messages list conditionally | No |
| `agents/reasoning.py` | Accept + forward `conversation_history` | No |
| `agents/coordinator.py` | Forward history to prompt builder | No |
| `agents/prompts.py` | Add multi-turn coordinator template | No |
| `orchestrator.py` | Thread `conversation_history` through | No |
| `api.py` | Add `conversation_history` param | No |
| `__init__.py` | Export `Message` | No |

Every change adds an optional parameter defaulting to `None`. Zero breaking changes.

---

## 7. What Does NOT Change

- CLI (`cli.py`) — untouched, still single-shot
- Config system — no new config keys
- Rate limiter — unchanged
- Output formatter and progress display — unchanged
- Session logger — unchanged
- Exception hierarchy — unchanged
- Orchestrator logic (spawn N, validate ≥50%, synthesize) — unchanged

---

## 8. Design Decisions

### Why native multi-turn over string concatenation?

Concatenating history into a single `user_message` string works functionally but:
- Loses structured turn awareness the model has at the architecture level
- Makes the system prompt bloated (coordinator embeds `{query}` in system prompt)
- Is fragile (requires prompt engineering to delimit "this is history" vs "this is the question")

Native multi-turn uses Bedrock Converse API's built-in message structure, which Claude models process with positional awareness of who said what.

### Why do reasoning agents see prior coordinator syntheses?

For follow-ups like "elaborate on point 3", agents must know what point 3 was. The prior coordinator synthesis is the only source of that context. Diversity is preserved because:
1. High temperature (0.9) ensures varied reasoning approaches
2. Agents are answering a *new* question, not re-answering the original
3. This mirrors how a study group works — you build on shared discussion

### Why is the coordinator prompt different from the agent messages approach?

The coordinator's task is fundamentally different: it receives a brief (conversation context + agent outputs + current question) and produces a synthesis. This entire brief belongs in the system prompt because the coordinator's "user message" is a fixed instruction. Mixing native multi-turn with the synthesis task would blur the coordinator's role framing.

---

## 9. Future Enhancements (Out of Scope)

- **History summarization**: For long conversations (10+ turns), summarize older turns to reduce token cost. The Pipe or a future TOXP middleware could do this.
- **Selective agent context**: Pass only user turns to agents (omitting prior syntheses) to maximize diversity at the cost of context. Could be a config flag.
- **Token budget management**: Cap the total history tokens sent to agents based on model context limits and num_agents.

---

## 10. Open WebUI Integration (Separate Project)

The TOXP changes enable but do not implement the Open WebUI integration. The Pipe function (living outside TOXP) will:

- Receive Open WebUI's `messages[]` array
- Split into `history` (all but last) and `query` (last message content)
- Call `run_query(query, conversation_history=history, config_overrides=...)`
- Stream coordinator tokens back via async generator
- Expose `num_agents`, `max_concurrency`, `temperature` as Valves (UI-configurable settings)

This is a separate deliverable after the TOXP changes land.
