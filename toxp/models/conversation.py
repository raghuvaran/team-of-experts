"""Conversation types for multi-turn support."""

from typing import Literal, TypedDict


class Message(TypedDict):
    """A single message in a conversation.

    Matches the OpenAI/Open WebUI message format for seamless integration
    with UI frontends.
    """

    role: Literal["user", "assistant"]
    content: str
