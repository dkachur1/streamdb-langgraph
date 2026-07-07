"""The summarization message-id prefix contract.

An agent that runs compaction mints summary messages with a known id prefix, and
this package filters those messages out of the chat stream by that prefix (so a
"note to self" summary never renders as a fake user/assistant bubble). The two
sides deliberately do not depend on each other, so the literal exists on both.

Here that invariant is expressed two ways: the shipped default prefix is a stable
literal a paired agent can hard-code against, and every filter site accepts an
override so a consumer whose middleware mints a different prefix can align them
without forking the package.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from streamdb_langgraph import SUMMARIZATION_MESSAGE_ID_PREFIX
from streamdb_langgraph.history_replay import next_live_ordinal
from streamdb_langgraph.state_translator import (
    SUMMARIZATION_MESSAGE_ID_PREFIX as TRANSLATOR_PREFIX,
)


def test_default_prefix_is_the_stable_literal() -> None:
    assert SUMMARIZATION_MESSAGE_ID_PREFIX == "summarization-"


def test_prefix_constant_is_shared_across_modules() -> None:
    assert TRANSLATOR_PREFIX == SUMMARIZATION_MESSAGE_ID_PREFIX


def test_default_prefix_excludes_summary_messages_from_ordinals() -> None:
    messages = [
        HumanMessage(id="u1", content="hi"),
        AIMessage(id="summarization-abc", content="(summary)"),
        AIMessage(id="a1", content="hello"),
    ]
    assert next_live_ordinal(messages) == 2


def test_prefix_is_overridable_at_the_filter_site() -> None:
    messages = [
        HumanMessage(id="u1", content="hi"),
        AIMessage(id="recap-abc", content="(summary)"),
        AIMessage(id="a1", content="hello"),
    ]
    assert next_live_ordinal(messages) == 3
    assert (
        next_live_ordinal(messages, summarization_message_id_prefix="recap-") == 2
    )
