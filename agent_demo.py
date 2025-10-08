from __future__ import annotations
"""
Demo runner: starts a Letta client and wires in QMem-backed archival memory tools.
- Registers tools
- Creates an agent (disables base tools to avoid collision; falls back if needed)
- Smoke tests insert/search and runs a short chat that exercises memory
"""

import os
from typing import Any

from letta_client import Letta
from letta_client.errors import UnprocessableEntityError

from qmem.letta_tools import QMemInsertTool, QMemSearchTool


def _mk_client() -> Letta:
    """
    Prefer local server if running; else allow Letta Cloud via LETTA_API_KEY.
    """
    token = os.environ.get("LETTA_API_KEY")
    base_url = os.environ.get("LETTA_BASE_URL", "http://localhost:8283")
    if token:
        return Letta(token=token)
    return Letta(base_url=base_url)


def register_tools(client: Letta) -> None:
    """
    Register our two tools with the Letta server.
    Some versions provide tools.add(); others provide tools.create().
    """
    try:
        client.tools.add(tool=QMemInsertTool())
        client.tools.add(tool=QMemSearchTool())
    except Exception:
        client.tools.create(tool=QMemInsertTool())
        client.tools.create(tool=QMemSearchTool())


def create_agent(client: Letta):
    """
    Create an agent that uses our archival tools.
    If your Letta version rejects include_base_tools=False, fall back.
    """
    try:
        return client.agents.create(
            model="openai/gpt-4.1-mini",                 # choose a chat model you have keys for
            embedding="openai/text-embedding-3-small",   # MUST match qmem init (e.g., 1536 dims)
            include_base_tools=False,
            tools=[
                # essentials for conversation + in-context edits
                "send_message",
                "memory_insert", "memory_replace", "memory_rethink", "memory_finish_edits",
                "conversation_search",
                # QMem-backed archival tools (override)
                "archival_memory_insert", "archival_memory_search",
            ],
            memory_blocks=[{"label": "persona", "value": "I am a helpful agent with QMem long-term memory."}],
            enable_sleeptime=False,
        )
    except UnprocessableEntityError:
        # Fallback: keep base tools; just ensure our replacements are present
        return client.agents.create(
            model="openai/gpt-4.1-mini",
            embedding="openai/text-embedding-3-small",
            include_base_tools=True,
            tools=["archival_memory_insert", "archival_memory_search"],
            memory_blocks=[{"label": "persona", "value": "I am a helpful agent with QMem long-term memory."}],
            enable_sleeptime=False,
        )


def print_msgs(resp: Any) -> None:
    """
    Robustly print mixed Letta message types (AssistantMessage, ToolCall, ToolResult,
    ReasoningMessage, etc.) across versions.
    """
    msgs = getattr(resp, "messages", []) or []
    for m in msgs:
        # Pydantic models across versions
        if hasattr(m, "model_dump"):
            d = m.model_dump()
        elif hasattr(m, "dict"):
            d = m.dict()
        else:
            d = m if isinstance(m, dict) else {"value": str(m)}

        role = d.get("role") or d.get("sender") or d.get("author") or d.get("type") or "message"
        content = d.get("content") or d.get("text") or d.get("value") or d
        print(f"{role}: {content}")


def main() -> None:
    client = _mk_client()

    # 1) Register our QMem tools
    register_tools(client)

    # 2) Create an agent configured to use them
    agent = create_agent(client)
    print("Agent:", agent.id)

    # 3) Quick smoke test: store & retrieve via QMem tools directly
    QMemInsertTool().run(content="User lives in Bengaluru and loves filter coffee.")
    print("\nQMem search:\n", QMemSearchTool().run(query="Where does the user live?", k=3))

    # 4) Chat: ask the agent to remember something
    r1 = client.agents.messages.create(
        agent_id=agent.id,
        messages=[
            {"role": "user", "content": "Remember: my favorite color is blue."}
        ],
    )
    print()
    print_msgs(r1)

    # 5) Ask it to recall
    r2 = client.agents.messages.create(
        agent_id=agent.id,
        messages=[
            {"role": "user", "content": "What is my favorite color? Check your memory."}
        ],
    )
    print("\nRecall:")
    print_msgs(r2)


if __name__ == "__main__":
    # Expect OPENAI_API_KEY (or another provider key) in env.
    # If using Letta Cloud instead of local server, set LETTA_API_KEY in env.
    main()
