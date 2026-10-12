"""metaflow-durability extension configuration for Metaflow.

Attaches the `agent_journal` step decorator to every step so pydantic-ai
agents using `MetaflowDurability` are journaled without any extra user decorator.
"""

TOGGLE_DECOSPECS = ["agent_journal"]
