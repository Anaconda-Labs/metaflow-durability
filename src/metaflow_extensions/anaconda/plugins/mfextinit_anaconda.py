"""metaflow-durability extension registration for Metaflow.

Registers the @agent step decorator via the metaflow_extensions plugin system.
CLI commands (fork, continue) will be added here via CLIS_DESC when ready.
"""

STEP_DECORATORS_DESC = [
    ("agent", "metaflow_durability.agent_decorator.AgentDecorator"),
    (
        "agent_journal",
        "metaflow_durability.journal_decorator.AgentJournalDecorator",
    ),
]

# CLI commands — uncomment when fork/continue are implemented
# CLIS_DESC = [
#     ("fork", "metaflow_durability.cli_ext.fork_cli"),
#     ("continue", "metaflow_durability.cli_ext.continue_cli"),
# ]
