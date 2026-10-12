from typing import TYPE_CHECKING
from typing import Any

try:
    from metaflow_durability._version import version as __version__
except ImportError:
    __version__ = "unknown"

if TYPE_CHECKING:
    from metaflow_durability.durability import MetaflowDurability

__all__ = ["MetaflowDurability"]


def __getattr__(name: str) -> Any:
    # Lazy so that importing metaflow_durability does not require pydantic-ai.
    if name == "MetaflowDurability":
        from metaflow_durability.durability import MetaflowDurability

        return MetaflowDurability
    raise AttributeError(f"module 'metaflow_durability' has no attribute {name!r}")
