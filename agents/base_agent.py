"""
Base agent for MARS (Multi-Agent Recommender System).

All agents inherit from BaseAgent. Communication happens via
synchronous Python calls routed through the Orchestrator —
no Redis/Celery needed for the research prototype.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .orchestrator import Orchestrator


@dataclass
class Message:
    """Inter-agent message passed through the Orchestrator."""

    sender: str
    target: str
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class BaseAgent(ABC):
    """
    Abstract base class for every MARS agent.

    Attributes
    ----------
    name : str
        Unique agent identifier (e.g. "diagnostic", "confidence").
    orchestrator : Orchestrator | None
        Back-reference set by ``Orchestrator.register_agent()``.
    status : str
        Current lifecycle state: "idle" or "processing".
    """

    name: str = "base"
    REQUIRED_COLUMNS: dict[str, list[str]] = {}  # Override in subclasses

    def __init__(self, name: str | None = None, config_path: str = "configs/config.yaml"):
        if name is not None:
            self.name = name
        self.logger = logging.getLogger(f"mars.agent.{self.name}")
        self.orchestrator: Orchestrator | None = None
        self.status: str = "idle"
        self._message_log: list[Message] = []
        self._config = self._load_config(config_path)

    def _load_config(self, path: str) -> dict:
        """Load agent-specific config section."""
        from .utils import load_config
        full = load_config(path)
        return full.get(self.name, {})

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self, **kwargs: Any) -> None:
        """
        Load models, build indices, or perform any one-time setup.
        Called by the Orchestrator after registration.
        """

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send_message(self, target_agent: str, data: dict[str, Any]) -> Any:
        """
        Send a message to another agent via the Orchestrator.

        Returns whatever the target agent's ``receive_message`` produces.
        """
        if self.orchestrator is None:
            raise RuntimeError(
                f"Agent '{self.name}' is not registered with an Orchestrator."
            )
        return self.orchestrator.route_message(self.name, target_agent, data)

    def receive_message(self, message: Message) -> Any:
        """
        Handle an incoming message from another agent.

        Override in subclasses to implement custom message handling.
        The default implementation logs the message and returns None.
        """
        if not isinstance(message, Message):
            raise TypeError(f"Expected Message, got {type(message).__name__}")
        if not isinstance(message.data, dict):
            raise ValueError(f"Message.data must be a dict, got {type(message.data).__name__}")

        self._message_log.append(message)
        self.logger.debug(
            "Received message from '%s': %s",
            message.sender,
            list(message.data.keys()),
        )
        return None

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_dataframe(self, df: "pd.DataFrame", context: str = "") -> None:
        """Check that required columns are present."""
        import pandas as pd
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{self.name}: expected DataFrame, got {type(df).__name__}")
        required = self.REQUIRED_COLUMNS.get(context, [])
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"{self.name}: missing columns {missing} for {context}. "
                f"Available: {list(df.columns)}"
            )

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_processing(self) -> None:
        self.status = "processing"

    def _set_idle(self) -> None:
        self.status = "idle"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r} status={self.status!r}>"
