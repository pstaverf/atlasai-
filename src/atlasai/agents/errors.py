"""Explicit agent errors; no silent fallbacks or hidden command execution."""


class AgentError(ValueError):
    """An agent request or configuration is invalid."""


class ProtocolError(AgentError):
    """The model output does not match the strict action protocol."""
