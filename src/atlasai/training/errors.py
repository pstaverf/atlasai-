"""Torch-free exceptions shared by the CLI and optional training modules."""


class TrainingError(ValueError):
    """Training or evaluation cannot produce a trustworthy result."""


class CheckpointError(TrainingError):
    """A checkpoint is missing, incompatible, corrupt, or unsafe to load."""
