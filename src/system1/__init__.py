"""Typed decisions from the next token distribution."""

from system1.core import SystemOne, system_one

__all__ = ["SystemOne", "system_one"]


def main() -> None:
    """Run the command line interface."""
    from system1.cli import app

    app()
