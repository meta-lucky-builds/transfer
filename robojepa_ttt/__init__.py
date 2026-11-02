"""Persistent test-time adaptation for RoboJEPA on SO-101.

Implements baseline B1 of SPECIFICATION.md: RoboJEPA + nominal command bridge +
persistent replay-based adaptation of LoRA and token-residual layers, without
the recurrent belief. Belief inputs are reserved in the residual so the belief
can be added later without changing checkpoints of the other components.
"""

__version__ = "0.1.0"
