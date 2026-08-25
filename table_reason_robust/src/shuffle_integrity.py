from __future__ import annotations

from typing import Any, Optional

from src.attack_integrity import SHUFFLE_ATTACKS, validate_attack_integrity


def validate_shuffle_integrity(workspace: Any, attack_type: str) -> Optional[str]:
    """Backward-compatible shim for callers and older tests."""
    if attack_type not in SHUFFLE_ATTACKS:
        return None
    return validate_attack_integrity(workspace, attack_type).error_message()
