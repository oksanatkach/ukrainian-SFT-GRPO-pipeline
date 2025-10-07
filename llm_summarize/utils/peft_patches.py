"""Patches for PEFT library to fix serialization issues."""

import json
from peft.config import PeftConfigMixin

_original_peft_save = PeftConfigMixin.save_pretrained


def patched_save_pretrained(self, save_directory, **kwargs):
    """Patched version that handles non-JSON-serializable attributes."""
    # Temporarily convert non-serializable attributes
    original_values = {}

    for attr_name in dir(self):
        if attr_name.startswith('_'):
            continue
        try:
            attr_value = getattr(self, attr_name, None)
            if attr_value is None or callable(attr_value):
                continue

            # Try to serialize it
            json.dumps(attr_value)
        except (TypeError, ValueError):
            # Can't serialize - save original and convert to dict
            original_values[attr_name] = attr_value
            if hasattr(attr_value, 'to_dict'):
                setattr(self, attr_name, attr_value.to_dict())
            elif hasattr(attr_value, '__dict__'):
                setattr(self, attr_name, vars(attr_value))

    try:
        # Call original save
        result = _original_peft_save(self, save_directory, **kwargs)
    finally:
        # Restore original values
        for attr_name, attr_value in original_values.items():
            setattr(self, attr_name, attr_value)

    return result


def apply_peft_patches():
    """Apply all PEFT patches."""
    PeftConfigMixin.save_pretrained = patched_save_pretrained
