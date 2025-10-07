import json
import types
from peft.config import PeftConfigMixin

_original_peft_save = PeftConfigMixin.save_pretrained

def _safe_serialize(obj, seen):
    """
    Recursively convert obj into JSON-serializable form.
    Replaces circular refs with a short placeholder string.
    """
    obj_id = id(obj)
    if obj_id in seen:
        return f"<CIRCULAR_REF:{type(obj).__name__}>"
    # primitive JSON types pass through
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        seen.add(obj_id)
        return [_safe_serialize(el, seen) for el in obj]
    if isinstance(obj, dict):
        seen.add(obj_id)
        serialized = {}
        for k, v in obj.items():
            # JSON keys must be strings; convert if necessary
            key = k if isinstance(k, str) else str(k)
            serialized[key] = _safe_serialize(v, seen)
        return serialized
    # if object provides to_dict, prefer it
    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict")):
        try:
            seen.add(obj_id)
            return _safe_serialize(obj.to_dict(), seen)
        except Exception:
            pass
    # if it has a __dict__, serialize that
    if hasattr(obj, "__dict__"):
        seen.add(obj_id)
        return _safe_serialize(vars(obj), seen)
    # last-resort: try JSON-dumpable via repr
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def patched_save_pretrained(self, save_directory, **kwargs):
    """Patched version that handles non-JSON-serializable attributes safely."""
    original_values = {}

    # Only iterate instance attributes (safer than dir(self))
    inst_dict = getattr(self, "__dict__", {})
    for attr_name, attr_value in list(inst_dict.items()):
        # skip private/internal attributes (optional)
        if attr_name.startswith("_"):
            continue
        # skip callables or modules/types
        if callable(attr_value) or isinstance(attr_value, (types.ModuleType, type)):
            continue
        # small fastpath: already json-serializable?
        try:
            json.dumps(attr_value)
            continue
        except (TypeError, ValueError):
            pass

        # prepare a JSON-safe serialized substitute using cycle-aware conversion
        try:
            serialized = _safe_serialize(attr_value, seen=set())
            # store original and set the safe version
            original_values[attr_name] = attr_value
            setattr(self, attr_name, serialized)
        except Exception as exc:
            # if something goes wrong, fall back to string repr (but still store + set)
            original_values[attr_name] = attr_value
            try:
                setattr(self, attr_name, repr(attr_value))
            except Exception:
                # give up modifying this attribute
                original_values.pop(attr_name, None)

    try:
        result = _original_peft_save(self, save_directory, **kwargs)
    finally:
        # restore original values
        for attr_name, attr_value in original_values.items():
            try:
                setattr(self, attr_name, attr_value)
            except Exception:
                # ignore restoration failures
                pass

    return result


def apply_peft_patches():
    """Apply all PEFT patches."""
    PeftConfigMixin.save_pretrained = patched_save_pretrained
