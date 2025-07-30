from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer


def qwen_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Qwen model."""
    return _MODEL_PROFILE

_INLINE_DEFS_JSON_SCHEMA_TRANSFORMER = InlineDefsJsonSchemaTransformer

_MODEL_PROFILE = ModelProfile(json_schema_transformer=_INLINE_DEFS_JSON_SCHEMA_TRANSFORMER)
