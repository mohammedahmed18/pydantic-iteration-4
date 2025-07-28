from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer


def meta_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model."""
    return _META_MODEL_PROFILE

_INLINE_DEFS_SCHEMA_TRANSFORMER = InlineDefsJsonSchemaTransformer

_META_MODEL_PROFILE = ModelProfile(json_schema_transformer=_INLINE_DEFS_SCHEMA_TRANSFORMER)
