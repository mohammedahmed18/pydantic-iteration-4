from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer

# Cache the ModelProfile instance for reuse since it is effectively immutable and constructed identically every call
_cached_profile: ModelProfile = ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


def meta_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model."""
    return _cached_profile
