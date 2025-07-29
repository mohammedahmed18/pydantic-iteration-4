from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer

# Cache a single ModelProfile instance since parameters are constant
_cached_profile: ModelProfile = ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


def amazon_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Amazon model."""
    return _cached_profile
