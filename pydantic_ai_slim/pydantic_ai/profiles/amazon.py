from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer

# Cache the ModelProfile instance since it's stateless and always constructed the same way
_amazon_model_profile: ModelProfile = ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


def amazon_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Amazon model."""
    return _amazon_model_profile
