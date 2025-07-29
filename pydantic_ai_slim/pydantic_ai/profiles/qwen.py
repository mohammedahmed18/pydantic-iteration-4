from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer

# Cache the ModelProfile instance, since it never changes and
# constructing it is the time-consuming part.
_qwen_model_profile_instance: ModelProfile = ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


def qwen_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Qwen model."""
    return _qwen_model_profile_instance
