from __future__ import annotations as _annotations

from dataclasses import dataclass, replace
from textwrap import dedent
from typing import Callable, Union

from typing_extensions import Self

from ..output import StructuredOutputMode
from ._json_schema import JsonSchemaTransformer


@dataclass
class ModelProfile:
    """Describes how requests to a specific model or family of models need to be constructed to get the best results, independent of the model and provider classes used."""

    supports_tools: bool = True
    """Whether the model supports tools."""
    supports_json_schema_output: bool = False
    """Whether the model supports JSON schema output."""
    supports_json_object_output: bool = False
    """Whether the model supports JSON object output."""
    default_structured_output_mode: StructuredOutputMode = 'tool'
    """The default structured output mode to use for the model."""
    prompted_output_template: str = dedent(
        """
        Always respond with a JSON object that's compatible with this schema:

        {schema}

        Don't include any text or Markdown fencing before or after.
        """
    )
    """The instructions template to use for prompted structured output. The '{schema}' placeholder will be replaced with the JSON schema for the output."""
    json_schema_transformer: type[JsonSchemaTransformer] | None = None
    """The transformer to use to make JSON schemas for tools and structured output compatible with the model."""

    @classmethod
    def from_profile(cls, profile: ModelProfile | None) -> Self:
        """Build a ModelProfile subclass instance from a ModelProfile instance."""
        if isinstance(profile, cls):
            return profile
        return cls().update(profile)

    def update(self, profile: ModelProfile | None) -> Self:
        """Update this ModelProfile (subclass) instance with the non-default values from another ModelProfile instance."""
        if not profile:
            return self

        # Use __dataclass_fields__ for fast lookups
        self_fields = self.__dataclass_fields__
        profile_fields = profile.__dataclass_fields__

        # Only update matching fields
        field_names = set(self_fields.keys())
        profile_field_objs = [profile_fields[name] for name in profile_fields if name in field_names]

        # Collect all defaults for profile fields in advance to avoid attribute access inside the loop
        defaults = {
            f.name: f.default
            for f in profile_field_objs
        }

        # Use vars(profile) for efficient attribute lookup
        profile_vars = vars(profile)

        # Fast dict comprehension: only keep non-default values
        non_default_attrs = {
            name: profile_vars[name]
            for name in profile_vars
            if name in defaults and profile_vars[name] != defaults[name]
        }

        return replace(self, **non_default_attrs)


ModelProfileSpec = Union[ModelProfile, Callable[[str], Union[ModelProfile, None]]]

DEFAULT_PROFILE = ModelProfile()
