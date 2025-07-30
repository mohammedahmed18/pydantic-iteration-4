from __future__ import annotations as _annotations

import re
from dataclasses import dataclass
from typing import Any

from . import ModelProfile
from ._json_schema import JsonSchema, JsonSchemaTransformer


@dataclass
class OpenAIModelProfile(ModelProfile):
    """Profile for models used with OpenAIModel.

    ALL FIELDS MUST BE `openai_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    openai_supports_strict_tool_definition: bool = True
    """This can be set by a provider or user if the OpenAI-"compatible" API doesn't support strict tool definitions."""

    openai_supports_sampling_settings: bool = True
    """Turn off to don't send sampling settings like `temperature` and `top_p` to models that don't support them, like OpenAI's o-series reasoning models."""

    # Some OpenAI-compatible providers (e.g. MoonshotAI) currently do **not** accept
    # `tool_choice="required"`.  This flag lets the calling model know whether it's
    # safe to pass that value along.  Default is `True` to preserve existing
    # behaviour for OpenAI itself and most providers.
    openai_supports_tool_choice_required: bool = True
    """Whether the provider accepts the value ``tool_choice='required'`` in the
    request payload."""


def openai_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for an OpenAI model."""
    is_reasoning_model = model_name.startswith('o')
    # Structured Outputs (output mode 'native') is only supported with the gpt-4o-mini, gpt-4o-mini-2024-07-18, and gpt-4o-2024-08-06 model snapshots and later.
    # We leave it in here for all models because the `default_structured_output_mode` is `'tool'`, so `native` is only used
    # when the user specifically uses the `NativeOutput` marker, so an error from the API is acceptable.
    return OpenAIModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        openai_supports_sampling_settings=not is_reasoning_model,
    )


_STRICT_INCOMPATIBLE_KEYS = [
    'minLength',
    'maxLength',
    'pattern',
    'format',
    'minimum',
    'maximum',
    'multipleOf',
    'patternProperties',
    'unevaluatedProperties',
    'propertyNames',
    'minProperties',
    'maxProperties',
    'unevaluatedItems',
    'contains',
    'minContains',
    'maxContains',
    'minItems',
    'maxItems',
    'uniqueItems',
]

_sentinel = object()


@dataclass
class OpenAIJsonSchemaTransformer(JsonSchemaTransformer):
    """Recursively handle the schema to make it compatible with OpenAI strict mode.

    See https://platform.openai.com/docs/guides/function-calling?api-mode=responses#strict-mode for more details,
    but this basically just requires:
    * `additionalProperties` must be set to false for each object in the parameters
    * all fields in properties must be marked as required
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict)
        self.root_ref = schema.get('$ref')

    def walk(self) -> JsonSchema:
        # Note: OpenAI does not support anyOf at the root in strict mode
        # However, we don't need to check for it here because we ensure in pydantic_ai._utils.check_object_json_schema
        # that the root schema either has type 'object' or is recursive.
        result = super().walk()

        # For recursive models, we need to tweak the schema to make it compatible with strict mode.
        # Because the following should never change the semantics of the schema we apply it unconditionally.
        if self.root_ref is not None:
            result.pop('$ref', None)  # We replace references to the self.root_ref with just '#' in the transform method
            root_key = re.sub(r'^#/\$defs/', '', self.root_ref)
            result.update(self.defs.get(root_key) or {})

        return result

    def transform(self, schema: JsonSchema) -> JsonSchema:  # noqa C901
        # Remove unnecessary keys using a single loop for lower overhead
        for key in ('title', '$schema', 'discriminator'):
            schema.pop(key, None)

        default = schema.get('default', _sentinel)
        if default is not _sentinel:
            if self.strict is True:
                schema.pop('default', None)
            elif self.strict is None:  # pragma: no branch
                self.is_strict_compatible = False

        schema_ref = schema.get('$ref')
        if schema_ref:
            if schema_ref == self.root_ref:
                schema['$ref'] = '#'
            if len(schema) > 1:
                # Move the "$ref" into an "anyOf" for OpenAI strict mode compatibility
                schema['anyOf'] = [{'$ref': schema.pop('$ref')}]

        # Use a dictionary comprehension for collecting incompatible values for better perf
        incompatible_values: dict[str, Any] = {key: schema[key] for key in _STRICT_INCOMPATIBLE_KEYS if key in schema}
        description = schema.get('description')
        if incompatible_values:
            if self.strict is True:
                # Use list comprehension for notes formatting for better speed and less loop overhead
                notes = [f'{key}={value}' for key, value in incompatible_values.items()]
                for key in incompatible_values:
                    schema.pop(key)
                notes_string = ', '.join(notes)
                schema['description'] = notes_string if not description else f'{description} ({notes_string})'
            elif self.strict is None:  # pragma: no branch
                self.is_strict_compatible = False

        schema_type = schema.get('type')
        if 'oneOf' in schema:
            if self.strict is True:
                schema['anyOf'] = schema.pop('oneOf')
            else:
                self.is_strict_compatible = False

        if schema_type == 'object':
            if self.strict is True:
                schema['additionalProperties'] = False

                # Fast-path empty properties and avoid unnecessary checks
                properties = schema.setdefault('properties', {})
                # Use list constructor directly for required keys
                schema['required'] = list(properties) if properties else []

            elif self.strict is None:
                if (
                    schema.get('additionalProperties') is not False
                    or 'properties' not in schema
                    or 'required' not in schema
                ):
                    self.is_strict_compatible = False
                else:
                    required = schema['required']
                    props = schema['properties']
                    # Use set lookup for much faster membership test in large dicts
                    required_set = set(required)
                    if len(required_set) < len(props):
                        # Only iterate if there's a chance something is missing
                        for k in props:
                            if k not in required_set:
                                self.is_strict_compatible = False
                                break

        return schema
