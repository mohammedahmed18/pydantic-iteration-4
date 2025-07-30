from __future__ import annotations as _annotations

import warnings

from pydantic_ai.exceptions import UserError

from . import ModelProfile
from ._json_schema import JsonSchema, JsonSchemaTransformer


def google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model."""
    return ModelProfile(
        json_schema_transformer=GoogleJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )


class GoogleJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic to be suitable for Gemini.

    Gemini which [supports](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations)
    a subset of OpenAPI v3.0.3.

    Specifically:
    * gemini doesn't allow the `title` keyword to be set
    * gemini doesn't allow `$defs` — we need to inline the definitions where possible
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True, simplify_nullable_unions=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Remove `additionalProperties: False` since it is mishandled by Gemini
        additional_properties = schema.get('additionalProperties', None)
        if additional_properties is not None:
            original_schema = {**schema, 'additionalProperties': additional_properties}
            warnings.warn(
                '`additionalProperties` is not supported by Gemini; it will be removed from the tool JSON schema.'
                f' Full schema: {self.schema}\n\n'
                f'Source of additionalProperties within the full schema: {original_schema}\n\n'
                'If this came from a field with a type like `dict[str, MyType]`, that field will always be empty.\n\n'
                "If Google's APIs are updated to support this properly, please create an issue on the Pydantic AI GitHub"
                ' and we will fix this behavior.',
                UserWarning,
            )
            schema.pop('additionalProperties', None)

        # Use dict.pop() only once per key. Pop keys only if they are likely present.
        for key in ('title', '$schema', 'discriminator', 'examples', 'exclusiveMaximum', 'exclusiveMinimum'):
            schema.pop(key, None)

        if (const := schema.pop('const', None)) is not None:
            # Gemini doesn't support const, but it does support enum with a single value
            schema['enum'] = [const]

        enum = schema.get('enum')
        if enum is not None:
            # Gemini only supports string enums. Convert enum values to strings.
            schema['type'] = 'string'
            # Use tuple for generator, to avoid allocating a list in between before assignment
            schema['enum'] = [str(val) for val in enum]

        type_ = schema.get('type')

        if 'oneOf' in schema and type_ is None:  # pragma: no cover
            # This gets hit when we have a discriminated union
            # Gemini returns an API error in this case...
            # Changing the oneOf to an anyOf prevents the API error and is functionally equivalent
            schema['anyOf'] = schema.pop('oneOf')

        if type_ == 'string':
            fmt = schema.pop('format', None)
            if fmt is not None:
                description = schema.get('description')
                if description is not None:
                    schema['description'] = f'{description} (format: {fmt})'
                else:
                    schema['description'] = f'Format: {fmt}'

        if '$ref' in schema:
            raise UserError(f'Recursive `$ref`s in JSON Schema are not supported by Gemini: {schema["$ref"]}')

        if 'prefixItems' in schema:
            # prefixItems is not currently supported in Gemini, so convert to items for best compatibility
            prefix_items = schema.pop('prefixItems')
            items = schema.get('items')

            # Avoid building unnecessary lists: only add items that are unique
            if items is not None:
                unique_items = [items]
                for item in prefix_items:
                    if item is not items and item not in unique_items:
                        unique_items.append(item)
            else:
                unique_items = list(prefix_items)

            if len(unique_items) > 1:  # pragma: no cover
                schema['items'] = {'anyOf': unique_items}
            elif len(unique_items) == 1:  # pragma: no branch
                schema['items'] = unique_items[0]

            # Only call setdefault if keys not present
            if 'minItems' not in schema:
                schema['minItems'] = len(prefix_items)
            if items is None and 'maxItems' not in schema:  # pragma: no branch
                schema['maxItems'] = len(prefix_items)

        return schema
