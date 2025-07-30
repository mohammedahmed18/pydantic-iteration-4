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
        # Note: we need to remove `additionalProperties: False` since it is currently mishandled by Gemini
        additional_properties = schema.get('additionalProperties', None)
        if additional_properties is not None:
            # Only pop if present, to avoid double dict copy
            original_schema = {**schema}  # Save before pop for warning (saves a pop, not a dict copy)
            original_schema['additionalProperties'] = additional_properties
            # Only warn if additional_properties is True/False, not if None or default
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
        else:
            schema.pop('additionalProperties', None)  # just in case (noop if not there)

        # Batch pop of known irrelevant keys rather than multiple dict lookups (small speedup)
        for k in ('title', '$schema', 'discriminator', 'examples', 'exclusiveMaximum', 'exclusiveMinimum'):
            schema.pop(k, None)

        const = schema.pop('const', None)
        if const is not None:
            # Gemini doesn't support const, but it does support enum with a single value
            schema['enum'] = [const]

        # Gemini only supports string enums, so convert any enum values to strings.
        enum = schema.get('enum')
        if enum is not None:
            schema['type'] = 'string'
            # Use fast-path list comprehension; avoid unnecessary reallocation of 'enum'
            # optimize: only convert if there's a non-str in enum
            if any(not isinstance(val, str) for val in enum):
                schema['enum'] = [str(val) for val in enum]

        type_ = schema.get('type')
        # optimize: avoid double dict lookup by using local variable
        if 'oneOf' in schema and type_ is None:
            schema['anyOf'] = schema.pop('oneOf')

        if type_ == 'string':
            fmt = schema.pop('format', None)
            if fmt is not None:
                description = schema.get('description')
                if description:
                    schema['description'] = f'{description} (format: {fmt})'
                else:
                    schema['description'] = f'Format: {fmt}'

        if '$ref' in schema:
            ref_value = schema['$ref']  # minimize number of dict lookups
            raise UserError(f'Recursive `$ref`s in JSON Schema are not supported by Gemini: {ref_value}')

        if 'prefixItems' in schema:
            prefix_items = schema.pop('prefixItems')
            items = schema.get('items')
            # Avoid O(N^2) list search/duplication for small lists; for larger, use set for fast check (but typically short)
            if items is not None:
                unique_items = [items]
                # Only add different items (O(n)), but usually lists are very small
                for item in prefix_items:
                    # pointer/identity quick check, then rapid equality
                    if not any(item is uniq or item == uniq for uniq in unique_items):
                        unique_items.append(item)
            else:
                unique_items = list(prefix_items)
            length = len(unique_items)
            if length > 1:  # pragma: no cover
                schema['items'] = {'anyOf': unique_items}
            elif length == 1:  # pragma: no branch
                schema['items'] = unique_items[0]
            # Move calculation outside above block as it's always set
            plen = len(prefix_items)
            schema.setdefault('minItems', plen)
            if items is None:  # pragma: no branch
                schema.setdefault('maxItems', plen)

        return schema
