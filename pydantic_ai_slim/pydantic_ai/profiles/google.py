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
        # Optimize by checking without pop for most cases (avoids dict mutation and redundant allocations)
        if 'additionalProperties' in schema:
            additional_properties = schema.pop('additionalProperties')
            if additional_properties:
                # Only create original_schema if we actually need to warn
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
        # Fast pops
        for key in ('title', '$schema', 'discriminator', 'examples', 'exclusiveMaximum', 'exclusiveMinimum'):
            schema.pop(key, None)
        # Fast const->enum conversion
        const = schema.pop('const', None)
        if const is not None:
            schema['enum'] = [const]

        # Optimize enum conversion (avoid unnecessary list-comprehension if type already string)
        enum = schema.get('enum')
        if enum is not None:
            schema['type'] = 'string'
            # Fast path if all items already str avoids unnecessary str(val) calls
            if not all(isinstance(val, str) for val in enum):
                schema['enum'] = [str(val) for val in enum]

        type_ = schema.get('type')

        # Inlined logic for special unions
        if 'oneOf' in schema and 'type' not in schema:  # pragma: no cover
            schema['anyOf'] = schema.pop('oneOf')

        # Optimize format-handling block
        if type_ == 'string':
            fmt = schema.pop('format', None)
            if fmt is not None:
                description = schema.get('description')
                if description:
                    schema['description'] = f'{description} (format: {fmt})'
                else:
                    schema['description'] = f'Format: {fmt}'

        if '$ref' in schema:
            raise UserError(f'Recursive `$ref`s in JSON Schema are not supported by Gemini: {schema["$ref"]}')

        # Optimize prefixItems transformation: avoid O(N^2) for-loop using set-based deduplication for non-None "items"
        if 'prefixItems' in schema:
            prefix_items = schema.pop('prefixItems')
            items = schema.get('items')
            if items is not None:
                # Use a set for deduplication (type will be hashable, as required for dict keys)
                # Only deduplicate by identity if values are not dict/list, fallback to slow method otherwise
                try:
                    # Attempt fast set-based deduplication for hashable types
                    seen = {items}
                    unique_items = [items]
                    append = unique_items.append  # local alias for speed
                    for item in prefix_items:
                        if item not in seen:
                            seen.add(item)
                            append(item)
                except TypeError:
                    # Fallback for unhashable children (dicts/lists): O(N^2) but correct
                    unique_items = [items]
                    append = unique_items.append
                    for item in prefix_items:
                        if item not in unique_items:
                            append(item)
            else:
                unique_items = list(prefix_items)
            n_unique = len(unique_items)
            if n_unique > 1:  # pragma: no cover
                schema['items'] = {'anyOf': unique_items}
            elif n_unique == 1:  # pragma: no branch
                schema['items'] = unique_items[0]
            schema.setdefault('minItems', len(prefix_items))
            if items is None:  # pragma: no branch
                schema.setdefault('maxItems', len(prefix_items))

        return schema
