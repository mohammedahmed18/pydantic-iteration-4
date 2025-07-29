from typing import Any, Protocol

from pydantic.json_schema import JsonSchemaValue

from pydantic_ai.tools import Tool as _Tool, Tool
from pydantic_ai.toolsets.function import FunctionToolset


class LangChainTool(Protocol):
    # args are like
    # {'dir_path': {'default': '.', 'description': 'Subdirectory to search in.', 'title': 'Dir Path', 'type': 'string'},
    #  'pattern': {'description': 'Unix shell regex, where * matches everything.', 'title': 'Pattern', 'type': 'string'}}
    @property
    def args(self) -> dict[str, JsonSchemaValue]: ...

    def get_input_jsonschema(self) -> JsonSchemaValue: ...

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def run(self, *args: Any, **kwargs: Any) -> str: ...


__all__ = ('tool_from_langchain', 'LangChainToolset')


def tool_from_langchain(langchain_tool: LangChainTool) -> Tool:
    """Creates a Pydantic AI tool proxy from a LangChain tool.

    Args:
        langchain_tool: The LangChain tool to wrap.

    Returns:
        A Pydantic AI tool that corresponds to the LangChain tool.
    """
    # Pre-fetch where possible
    function_name = langchain_tool.name
    function_description = langchain_tool.description
    # Only one pass through items for both required and defaults
    inputs = langchain_tool.args
    required = []
    defaults = {}
    # Avoid copy: just check for 'default' field
    for name, detail in inputs.items():
        if 'default' in detail:
            defaults[name] = detail['default']
        else:
            required.append(name)
    required.sort()
    # Do not copy input schema unless absolutely necessary
    schema: JsonSchemaValue = langchain_tool.get_input_jsonschema()
    # Defensive, but only update if missing to retain shared structure
    if 'additionalProperties' not in schema:
        schema = dict(schema)  # shallow copy only if needed
        schema['additionalProperties'] = False
    if required:
        if schema is not dict:
            # in very rare cases where schema was not updated above
            schema = dict(schema)
        schema['required'] = required

    def proxy(*args: Any, **kwargs: Any) -> str:
        # assert not args, 'This should always be called with kwargs'
        if args:
            raise AssertionError('This should always be called with kwargs')
        if kwargs:
            kw = defaults.copy()
            kw.update(kwargs)
            return langchain_tool.run(kw)
        return langchain_tool.run(defaults)

    return _Tool.from_schema(
        function=proxy,
        name=function_name,
        description=function_description,
        json_schema=schema,
    )


class LangChainToolset(FunctionToolset):
    """A toolset that wraps LangChain tools."""

    def __init__(self, tools: list[LangChainTool]):
        super().__init__([tool_from_langchain(tool) for tool in tools])
