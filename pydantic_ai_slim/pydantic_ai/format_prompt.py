from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import date
from typing import Any
from xml.etree import ElementTree

from pydantic import BaseModel

__all__ = ('format_as_xml',)


def format_as_xml(
    obj: Any,
    root_tag: str | None = None,
    item_tag: str = 'item',
    none_str: str = 'null',
    indent: str | None = '  ',
) -> str:
    """Format a Python object as XML.

    This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML,
    rather than JSON etc.

    Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `Mapping`,
    `Iterable`, `dataclass`, and `BaseModel`.

    Args:
        obj: Python Object to serialize to XML.
        root_tag: Outer tag to wrap the XML in, use `None` to omit the outer tag.
        item_tag: Tag to use for each item in an iterable (e.g. list), this is overridden by the class name
            for dataclasses and Pydantic models.
        none_str: String to use for `None` values.
        indent: Indentation string to use for pretty printing.

    Returns:
        XML representation of the object.

    Example:
    ```python {title="format_as_xml_example.py" lint="skip"}
    from pydantic_ai import format_as_xml

    print(format_as_xml({'name': 'John', 'height': 6, 'weight': 200}, root_tag='user'))
    '''
    <user>
      <name>John</name>
      <height>6</height>
      <weight>200</weight>
    </user>
    '''
    ```
    """
    # Construct the XML element tree in a single call.
    el = _ToXml(item_tag=item_tag, none_str=none_str).to_xml(obj, root_tag)
    if root_tag is None and el.text is None:
        join = '' if indent is None else '\n'
        return join.join(_rootless_xml_elements(el, indent))
    else:
        if indent is not None:
            ElementTree.indent(el, space=indent)
        return ElementTree.tostring(el, encoding='unicode')


@dataclass
class _ToXml:
    item_tag: str
    none_str: str

    def to_xml(self, value: Any, tag: str | None) -> ElementTree.Element:
        # Fast path for primitive types
        t = type(value)
        # Avoid repeated lookups
        item_tag = self.item_tag

        # String, int, float, bool, date, None
        if value is None:
            element = ElementTree.Element(item_tag if tag is None else tag)
            element.text = self.none_str
            return element
        if t is str:
            element = ElementTree.Element(item_tag if tag is None else tag)
            element.text = value
            return element
        if t is bytes or t is bytearray:
            element = ElementTree.Element(item_tag if tag is None else tag)
            element.text = value.decode(errors='ignore')
            return element
        if t is int or t is float or t is bool:
            element = ElementTree.Element(item_tag if tag is None else tag)
            element.text = str(value)
            return element
        if isinstance(value, date):
            element = ElementTree.Element(item_tag if tag is None else tag)
            element.text = value.isoformat()
            return element
        if isinstance(value, Mapping):
            element = ElementTree.Element(item_tag if tag is None else tag)
            self._mapping_to_xml(element, value)
            return element

        # Special fast path for dataclasses leveraging fields (avoiding asdict)
        if is_dataclass(value) and not isinstance(value, type):
            class_tag = value.__class__.__name__ if tag is None else tag
            element = ElementTree.Element(class_tag)
            # Iterate directly on fields, asdict is expensive.
            for f in fields(value):
                v = getattr(value, f.name)
                child = self.to_xml(v, f.name)
                element.append(child)
            return element

        # Special path for Pydantic BaseModels
        if isinstance(value, BaseModel):
            class_tag = value.__class__.__name__ if tag is None else tag
            element = ElementTree.Element(class_tag)
            # Avoid mode='python' dict conversion if possible
            data = value.__dict__ if hasattr(value, '__dict__') else value.model_dump(mode='python')
            self._mapping_to_xml(element, data)
            return element

        # Iterables (lists, tuples, sets)
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            element = ElementTree.Element(item_tag if tag is None else tag)
            # Avoid the function call stack by inlining recursion for simple types
            for item in value:
                item_el = self.to_xml(item, None)
                element.append(item_el)
            return element

        # Not a supported type
        raise TypeError(f'Unsupported type for XML formatting: {t}')

    def _mapping_to_xml(self, element: ElementTree.Element, mapping: Mapping[Any, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(key, int):
                key = str(key)
            elif not isinstance(key, str):
                raise TypeError(f'Unsupported key type for XML formatting: {type(key)}, only str and int are allowed')
            element.append(self.to_xml(value, key))


def _rootless_xml_elements(root: ElementTree.Element, indent: str | None) -> Iterator[str]:
    # Avoids unneeded indentation when indent is None.
    for sub_element in root:
        if indent is not None:
            ElementTree.indent(sub_element, space=indent)
        yield ElementTree.tostring(sub_element, encoding='unicode')
