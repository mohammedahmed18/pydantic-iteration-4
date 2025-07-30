from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
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
    to_xml_inst = _ToXml(item_tag=item_tag, none_str=none_str)
    el = to_xml_inst.to_xml(obj, root_tag)
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
        # Avoid unnecessary attribute lookups in hotpath
        item_tag = self.item_tag
        none_str = self.none_str

        # Optimized element creation and name inference
        tag_name = item_tag if tag is None else tag
        element = ElementTree.Element(tag_name)

        # Fast-path type checks (str, None, bytes/bytearray, basic types, date)
        if value is None:
            element.text = none_str
            return element
        v_type = type(value)
        if v_type is str:
            element.text = value
            return element
        if v_type is bytes or v_type is bytearray:
            element.text = value.decode(errors='ignore')
            return element
        if v_type is bool or v_type is int or v_type is float:
            element.text = str(value)
            return element
        if isinstance(value, date):
            element.text = value.isoformat()
            return element
        # Mapping handling
        if isinstance(value, Mapping):
            self._mapping_to_xml(element, value)
            return element
        # Dataclass handling
        if is_dataclass(value) and not isinstance(value, type):
            dc_class_name = value.__class__.__name__
            if tag is None:
                element = ElementTree.Element(dc_class_name)
            dc_dict = asdict(value)
            self._mapping_to_xml(element, dc_dict)
            return element
        # Pydantic BaseModel handling
        if isinstance(value, BaseModel):
            model_class_name = value.__class__.__name__
            if tag is None:
                element = ElementTree.Element(model_class_name)
            # Use model_dump with mode='python'
            self._mapping_to_xml(element, value.model_dump(mode='python'))
            return element
        # Iterable (but NOT str, bytes, bytearray, Mapping already handled)
        if isinstance(value, Iterable):
            # Avoid nesting generators or function call overhead
            append = element.append
            to_xml = self.to_xml
            for item in value:
                item_el = to_xml(item, None)
                append(item_el)
            return element
        # Fallback -- type not supported
        raise TypeError(f'Unsupported type for XML formatting: {type(value)}')

    def _mapping_to_xml(self, element: ElementTree.Element, mapping: Mapping[Any, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(key, int):
                key = str(key)
            elif not isinstance(key, str):
                raise TypeError(f'Unsupported key type for XML formatting: {type(key)}, only str and int are allowed')
            element.append(self.to_xml(value, key))


def _rootless_xml_elements(root: ElementTree.Element, indent: str | None) -> Iterator[str]:
    # Avoid repeated lookups in ElementTree
    etree_indent = ElementTree.indent if indent is not None else None
    etree_tostring = ElementTree.tostring
    for sub_element in root:
        if etree_indent is not None:
            etree_indent(sub_element, space=indent)
        yield etree_tostring(sub_element, encoding='unicode')
