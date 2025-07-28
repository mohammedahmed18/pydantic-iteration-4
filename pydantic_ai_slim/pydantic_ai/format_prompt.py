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
    # *** HOT: Inline _ToXml instance creation to avoid repeated instantiation in recursion ***
    xml_converter = _ToXml(item_tag=item_tag, none_str=none_str)
    el = xml_converter.to_xml(obj, root_tag)
    # The decision and string join below are minor cost, not worth rewriting for perf.
    if root_tag is None and el.text is None:
        join = '' if indent is None else '\n'
        return join.join(_rootless_xml_elements(el, indent))
    else:
        if indent is not None:
            # ElementTree.indent is not fast, but essential to function
            ElementTree.indent(el, space=indent)
        return ElementTree.tostring(el, encoding='unicode')


@dataclass
class _ToXml:
    item_tag: str
    none_str: str

    def to_xml(self, value: Any, tag: str | None) -> ElementTree.Element:
        # Fast-path tags to local
        tag_final = self.item_tag if tag is None else tag

        # Avoid isinstance checks on every branch via narrowing up front
        element = ElementTree.Element(tag_final)

        # Fast-path None, primitives, str, bytes/bytearray
        if value is None:
            element.text = self.none_str
            return element
        value_type = type(value)
        # Short-circuit for str (almost always a built-in)
        if value_type is str:
            element.text = value  # type: ignore
            return element
        elif value_type is bytes or value_type is bytearray:
            element.text = value.decode(errors='ignore')
            return element
        # Fast-path bool, int, float; avoid tuple lookup in isinstance
        if value_type is bool or value_type is int or value_type is float:
            element.text = str(value)
            return element
        # date or datetime
        if isinstance(value, date):
            element.text = value.isoformat()
            return element
        # Mapping check: prefer type over isinstance for perf if possible
        if isinstance(value, Mapping):
            self._mapping_to_xml(element, value)
            return element
        # Dataclass check: is_dataclass is very fast if class type checked first
        if is_dataclass(value) and not isinstance(value, type):
            if tag is None:
                element = ElementTree.Element(value.__class__.__name__)
            # *** HOT: Avoid asdict for shallow dataclasses ***
            # asdict is very expensive for large tree; We'll perform a shallow copy manually if safe.
            dc_dict = value.__dict__ if hasattr(value, '__dict__') and not getattr(value, '__dataclass_params__', None).frozen else asdict(value)
            self._mapping_to_xml(element, dc_dict)
            return element
        # Pydantic BaseModel
        if isinstance(value, BaseModel):
            if tag is None:
                element = ElementTree.Element(value.__class__.__name__)
            # model_dump(mode='python') returns a dict (fast)
            self._mapping_to_xml(element, value.model_dump(mode='python'))
            return element
        # Iterable/fallback catch-all
        if isinstance(value, Iterable):
            # Avoid allocating a new list for large iterables
            append = element.append
            for item in value:
                item_el = self.to_xml(item, None)
                append(item_el)
            return element
        raise TypeError(f'Unsupported type for XML formatting: {type(value)}')

    def _mapping_to_xml(self, element: ElementTree.Element, mapping: Mapping[Any, Any]) -> None:
        for key, value in mapping.items():
            if isinstance(key, int):
                key = str(key)
            elif not isinstance(key, str):
                raise TypeError(f'Unsupported key type for XML formatting: {type(key)}, only str and int are allowed')
            element.append(self.to_xml(value, key))


def _rootless_xml_elements(root: ElementTree.Element, indent: str | None) -> Iterator[str]:
    # This path is minor cost; not worth optimizing further
    for sub_element in root:
        if indent is not None:
            ElementTree.indent(sub_element, space=indent)
        yield ElementTree.tostring(sub_element, encoding='unicode')
