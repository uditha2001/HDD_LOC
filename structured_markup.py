"""Tree-preserving representation of XML inputs for HDD.

HDD operates on ordinary nested Python containers.  This module bridges that
representation to XML without treating the input as independent source lines:
each dictionary entry denotes one complete XML element, and removing it removes
the whole element subtree.  Element names, attributes, text, and sibling order
come from an immutable template and therefore cannot be corrupted by HDD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from xml.etree import ElementTree


MarkupTree = dict[str, "MarkupTree"]


@dataclass(frozen=True)
class ElementTemplate:
    element_id: str
    tag: str
    attributes: tuple[tuple[str, str], ...]
    text: str | None
    tail: str | None
    child_ids: tuple[str, ...]


@dataclass(frozen=True)
class StructuredMarkup:
    """An immutable XML template plus HDD's removable subtree view."""

    tree: MarkupTree
    elements: Mapping[str, ElementTemplate]
    root_id: str

    @classmethod
    def parse_xml(cls, source: str) -> "StructuredMarkup":
        root = ElementTree.fromstring(source)
        elements: dict[str, ElementTemplate] = {}
        counter = 0

        def visit(element: ElementTree.Element) -> tuple[str, MarkupTree]:
            nonlocal counter
            element_id = f"element_{counter}"
            counter += 1
            child_entries: MarkupTree = {}
            child_ids: list[str] = []
            for child in list(element):
                child_id, child_tree = visit(child)
                child_ids.append(child_id)
                child_entries[child_id] = child_tree
            elements[element_id] = ElementTemplate(
                element_id=element_id,
                tag=element.tag,
                attributes=tuple(element.attrib.items()),
                text=element.text,
                tail=element.tail,
                child_ids=tuple(child_ids),
            )
            return element_id, child_entries

        root_id, root_children = visit(root)
        # The outer singleton protects the document element: HDD may reduce its
        # descendants, but can never produce an XML document without a root.
        return cls(tree={root_id: root_children}, elements=elements, root_id=root_id)

    def serialize(self, candidate: MarkupTree) -> str:
        if set(candidate) != {self.root_id}:
            raise ValueError("candidate must contain exactly the original XML root")

        def materialize(element_id: str, children: MarkupTree) -> ElementTree.Element:
            try:
                template = self.elements[element_id]
            except KeyError as exc:
                raise ValueError(f"unknown XML element id: {element_id}") from exc
            if not isinstance(children, dict):
                raise ValueError(f"children of {element_id} must be a dictionary")
            unknown = set(children).difference(template.child_ids)
            if unknown:
                raise ValueError(f"unexpected children for {element_id}: {sorted(unknown)}")

            element = ElementTree.Element(template.tag, dict(template.attributes))
            element.text = template.text
            element.tail = template.tail
            for child_id in template.child_ids:
                if child_id in children:
                    element.append(materialize(child_id, children[child_id]))
            return element

        root = materialize(self.root_id, candidate[self.root_id])
        return ElementTree.tostring(root, encoding="unicode")
