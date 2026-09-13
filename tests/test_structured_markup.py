from __future__ import annotations

import os
import sys
from xml.etree import ElementTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from structured_markup import StructuredMarkup


def test_xml_tree_removes_complete_subtrees_and_remains_parseable():
    markup = StructuredMarkup.parse_xml(
        '<root keep="yes"><group><first/><second/></group><outside/></root>'
    )
    root_children = markup.tree[markup.root_id]
    group_id = next(iter(root_children))
    group_children = root_children[group_id]
    first_id = next(iter(group_children))

    candidate = {
        markup.root_id: {
            group_id: {
                child_id: children
                for child_id, children in group_children.items()
                if child_id != first_id
            }
        }
    }
    serialized = markup.serialize(candidate)
    root = ElementTree.fromstring(serialized)

    assert root.attrib == {"keep": "yes"}
    assert [element.tag for element in root] == ["group"]
    assert [element.tag for element in root[0]] == ["second"]


def test_xml_document_root_cannot_be_omitted():
    markup = StructuredMarkup.parse_xml("<root><child/></root>")

    try:
        markup.serialize({})
    except ValueError as exc:
        assert "XML root" in str(exc)
    else:
        raise AssertionError("missing XML document root should be rejected")
