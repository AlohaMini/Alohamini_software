"""The live description determines the TF frame set in both supported workflows."""

import pytest
from alohamini_validation.validate_tf import description_frames

from alohamini.model import get_robot_model


@pytest.mark.parametrize("description,root,count", [("collision", "base_link", 29), ("planning", "root", 32)])
def test_physical_and_planning_tf_trees_use_their_actual_roots(description, root, count):
    frame, frames = description_frames(get_robot_model("alohamini2pro").description_xml(description))
    assert frame == root and len(frames) == count
    assert {"base_link", "left_tcp", "right_tcp"} <= frames
    assert ("root" in frames) == (description == "planning")


def test_disconnected_description_is_rejected():
    with pytest.raises(ValueError, match="one named root"):
        description_frames('<robot name="bad"><link name="a"/><link name="b"/></robot>')
