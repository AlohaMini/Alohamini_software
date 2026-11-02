"""Explicit absolute/relative action coding shared by dataset views and policy processors."""

from copy import deepcopy

import numpy as np

from .rotations import RotationEncoding


class RelativeActions:
    """Every target is relative to one observation anchor, never the preceding target.

    Base-frame deltas use p_target - p_reference and R_target @ R_reference.T.
    Tool-frame deltas use T_reference^-1 @ T_target, matching GR00T's EEF path:
    NVIDIA/Isaac-GR00T c56f541, gr00t/data/state_action/pose.py.
    Unselected fields remain absolute.
    """

    def __init__(
        self,
        width,
        *,
        linear_dims=(),
        rotation_groups=(),
        reference="anchor",
        rotation_representation="rotvec",
        reference_frame="base",
        position_groups=(),
    ):
        if type(width) is not int or width < 1 or reference != "anchor":
            raise ValueError("Relative actions require a positive width and an anchor reference")
        self.width = width
        self.rotation = RotationEncoding(rotation_representation)
        self.linear_dims = tuple(linear_dims)
        self.rotation_groups = tuple(tuple(g) for g in rotation_groups)
        self.position_groups = tuple(tuple(g) for g in position_groups)
        self.reference_frame = reference_frame
        indices = [*self.linear_dims, *(i for group in self.rotation_groups for i in group)]
        if (
            any(len(g) != self.rotation.width for g in self.rotation_groups)
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < width for i in indices)
        ):
            raise ValueError("Relative action coordinates must be distinct valid dimensions")
        positions = [i for group in self.position_groups for i in group]
        if (
            reference_frame not in ("base", "tool")
            or (reference_frame == "base" and self.position_groups)
            or (
                reference_frame == "tool"
                and (
                    not self.position_groups
                    or len(self.position_groups) != len(self.rotation_groups)
                    or any(len(g) != 3 for g in self.position_groups)
                    or len(set(positions)) != len(positions)
                    or any(type(i) is not int or i not in self.linear_dims for i in positions)
                )
            )
        ):
            raise ValueError("Tool-frame relative actions require paired position/rotation groups")
        self.descriptor = dict(
            width=width,
            linear_dims=list(self.linear_dims),
            rotation_groups=[list(g) for g in self.rotation_groups],
            reference=reference,
        )
        if rotation_representation != "rotvec":
            self.descriptor["rotation_representation"] = rotation_representation
        if reference_frame != "base":
            self.descriptor.update(
                reference_frame=reference_frame,
                position_groups=[list(g) for g in self.position_groups],
            )

    def _transform(self, reference, actions, inverse):
        reference = np.asarray(reference, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        if (
            reference.shape[-1:] != (self.width,)
            or actions.shape[-1:] != (self.width,)
            or actions.ndim != reference.ndim + 1
            or actions.shape[:-2] != reference.shape[:-1]
            or not np.isfinite(reference).all()
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Relative action windows must match finite anchor coordinates")
        result = np.array(actions, copy=True)
        if not result.size:
            return result
        dims = list(self.linear_dims)
        if dims:
            result[..., dims] += (1 if inverse else -1) * reference[..., None, dims]
        for number, group in enumerate(self.rotation_groups):
            dims = list(group)
            anchor = np.broadcast_to(reference[..., None, dims], actions[..., dims].shape)
            a = self.rotation.decode(anchor)
            b = self.rotation.decode(actions[..., dims])
            if self.reference_frame == "tool":
                rotation = a @ b if inverse else np.swapaxes(a, -1, -2) @ b
                xyz = list(self.position_groups[number])
                if inverse:
                    result[..., xyz] = (a @ actions[..., xyz, None])[..., 0]
                    result[..., xyz] += reference[..., None, xyz]
                else:
                    result[..., xyz] = (np.swapaxes(a, -1, -2) @ result[..., xyz, None])[..., 0]
            else:
                rotation = b @ a if inverse else b @ np.swapaxes(a, -1, -2)
            result[..., dims] = self.rotation.encode(rotation)
        return result

    def encode(self, reference, actions):
        return self._transform(reference, actions, False)

    def decode(self, reference, actions):
        return self._transform(reference, actions, True)

    def configuration(self):
        return deepcopy(self.descriptor)

    def feature(self, absolute_feature):
        """Describe delta and unchanged coordinates without relabelling their units."""
        result = deepcopy(absolute_feature)
        if result.get("shape") != [self.width] or len(result.get("names", [])) != self.width:
            raise ValueError("Relative feature names/shape must match the action width")
        relative = [*self.linear_dims, *(i for group in self.rotation_groups for i in group)]
        for index in relative:
            result["names"][index] = "delta." + result["names"][index]
        result["representation"] = "relative_to_anchor"
        result["relative_actions"] = self.configuration()
        return result
