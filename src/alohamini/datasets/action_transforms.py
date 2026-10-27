"""Explicit absolute/relative action coding shared by dataset views and policy processors."""

from copy import deepcopy

import numpy as np
from scipy.spatial.transform import Rotation


class RelativeActions:
    """Every target is relative to one observation anchor, never the preceding target.

    Linear dimensions are subtracted. Rotation-vector triples use left relative
    rotations R_target @ R_reference.T. Unselected fields remain absolute.
    """

    def __init__(self, width, *, linear_dims=(), rotation_groups=(), reference="anchor"):
        if type(width) is not int or width < 1 or reference != "anchor":
            raise ValueError("Relative actions require a positive width and an anchor reference")
        self.width = width
        self.linear_dims = tuple(linear_dims)
        self.rotation_groups = tuple(tuple(g) for g in rotation_groups)
        indices = [*self.linear_dims, *(i for group in self.rotation_groups for i in group)]
        if (
            any(len(g) != 3 for g in self.rotation_groups)
            or len(set(indices)) != len(indices)
            or any(type(i) is not int or not 0 <= i < width for i in indices)
        ):
            raise ValueError("Relative action coordinates must be distinct valid dimensions")
        self.descriptor = dict(
            width=width,
            linear_dims=list(linear_dims),
            rotation_groups=[list(g) for g in rotation_groups],
            reference=reference,
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
        dims = list(self.linear_dims)
        if dims:
            result[..., dims] += (1 if inverse else -1) * reference[..., None, dims]
        for group in self.rotation_groups:
            dims = list(group)
            anchor = np.broadcast_to(reference[..., None, dims], actions[..., dims].shape)
            a = Rotation.from_rotvec(anchor.reshape(-1, 3))
            b = Rotation.from_rotvec(actions[..., dims].reshape(-1, 3))
            rotation = b * a if inverse else b * a.inv()
            result[..., dims] = rotation.as_rotvec().reshape(actions[..., dims].shape)
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
