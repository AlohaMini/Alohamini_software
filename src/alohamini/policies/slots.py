"""Explicit named-coordinate placement in fixed-width pretrained model inputs."""


class NamedSlots:
    """Slot indices place coordinates; they do not infer frames, units or joint semantics."""

    def __init__(self, names, mapping, width):
        if type(width) is not int or width < 1:
            raise ValueError("Model slot width must be a positive integer")
        if (
            not isinstance(names, (tuple, list))
            or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Slots require ordered, distinct field names")
        if not isinstance(mapping, dict) or set(mapping) != set(names):
            raise ValueError("Slot mapping must cover exactly the declared field names")
        values = [mapping[name] for name in names]
        if any(type(index) is not int or not 0 <= index < width for index in values):
            raise ValueError("Coordinate slots must be integer indices inside model width")
        if len(set(values)) != len(values):
            raise ValueError("Coordinate slots must be distinct")
        self.names, self.indices, self.width = tuple(names), tuple(values), width

    def _indices(self, device):
        import torch

        return torch.tensor(self.indices, dtype=torch.long, device=device)

    def pack(self, tensor):
        if tensor.shape[-1] != len(self.names):
            raise ValueError("Tensor width differs from the named coordinate layout")
        result = tensor.new_zeros((*tensor.shape[:-1], self.width))
        return result.index_copy(-1, self._indices(tensor.device), tensor)

    def unpack(self, tensor):
        if tensor.shape[-1] != self.width:
            raise ValueError("Tensor width differs from model slot width")
        return tensor.index_select(-1, self._indices(tensor.device))

    def mask(self, *shape, device):
        import torch

        result = torch.zeros(*shape, self.width, dtype=torch.bool, device=device)
        result[..., list(self.indices)] = True
        return result
