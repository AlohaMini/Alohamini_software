"""Private, file-backed training sequences with process-local bounded block caches.

These caches contain only values written by this process, never dataset-supplied
pickle files. Keep the creating Dataset alive until its workers have exited;
worker copies reopen files but do not own or remove the temporary directory.
The decoded cache budget is not an RSS limit: one working block, serialization
buffers and operating-system file pages can also be resident.
"""

import io
import mmap
import operator
import os
import pickle
import shutil
import struct
import sys
import tempfile
import threading
import weakref
from collections import OrderedDict
from collections.abc import Sequence
from itertools import islice
from pathlib import Path

_BLOCK = struct.Struct("<QQQ")  # exclusive row end, byte offset, byte length
_INTEGER = struct.Struct("<q")


def _remove_directory(path, pid):
    if os.getpid() == pid:
        shutil.rmtree(path, ignore_errors=True)


class _DirectoryOwner:
    def __init__(self, root):
        self.path = tempfile.mkdtemp(prefix="alohamini-data-", dir=root)
        self.cleanup = weakref.finalize(self, _remove_directory, self.path, os.getpid())


def _deep_size(value):
    """Count owned Python containers once, including shared/cyclic references."""
    seen = set()

    def measure(item):
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(item)
        if isinstance(item, dict):
            size += sum(measure(k) + measure(v) for k, v in item.items())
        elif isinstance(item, (list, tuple, set, frozenset)):
            size += sum(measure(v) for v in item)
        return size

    return measure(value)


def _index(index, length):
    index = operator.index(index)
    if index < 0:
        index += length
    if not 0 <= index < length:
        raise IndexError("sequence index out of range")
    return index


class _Files:
    def _initialize(self, root):
        self._owner = _DirectoryOwner(root)
        self.path = Path(self._owner.path)
        self._closed = False
        self._process_state()

    def _process_state(self):
        self._pid = os.getpid()
        self._lock = threading.RLock()
        self._handles = {}

    def _prepare(self, *, closing=False):
        if self._pid != os.getpid():
            # Do not acquire an inherited lock: its owning thread may not exist.
            if not self._finalized and not closing:
                raise RuntimeError("finalize file-backed data before forking workers")
            self._close_handles()
            self._process_state()
            self._reset_cache()
        if self._closed:
            raise ValueError("file-backed sequence is closed")
        return self._lock

    def _reset_cache(self):
        pass

    def _close_handles(self):
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def _file(self, name, mode):
        if name not in self._handles:
            # No inherited buffered writes can be flushed by a forked worker.
            self._handles[name] = (self.path / name).open(mode, buffering=0)
        return self._handles[name]

    def _mapping(self, name, *, writable=False):
        key = f"{name}:mmap"
        if key not in self._handles:
            source = self._file(name, "r+b" if writable else "rb")
            if not os.fstat(source.fileno()).st_size:
                return None
            self._handles[key] = mmap.mmap(
                source.fileno(), 0, access=mmap.ACCESS_WRITE if writable else mmap.ACCESS_READ
            )
        return self._handles[key]

    def _restore(self, state):
        self.__dict__.update(state)
        self._owner = None
        self._closed = False
        self._process_state()
        self._reset_cache()

    def close(self):
        if not hasattr(self, "_handles") or self._closed:
            return
        with self._prepare(closing=True):
            self._close_handles()
            self._reset_cache()
            self._closed = True
            owner = self._owner
            self._owner = None
            if owner is not None:
                owner.cleanup()

    def __del__(self):
        self.close()


class _Sequence(Sequence):
    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self._get(_index(index, len(self)))

    def __eq__(self, other):
        if self is other:
            return True
        try:
            if len(self) != len(other):
                return False
            return all(left == right for left, right in zip(self, other, strict=True))
        except TypeError:
            return NotImplemented


class RowStore(_Files, _Sequence):
    """Append values, then read immutable blocks through a bounded decoded LRU.

    ``finalize`` (also called by the first read or pickle) ends appends. Values
    are snapshotted on append; returned nested objects must be treated as read-only.
    ``currbytes`` includes conservative per-entry cache bookkeeping. Oversized
    blocks are read without caching. Separate stores can keep rows and sidecars
    from decoding one another.
    """

    def __init__(self, cache_bytes=64 * 1024**2, block_rows=128, *, root=None):
        if type(cache_bytes) is not int or cache_bytes < 0:
            raise ValueError("cache_bytes must be a nonnegative integer")
        if type(block_rows) is not int or block_rows < 1:
            raise ValueError("block_rows must be a positive integer")
        self.cache_bytes, self.block_rows = cache_bytes, block_rows
        self._length = self._blocks = 0
        self._finalized = False
        self._pending = []
        self._initialize(root)
        self._reset_cache()
        (self.path / "values").touch()
        (self.path / "index").touch()

    def _reset_cache(self):
        self._cache = OrderedDict()
        self._currbytes = 0
        self.hits = self.misses = 0

    @property
    def currbytes(self):
        with self._prepare():
            return self._currbytes

    def __len__(self):
        return self._length

    def append(self, value):
        with self._prepare():
            if self._finalized:
                raise ValueError("cannot append to a finalized RowStore")
            self._pending.append(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            self._length += 1
            if len(self._pending) == self.block_rows:
                self._flush()

    def extend(self, values):
        for value in values:
            self.append(value)

    def _flush(self):
        if not self._pending:
            return
        payload = b"".join(self._pending)
        stream = self._file("values", "ab")
        offset = stream.tell()
        stream.write(payload)
        self._file("index", "ab").write(_BLOCK.pack(self._length, offset, len(payload)))
        self._blocks += 1
        self._pending.clear()

    def finalize(self):
        with self._prepare():
            if not self._finalized:
                self._flush()
                self._close_handles()
                self._finalized = True
        return self

    def _block(self, number):
        with self._prepare():
            self.finalize()
            cached = self._cache.pop(number, None)
            if cached is not None:
                self.hits += 1
                self._cache[number] = cached
                return cached[0]
            self.misses += 1
            index = self._mapping("index")
            end, offset, size = _BLOCK.unpack_from(index, number * _BLOCK.size)
            start = _BLOCK.unpack_from(index, (number - 1) * _BLOCK.size)[0] if number else 0
            stream = self._file("values", "rb")
            stream.seek(offset)
            payload = io.BytesIO(stream.read(size))
            # Each appended value has its own pickle memo.
            block = tuple(pickle.load(payload) for _ in range(end - start))
            footprint = max(_deep_size(block), size) + 512
            if footprint <= self.cache_bytes:
                while self._currbytes + footprint > self.cache_bytes:
                    _, (_, removed) = self._cache.popitem(last=False)
                    self._currbytes -= removed
                self._cache[number] = (block, footprint)
                self._currbytes += footprint
            return block

    def _get(self, index):
        return self._block(index // self.block_rows)[index % self.block_rows]

    def __iter__(self):
        self.finalize()
        for number in range(self._blocks):
            yield from self._block(number)

    def view(self, component):
        return _RowView(self, operator.index(component))

    def __getstate__(self):
        self.finalize()
        return {
            key: self.__dict__[key]
            for key in ("path", "cache_bytes", "block_rows", "_length", "_blocks", "_finalized")
        }

    def __setstate__(self, state):
        self._pending = []
        self._restore(state)


class _RowView(_Sequence):
    def __init__(self, store, component):
        self.store, self.component = store, component

    def __len__(self):
        return len(self.store)

    def _get(self, index):
        return self.store[index][self.component]

    def __iter__(self):
        return (value[self.component] for value in self.store)


class FileBackedIntSequence(_Files, _Sequence):
    """Compact signed int64 sequence; read/search through a read-only mmap."""

    def __init__(self, *, root=None):
        self._length = 0
        self._finalized = False
        self._initialize(root)
        (self.path / "values").touch()

    def __len__(self):
        return self._length

    def append(self, value):
        if isinstance(value, bool):
            raise TypeError("integer sequence values must be integers, not booleans")
        value = operator.index(value)
        if not -(1 << 63) <= value < (1 << 63):
            raise OverflowError("integer sequence value is outside signed int64")
        encoded = _INTEGER.pack(value)
        with self._prepare():
            if self._finalized:
                raise ValueError("cannot append to a finalized integer sequence")
            self._file("values", "ab").write(encoded)
            self._length += 1

    def extend(self, values):
        for value in values:
            self.append(value)

    def finalize(self):
        with self._prepare():
            if not self._finalized:
                self._close_handles()
                self._finalized = True
        return self

    def _get(self, index):
        with self._prepare():
            self.finalize()
            return _INTEGER.unpack_from(self._mapping("values"), index * _INTEGER.size)[0]

    def searchsorted(self, value, side="left"):
        """Return a scalar insertion index in an already sorted sequence."""
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        left, right = 0, len(self)
        while left < right:
            middle = (left + right) // 2
            item = self[middle]
            if item < value or (side == "right" and item == value):
                left = middle + 1
            else:
                right = middle
        return left

    def __getstate__(self):
        self.finalize()
        return {key: self.__dict__[key] for key in ("path", "_length", "_finalized")}

    def __setstate__(self, state):
        self._restore(state)


class IndexSet(_Files):
    """Deduplicated row indices in an mmap byte mask, iterating in sorted order."""

    def __init__(self, size, root=None):
        if type(size) is not int or size < 0:
            raise ValueError("size must be a nonnegative integer")
        self.size, self._length = size, 0
        self._finalized = False
        self._initialize(root)
        with (self.path / "values").open("wb") as stream:
            stream.truncate(size)

    def __len__(self):
        return self._length

    def __contains__(self, index):
        try:
            index = operator.index(index)
        except TypeError:
            return False
        if not 0 <= index < self.size:
            return False
        with self._prepare():
            return bool(self._mapping("values", writable=not self._finalized)[index])

    def add(self, index):
        if isinstance(index, bool):
            raise TypeError("row indices must be integers, not booleans")
        index = operator.index(index)
        if not 0 <= index < self.size:
            raise IndexError("row index outside IndexSet size")
        with self._prepare():
            if self._finalized:
                raise ValueError("cannot update a finalized IndexSet")
            values = self._mapping("values", writable=True)
            if not values[index]:
                values[index] = 1
                self._length += 1

    def update(self, indices):
        iterator = iter(indices)
        while batch := tuple(islice(iterator, 4096)):
            unique = set()
            for index in batch:
                if isinstance(index, bool):
                    raise TypeError("row indices must be integers, not booleans")
                index = operator.index(index)
                if not 0 <= index < self.size:
                    raise IndexError("row index outside IndexSet size")
                unique.add(index)
            with self._prepare():
                if self._finalized:
                    raise ValueError("cannot update a finalized IndexSet")
                values = self._mapping("values", writable=True)
                start, end = min(unique), max(unique) + 1
                if len(unique) == end - start:
                    previous = values[start:end]
                    self._length += previous.count(b"\x00")
                    values[start:end] = b"\x01" * (end - start)
                else:
                    for index in unique:
                        if not values[index]:
                            values[index] = 1
                            self._length += 1

    def finalize(self):
        with self._prepare():
            if not self._finalized:
                self._close_handles()
                self._finalized = True
        return self

    def __iter__(self):
        self.finalize()
        # Bound transient Python buffers rather than materializing all true indices.
        for start in range(0, self.size, 65536):
            with self._prepare():
                block = self._mapping("values")[start : start + 65536]
            for offset, present in enumerate(block):
                if present:
                    yield start + offset

    def __getstate__(self):
        self.finalize()
        return {key: self.__dict__[key] for key in ("path", "size", "_length", "_finalized")}

    def __setstate__(self, state):
        self._restore(state)
