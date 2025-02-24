"""
Support for memory-mapping uncompressed .npz files.
"""

import os
import struct
from collections.abc import Mapping
from zipfile import ZipFile

import numpy as np


class MemmappedNpzFile(Mapping):
    """
    A dictionary-like object with lazy-loading of numpy arrays in the given
    uncompressed .npz file. Upon construction, creates a memory map of the
    full .npz file, returning views for the arrays within on request.

    Attributes
    ----------
    files : list of str
        List of all uncompressed files in the archive with a ``.npy`` extension
        (listed without the extension). These are supported as dictionary keys.
    mmap : np.memmap
        The memory map of the full .npz file.
    arrays : dict
        Preloaded or cached arrays.

    Parameters
    ----------
    filenames : list of str or str or Path
        The zipped archive to open. Supports multiple ``.npz`` files by keeping track of
        which arrays exist in which file and separate memory maps for each file. Loads
        arrays dynamically from the correct memory map.
    cache : bool, optional
        Whether to cache array objects in case they are requested again.
    preload : bool, optional
        Whether to precreate all array objects upon opening. Enforces caching.
    """

    def __init__(self, filenames: list, cache: bool = True, preload: bool = False):
        if isinstance(filenames, (str, bytes, os.PathLike)):
            filenames = [filenames]  # Convert single filename to list

        self._offsets = {}
        self.mmaps = []
        self.cache = cache or preload
        self.preload = preload
        self.arrays = {}

        for fn in filenames:
            with ZipFile(fn, mode="r") as f:
                offsets = {
                    zinfo.filename[:-4]: (zinfo.header_offset, zinfo.file_size, len(self.mmaps))
                    for zinfo in f.infolist()
                    if zinfo.filename.endswith(".npy") and zinfo.compress_type == 0
                }
                self._offsets.update(offsets)
            self.mmaps.append(np.memmap(fn, mode="r"))  # Create memory map for this file

        self.files = list(self._offsets.keys())
        if self.preload:
            self.arrays = {name: self.load(name) for name in self.files}

    def load(self, name: str):
        header_offset, file_size, mmap_idx = self._offsets[name]
        mmap = self.mmaps[mmap_idx]  # Select the correct memory map
        # parse lengths of local header file name and extra fields
        # (ZipInfo is based on the global directory, not local header)
        fn_len, extra_len = struct.unpack(
            "<2H", mmap[header_offset + 26 : header_offset + 30]
        )
        # compute offset of start and end of data
        npy_start = header_offset + 30 + fn_len + extra_len
        npy_end = npy_start + file_size
        # read NPY header
        fp = MemoryviewIO(mmap)
        fp.seek(npy_start)
        version = np.lib.format.read_magic(fp)
        np.lib.format._check_version(version)
        shape, fortran, dtype = np.lib.format._read_array_header(fp, version)
        # produce slice of memmap
        data_start = fp.tell()
        return (
            mmap[data_start:npy_end]
            .view(dtype=dtype)
            .reshape(shape, order="F" if fortran else "C")
        )

    def close(self):
        if hasattr(self, "mmap"):
            del self.mmap
        self.arrays = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, key: str):
        if self.cache:
            try:
                return self.arrays[key]
            except KeyError:
                pass
        array = self.load(key)
        if self.cache:
            self.arrays[key] = array
        return array

    def __contains__(self, key: str):
        # Mapping.__contains__ calls __getitem__, which could be expensive
        return key in self._offsets


class MemoryviewIO(object):
    """
    Wraps an object supporting the buffer protocol to be a readonly file-like.
    """

    def __init__(self, buffer):
        self._buffer = memoryview(buffer).cast("B")
        self._pos = 0
        self.seekable = lambda: True
        self.readable = lambda: True
        self.writable = lambda: False

    def seek(self, offset, whence=0):
        if whence == 0:
            self._pos = offset
        elif whence == 1:
            self._pos += offset
        elif whence == 2:
            self._pos = self._buffer.nbytes + offset

    def read(self, size=-1):
        data = self._buffer[
            self._pos : self._pos + size if size >= 0 else None
        ].tobytes()
        self._pos += len(data)
        return data

    def tell(self):
        return self._pos
