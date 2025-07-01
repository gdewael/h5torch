from scipy import sparse
import h5py
import numpy as np
from typing import Optional, Literal, Union, List, Tuple, Sequence
import warnings
import collections


class File(h5py.File):
    """Initializes a file handle to a HDF5 file.

    Parameters
    ----------
    path : str
        path to HDF5 file to save (or read) to (or from).
    mode : Literal["r", "r+", "x", "w-", "a", "w"], optional
        load in the file in read, write, append, ..., mode, by default "r".
    driver : Optional[str]
        see https://docs.h5py.org/en/stable/high/file.html#file-drivers.
        By default None.
        Choose "core" to load file in an in-memory format.
    """

    def __init__(
        self,
        path: str,
        mode: Literal["r", "r+", "x", "w-", "a", "w"] = "r",
        driver=None,
    ) -> None:
        super().__init__(path, mode, driver=driver)

        if mode == "r":
            for key in list(self.keys()):
                if key.isdigit():
                    for axis_key in list(self[key].keys()):
                        name = "%s/%s" % (key, axis_key)
                        filled_to = self[name].attrs["filled_to"]
                        shape = self[name].attrs["shape"][0]
                        if filled_to != shape:
                            warnings.warn(
                                'object "%s" has not been completely filled to its pre-specified length (%s / %s filled)'
                                % (name, filled_to, shape)
                            )
                elif key == "central":
                    name = key
                    filled_to = self[name].attrs["filled_to"]
                    shape = self[name].attrs["shape"][0]
                    if filled_to != shape:
                        warnings.warn(
                            'object "%s" has not been completely filled to its pre-specified length (%s / %s filled)'
                            % (name, filled_to, shape)
                        )

    def register(
        self,
        data: Union[List, np.ndarray, Tuple[np.ndarray, np.ndarray, Sequence]],
        axis: Union[int, Literal["central", "unstructured"]],
        length: Optional[int] = None,
        name: Optional[str] = None,
        mode: Literal["N-D", "csr", "coo", "vlen", "separate"] = "N-D",
        dtype_save: Optional[str] = None,
        dtype_load: Optional[str] = None,
        csr_load_sparse: bool = False,
    ) -> None:
        """Registers a new dataset to a HDF5 file.

        Parameters
        ----------
        data : Union[List, np.ndarray, Tuple[np.ndarray, np.ndarray, Sequence]]
            Data to save.

            If mode == "N-D" or "csr", expects an `np.ndarray`

            If mode == "coo", expects either a 2D `np.ndarray` or a Tuple: indices (N, M), values (M) and shape (..)*N

            If mode == "vlen", expects a List of 1D np.ndarrays

            If mode == "separate", expects a List of np.ndarrays

        axis : Union[int, Literal["central", "unstructured"]]
            Axis to align the dataset to. The first dataset that should be registered to a HDF5 file should always be the central dataset.
        length : Optional[int], optional
            length of the dataset, useful when registering a dataset to which you want to append later
            by default None, which means you will not be able to append to the dataset later.
        name : Optional[str], optional
            name of the dataset, ignored in the case of axis == "central", mandatory in the case of an alignment axis, by default None
        mode : Literal["N-D", "csr", "coo", "vlen", "separate"], optional
            mode in which to save the data, by default "N-D"
        dtype_save : Optional[str], optional
            data type in which to save the data, by default None, which means it uses the datatype as given
        dtype_load : Optional[str], optional
            data type in which to load the data, by default None, which means it uses the datatype as given
        csr_load_sparse : bool, optional
            For csr objects, whether it should be loaded as a dense array or a sparse collection of indices and data.
        """
        if isinstance(name, str) and ("/" in name):
            raise ValueError(
                '"/" not allowed in name as it creates a different group structure'
            )
        if mode not in ["N-D", "csr", "coo", "vlen", "separate"]:
            raise ValueError('mode not in ["N-D", "csr", "coo", "vlen", "separate"]')
        if (axis not in ["central", "unstructured"]) and not isinstance(axis, int):
            raise TypeError('axis should be an `int` or "central" or "unstructured')
        if axis != "central" and "central" not in self:
            raise AssertionError(
                '"central" data object should exist before other items are added'
            )
        if isinstance(axis, int) and (axis >= len(self["central"].attrs["shape"])):
            raise ValueError(
                "given alignment axis exceeds the number of axes in central data object"
            )
        if (mode == "csr") and (not isinstance(data, sparse.csr_matrix)):
            data = sparse.csr_matrix(data)
        len_ = (
            (len(data) if not isinstance(data, sparse.csr_matrix) else data.shape[0])
            if length is None
            else length
        )
        if (
            mode != "coo"
            and isinstance(axis, int)
            and (len_ != self["central"].attrs["shape"][axis])
        ):
            raise ValueError(
                """the number of rows in the given data does not equal the number of elements in the
                central data object along its alignment axis."""
            )
        if axis != "central" and name is None:
            raise TypeError(
                "`name` should be specified for data objects that are not the central object."
            )
        if (mode == "coo") and (axis not in ["central", "unstructured"]):
            raise ValueError("COO sparse matrix not supported for `axis` data objects.")
        if (mode != "csr") and (csr_load_sparse == True):
            raise ValueError(
                "`csr_load_sparse = True` can only be used with csr objects."
            )

        if mode == "N-D":
            register_fun = self._ND_register
        elif mode == "csr":
            register_fun = self._csr_register
        elif mode == "coo":
            register_fun = self._coo_register
        elif mode == "vlen":
            register_fun = self._vlen_register
        elif mode == "separate":
            register_fun = self._separate_register

        if axis == "central":
            name = "central"
        else:
            name = "%s/%s" % (axis, name)

        if csr_load_sparse:
            register_fun(
                data, name, dtype_save, dtype_load, length, csr_load_sparse=True
            )
        else:
            register_fun(data, name, dtype_save, dtype_load, length)

    def _ND_register(self, data, name, dtype_save, dtype_load, length):
        dtype_save_np = default_dtype(data, dtype_save)
        dtype_load_np = default_dtype(data, dtype_load)

        shape = list(data.shape)
        if length is not None:
            if np.issubdtype(dtype_save_np, np.dtype("bytes")):
                raise ValueError("pre-specifying length does not work for bytes data types.")
            shape[0] = length
            self.create_dataset(name, shape=shape, dtype=dtype_save_np)
            self[name][: data.shape[0]] = data.astype(dtype_save_np)
        else:
            self.create_dataset(name, data=data.astype(dtype_save_np), shape=shape)

        self[name].attrs["shape"] = shape
        self[name].attrs["mode"] = "N-D"
        self[name].attrs["dtypes"] = [str(dtype_save_np), str(dtype_load_np)]
        self[name].attrs["filled_to"] = data.shape[0]

    def _csr_register(
        self, data, name, dtype_save, dtype_load, length, csr_load_sparse=False
    ):
        dtype_save_np = default_dtype(data.data, dtype_save)
        dtype_load_np = default_dtype(data.data, dtype_load)

        shape = list(data.shape)

        if length is not None:
            shape[0] = length
            self.create_dataset(
                "%s/indptr" % name, shape=shape[0] + 1, dtype=data.indptr.dtype
            )
            self["%s/indptr" % name][: len(data.indptr)] = data.indptr
            self.create_dataset(
                "%s/data" % name, data=data.data.astype(dtype_save_np), maxshape=(None,)
            )
            self.create_dataset(
                "%s/indices" % name, data=data.indices, maxshape=(None,)
            )
        else:
            self.create_dataset("%s/indptr" % name, data=data.indptr)
            self.create_dataset("%s/data" % name, data=data.data.astype(dtype_save_np))
            self.create_dataset("%s/indices" % name, data=data.indices)

        self[name].attrs["shape"] = shape
        self[name].attrs["mode"] = "csr"
        self[name].attrs["dtypes"] = [str(dtype_save_np), str(dtype_load_np)]
        self[name].attrs["filled_to"] = data.shape[0]
        self[name].attrs["load_sparse"] = csr_load_sparse

    def _coo_register(self, data, name, dtype_save, dtype_load, length):
        if length is not None:
            raise ValueError("pre-specifying length is ambiguous for coo-type objects")

        if isinstance(data, np.ndarray):
            data = sparse.coo_matrix(data)
            data = (np.stack([data.row, data.col]), data.data, data.shape)
        dtype_save_np = default_dtype(data[1], dtype_save)
        dtype_load_np = default_dtype(data[1], dtype_load)

        self.create_dataset("%s/indices" % name, data=data[0])
        self.create_dataset("%s/data" % name, data=data[1].astype(dtype_save_np))

        shape = list(data[2])

        self[name].attrs["shape"] = shape
        self[name].attrs["mode"] = "coo"
        self[name].attrs["dtypes"] = [str(dtype_save_np), str(dtype_load_np)]
        self[name].attrs["filled_to"] = shape[0]

    def _vlen_register(self, data, name, dtype_save, dtype_load, length):
        if not all([len(elem.shape) == 1 for elem in data]):
            raise ValueError("All elements in data should be 1D for `mode=vlen`")
        if not all([elem.dtype == data[0].dtype for elem in data]):
            raise ValueError(
                "All elements in `vlen` data object should have same data type."
            )
        dtype_save_np = default_dtype(data[0], dtype_save)
        dtype_load_np = default_dtype(data[0], dtype_load)

        shape = [len(data)]
        if length is not None:
            shape[0] = length
        self.create_dataset(name, dtype=h5py.vlen_dtype(dtype_save_np), shape=shape)
        self[name][: len(data)] = [d.astype(dtype_save_np) for d in data]

        self[name].attrs["shape"] = shape
        self[name].attrs["mode"] = "vlen"
        self[name].attrs["dtypes"] = [str(dtype_save_np), str(dtype_load_np)]
        self[name].attrs["filled_to"] = len(data)

    def _separate_register(self, data, name, dtype_save, dtype_load, length):
        if not all([elem.dtype == data[0].dtype for elem in data]):
            raise ValueError(
                "All elements in `separate` data object should have same data type."
            )
        dtype_save_np = default_dtype(data[0], dtype_save)
        dtype_load_np = default_dtype(data[0], dtype_load)

        for ix, elem in enumerate(data):
            self.create_dataset("%s/%s" % (name, ix), data=elem.astype(dtype_save_np))

        if length is None:
            shape = [len(data)]
        else:
            shape = [length]

        self[name].attrs["shape"] = shape
        self[name].attrs["mode"] = "separate"
        self[name].attrs["dtypes"] = [str(dtype_save_np), str(dtype_load_np)]
        self[name].attrs["filled_to"] = len(data)

    def append(self, data: str, name: str) -> None:
        """Append data to an existing group.

        Parameters
        ----------
        data : str
            The data to append
        name : str
            The key / name of the HDF5 dataset to append data to.
        """
        if self[name].attrs["mode"] not in ["vlen", "separate", "N-D", "csr"]:
            raise ValueError(
                "Appending is only possible for `N-D`, `vlen`, `csr`, or `separate` type objects."
            )
        if (self[name].attrs["mode"] == "csr") and (
            not isinstance(data, sparse.csr_matrix)
        ):
            data = sparse.csr_matrix(data)

        start_ix = self[name].attrs["filled_to"]
        end_ix = start_ix + (
            len(data) if self[name].attrs["mode"] != "csr" else data.shape[0]
        )
        if end_ix > self[name].attrs["shape"][0]:
            raise ValueError(
                "Appended data would exceed data size limits: slice = %s:%s, dataset length = %s"
                % (start_ix, end_ix, self[name].attrs["shape"][0])
            )

        if self[name].attrs["mode"] == "N-D":
            self[name][start_ix:end_ix] = data.astype(self[name].attrs["dtypes"][0])
            self[name].attrs["filled_to"] = end_ix

        if self[name].attrs["mode"] == "vlen":
            self[name][start_ix:end_ix] = [
                d.astype(self[name].attrs["dtypes"][0]) for d in data
            ]
            self[name].attrs["filled_to"] = end_ix

        if self[name].attrs["mode"] == "separate":
            for ix, elem in zip(range(start_ix, end_ix), data):
                self.create_dataset(
                    "%s/%s" % (name, ix),
                    data=elem.astype(self[name].attrs["dtypes"][0]),
                )
            self[name].attrs["filled_to"] = end_ix

        if self[name].attrs["mode"] == "csr":
            indptr_fill = data.indptr[1:] + self["%s/indptr" % name][start_ix]
            self["%s/indptr" % name][start_ix+1:end_ix+1] = indptr_fill
            self[name].attrs["filled_to"] = end_ix

            self["%s/data" % name].resize(
                self["%s/data" % name].shape[0] + len(data.data), axis=0
            )
            self["%s/data" % name][-len(data.data) :] = data.data.astype(
                self[name].attrs["dtypes"][0]
            )

            self["%s/indices" % name].resize(
                self["%s/indices" % name].shape[0] + len(data.indices), axis=0
            )
            self["%s/indices" % name][-len(data.indices) :] = data.indices

    def __repr__(self):
        f = h5py.File.__repr__(self).split("HDF5 file")
        return f[0] + "h5torch file" + f[1]

    def to_dict(self):
        """Read the whole hdf5 in memory and returns a dict-like object (h5torch.file.h5pyDict).
        This dict-like object is interoperable with h5torch.Dataset.
        """
        return hdf5_to_dict(self)


def hdf5_to_dict(hdf5):
    if isinstance(hdf5, h5py.Dataset):
        return AttrArray(hdf5[()], attrs=dict(hdf5.attrs))
    dataset = h5pyDict()
    for k, v in hdf5.items():
        dataset[k] = hdf5_to_dict(v)
        try:
            dataset[k].attrs = dict(v.attrs)
        except:
            continue
    return dataset


# this subclass of a normal dict allows to set "attr" attributes
class h5pyDict(collections.UserDict):
    def __getitem__(self, key):
        if "/" in key:
            return (
                super()
                .__getitem__(key.split("/")[0])
                .__getitem__("/".join(key.split("/")[1:]))
            )
        else:
            return super().__getitem__(key)

    def close(self):
        pass


# this subclass of a normal ndarray allows to set "attr" attributes
class AttrArray(np.ndarray):
    def __new__(cls, input_array, attrs=None):
        obj = np.asarray(input_array).view(cls)
        obj.attrs = attrs
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.attrs = getattr(obj, "attrs", None)


def default_dtype(data, dtype):
    if dtype is None:
        return data.dtype
    else:
        return np.dtype(dtype)
