from scipy import sparse
import h5py
import numpy as np
from typing import Optional, Literal, Union, List, Tuple, Sequence


class File(h5py.File):
    def __init__(self, path: str, mode: Literal["r", "r+", "x", "w-", "a", "w"] = "r"):
        super().__init__(path, mode)
        """Initializes a file handle to a HDF5 file.
        """

    def register(
        self,
        data: Union[List, np.ndarray, Tuple[np.ndarray, np.ndarray, Sequence]],
        axis: Union[int, Literal["central", "unstructured"]],
        name: Optional[str] = None,
        mode: Literal["N-D", "csr", "coo", "vlen", "separate"] = "N-D",
        dtype_save: Optional[str] = None,
        dtype_load: Optional[str] = None,
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

        axis : Union[int, Literal[&quot;central&quot;, &quot;unstructured&quot;]]
            Axis to align the dataset to. The first dataset that should be registered to a HDF5 file should always be the central dataset.
        name : Optional[str], optional
            name of the dataset, ignored in the case of axis == "central", mandatory in the case of an alignment axis, by default None
        mode : Literal[&quot;N-D&quot;, &quot;csr&quot;, &quot;coo&quot;, &quot;vlen&quot;, &quot;separate&quot;], optional
            mode in which to save the data, by default "N-D"
        dtype_save : Optional[str], optional
            data type in which to save the data, by default None, which means it uses the datatype as given
        dtype_load : Optional[str], optional
            data type in which to load the data, by default None, which means it uses the datatype as given
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
        if mode == "csr":
            data = sparse.csr_matrix(data)
        len_ = len(data) if not isinstance(data, sparse.csr_matrix) else data.shape[0]
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

        register_fun(data, name, dtype_save, dtype_load)

        # data = data.astype(np.dtype(dtype)))

    def _ND_register(self, data, name, dtype_save, dtype_load):
        dtype_save_np = default_dtype(data, dtype_save)
        dtype_load_np = default_dtype(data, dtype_load)

        self.create_dataset(name, data=data.astype(dtype_save_np))
        self[name].attrs["shape"] = data.shape
        self[name].attrs["mode"] = "N-D"
        self[name].attrs["dtypes"] = [str(dtype_save_np), str(dtype_load_np)]

    def _csr_register(self, data, name, dtype_save, dtype_load):
        dtype_save_np = default_dtype(data.data, dtype_save)
        dtype_load_np = default_dtype(data.data, dtype_load)

        self.create_dataset("%s/data" % name, data=data.data.astype(dtype_save_np))
        self.create_dataset("%s/indices" % name, data=data.indices)
        self.create_dataset("%s/indptr" % name, data=data.indptr)
        self[name].attrs["shape"] = data.shape
        self[name].attrs["mode"] = "csr"
        self[name].attrs["dtypes"] = [str(dtype_save_np), str(dtype_load_np)]

    def _coo_register(self, data, name, dtype_save, dtype_load):
        if isinstance(data, np.ndarray):
            data = sparse.coo_matrix(data)
            data = (np.stack([data.row, data.col]), data.data, data.shape)
        dtype_save_np = default_dtype(data[1], dtype_save)
        dtype_load_np = default_dtype(data[1], dtype_load)

        self.create_dataset("%s/indices" % name, data=data[0])
        self.create_dataset("%s/data" % name, data=data[1].astype(dtype_save_np))
        self[name].attrs["shape"] = data[2]
        self[name].attrs["mode"] = "coo"
        self[name].attrs["dtypes"] = [str(dtype_save_np), str(dtype_load_np)]

    def _vlen_register(self, data, name, dtype_save, dtype_load):
        if not all([len(elem.shape) == 1 for elem in data]):
            raise ValueError("All elements in data should be 1D for `mode=vlen`")
        if not all([elem.dtype == data[0].dtype for elem in data]):
            raise ValueError(
                "All elements in `vlen` data object should have same data type."
            )
        dtype_save_np = default_dtype(data[0], dtype_save)
        dtype_load_np = default_dtype(data[0], dtype_load)

        self.create_dataset(name, data=data, dtype=h5py.vlen_dtype(dtype_save_np))
        self[name].attrs["shape"] = (len(data),)
        self[name].attrs["mode"] = "vlen"
        self[name].attrs["dtypes"] = [str(dtype_save_np), str(dtype_load_np)]

    def _separate_register(self, data, name, dtype_save, dtype_load):
        if not all([elem.dtype == data[0].dtype for elem in data]):
            raise ValueError(
                "All elements in `separate` data object should have same data type."
            )
        dtype_save_np = default_dtype(data[0], dtype_save)
        dtype_load_np = default_dtype(data[0], dtype_load)

        for ix, elem in enumerate(data):
            self.create_dataset("%s/%s" % (name, ix), data=elem.astype(dtype_save_np))

        self[name].attrs["shape"] = (len(data),)
        self[name].attrs["mode"] = "separate"
        self[name].attrs["dtypes"] = [str(dtype_save_np), str(dtype_load_np)]

    def __repr__(self):
        f = h5py.File.__repr__(self).split("HDF5 file")
        return f[0] + "h5torch file" + f[1]


def default_dtype(data, dtype):
    if dtype is None:
        return data.dtype
    else:
        return np.dtype(dtype)
