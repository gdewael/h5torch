from scipy import sparse
import h5py
import numpy as np


class File(h5py.File):
    def __init__(self, path, mode = "r"):
        super().__init__(path, mode)

    def register(self, data, axis, name = None, mode = "N-D", dtype = None):
        """
        Note: if mode = "coo": then data should be a tuple consisting of indices, data and a shape.
        If it's a np.ndarray, it will be first converted to a 2D scipy.sparse.coo_matrix
        """
        if isinstance(name, str) and ("/" in name): raise ValueError("\"/\" not allowed in name as it creates a different group structure")
        if mode not in ["N-D", "csr", "coo", "vlen", "separate"]:
            raise ValueError('mode not in ["N-D", "csr", "coo", "vlen", "separate"]')
        if (axis not in ["central", "unstructured"]) and not isinstance(axis, int):
            raise TypeError('axis should be an `int` or "central" or "unstructured')
        if axis != "central" and "central" not in self:
            raise AssertionError("\"central\" data object should exist before other items are added")
        if isinstance(axis, int) and (axis >= len(self["central"].attrs["shape"])): # type: ignore
            raise ValueError("given alignment axis exceeds the number of axes in central data object")
        if mode == "csr": data = sparse.csr_matrix(data)
        len_ = (len(data) if not isinstance(data, sparse.csr_matrix) else data.shape[0])
        if mode != "coo" and isinstance(axis, int) and (len_ != self["central"].attrs["shape"][axis]):  # type: ignore
            raise ValueError(
                """the number of rows in the given data does not equal the number of elements in the
                central data object along its alignment axis.""")
        if axis != "central" and name is None: 
            raise TypeError("`name` should be specified for data objects that are not the central object.")
        if (mode == "coo") and (axis not in ["central", "unstructured"]):
            raise ValueError("COO sparse matrix not supported for `axis` data objects.")

        if mode == "N-D": register_fun = self._ND_register
        elif mode == "csr": register_fun = self._csr_register
        elif mode == "coo": register_fun = self._coo_register
        elif mode == "vlen": register_fun = self._vlen_register
        elif mode == "separate": register_fun = self._separate_register

        if axis == "central":
            name = "central"
        else:
            name = "%s/%s"% (axis, name)
    

        register_fun(data, name, dtype) # type: ignore

        # data = data.astype(np.dtype(dtype)))

    def __repr__(self):
        f = h5py.File.__repr__(self)
        return f[0] + "h5torch" + f[5:]

    def _ND_register(self, data, name, dtype):
        self.create_dataset(name, data = data)
        self[name].attrs["shape"] = data.shape
        self[name].attrs["mode"] = "N-D"

    def _csr_register(self, data, name, dtype):
        self.create_dataset("%s/data" % name, data = data.data)
        self.create_dataset("%s/indices" % name, data = data.indices)
        self.create_dataset("%s/indptr" % name, data = data.indptr)
        self[name].attrs["shape"] = data.shape
        self[name].attrs["mode"] = "csr"

    def _coo_register(self, data, name, dtype):
        if isinstance(data, np.ndarray):
            data = sparse.coo_matrix(data)
            data = (np.stack([data.row, data.col]), data.data, data.shape)
            
        self.create_dataset("%s/indices" % name, data = data[0])
        self.create_dataset("%s/data" % name, data = data[1])
        self[name].attrs["shape"] = data[2]
        self[name].attrs["mode"] = "coo"


    def _vlen_register(self, data, name, dtype):
        if not all([len(elem.shape)==1 for elem in data]):
            raise ValueError("All elements in data should be 1D for `mode=vlen`")
        
        self.create_dataset(name, data = data, dtype = h5py.vlen_dtype(data[0].dtype))
        self[name].attrs["shape"] = (len(data), )
        self[name].attrs["mode"] = "vlen"


    def _separate_register(self, data, name, dtype):
        for ix, elem in enumerate(data):
            self.create_dataset("%s/%s" % (name, ix), data = elem)

        self[name].attrs["shape"] = (len(data), )
        self[name].attrs["mode"] = "separate"

