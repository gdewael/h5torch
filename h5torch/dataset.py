from torch.utils import data
import h5torch
import numpy as np
from typing import Union, Literal

class Dataset(data.Dataset):
    def __init__(self, path: str, sampling: Union[int, Literal["coo"]] = 0):
        """
        h5torch.Dataset object.

        Parameters
        ----------
        path: str
            Path to the saved HDF5 file.
        sampling: Union[int, Literal[&quot;coo&quot;]]
            Sampling axis.
        """
        self.f = h5torch.File(path)
        if "central" not in self.f:
            raise ValueError("\"central\" data object was not found in input file.")
        if (sampling != "coo") and not isinstance(sampling, int):
            raise TypeError('`sampling` should be either "coo" or `int`')
        if (sampling == "coo") and (self.f["central"].attrs["mode"] not in ["coo", "N-D"]):
            raise ValueError("`coo` sampling only works with central objects stored in `coo` or `N-D` mode")
        if isinstance(sampling, int) and (sampling >= len(self.f["central"].attrs["shape"])): # type: ignore
            raise ValueError("given sampling axis exceeds the number of axes in central data object")
        if isinstance(sampling, int) and (sampling > 0) and (self.f["central"].attrs["mode"] == "csr"):
            raise ValueError("sampling axis can maximally be `0` for central objects in `csr` mode ")
        self.sampling = sampling

    def __len__(self):
        if (self.sampling == "coo") and (self.f["central"].attrs["mode"] == "coo"):
            return self.f["central"]["data"].shape[0] # type: ignore
        elif (self.sampling == "coo") and (self.f["central"].attrs["mode"] == "N-D"):
            return self.f["central"].size # type: ignore
        else:
            return self.f["central"].attrs["shape"][self.sampling] # type: ignore
    def __getitem__(self, index):
        sample = {}

        if self.sampling == "coo":
            if self.f["central"].attrs["mode"] == "N-D":
                indices = np.unravel_index(index, self.f["central"].attrs["shape"]) # type: ignore
                sample["central"] = self.f["central"][indices] # type: ignore
            else:
                indices, data = sample_coo(self.f["central"], index)
                sample["central"] = data
            for axis, ix in enumerate(indices):
                sample |= self.sample_axis(axis, ix, self.f[str(axis)].keys()) # type: ignore
            return sample
        
        else:
            sample["central"] = np.take(self.f["central"], index, self.sampling) # type: ignore
            sample |= self.sample_axis(self.sampling, index, self.f[str(self.sampling)].keys()) # type: ignore
        return sample

    def sample_axis(self, axis, index, keys):
        sample = {}
        for key in keys:
            h5object = self.f["%s/%s" % (axis, key)]
            sample["%s/%s" % (axis, key)] = mode_to_sampler[h5object.attrs["mode"]](h5object, index) # type: ignore
        return sample

    def close(self):
        self.f.close()

sample_default = lambda h5object, index : h5object[index]
sample_separate = lambda h5object, index : h5object[str(index)][()]
def sample_csr(h5object, index):
    ix0, ix1 = h5object["indptr"][index:index+2]
    x = np.zeros(h5object.attrs["shape"][1], dtype= h5object["data"].dtype)
    x[h5object["indices"][ix0:ix1]] = h5object["data"][ix0:ix1]
    return x
def sample_coo(h5object, index):
    indices = tuple(h5object["indices"][:, index])
    data = h5object["data"][index]
    return indices, data

def sample_ND_as_coo(h5object, index):
    np.unravel_index(index, h5object.attrs["shape"])

mode_to_sampler = {
    "N-D": sample_default,
    "csr": sample_csr,
    "coo": sample_coo,
    "vlen": sample_default,
    "separate": sample_separate,

}