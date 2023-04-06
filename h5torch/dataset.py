from torch.utils import data
import h5torch
from scipy import sparse
import numpy as np
from typing import Union, Literal, Tuple, Optional, Callable, List, Sequence
import re


class Dataset(data.Dataset):
    """h5torch.Dataset object.

    Parameters
    ----------
    path : str
        Path to the saved HDF5 file. Has to follow the logic defined by `h5torch.File`
    sampling : Union[int, Literal["coo"]], optional
        Sampling axis, by default 0
    subset : Optional[Union[Tuple[str, str], np.ndarray]], optional
        subset of data to use in dataset.
        Either: a np.ndarray of indices or np.ndarray containing booleans.
        Or: a tuple of 2 strings with the first specifying a key in the dataset and the second a regex that must match in that dataset.
        By default None, specifying to use the whole dataset as is.
    sample_processor : Optional[Callable], optional
        A callable that takes as input arguments `f` (the file handle to the HDF5 file) and `sample` (the output of this Dataset's __getitem__).
        Can be used to postprocess samples
        By default None
    """
    def __init__(
        self,
        path: str,
        sampling: Union[int, Literal["coo"]] = 0,
        subset: Optional[Union[Tuple[str, str], np.ndarray]] = None,
        sample_processor: Optional[Callable] = None,
    ):
        self.f = h5torch.File(path)
        if "central" not in self.f:
            raise ValueError('"central" data object was not found in input file.')
        if (sampling != "coo") and not isinstance(sampling, int):
            raise TypeError('`sampling` should be either "coo" or `int`')
        if (sampling != "coo") and (self.f["central"].attrs["mode"] == "coo"):
            raise ValueError("`coo` central objects require `coo` sampling")
        if (sampling == "coo") and (
            self.f["central"].attrs["mode"] not in ["coo", "N-D"]
        ):
            raise ValueError(
                "`coo` sampling only works with central objects stored in `coo` or `N-D` mode"
            )
        if isinstance(sampling, int) and (
            sampling >= len(self.f["central"].attrs["shape"])
        ):
            raise ValueError(
                "given sampling axis exceeds the number of axes in central data object"
            )
        if (
            isinstance(sampling, int)
            and (sampling > 0)
            and (self.f["central"].attrs["mode"] == "csr")
        ):
            raise ValueError(
                "sampling axis can maximally be `0` for central objects in `csr` mode "
            )
        self.sampling = sampling

        if isinstance(subset, np.ndarray) and subset.ndim > 1:
            raise ValueError("`subset` can not have more than one dimension.")
        if isinstance(subset, np.ndarray) and (subset.dtype == np.bool_):
            subset = np.where(subset)[0]
        elif isinstance(subset, tuple):
            matcher = np.vectorize(lambda x: bool(re.match(subset[1], x)))
            subset = np.where(matcher(self.f[subset[0]][:].astype(str)))[0]
        elif subset is None:
            subset = np.arange(self.__len_without_subset__())
        self.indices = subset

        if sample_processor is None:
            self.sample_processor = lambda f, sample: sample
        else:
            self.sample_processor = sample_processor

    def __len_without_subset__(self):
        if (self.sampling == "coo") and (self.f["central"].attrs["mode"] == "coo"):
            return self.f["central"]["data"].shape[0]
        elif (self.sampling == "coo") and (self.f["central"].attrs["mode"] == "N-D"):
            return self.f["central"].size
        else:
            return self.f["central"].attrs["shape"][self.sampling]

    def __len__(self):
        return len(self.indices)

    def _indices_to_index(self, index):
        return self.indices[index]

    def __getitem__(self, index):
        index = self._indices_to_index(index)
        sample = {}

        if self.sampling == "coo":
            if self.f["central"].attrs["mode"] == "N-D":
                indices = np.unravel_index(index, self.f["central"].attrs["shape"])
                sample["central"] = apply_dtype(
                    self.f["central"], self.f["central"][indices]
                )
            else:
                indices, data = sample_coo(self.f["central"], index)
                sample["central"] = data
            for axis, ix in enumerate(indices):
                if str(axis) in self.f:
                    sample |= self.sample_axis(axis, ix, self.f[str(axis)].keys())
            return sample

        else:
            if self.f["central"].attrs["mode"] == "N-D":
                sample["central"] = apply_dtype(
                    self.f["central"], np.take(self.f["central"], index, self.sampling)
                )
            else:
                sample["central"] = mode_to_sampler[self.f["central"].attrs["mode"]](
                    self.f["central"], index
                )
            if str(self.sampling) in self.f:
                sample |= self.sample_axis(
                    self.sampling, index, self.f[str(self.sampling)].keys()
                )
        return self.sample_processor(self.f, sample)

    def sample_axis(self, axis, index, keys):
        sample = {}
        for key in keys:
            h5object = self.f["%s/%s" % (axis, key)]
            sample["%s/%s" % (axis, key)] = mode_to_sampler[h5object.attrs["mode"]](
                h5object, index
            )
        return sample

    def close(self):
        self.f.close()


class SliceDataset(Dataset):
    """h5torch.SliceDataset object.
    Takes slices from the central object (and the sampled axis) as samples.
    The default behavior is to take slices starting from the first element with size `window_size` and optionally overlapping by `overlap` elements.
    If the last slice of the data would be an incomplete sample, it would be thrown away.

    If `window_indices` is specified, then `window_size` and `overlap` is ignored.

    Parameters
    ----------
    path : str
        Path to the saved HDF5 file. Has to follow the logic defined by `h5torch.File`
    sampling : Union[int, Literal["coo"]], optional
        Sampling axis, by default 0
    sample_processor : Optional[Callable], optional
        A callable that takes as input arguments `f` (the file handle to the HDF5 file) and `sample` (the output of this Dataset's __getitem__).
        Can be used to postprocess samples
        By default None
    window_size : int, optional
        Size of the slices in number of elements, by default 501
    overlap : int, optional
        Overlap of each slice in number of elements, by default 0
    window_indices : Optional[np.ndarray], optional
        A np.ndarray of size N x 2 with N the number of slices. Each row specifies the start and end index of each slice.
        (End indices are not included in python-slicing style)
        Can be used to overwrite `window_size` and `overlap` default behavior and/or to specify subsets as training/validation/test sets.
        By default None
    """
    def __init__(
        self,
        path: str,
        sampling: Union[int, Literal["coo"]] = 0,
        sample_processor: Optional[Callable] = None,
        window_size: int = 501,
        overlap: int = 0,
        window_indices: Optional[np.ndarray] = None,
    ):
        super().__init__(
            path, sampling=sampling, subset=None, sample_processor=sample_processor
        )
        if not isinstance(sampling, int):
            raise TypeError("`sampling` should be `int`")
        if self.f["central"].attrs["mode"] == "coo":
            raise ValueError(
                "`SliceDataset` is incompatible with `coo` central objects"
            )

        if window_indices is not None:
            if (window_indices.ndim != 2) or (window_indices.shape[1] != 2):
                raise ValueError("`window_indices` must be an Nx2 array")
            self.indices = window_indices

        else:
            len_ = self.f["central"].attrs["shape"][self.sampling]
            indices = np.stack(
                [
                    np.arange(0, len_, window_size - overlap),
                    np.arange(0, len_, window_size - overlap) + window_size,
                ]
            ).T

            indices = indices[~(indices > len_).sum(1).astype(bool)]
            self.indices = indices

    def __len__(self):
        return len(self.indices)

    def _indices_to_index(self, index):
        index = self.indices[index]
        return np.arange(index[0], index[1])


sample_default = lambda h5object, index: apply_dtype(h5object, h5object[index])


def sample_vlen(h5object, index):
    if isinstance(index, (int, np.integer)):
        return apply_dtype(h5object, h5object[index])
    else:
        res = np.empty(len(index), object)
        res[:] = [apply_dtype(h5object, i) for i in h5object[index]]
        return res


def sample_separate(h5object, index):
    if isinstance(index, (int, np.integer)):
        return apply_dtype(h5object, h5object[str(index)][()])
    else:
        res = np.empty(len(index), object)
        res[:] = [apply_dtype(h5object, h5object[str(i)][()]) for i in index]
        return res


def sample_csr_oneindex(h5object, index):
    ix0, ix1 = h5object["indptr"][index : index + 2]
    x = np.zeros(h5object.attrs["shape"][1], dtype=h5object.attrs["dtypes"][1])
    x[h5object["indices"][ix0:ix1]] = h5object["data"][ix0:ix1]
    return x


def sample_csr_slice(h5object, ix0, ix1):
    t = h5object["indptr"][ix0 : ix1 + 2]
    r = np.repeat(np.arange(ix1 - ix0 + 1), np.diff(t))
    c = h5object["indices"][t[0] : t[-1]]
    x = np.zeros(
        (ix1 - ix0 + 1, h5object.attrs["shape"][1]), dtype=h5object.attrs["dtypes"][1]
    )
    x[r, c] = h5object["data"][t[0] : t[-1]]
    return x


def sample_csr(h5object, index):
    if isinstance(index, (int, np.integer)):
        return apply_dtype(h5object, sample_csr_oneindex(h5object, index))
    else:
        if (index == np.arange(index[0], index[-1] + 1)).all():
            return apply_dtype(
                h5object, sample_csr_slice(h5object, index[0], index[-1])
            )
        else:
            return apply_dtype(
                h5object, np.stack([sample_csr_oneindex(h5object, i) for i in index])
            )


def sample_coo(h5object, index):
    indices = tuple(h5object["indices"][:, index])
    data = apply_dtype(h5object, h5object["data"][index])
    return indices, data


def apply_dtype(h5object, data):
    return data.astype(h5object.attrs["dtypes"][1])


mode_to_sampler = {
    "N-D": sample_default,
    "csr": sample_csr,
    "coo": sample_coo,
    "vlen": sample_vlen,
    "separate": sample_separate,
}
