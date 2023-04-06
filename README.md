<div align="center">
<h1>h5torch</h1>

HDF5 data utilities for PyTorch.


[![PyPi Version](https://img.shields.io/pypi/v/h5torch.svg)](https://pypi.python.org/pypi/h5torch/)
[![GitHub license](https://img.shields.io/github/license/gdewael/h5torch)](https://github.com/gdewael/h5torch/blob/main/LICENSE)
[![Documentation](https://readthedocs.org/projects/h5torch/badge/?version=latest&style=flat-default)](https://h5torch.readthedocs.io/en/latest/index.html)

</div>

`h5torch` consists of two main parts: (1) `h5torch.File`: a wrapper around `h5py.File` as an interface to create HDF5 files compatible with (2) `h5torch.Dataset`, a wrapper around `torch.utils.data.Dataset`. As a library, `h5torch` establishes a "code" for linking [h5py] and [torch]. To do this, this package has to formulate a vocabulary for how datasets generally look, unifying as many ML settings to the best of its abilities. In turn, this vocabulary allows dataloading of various machine learning data settings from a single dataset class definition, reducing boilerplate in your projects.

### Who is this package for?
Loading data from HDF5 files allows for efficient data-loading from an **on-disk** format, drastically reducing memory overhead. Additionally, you will find your datasets to be more **organized** using the HDF5 format, as everything is neatly arrayed in a single file.

If you want to use this package but are not sure your use-case is covered by the current formulation of the package, feel free to open an issue.

## Install
Since PyTorch is a dependency of `h5torch`, we recommend [installing PyTorch](https://pytorch.org/get-started/locally/) independently first, as your system may require a specific version (e.g. CUDA drivers).

After PyTorch installation, `h5torch` can be installed using `pip`
```bash
pip install h5torch
```


### Package concepts

#### Storing

The main idea behind `h5torch` is that datasets can usually be formulated as being aligned to a central object. E.g. in a classical supervised learning setup, features/inputs are aligned to a label vector/matrix. In recommender systems, a score matrix is the central object, with features aligned to rows and columns.

<p align="center">
    <img src="https://raw.githubusercontent.com/gdewael/h5torch/main/img/centralvsaligned.svg" width="750">
</p>

`h5torch` allows creating and reading HDF5 datasets for use in PyTorch using this dogma. When creating a new dataset, the first data object that should be registered is the `central` object. The type of `central` object is flexible:

- `N-D`: for regular dense data. The number of dimensions in this object will dictate how many possible aligned axes can exist.
- `coo`: The sparse variant as `N-D`. The number of dimensions here can be arbitrary high.
- `csr`: For sparse 2D arrays, this central data type can only have 2 aligned axes and can only be sampled along the first dimension
- `vlen`: For variable length 1D arrays. This central data type can only have one aligned axis (0).
- `separate`: For objects that are better stored in separate groups instead of as one dataset. An example is variable shape N-D objects such as variably-sized images. This central data type can only have one aligned axis (0).

Along this central object, axis objects can be aligned. The first dimension length of any axis object must correspond to the length of the central data object to that dimension. For example, a central data object of shape (50, 40) can only have 50-length and 40-length objects aligned to its first and second axis, respectively. For axis objects, these possibilities are available:

- `N-D`: Can have arbitrary number of dimensions. E.g. equally-sized images: `(N, 3, H, W)`.
- `csr`: Max 2 dimensions, rows will be sampled. E.g. A sparse scRNA-seq count matrix
- `vlen`: Variable length 1D arrays. E.g. Tokenized text as variable length arrays of integers.
- `separate`: For objects that are better stored in separate groups instead of as one dataset. An example is variable shape N-D objects such as variably-sized images.

Note there is no support for `coo` data type for aligned objects, that is because aligned axis objects require efficient indexing along their first dimension.

<p align="center">
    <img src="https://raw.githubusercontent.com/gdewael/h5torch/main/img/multidim.svg" width="750">
</p>

Also note that there is no limit on the number of data objects aligned to an axis. For example, in the case of images aligned to a central label vector, extra information of every image can be added such as the URL, the date said image was taken, the geolocation of that image, ...

Besides the central and axis objects, you can also store `unstructured` data which can be any length or dimension and follow any of the above-mentioned data types (including `coo`). This could for example be a vocabularium vector or the names of classes...

#### Sampling

Once a dataset is created using `h5torch.File`, it can be used as a PyTorch Dataset using `h5torch.Dataset`. Sampling can occur along any of the axes in the central object, upon which the corresponding indices in the objects aligned to that axis are also sampled. Alternatively, `coo` sampling (available for `N-D` and `coo`-type central objects) samples one specific element of the central dataset, along with the corresponding indices of all axis-aligned objects.


<p align="center">
    <img src="https://raw.githubusercontent.com/gdewael/h5torch/main/img/sampling.svg" width="750">
</p>

## Usage

Refer to the [tutorial on the documentation page](https://h5torch.readthedocs.io/en/latest/tutorial.html).


# Package roadmap
- [x] Implement typing
- [x] Provide data type conversion capabilities for registering datasets
- [x] Add support for custom samplers
- [x] Add support for making data splits
- [ ] Implement a collater for variable length objects
- [x] Add a slice sampler
- [x] Implement a way to pre-specify dataset size and append to it
- [ ] Add tests
- [x] Add better docs