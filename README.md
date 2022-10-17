# h5torch
HDF5 data utilities for PyTorch.

`h5torch` consists of two main parts: (1) `h5torch.File`: a wrapper around `h5py.File` as an interface to create HDF5 files compatible with (2) `h5torch.Dataset`, a wrapper around `torch.utils.data.Dataset`. As a library, `h5torch` establishes a "code" for how datasets should be saved, hence allowing dataloading of various machine learning data settings from a single dataset object, reducing boilerplate in your projects.

### :weary: but y tho?
Loading data from HDF5 files allows for efficient data-loading from an **on-disk** format, drastically reducing memory overhead. Additionally, you will find your datasets to be more **organized** using the HDF5 format, as everything is neatly arrayed in a single file.

## Install
```bash
conda create -n "gaetan_h5torch" python=3.10
# install torch manually for your specific system
git clone https://github.com/gdewael/h5torch
cd h5torch
pip install -e .
```

## Usage

The most simple use-case is a ML setting with a 2-D `X` matrix as central object with corresponding labels `y` along the first axis.

```python
import h5torch
import numpy as np
f = h5torch.File("example.h5t", "w")
X = np.random.randn(100, 15)
y = np.random.rand(100)
f.register(X, "central")
f.register(y, 0, name = "y")
f.close()

dataset = h5torch.Dataset("example.h5t")
dataset[5], len(dataset)
```

Note that labels `y` can also play the role of central object. Both are equivalent in this simple case.
```python
import h5torch
import numpy as np
f = h5torch.File("example.h5t", "w")
X = np.random.randn(100, 15)
y = np.random.rand(100)
f.register(y, "central")
f.register(X, 0, name = "X")
f.close()

dataset = h5torch.Dataset("example.h5t")
dataset[5], len(dataset)
```

An example with a 2-dimensional Y matrix (such as a score matrix), with objects aligned to both axes of the central matrix. Storing Y and sampling is performed in `"coo"` mode, meaning the length of the dataset is the number of nonzero elements in the score matrix, and a sample constitutes such a nonzero element, along with the stored information of the row and col of said element.
```python
import h5torch
import numpy as np
f = h5torch.File("example.h5t", "w")

Y = (np.random.rand(1000, 500) > 0.95).astype(int)
row_features = np.random.randn(1000, 15)
col_names = np.arange(500).astype(bytes)


f.register(Y, "central", mode = "coo")
f.register(row_features, 0, name = "row_features")
f.register(col_names, 1, name = "col_names")
f.close()

dataset = h5torch.Dataset("example.h5t", sampling = "coo")
dataset[5], len(dataset)
```
Note: `h5torch` does not limit the number of possible dimensions along its central data object (and hence also the number of axes to align objects to).





# Package roadmap
- [ ] Implement typing
- [ ] Provide data type conversion capabilities for registering datasets
- [ ] Add support for custom samplers
- [ ] Add support for making data splits