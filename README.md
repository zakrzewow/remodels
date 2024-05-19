# remodels

[![PyPI](https://img.shields.io/pypi/v/remodels.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/remodels.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/remodels)][python version]
[![License](https://img.shields.io/pypi/l/remodels)][license]

[![Documentation Status](https://readthedocs.org/projects/remodels/badge/?version=latest)][read the docs]
[![Tests](https://github.com/zakrzewow/remodels/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/zakrzewow/remodels/branch/main/graph/badge.svg)][codecov]

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi_]: https://pypi.org/project/remodels/
[status]: https://pypi.org/project/remodels/
[python version]: https://pypi.org/project/remodels
[read the docs]: https://remodels.readthedocs.io/en/latest/
[tests]: https://github.com/zakrzewow/remodels/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/zakrzewow/remodels
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

ReModels is a Python package for probabilistic energy price forecasting using Quantile Regression Averaging (QRA) methods.

## Installation

You can install _remodels_ via [pip] from [PyPI]:

```console
$ pip install remodels
```

Alternatively, you can install from source:

```console
$ git clone https://github.com/zakrzewow/remodels.git
$ cd remodels
$ pip install .
```

## Features

- **Dataset Download**: access commonly used public datasets for transparent data acquisition.
- **Data Preprocessing**: apply variance stabilizing transformation for improved data quality.
- **Forecast Generation**: produce point and probabilistic forecasts with reference implementations of QRA variants.
- **Result Evaluation**: compare predictions using dedicated metrics for fair and consistent evaluation.

ReModels provides a robust framework for researchers to compare different QRA methods and other forecasting techniques. It supports the development of new forecasting methods, extending beyond energy price forecasting.

ReModels simplifies and enhances energy price forecasting research with comprehensive tools and transparent methodologies.

The main functionality of the ReModels package are Variance Stabilizing Transformations and Quantile Regression Averaging variants.

Implemented VSTs:

- clipping,
- clipping with the logarithm function
- logistic transform
- inverse hyperbolic sine transform
- BoxCox transform
- polynomial transform
- mirror-log transformation
- probability integral transform

Implemented QRA variants:

- QRA
- QRM
- FQRA
- FQRM
- sFQRA
- sFQRM
- LQRA
- SQRA
- SQRM

## Usage

VST basic usage example:

```py
import numpy as np
from remodels.transformers.VSTransformers import ArcsinhScaler, BoxCoxScaler, ClippingScaler, LogClippingScaler, LogisticScaler, MLogScaler, PITScaler, PolyScaler

X = np.random.normal(0, 1, size=(100, 10))
ArcsinhScaler().fit_transform(X)
```

QRA basic usage example:

```py
import numpy as np
from remodels.qra import QRA, QRM, FQRA, FQRM, sFQRA, sFQRM, LQRA, SQRA, SQRM

X = np.random.normal(0, 1, size=(100, 10))
y = X.sum(axis=1)

qra = QRA(quantile=0.5, fit_intercept=True)
qra.fit(X, y).predict(X)
```

For more advanced examples of high-level functionalities, please see the [Usage] or [Reference].

## License

Distributed under the terms of the [MIT license][license],
_remodels_ is free and open source software.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/project/remodels/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/zakrzewow/remodels/blob/main/LICENSE
[usage]: https://remodels.readthedocs.io/en/latest/usage.html
[reference]: https://remodels.readthedocs.io/en/latest/reference.html
