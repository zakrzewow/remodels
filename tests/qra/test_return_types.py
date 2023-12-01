"""Test cases for all QR* models."""
from typing import Tuple

import numpy as np
import pytest

from remodels.qra import FQRA
from remodels.qra import FQRM
from remodels.qra import LQRA
from remodels.qra import QRA
from remodels.qra import QRM
from remodels.qra import SQRA
from remodels.qra import SQRM
from remodels.qra import sFQRA
from remodels.qra import sFQRM

from . import sample_data


@pytest.mark.parametrize(
    "qr_class", [QRA, QRM, FQRA, FQRM, sFQRA, sFQRM, LQRA, SQRA, SQRM]
)
def test_return_types(
    sample_data: Tuple[np.array, np.array],
    qr_class,
) -> None:
    """Fit method returns model instance, predict method returns np.ndarray with proper shape."""
    X_, y_ = sample_data
    qr_model = qr_class(quantile=0.5, fit_intercept=True)
    assert isinstance(qr_model.fit(X_, y_), QRA)

    y_pred = qr_model.predict(X_)
    assert isinstance(y_pred, np.ndarray)
    assert y_pred.shape == y_.shape
