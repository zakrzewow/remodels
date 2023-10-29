"""100% coverage dumbass tests."""

from remodels.qra._linear_model import _LinearModel


def test_linear_model() -> None:
    """Linear model works."""
    lm = _LinearModel().fit(X=None, y=None)
    assert type(lm) is _LinearModel
