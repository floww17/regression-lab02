# Быстрые проверки моделей. Обучение маленькое, без перекрёстной проверки.
import numpy as np
import pytest

from src.data import load_train_test
from src.evaluate import regression_metrics
from src.experiments import EXPERIMENTS


@pytest.fixture(scope="module")
def data():
    X_train, y_train, _, _ = load_train_test()
    return X_train, y_train


@pytest.mark.parametrize("name", list(EXPERIMENTS))
def test_predictions_count_and_finite(name, data):
    X, y = data
    spec = EXPERIMENTS[name]
    Xs = X[spec["numeric"] + ["Species"]]
    pred = spec["build"]().fit(Xs, y).predict(Xs)
    assert len(pred) == len(Xs)
    assert np.isfinite(pred).all()


def test_unknown_species_does_not_break_pipeline(data):
    X, y = data
    spec = EXPERIMENTS["E1c-author-baseline"]
    cols = spec["numeric"] + ["Species"]
    model = spec["build"]().fit(X[cols], y)
    new = X[cols].iloc[:1].copy()
    new["Species"] = "UnknownFish"
    assert np.isfinite(model.predict(new)).all()


def test_metrics_match_manual_calculation():
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 4.0]
    m = regression_metrics(y_true, y_pred)
    assert m["mse"] == pytest.approx(1 / 3)
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["r2"] == pytest.approx(0.5)
