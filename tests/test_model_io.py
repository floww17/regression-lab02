"""Проверки сохранения модели и согласованности данных."""
import joblib
import numpy as np

from src.data import load_train_test
from src.experiments import EXPERIMENTS
from src.split import load_split


def test_saved_pipeline_reproduces_predictions(tmp_path):
    X, y, _, _ = load_train_test()
    spec = EXPERIMENTS["E1c-author-baseline"]
    cols = spec["numeric"] + ["Species"]
    model = spec["build"]().fit(X[cols], y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(model.predict(X[cols]), loaded.predict(X[cols]), rtol=1e-9)


def test_loaded_train_and_test_match_saved_split():
    X_train, y_train, X_test, y_test = load_train_test()
    saved = load_split()
    assert len(X_train) == saved["n_train"] and len(X_test) == saved["n_test"]
    assert set(X_train.index).isdisjoint(X_test.index)
    assert (y_train > 0).all() and (y_test > 0).all()
