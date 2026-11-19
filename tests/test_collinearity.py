# Проверки расчёта VIF на искусственных данных.
import numpy as np
import pandas as pd

from src.collinearity import vif


def test_vif_of_independent_columns_is_close_to_one():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(2000, 3)), columns=list("abc"))
    assert (vif(df) < 1.1).all()


def test_vif_of_duplicated_column_is_huge():
    rng = np.random.default_rng(0)
    a = rng.normal(size=500)
    df = pd.DataFrame({"a": a, "b": a + 1e-3 * rng.normal(size=500), "c": rng.normal(size=500)})
    v = vif(df)
    assert v["a"] > 1e4 and v["b"] > 1e4 and v["c"] < 1.1
