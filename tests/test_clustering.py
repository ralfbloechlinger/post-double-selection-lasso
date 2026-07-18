"""Regression tests for one-way clustered final-stage inference."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from pdslasso import PDSLasso


def _make_cluster_data(
    n_clusters: int = 12,
    observations_per_cluster: int = 10,
    seed: int = 20260718,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_clusters), observations_per_cluster)
    cluster_shock = rng.normal(scale=0.8, size=n_clusters)[groups]
    x = rng.normal(size=groups.size)
    d_latent = 0.8 * x + 0.4 * cluster_shock + rng.normal(size=groups.size)
    d = (d_latent > 0).astype(int)
    y = 1.5 * d + 0.7 * x + cluster_shock + rng.normal(size=groups.size)
    return pd.DataFrame({"y": y, "d": d, "x": x, "cluster": groups})


def _fit_direct_clustered_ols(df: pd.DataFrame):
    design = sm.add_constant(df[["d"]], has_constant="add")
    return sm.OLS(df["y"], design).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["cluster"]},
    )


def test_valid_cluster_column_matches_statsmodels() -> None:
    df = _make_cluster_data()

    result = PDSLasso(
        data=df,
        y="y",
        d="d",
        control_cols=None,
        cluster_cov="cluster",
    ).fit()
    expected = _fit_direct_clustered_ols(df)

    assert result.cov_type == "cluster"
    np.testing.assert_allclose(result.params, expected.params)
    np.testing.assert_allclose(result.bse, expected.bse)
    np.testing.assert_allclose(result.cov_params(), expected.cov_params())


def test_missing_cluster_column_raises_value_error() -> None:
    df = _make_cluster_data().drop(columns="cluster")
    model = PDSLasso(
        data=df,
        y="y",
        d="d",
        control_cols=None,
        cluster_cov="cluster",
    )

    with pytest.raises(ValueError, match="Cluster column 'cluster' not found"):
        model.fit()


def test_missing_cluster_identifier_raises_value_error() -> None:
    df = _make_cluster_data()
    df.loc[df.index[3], "cluster"] = np.nan
    model = PDSLasso(
        data=df,
        y="y",
        d="d",
        control_cols=None,
        cluster_cov="cluster",
    )

    with pytest.raises(
        ValueError,
        match="Cluster column 'cluster' contains missing values",
    ):
        model.fit()


def test_single_cluster_raises_value_error() -> None:
    df = _make_cluster_data(n_clusters=1, observations_per_cluster=40)
    model = PDSLasso(
        data=df,
        y="y",
        d="d",
        control_cols=None,
        cluster_cov="cluster",
    )

    with pytest.raises(
        ValueError,
        match="must contain at least two distinct clusters",
    ):
        model.fit()


def test_two_clusters_are_accepted() -> None:
    df = _make_cluster_data(n_clusters=2, observations_per_cluster=40)

    result = PDSLasso(
        data=df,
        y="y",
        d="d",
        control_cols=None,
        cluster_cov="cluster",
    ).fit()

    assert result.cov_type == "cluster"
    assert np.isfinite(result.bse["d"])


def test_cluster_groups_follow_shuffled_nondefault_index() -> None:
    df = _make_cluster_data().sample(frac=1.0, random_state=42)
    df.index = np.arange(1000, 1000 + len(df)) * 3

    result = PDSLasso(
        data=df,
        y="y",
        d="d",
        control_cols=None,
        cluster_cov="cluster",
    ).fit()
    expected = _fit_direct_clustered_ols(df)

    np.testing.assert_allclose(result.bse, expected.bse)
    np.testing.assert_allclose(result.cov_params(), expected.cov_params())
