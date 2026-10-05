"""固定八维 v1 特征变换测试（任务 1.3）。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hotel_model.data import Sample
from hotel_model.features import (
    FEATURE_CONSTANTS,
    FEATURE_COUNT,
    FEATURE_NAMES,
    PREPROCESSING_VERSION,
    FeatureConfig,
    feature_matrix,
    to_features,
)

from .conftest import make_samples, row


def make_sample(**overrides: object) -> Sample:
    return make_samples([row("x", **overrides)])[0]


def test_feature_order_and_version() -> None:
    assert FEATURE_NAMES == (
        "grade",
        "renovation",
        "rating",
        "subway",
        "mall",
        "attraction",
        "area",
        "price",
    )
    assert FEATURE_COUNT == 8
    assert PREPROCESSING_VERSION == "v1"


def test_grade_centered_on_four() -> None:
    for grade in range(1, 10):
        vector = to_features(make_sample(grade=grade))
        assert vector[0] == pytest.approx(grade - 4)


def test_renovation_capped_at_fifteen_years() -> None:
    assert to_features(make_sample(renovation_age_years=15))[1] == pytest.approx(-3.0)
    assert to_features(make_sample(renovation_age_years=20))[1] == pytest.approx(-3.0)
    assert to_features(make_sample(renovation_age_years=5))[1] == pytest.approx(-1.0)
    assert to_features(make_sample(renovation_age_years=0))[1] == pytest.approx(0.0)


def test_rating_centered_on_4_7_with_0_2_scale() -> None:
    assert to_features(make_sample(ctrip_rating=4.6))[2] == pytest.approx(-0.5)
    assert to_features(make_sample(ctrip_rating=4.7))[2] == pytest.approx(0.0)
    assert to_features(make_sample(ctrip_rating=4.8))[2] == pytest.approx(0.5)


def test_zero_minutes_give_zero_cost_feature() -> None:
    vector = to_features(
        make_sample(subway_walk_minutes=0, mall_transit_minutes=0, attraction_taxi_minutes=0)
    )
    assert vector[3] == pytest.approx(0.0)
    assert vector[4] == pytest.approx(0.0)
    assert vector[5] == pytest.approx(0.0)


def test_minutes_use_log1p_over_five() -> None:
    vector = to_features(make_sample(subway_walk_minutes=5))
    assert vector[3] == pytest.approx(-math.log(2.0))
    vector = to_features(make_sample(mall_transit_minutes=15))
    assert vector[4] == pytest.approx(-math.log(4.0))


def test_area_and_price_logarithms() -> None:
    assert to_features(make_sample(room_area_sqm=25))[6] == pytest.approx(0.0)
    assert to_features(make_sample(room_area_sqm=50))[6] == pytest.approx(math.log(2.0))
    assert to_features(make_sample(nightly_price=300))[7] == pytest.approx(0.0)
    assert to_features(make_sample(nightly_price=600))[7] == pytest.approx(-math.log(2.0))


def test_price_ratio_sensitivity() -> None:
    """200→300 的价格特征下降幅度约为 800→900 的 3.44 倍。"""

    low = to_features(make_sample(nightly_price=200))[7] - to_features(
        make_sample(nightly_price=300)
    )[7]
    high = to_features(make_sample(nightly_price=800))[7] - to_features(
        make_sample(nightly_price=900)
    )[7]
    assert low == pytest.approx(math.log(1.5))
    assert high == pytest.approx(math.log(900 / 800))
    assert low / high == pytest.approx(3.44, abs=0.01)


def test_monotonic_directions() -> None:
    """正向指标随数值上升，成本指标随数值下降。"""

    better = to_features(
        make_sample(
            grade=5,
            renovation_age_years=1,
            ctrip_rating=4.8,
            subway_walk_minutes=2,
            mall_transit_minutes=3,
            attraction_taxi_minutes=4,
            room_area_sqm=40,
            nightly_price=200,
        )
    )
    worse = to_features(
        make_sample(
            grade=3,
            renovation_age_years=14,
            ctrip_rating=4.5,
            subway_walk_minutes=20,
            mall_transit_minutes=30,
            attraction_taxi_minutes=40,
            room_area_sqm=20,
            nightly_price=500,
        )
    )
    assert np.all(better > worse), "正向指标应更大，成本指标应更接近零"


def test_preprocessing_independent_of_city_and_pool() -> None:
    same = dict(
        grade=4,
        renovation_age_years=3.5,
        ctrip_rating=4.65,
        subway_walk_minutes=7,
        mall_transit_minutes=11,
        attraction_taxi_minutes=19,
        room_area_sqm=31,
        nightly_price=351,
    )
    a = make_samples([row("x", city="杭州", **same)])[0]
    b = make_samples([row("x", city="成都", **same)])[0]
    assert np.array_equal(to_features(a), to_features(b))

    # 同一原值放在不同规模的样本池中，向量不变。
    pool = make_samples([row("x", city="成都", **same), row("y", city="成都", grade=9)])
    single = make_samples([row("x", city="成都", **same)])
    assert np.array_equal(feature_matrix(pool)[0], feature_matrix(single)[0])


def test_feature_config_rejects_unknown_version_and_names() -> None:
    with pytest.raises(ValueError, match="预处理版本"):
        FeatureConfig(version="v2")
    with pytest.raises(ValueError, match="特征顺序"):
        FeatureConfig(names=("grade",))
    with pytest.raises(ValueError, match="常数"):
        FeatureConfig(constants={"grade_center": 99.0})


def test_feature_config_round_trip() -> None:
    config = FeatureConfig()
    restored = FeatureConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()
    assert restored.constants == FEATURE_CONSTANTS
    with pytest.raises(ValueError):
        FeatureConfig.from_dict({"version": "v9", "names": list(FEATURE_NAMES), "constants": {}})


def test_vector_is_finite_for_extreme_inputs() -> None:
    vector = to_features(
        make_sample(
            grade=9,
            renovation_age_years=100000,
            ctrip_rating=5.0,
            subway_walk_minutes=100000,
            mall_transit_minutes=0,
            attraction_taxi_minutes=0.001,
            room_area_sqm=1e-6,
            nightly_price=1e9,
        )
    )
    assert np.all(np.isfinite(vector))
