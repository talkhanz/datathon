"""Optimised RecommendationIndex must equal the reference `recommend` (results AND summary)."""
import os

import pandas as pd
import pytest

from careproof import pipeline as P
from careproof.config import WEIGHT_SCENARIOS
from careproof.geography import ResolvedLocation, resolve_location
from careproof.service_matching import TAXONOMY

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "derived")


def assert_same(a, b):
    ra, sa = a
    rb, sb = b
    assert [r["facility_id"] for r in ra] == [r["facility_id"] for r in rb]
    for x, y in zip(ra, rb):
        assert x == y
    assert sa == sb


def test_index_equals_reference_on_synthetic(built_gold_for_index):
    gold = built_gold_for_index
    idx = P.RecommendationIndex(gold)
    for lat, lon, km, k in [(25.61, 85.14, 5, 3), (25.61, 85.14, 100, 10), (25.61, 85.14, 100, 1), (20, 80, 10, 5)]:
        loc = ResolvedLocation(lat, lon, "t", "user_coordinates")
        assert_same(idx.recommend(loc, "dialysis", km, k), P.recommend(gold, loc, "dialysis", km, k))


@pytest.fixture(scope="module")
def built_gold_for_index():
    from test_pipeline import make_synthetic
    return make_synthetic()[2]


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "careproof_facility_service_gold.parquet")),
                    reason="local derived data not built")
def test_index_equals_reference_on_real_data():
    gold = P.coerce_gold_types(pd.read_parquet(os.path.join(DATA, "careproof_facility_service_gold.parquet")))
    pin = P.coerce_pin_types(pd.read_parquet(os.path.join(DATA, "careproof_pincode_silver.parquet")))
    places = P.place_index_from_pin_silver(pin)
    locs = [resolve_location("pincode", p, index=places) for p in ("400001", "800001", "110001", "560001", "226001")]
    locs.append(ResolvedLocation(11.938235, 79.485443, "merged", "user_coordinates"))
    for w_name in ("default", "distance_first"):
        idx = P.RecommendationIndex(gold, WEIGHT_SCENARIOS[w_name])
        for loc in locs:
            for svc in TAXONOMY:
                for km, k in ((25, 5), (150, 10)):
                    assert_same(idx.recommend(loc, svc, km, k), P.recommend(gold, loc, svc, km, k, WEIGHT_SCENARIOS[w_name]))
