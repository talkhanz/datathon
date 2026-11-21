import pytest
from careproof.geography import (
    LocationError, PlaceIndex, aggregate_pin, haversine_km, resolve_location,
)


def test_haversine_known_distance():
    import math
    # analytic checks: quarter of a great circle, and 1 degree of latitude
    assert haversine_km(0, 0, 0, 90) == pytest.approx(math.pi / 2 * 6371.0088, rel=1e-9)
    assert haversine_km(20, 77, 21, 77) == pytest.approx(math.pi / 180 * 6371.0088, rel=1e-9)
    assert haversine_km(28.6, 77.2, 19.0, 72.8) == pytest.approx(haversine_km(19.0, 72.8, 28.6, 77.2))
    assert haversine_km(10, 78, 10, 78) == 0
    assert haversine_km(0, 0, 0, 1) == pytest.approx(111.2, abs=0.2)


def test_haversine_requires_coordinates():
    with pytest.raises(ValueError):
        haversine_km(None, 77, 28, 77)


def _o(d, s, lat, lon):
    return {"district": d, "statename": s, "latitude": lat, "longitude": lon}


def test_aggregate_pin_rejects_outliers_and_swaps():
    offices = [_o("PATNA", "BIHAR", "25.61", "85.14"), _o("PATNA", "BIHAR", "25.62", "85.15"),
               _o("PATNA", "BIHAR", "25.60", "85.13"), _o("PATNA", "BIHAR", "85.14", "25.61"),
               _o("PATNA", "BIHAR", "19.0", "72.8"), _o("PATNA", "BIHAR", "NA", "NA")]
    s = aggregate_pin("800001", offices)
    assert s.post_office_count == 6 and s.n_offices_with_valid_coords == 4
    assert s.n_offices_coord_outliers == 1 and s.coords_defensible
    assert s.centroid_lat == pytest.approx(25.61, abs=0.01)
    assert s.coord_issue_counts == {"swapped": 1, "missing": 1}
    assert not s.is_ambiguous and s.primary_district == "PATNA"


def test_aggregate_pin_ambiguous_and_missing():
    s = aggregate_pin("121004", [_o("FARIDABAD", "HARYANA", "NA", "NA"), _o("PALWAL", "UTTAR PRADESH", "NA", "NA")])
    assert s.is_ambiguous and s.missing_coordinates and not s.coords_defensible
    assert s.centroid_lat is None and s.primary_district is None


def test_aggregate_pin_wide_spread_not_defensible():
    s = aggregate_pin("111111", [_o("A", "BIHAR", "25.0", "85.0"), _o("A", "BIHAR", "25.3", "85.3")])
    assert not s.coords_defensible  # two offices ~45 km apart: both are >15 km from the median


def _index():
    rows = [
        {"pincode": "800001", "coords_defensible": True, "is_ambiguous": False, "centroid_lat": 25.61,
         "centroid_lon": 85.14, "primary_district": "PATNA", "primary_state": "BIHAR", "districts": ["PATNA"],
         "states": ["BIHAR"], "post_office_count": 5, "n_offices_with_valid_coords": 5},
        {"pincode": "110025", "coords_defensible": True, "is_ambiguous": True, "centroid_lat": 28.55,
         "centroid_lon": 77.28, "primary_district": None, "primary_state": None,
         "districts": ["SOUTH", "SOUTH EAST"], "states": ["DELHI", "HARYANA"], "post_office_count": 5,
         "n_offices_with_valid_coords": 5},
        {"pincode": "999999", "coords_defensible": False, "is_ambiguous": False, "centroid_lat": None,
         "centroid_lon": None, "primary_district": "X", "primary_state": "BIHAR", "districts": ["X"],
         "states": ["BIHAR"], "post_office_count": 2, "n_offices_with_valid_coords": 0},
        {"pincode": "495001", "coords_defensible": True, "is_ambiguous": False, "centroid_lat": 22.08,
         "centroid_lon": 82.15, "primary_district": "BILASPUR", "primary_state": "CHHATTISGARH",
         "districts": ["BILASPUR"], "states": ["CHHATTISGARH"], "post_office_count": 3, "n_offices_with_valid_coords": 3},
        {"pincode": "174001", "coords_defensible": True, "is_ambiguous": False, "centroid_lat": 31.33,
         "centroid_lon": 76.76, "primary_district": "BILASPUR", "primary_state": "HIMACHAL PRADESH",
         "districts": ["BILASPUR"], "states": ["HIMACHAL PRADESH"], "post_office_count": 3, "n_offices_with_valid_coords": 3},
    ]
    return PlaceIndex.from_pin_rows(rows)


def test_resolve_pin_and_coordinates():
    idx = _index()
    loc = resolve_location("pincode", "800 001", index=idx)
    assert loc.method == "pin_centroid" and loc.district == "PATNA" and not loc.warnings
    loc = resolve_location("coordinates", "", "25.6", "85.1")
    assert (loc.lat, loc.lon) == (25.6, 85.1)
    assert resolve_location("coordinates", "25.6, 85.1").lat == 25.6


def test_ambiguous_pin_is_surfaced():
    loc = resolve_location("pincode", "110025", index=_index())
    assert loc.warnings and "multiple districts" in loc.warnings[0]


@pytest.mark.parametrize("args,msg", [
    (("pincode", "12345"), "not a valid"), (("pincode", "560001"), "not in the India Post"),
    (("pincode", "999999"), "no reliable location"), (("district", "Bilaspur"), "several states"),
    (("district", "Atlantis"), "Could not find"),
])
def test_actionable_errors(args, msg):
    with pytest.raises(LocationError, match=msg):
        resolve_location(*args, index=_index())


def test_coordinate_errors():
    with pytest.raises(LocationError, match="swapped"):
        resolve_location("coordinates", "", "77.2", "28.6")
    with pytest.raises(LocationError, match="decimal degrees"):
        resolve_location("coordinates", "", "", "")


def test_district_with_state_disambiguates():
    loc = resolve_location("district", "Bilaspur, Chhattisgarh", index=_index())
    assert loc.state == "CHHATTISGARH"


def test_free_text_location_detection():
    from careproof.geography import resolve_free_text
    idx = _index()
    assert resolve_free_text("800 001", idx).method == "pin_centroid"
    assert resolve_free_text("25.6, 85.1", idx).method == "user_coordinates"
    assert resolve_free_text("Bilaspur, Chhattisgarh", idx).state == "CHHATTISGARH"
    with pytest.raises(LocationError, match="several states"):
        resolve_free_text("Bilaspur", idx)
    with pytest.raises(LocationError, match="not a valid"):
        resolve_free_text("12345", idx)
    with pytest.raises(LocationError, match="Enter a PIN"):
        resolve_free_text("  ", idx)
