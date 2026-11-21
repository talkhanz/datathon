import pytest
from careproof.normalization import (
    clean_str, normalize_name, normalize_pin, normalize_state, parse_array, parse_coordinate,
    parse_nfhs_value, validate_coordinates, is_valid_uuid,
)


@pytest.mark.parametrize("raw,expected", [
    ("110002", "110002"), (" 110 002 ", "110002"), ("110-002", "110002"), (110002, "110002"),
    (110002.0, "110002"), ("110002.0", "110002"),
])
def test_pin_valid_and_kept_as_string(raw, expected):
    p = normalize_pin(raw)
    assert p.pin == expected and isinstance(p.pin, str) and p.issue is None


@pytest.mark.parametrize("raw,issue", [
    ("011000", "invalid_first_digit"), ("11000", "wrong_length"), ("1100021", "wrong_length"),
    ("COCHIN-68", "invalid_format"), ("null", "missing"), (None, "missing"), ("", "missing"),
])
def test_pin_invalid(raw, issue):
    assert normalize_pin(raw) == normalize_pin(raw) and normalize_pin(raw).pin is None
    assert normalize_pin(raw).issue == issue


def test_pin_leading_zero_string_is_never_int_converted():
    # Indian PINs never start with 0, but a leading-zero string must be preserved, not silently stripped
    assert normalize_pin("012345").issue == "invalid_first_digit"


def test_name_normalization():
    assert normalize_name("  Dr. Ram's  Eye & ENT Hospital (P) Ltd. ") == "dr ram s eye and ent hospital p ltd"
    assert normalize_name("null") is None
    assert normalize_name("Clínica São") == "clinica sao"


@pytest.mark.parametrize("raw,canon", [
    ("Maharastra", "MAHARASHTRA"), ("NCT of Delhi", "DELHI"), ("Tamilnadu", "TAMIL NADU"),
    ("Orissa", "ODISHA"), ("Jammu & Kashmir", "JAMMU AND KASHMIR"), (" Lakshadweep ", "LAKSHADWEEP"),
    ("Andaman & Nicobar Islands", "ANDAMAN AND NICOBAR ISLANDS"), ("Mumbai", None), ("null", None),
])
def test_state_normalization(raw, canon):
    assert normalize_state(raw) == canon


def test_parse_array_variants():
    assert parse_array('["a","b","a",null,"",5]') == ["a", "b"]
    assert parse_array('["a","a"]', dedupe=False) == ["a", "a"]
    assert parse_array(["x", None, " y "]) == ["x", "y"]
    assert parse_array("null") == [] and parse_array(None) == []
    assert parse_array("plain text claim") == ["plain text claim"]


def test_coordinates():
    assert validate_coordinates("28.6", "77.2").issue is None
    assert validate_coordinates("77.2", "28.6").issue == "swapped"
    assert validate_coordinates("-81.7", "26.9").issue == "outside_india"
    assert validate_coordinates("NA", "NA").issue == "missing"
    assert validate_coordinates("17°57'17.7", "78.1").issue == "unparseable"
    assert parse_coordinate("28.430354° N") == pytest.approx(28.430354)
    assert parse_coordinate("28.4 32628") is None


def test_nfhs_values():
    v = parse_nfhs_value("*")
    assert v.value is None and v.suppressed and not v.small_sample
    v = parse_nfhs_value("(29.5)")
    assert v.value == 29.5 and v.small_sample and v.raw == "(29.5)"
    v = parse_nfhs_value("83.2 ")
    assert v.value == 83.2 and not v.small_sample
    assert parse_nfhs_value("abc").parse_error


def test_misc():
    assert clean_str(" NULL ") is None
    assert is_valid_uuid("b8a5401f-42f1-422a-8cd9-686a15b4cb76")
    assert not is_valid_uuid("| **Dr. Anvita Verma**")
