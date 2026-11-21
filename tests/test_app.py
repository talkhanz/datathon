"""Headless Streamlit tests of the redesigned app against local Parquet (skipped if data or streamlit absent)."""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "derived")
st_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "careproof_facility_service_gold.parquet")),
                                reason="local derived data not built")


def search(svc="dialysis", loc="800001", radius=25):
    os.environ["CAREPROOF_LOCAL_DATA"] = DATA
    at = st_testing.AppTest.from_file(os.path.join(ROOT, "app", "app.py"), default_timeout=180)
    at.run()
    at.selectbox(key="svc").set_value(svc)
    at.text_input(key="loc_text").set_value(loc)
    at.selectbox(key="radius").set_value(radius)
    next(b for b in at.button if b.label == "Search").click().run()
    return at


def text(at):
    return " ".join([m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning]
                    + [c.value for c in at.caption] + [i.value for i in at.info])


def test_strong_results_and_safety_line():
    at = search()
    assert not at.exception
    t = text(at)
    assert "Patna Dialysis Centre" in t and "Strong evidence" in t and "Not verified" in t
    assert "Call the facility to confirm" in t


def test_coordinates_in_one_box_and_abstention():
    at = search(loc="11.938235, 79.485443", radius=10)
    assert not at.exception
    t = text(at)
    assert "none had sufficiently consistent identity" in t and "is not shown" in t  # nearest keyword match explained


def test_invalid_location():
    at = search(loc="12345")
    assert not at.exception and "not a valid Indian PIN code" in text(at)


def test_call_and_refer_flow_promotes_alternative():
    at = search()
    first = at.session_state["res"][0]["facility_id"]
    at.button(key=f"pick_{first}").click().run()
    at.radio(key=f"o_{first}").set_value("Unreachable").run()
    assert not at.exception
    assert "Next option" in text(at)
    assert "CAREPROOF REFERRAL SUMMARY" in " ".join(c.value for c in at.code)
