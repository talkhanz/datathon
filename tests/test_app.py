"""Headless Streamlit tests of the app against local Parquet (skipped if derived data or streamlit is absent)."""
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "derived")
st_testing = pytest.importorskip("streamlit.testing.v1")
pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "careproof_facility_service_gold.parquet")),
                                reason="local derived data not built")


def run_app(**widgets):
    os.environ["CAREPROOF_LOCAL_DATA"] = DATA
    at = st_testing.AppTest.from_file(os.path.join(ROOT, "app", "app.py"), default_timeout=180)
    at.run()
    for key, val in widgets.items():  # set one at a time: some inputs appear only after an earlier choice
        w = next((x for x in list(at.selectbox) + list(at.radio) + list(at.text_input) + list(at.slider) if x.key == key))
        w.set_value(val)
        at.run()
    at.button(key="go").click().run()
    return at


def all_text(at):
    return " ".join([m.value for m in at.markdown] + [e.value for e in at.error] + [w.value for w in at.warning]
                    + [c.value for c in at.caption] + [i.value for i in at.info])


def test_safety_banner_and_strong_results():
    at = run_app(svc="dialysis", ltype="pincode", lval="800001", radius=25)
    assert not at.exception
    t = all_text(at)
    assert "does not provide medical advice" in t and "Patna Dialysis Centre" in t
    assert "Current availability not verified" in t and "approx. straight-line" in t


def test_merged_record_location_abstains():
    at = run_app(svc="dialysis", ltype="coordinates", lat="11.938235", lon="79.485443", radius=5)
    assert not at.exception
    t = all_text(at)
    assert "CareProof abstains" in t and "none had sufficiently consistent identity" in t


def test_invalid_location_error():
    at = run_app(svc="mri", ltype="pincode", lval="12345")
    assert not at.exception and "not a valid Indian PIN code" in all_text(at)


def test_verify_unreachable_promotes_alternative():
    at = run_app(svc="dialysis", ltype="pincode", lval="800001", radius=25)
    first = at.session_state["results"][0]["facility_id"]
    at.selectbox(key=f"status_{first}").set_value("Unreachable").run()
    assert not at.exception
    assert "Next eligible alternative" in all_text(at)
    refer_code = " ".join(c.value for c in at.code)
    assert "CAREPROOF REFERRAL SUMMARY" in refer_code and "Unreachable" in refer_code
