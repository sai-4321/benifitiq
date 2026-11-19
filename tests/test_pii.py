"""PII/PHI masking tests. All values are fake (555 phone numbers, example.com, synthetic IDs)."""
from __future__ import annotations

import pytest

from app.guardrails import extract_member_id, mask_pii, unmask


@pytest.mark.parametrize(
    "label, text, secret",
    [
        ("SSN", "SSN 123-45-6789 on file", "123-45-6789"),
        ("MBI", "Medicare number 1EG4-TE5-MK73", "1EG4-TE5-MK73"),
        ("MBI", "MBI 1EG4TE5MK73 without dashes", "1EG4TE5MK73"),
        ("EMAIL", "email jane.doe@example.com today", "jane.doe@example.com"),
        ("PHONE", "call 469-555-0134", "469-555-0134"),
        ("PHONE", "call (214) 555 0199", "(214) 555 0199"),
        ("PHONE", "call +1 972.555.0100", "+1 972.555.0100"),
        ("DOB", "born 03/14/1952", "03/14/1952"),
        ("DOB", "born 1952-03-14", "1952-03-14"),
        ("MEMBER_ID", "member id M1003 please", "M1003"),
    ],
)
def test_each_identifier_is_masked(label, text, secret):
    r = mask_pii(text)
    assert secret not in r.text
    assert f"[{label}_1]" in r.text
    assert r.mapping[f"[{label}_1]"] == secret


def test_mixed_message_counts():
    text = "SSN 123-45-6789, call 469-555-0134, email jane.doe@example.com, DOB 03/14/1952."
    assert mask_pii(text).counts == {"SSN": 1, "EMAIL": 1, "PHONE": 1, "DOB": 1}


@pytest.mark.parametrize(
    "text, name",
    [
        ("Patient John Smith needs a walker", "John Smith"),
        ("the member named Mary Lee called", "Mary Lee"),
        ("Ask Dr. Patel about it", "Patel"),
        ("PATIENT Rosa Diaz was discharged", "Rosa Diaz"),     # cue word in any case
    ],
)
def test_cued_names_masked_but_cue_word_kept(text, name):
    r = mask_pii(text)
    assert name not in r.text
    assert "[NAME_1]" in r.text
    assert text.split()[0] in r.text or text.split()[1] in r.text   # sentence still readable


def test_same_value_gets_same_placeholder():
    r = mask_pii("M1001 asked again about M1001 and M1002")
    assert r.text == "[MEMBER_ID_1] asked again about [MEMBER_ID_1] and [MEMBER_ID_2]"
    assert r.counts["MEMBER_ID"] == 2


def test_ssn_is_not_mistaken_for_phone():
    r = mask_pii("123-45-6789")
    assert "SSN" in r.counts and "PHONE" not in r.counts


def test_unmask_roundtrip():
    original = "Patient John Smith (M1002) can be reached at 214-555-0199"
    r = mask_pii(original)
    assert unmask(r.text, r.mapping) == original


@pytest.mark.parametrize(
    "question",
    [
        "How often does Medicare cover a screening mammogram?",
        "Does Part B cover durable medical equipment in 2026?",
        "What is the Part D late enrollment penalty after 63 days?",
        "Is a 20% coinsurance normal for Part B services?",
    ],
)
def test_plain_policy_questions_untouched(question):
    r = mask_pii(question)
    assert r.text == question and r.mapping == {}


def test_extract_member_id():
    assert extract_member_id("Does M1042 need prior auth?") == "M1042"
    assert extract_member_id("No id here") is None
    assert extract_member_id("model M10425 is not an id") is None
