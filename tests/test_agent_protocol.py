from __future__ import annotations

import pytest

from atlasai.agents.errors import ProtocolError
from atlasai.agents.protocol import parse_action


def test_parse_respond_and_think() -> None:
    assert parse_action('{"action":"respond","text":"Привет"}').action == "respond"
    assert parse_action(' {"action":"think","text":"План"} ').text == "План"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        "[]",
        '{"action":"respond"}',
        '{"action":"respond","text":"ok","extra":1}',
        '{"action":"shell","text":"ls"}',
        '{"action":"respond","text":""}',
        '{"action":"respond","text":123}',
        '```json\n{"action":"respond","text":"ok"}\n```',
        '{"action":"respond","text":"ok"}\n{"action":"think","text":"more"}',
    ],
)
def test_invalid_protocol_is_rejected(raw: str) -> None:
    with pytest.raises(ProtocolError):
        parse_action(raw)


def test_text_limit_is_enforced() -> None:
    with pytest.raises(ProtocolError, match="exceeds"):
        parse_action('{"action":"respond","text":"abcdef"}', max_text_chars=3)


def test_non_string_output_is_rejected() -> None:
    with pytest.raises(ProtocolError, match="string"):
        parse_action(None)  # type: ignore[arg-type]
