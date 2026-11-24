"""MCP tool tests: direct calls, schema validation and a real stdio subprocess round trip."""
from __future__ import annotations

import pytest

from app.agents.mcp_client import ToolCallError, run_member_lookup
from app.mcp_server import db, server


def test_member_plan_is_minimum_necessary(member_db):
    data = server.get_member_plan("M1001")
    assert data["plan_id"] == "PLN-001" and data["status"] == "active"
    assert not {"first_name", "last_name", "dob", "phone"} & data.keys()  # no direct identifiers returned


def test_plan_benefits_values(member_db):
    data = server.get_plan_benefits("PLN-002")
    assert data["specialist_copay"] == 35 and data["referral_required"] is True


def test_prior_auth_rule(member_db):
    assert server.check_prior_auth("PLN-002", "mri")["prior_auth_required"] is True
    assert server.check_prior_auth("PLN-001", "ct_scan")["prior_auth_required"] is False


def test_unknown_member_raises(member_db):
    with pytest.raises(db.NotFoundError):
        server.get_member_plan("M9999")


async def test_server_rejects_bad_input_schema(member_db):
    """Typed schema: the MCP server validates the pattern before running the tool."""
    from mcp.server.fastmcp.exceptions import ToolError

    with pytest.raises(ToolError):
        await server.mcp.call_tool("get_member_plan", {"member_id": "DROP TABLE"})


async def test_tool_list_has_typed_schemas(member_db):
    tools = {t.name: t for t in await server.mcp.list_tools()}
    assert set(tools) == {"get_member_plan", "get_plan_benefits", "check_prior_auth", "get_claims_summary"}
    assert tools["get_member_plan"].inputSchema["properties"]["member_id"]["pattern"] == r"^M\d{4}$"
    assert "mri" in tools["check_prior_auth"].inputSchema["properties"]["service"]["enum"]


def test_end_to_end_stdio_lookup(member_db):
    """Starts the real MCP server process and calls it over stdio."""
    result = run_member_lookup("M1002", "mri")
    assert result["member"]["plan_id"] == "PLN-002"
    assert result["prior_auth"]["prior_auth_required"] is True
    assert [c["tool"] for c in result["tool_calls"]] == ["get_member_plan", "get_plan_benefits", "check_prior_auth"]


def test_stdio_lookup_unknown_member_gives_clean_error(member_db):
    with pytest.raises(ToolCallError, match="not found"):
        run_member_lookup("M9999")


def test_database_is_opened_read_only(member_db):
    import sqlite3

    with db.connect() as con, pytest.raises(sqlite3.OperationalError, match="readonly"):
        con.execute("DELETE FROM members")


def test_sql_injection_text_is_just_a_value(member_db):
    with pytest.raises(db.NotFoundError):
        db.get_member_plan("M1001' OR '1'='1")


def test_unknown_service_rejected_by_schema(member_db):
    import asyncio

    from mcp.server.fastmcp.exceptions import ToolError

    with pytest.raises(ToolError):
        asyncio.run(server.mcp.call_tool("check_prior_auth", {"plan_id": "PLN-001", "service": "massage"}))
