"""MCP tool server: exposes 4 typed, read-only tools over the Model Context Protocol.

Why MCP instead of letting the agent query the database?
- The agent can only call these 4 tools - no free-form SQL, no other tables.
- Every input has a typed schema (regex pattern / allowed values). Bad input is
  rejected by the server BEFORE any code runs.
- Any MCP client (Claude Desktop, VS Code, another agent) can reuse the same tools.

Run standalone (it talks over stdin/stdout):  python -m app.mcp_server.server
"""
from __future__ import annotations

from typing import Annotated

import requests
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from app.integrations import bluebutton, claims
from app.mcp_server import db
from app.schemas import Service

mcp = FastMCP("benefitiq-member-tools", log_level="WARNING")

MemberId = Annotated[str, Field(pattern=r"^M\d{4}$", description="Synthetic member ID, e.g. M1001")]
PlanId = Annotated[str, Field(pattern=r"^PLN-\d{3}$", description="Plan ID, e.g. PLN-002")]
ServiceText = Annotated[str | None, Field(default=None, max_length=40, pattern=r"^[A-Za-z][A-Za-z \-]*$",
                                          description="Optional service to look for, e.g. 'mammogram', 'flu shot'")]
Months = Annotated[int, Field(default=24, ge=1, le=60, description="Look-back window in months")]


def _http_session() -> requests.Session:
    """Factory so tests can swap in a fake session (no network)."""
    return requests.Session()


@mcp.tool()
def get_member_plan(member_id: MemberId) -> dict:
    """Return the member's plan ID, plan name, plan type, coverage start and status (active/inactive)."""
    return db.get_member_plan(member_id)


@mcp.tool()
def get_plan_benefits(plan_id: PlanId) -> dict:
    """Return cost-sharing for a plan: deductible, copays (PCP, specialist, ER, urgent care),
    out-of-pocket maximum, coinsurance percent and whether referrals are required."""
    return db.get_plan_benefits(plan_id)


@mcp.tool()
def check_prior_auth(plan_id: PlanId, service: Service) -> dict:
    """Return whether a service needs prior authorization on a plan, with notes."""
    return db.check_prior_auth(plan_id, service)


@mcp.tool()
def get_claims_summary(member_id: MemberId, service: ServiceText = None, months: Months = 24) -> dict:
    """Return the member's recent Medicare claim lines from CMS Blue Button (synthetic sandbox data):
    service date, claim type, procedure/drug code and description only. Optionally filter by a service
    such as 'mammogram' or 'flu shot'. Fails with a clear message if the member hasn't connected Blue Button."""
    try:
        with _http_session() as session:
            return claims.get_claims_summary(session, member_id, service=service, months=months)
    except bluebutton.BlueButtonError as exc:
        raise ValueError(str(exc)) from exc


if __name__ == "__main__":
    mcp.run()  # stdio transport by default
