"""Smoke test: proves the package imports and the test runner works in CI."""
import app


def test_package_imports():
    assert app.__version__.startswith("0.")


def test_subpackages_import():
    import app.agents  # noqa: F401
    import app.api  # noqa: F401
    import app.core  # noqa: F401
    import app.guardrails  # noqa: F401
    import app.mcp_server  # noqa: F401
    import app.rag  # noqa: F401
