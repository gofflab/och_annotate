"""Shared test fixtures."""

import pytest

# Token env vars that a developer's local .env (auto-loaded by python-dotenv in
# config.py) could otherwise leak into the suite, making token tests non-hermetic.
_TOKEN_ENV = [
    "BASEROW_TOKEN",
    "BIOHUB_API_TOKEN",
    "BIOHUB_API_TOKENS",
    "ESM_API_KEY",
    "ESM_API_KEYS",
]


@pytest.fixture(autouse=True)
def _isolate_token_env(monkeypatch):
    """Clear all token env vars before each test so a local .env can't leak in.

    Each test then sets exactly the credentials it needs via ``monkeypatch.setenv``.
    """
    for name in _TOKEN_ENV:
        monkeypatch.delenv(name, raising=False)
