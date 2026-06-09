"""Minimal Baserow REST client: read rows, ensure result columns, write back.

Only the endpoints we need, with token auth and pagination. Uses
``user_field_names=true`` everywhere so we work with human column names.
"""

from __future__ import annotations

from typing import Any, Iterator

import requests

# Baserow field types we create for results.
_LONG_TEXT = "long_text"


class BaserowClient:
    def __init__(self, base_url: str, token: str, *, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {token}"})

    # ---- low-level ---------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}/api/{path.lstrip('/')}"

    def _request(self, method: str, path: str, **kwargs) -> Any:
        resp = self.session.request(method, self._url(path), timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return None
        return resp.json()

    # ---- fields ------------------------------------------------------------
    def list_fields(self, table_id: int) -> list[dict[str, Any]]:
        return self._request("GET", f"database/fields/table/{table_id}/")

    def field_names(self, table_id: int) -> set[str]:
        return {f["name"] for f in self.list_fields(table_id)}

    def ensure_fields(
        self, table_id: int, names: list[str], *, field_type: str = _LONG_TEXT
    ) -> dict[str, bool]:
        """Create any missing long-text columns. Returns {name: created?}."""
        existing = self.field_names(table_id)
        result: dict[str, bool] = {}
        for name in names:
            if name in existing:
                result[name] = False
                continue
            self._request(
                "POST",
                f"database/fields/table/{table_id}/",
                json={"name": name, "type": field_type},
            )
            result[name] = True
        return result

    # ---- rows --------------------------------------------------------------
    def iter_rows(
        self, table_id: int, *, page_size: int = 200, fields: list[str] | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield every row, transparently following pagination."""
        page = 1
        params: dict[str, Any] = {"user_field_names": "true", "size": page_size}
        if fields:
            # Baserow accepts repeated ?include= entries via comma list.
            params["include"] = ",".join(fields)
        while True:
            params["page"] = page
            data = self._request("GET", f"database/rows/table/{table_id}/", params=params)
            for row in data["results"]:
                yield row
            if not data.get("next"):
                break
            page += 1

    def fetch_rows(
        self, table_id: int, *, page_size: int = 200, fields: list[str] | None = None
    ) -> list[dict[str, Any]]:
        return list(self.iter_rows(table_id, page_size=page_size, fields=fields))

    def update_rows(self, table_id: int, items: list[dict[str, Any]]) -> None:
        """Batch-update rows. Each item must include the integer ``id``."""
        if not items:
            return
        # Baserow batch endpoint caps at 200 items per call.
        for start in range(0, len(items), 200):
            chunk = items[start : start + 200]
            self._request(
                "PATCH",
                f"database/rows/table/{table_id}/batch/",
                params={"user_field_names": "true"},
                json={"items": chunk},
            )
