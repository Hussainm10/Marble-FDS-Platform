"""Checkmarble API client — ingest, decide, list management."""

import logging
from datetime import datetime, timezone
from typing import Any, Optional

import httpx

from .config import settings
from .models import CheckmarbleDecision, RuleTriggered

logger = logging.getLogger(__name__)


class CheckmarbleClient:
    def __init__(self):
        self.base_url = settings.CHECKMARBLE_API_URL.rstrip("/")
        self.api_key = settings.CHECKMARBLE_API_KEY
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=30.0,
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def ingest(self, table_name: str, object_id: str, data: dict[str, Any]) -> dict:
        """Ingest a record. Marble expects flat fields alongside object_id/updated_at."""
        client = await self._get_client()
        payload = {
            "object_id": object_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            **data,
        }
        response = await client.post(f"/ingestion/{table_name}", json=payload)
        response.raise_for_status()
        return {"status": "ok"} if not response.text else response.json()

    async def decide(
        self,
        scenario_id: str,
        trigger_object_type: str,
        trigger_object: dict[str, Any],
    ) -> CheckmarbleDecision:
        """Trigger a decision. trigger_object must include all fields for rule evaluation."""
        client = await self._get_client()
        if "object_id" not in trigger_object:
            raise ValueError("trigger_object must include 'object_id'")
        if "updated_at" not in trigger_object:
            trigger_object["updated_at"] = datetime.now(timezone.utc).isoformat()

        payload = {
            "scenario_id": scenario_id,
            "object_type": trigger_object_type,
            "trigger_object": trigger_object,
        }
        response = await client.post("/decisions", json=payload)
        response.raise_for_status()
        result = response.json()

        decision_data = result.get("decision", result)
        rules = [
            RuleTriggered(
                rule_id=r.get("rule_id", r.get("id", "")),
                score=r.get("score_modifier", r.get("score", 0)),
                description=r.get("description", ""),
            )
            for r in decision_data.get("rules", [])
        ]

        trigger_object_id = trigger_object.get("object_id", "")
        return CheckmarbleDecision(
            decision_id=decision_data.get("id", ""),
            scenario_id=scenario_id,
            outcome=decision_data.get("outcome", ""),
            score=decision_data.get("score", 0),
            rules_triggered=rules,
            trigger_object_type=trigger_object_type,
            trigger_object_id=trigger_object_id,
        )

    async def get_decision(self, decision_id: str) -> dict:
        client = await self._get_client()
        response = await client.get(f"/decisions/{decision_id}")
        response.raise_for_status()
        return response.json()

    async def create_list(self, name: str, description: str) -> dict:
        client = await self._get_client()
        response = await client.post(
            "/custom-lists",
            json={"name": name, "description": description},
        )
        response.raise_for_status()
        return response.json()

    async def add_list_entry(self, list_id: str, value: dict) -> dict:
        client = await self._get_client()
        response = await client.post(
            f"/custom-lists/{list_id}/values",
            json={"value": value},
        )
        response.raise_for_status()
        return response.json()

    async def health(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get("/liveness")
            return response.status_code == 200
        except Exception:
            return False
