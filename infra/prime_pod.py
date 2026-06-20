"""Prime Intellect GPU pod provisioning via REST API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from paths import REPO_ROOT, get_logger

logger = get_logger("burnttt.infra.prime_pod")

try:
    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

API_BASE = "https://api.primeintellect.ai/api/v1"


def _api_key() -> str:
    key = os.environ.get("PRIME_API_KEY", "").strip()
    if not key:
        raise RuntimeError("PRIME_API_KEY is not set")
    return key


def _request(method: str, path: str, body: dict | None = None) -> Any:
    url = f"{API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()
        raise RuntimeError(f"Prime API {method} {path} failed ({exc.code}): {detail}") from exc


def list_gpu_offers(
    gpu_type: str = "H100_80GB",
    gpu_count: int = 1,
    regions: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return available GPU offers from the availability endpoint."""
    regions = regions or ["united_states", "canada"]
    qs = f"gpu_type={gpu_type}&gpu_count={gpu_count}"
    for r in regions:
        qs += f"&regions={r}"
    data = _request("GET", f"/availability/gpus?{qs}")
    if isinstance(data, list):
        return data
    return data.get("items", data.get("offers", data.get("data", [])))


def _offer_to_pod_body(offer: dict[str, Any], name: str, image: str) -> dict[str, Any]:
    provider_type = offer.get("provider", offer.get("providerType", "hyperstack"))
    pod: dict[str, Any] = {
        "name": name,
        "cloudId": offer["cloudId"],
        "gpuType": offer.get("gpuType", "H100_80GB"),
        "gpuCount": offer.get("gpuCount", 1),
        "image": image,
    }
    if offer.get("socket"):
        pod["socket"] = offer["socket"]
    if offer.get("dataCenter"):
        pod["dataCenterId"] = offer["dataCenter"]
    elif offer.get("dataCenterId"):
        pod["dataCenterId"] = offer["dataCenterId"]
    if offer.get("country"):
        pod["country"] = offer["country"]
    if offer.get("security"):
        pod["security"] = offer["security"]
    return {"pod": pod, "provider": {"type": provider_type}}


def create_pod(
    offer: dict[str, Any],
    name: str = "burnttt-ttt",
    image: str = "ubuntu_22_cuda_12",
) -> dict[str, Any]:
    """Create a pod from an availability offer dict."""
    return _request("POST", "/pods/", _offer_to_pod_body(offer, name, image))


def list_pods() -> list[dict[str, Any]]:
    data = _request("GET", "/pods/")
    if isinstance(data, list):
        return data
    return data.get("pods", data.get("data", []))


def get_pod(pod_id: str) -> dict[str, Any]:
    pods = list_pods()
    for p in pods:
        if p.get("id") == pod_id:
            return p
    return _request("GET", f"/pods/{pod_id}")


def get_pod_status(pod_ids: list[str]) -> dict[str, Any]:
    try:
        ids = ",".join(pod_ids)
        return _request("GET", f"/pods/status?ids={ids}")
    except RuntimeError:
        return {"pods": [get_pod(pid) for pid in pod_ids]}


def delete_pod(pod_id: str) -> dict[str, Any]:
    return _request("DELETE", f"/pods/{pod_id}")
