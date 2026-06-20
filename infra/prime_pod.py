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


def best_offer(gpu_type: str, gpu_count: int) -> dict[str, Any] | None:
    offers = list_gpu_offers(gpu_type, gpu_count)
    return offers[0] if offers else None


def create_pod_for_model(
    model_id: str,
    name: str = "burnttt-glm52-ttt",
    image: str = "ubuntu_22_cuda_12",
) -> dict[str, Any]:
    """Pick GPU shape from :mod:`glm.model_specs` and create a pod."""
    from glm.model_specs import pod_spec_for_model

    spec = pod_spec_for_model(model_id)
    offer = best_offer(spec.gpu_type, spec.gpu_count)
    if offer is None and spec.gpu_count > 1:
        logger.warning(
            "No %dx %s offer; trying 2x then 1x (model %s may OOM).",
            spec.gpu_count,
            spec.gpu_type,
            model_id,
        )
        offer = best_offer(spec.gpu_type, 2) or best_offer(spec.gpu_type, 1)
    if offer is None:
        raise RuntimeError(f"No Prime GPU offers for {spec.gpu_type} x{spec.gpu_count}")
    logger.info(
        "Creating pod for %s: %s x%d (~$%s/hr) — %s",
        model_id,
        offer.get("gpuType"),
        offer.get("gpuCount"),
        offer.get("prices", {}).get("onDemand"),
        spec.notes,
    )
    return create_pod(offer, name=name, image=image)


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


def list_ssh_keys() -> list[dict[str, Any]]:
    data = _request("GET", "/ssh_keys/")
    if isinstance(data, list):
        return data
    return data.get("sshKeys", data.get("data", []))


def upload_ssh_key(name: str, public_key: str) -> dict[str, Any]:
    """Register a public key with Prime (required before pod SSH works)."""
    return _request("POST", "/ssh_keys/", {"name": name, "publicKey": public_key.strip()})


def set_primary_ssh_key(key_id: str) -> None:
    """Mark an uploaded key as primary (best-effort; API shape varies)."""
    for method, path, body in (
        ("PUT", f"/ssh_keys/{key_id}", {"isPrimary": True}),
        ("PATCH", f"/ssh_keys/{key_id}", {"isPrimary": True}),
        ("POST", "/ssh_keys/set-primary-key", {"sshKeyId": key_id}),
    ):
        try:
            _request(method, path, body)
            return
        except RuntimeError:
            continue
    logger.warning("Could not set primary SSH key via API; key may still work if it is the only one.")


def ensure_ssh_key(name: str = "cursor-burnttt", public_key_path: Path | None = None) -> str | None:
    """Upload local ~/.ssh/id_ed25519.pub if Prime has no matching key."""
    pub_path = public_key_path or (Path.home() / ".ssh" / "id_ed25519.pub")
    if not pub_path.is_file():
        logger.warning("No SSH public key at %s; pod SSH may fail.", pub_path)
        return None
    pub = pub_path.read_text().strip()
    for key in list_ssh_keys():
        if key.get("publicKey", "").strip() == pub:
            return key.get("id")
    uploaded = upload_ssh_key(name, pub)
    key_id = uploaded.get("id")
    if key_id:
        set_primary_ssh_key(key_id)
    return key_id


def wait_for_pod(
    pod_id: str,
    *,
    timeout_sec: int = 600,
    poll_sec: int = 10,
) -> dict[str, Any]:
    """Poll until pod is ACTIVE and has an IP."""
    import time

    deadline = time.time() + timeout_sec
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = get_pod(pod_id)
        if last.get("status") == "ACTIVE" and last.get("ip"):
            return last
        time.sleep(poll_sec)
    raise TimeoutError(f"Pod {pod_id} not ACTIVE with IP after {timeout_sec}s (last={last.get('status')})")


def write_pod_env(
    path: Path,
    *,
    pod_id: str,
    ssh_host: str,
    ssh_user: str = "ubuntu",
    ssh_key: str | None = None,
) -> None:
    """Persist pod connection info for reuse (gitignored .env.pod)."""
    lines = [
        f"PRIME_POD_ID={pod_id}",
        f"SSH_HOST={ssh_host}",
        f"SSH_USER={ssh_user}",
    ]
    if ssh_key:
        lines.append(f"SSH_KEY={ssh_key}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote pod env: %s", path)


def read_pod_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip().strip("'\"")
    return out
