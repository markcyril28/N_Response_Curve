from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


class RBridgeError(RuntimeError):
    """A contract, dispatch, or result validation violation at the Python/R boundary."""


@dataclass(frozen=True)
class RStageContract:
    stage_root: Path
    contract_path: Path
    input_path: Path
    output_path: Path
    contract_sha256: str
    input_sha256: str
    row_count: int
    stable_key: str
    stable_key_count: int


@dataclass(frozen=True)
class RStageResult:
    status: str
    return_code: int
    command: tuple[str, ...]
    stdout: str
    stderr: str
    results: tuple[Mapping[str, Any], ...]
    metadata: Mapping[str, Any]


_FORBIDDEN_SPECIFICATION_KEY_FRAGMENTS = (
    "operator_config",
    "scriptconfig",
    "raw_source",
    "source_file_path",
    "toml_path",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RBridgeError("R stage specification must be JSON serializable without NaN values") from exc


def _contains_forbidden_reference(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if any(fragment in normalized for fragment in _FORBIDDEN_SPECIFICATION_KEY_FRAGMENTS):
                return True
            if _contains_forbidden_reference(nested):
                return True
    elif isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_forbidden_reference(item) for item in value)
    return False


def _contract_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    return hashlib.sha256(_canonical_json_bytes(canonical)).hexdigest()


def _path_within(root: Path, candidate: Path) -> Path:
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root):
        raise RBridgeError("R stage artifact path must remain inside its isolated stage root")
    return resolved


def write_r_stage_contract(
    stage_root: str | Path,
    *,
    specification: Mapping[str, Any],
    rows: Iterable[Mapping[str, Any]],
    stable_key: str,
) -> RStageContract:
    """Write the sole Python-to-R handoff: normalized JSON contract plus Parquet input."""

    if not isinstance(specification, Mapping) or not specification:
        raise RBridgeError("R stage specification must be a nonempty mapping")
    if _contains_forbidden_reference(specification):
        raise RBridgeError("R stage specification contains a forbidden raw-source or operator-config reference")
    if not isinstance(stable_key, str) or not stable_key:
        raise RBridgeError("R stage stable_key must be a nonempty string")
    root = Path(stage_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    input_path = _path_within(root, root / "input.parquet")
    contract_path = _path_within(root, root / "contract.json")
    output_path = _path_within(root, root / "result.json")
    normalized_rows = [dict(row) for row in rows]
    if not normalized_rows:
        raise RBridgeError("R stage cannot be invoked with zero normalized rows")
    if any(stable_key not in row or row[stable_key] in {None, ""} for row in normalized_rows):
        raise RBridgeError(f"Each normalized R-stage row must contain nonempty stable key {stable_key!r}")
    try:
        frame = pd.DataFrame(normalized_rows)
        frame.to_parquet(input_path, index=False)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        raise RBridgeError("Unable to write normalized R-stage Parquet input") from exc
    input_sha256 = _sha256_path(input_path)
    stable_values = {str(row[stable_key]) for row in normalized_rows}
    normalized_specification = json.loads(_canonical_json_bytes(specification).decode("utf-8"))
    payload: dict[str, Any] = {
        "contract_version": 1,
        "specification": normalized_specification,
        "input": {
            "path": str(input_path),
            "sha256": input_sha256,
            "row_count": len(normalized_rows),
            "stable_key": stable_key,
            "stable_key_count": len(stable_values),
        },
        "schema": [{"name": str(column), "dtype": str(dtype)} for column, dtype in frame.dtypes.items()],
        "output": {"path": str(output_path), "relative_path": "result.json"},
    }
    contract_sha256 = _contract_digest(payload)
    payload["contract_sha256"] = contract_sha256
    contract_path.write_bytes(_canonical_json_bytes(payload))
    return RStageContract(
        stage_root=root,
        contract_path=contract_path,
        input_path=input_path,
        output_path=output_path,
        contract_sha256=contract_sha256,
        input_sha256=input_sha256,
        row_count=len(normalized_rows),
        stable_key=stable_key,
        stable_key_count=len(stable_values),
    )


def validate_r_stage_result(contract: RStageContract) -> RStageResult:
    """Reject partial, stale, schema-invalid, or mismatched R-stage output."""

    if not contract.output_path.is_file():
        raise RBridgeError("R stage did not produce its required result.json output")
    try:
        payload = json.loads(contract.output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RBridgeError("R stage result is not valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RBridgeError("R stage result must be a JSON object")
    if payload.get("contract_sha256") != contract.contract_sha256:
        raise RBridgeError("R stage result is stale or belongs to a different contract")
    if payload.get("contract_version") != 1:
        raise RBridgeError("R stage result has an unsupported or missing contract version")
    if payload.get("input_sha256") != contract.input_sha256:
        raise RBridgeError("R stage result input hash does not match its contract")
    if payload.get("input_row_count") != contract.row_count:
        raise RBridgeError("R stage result input row count does not match its contract")
    status = payload.get("status")
    if status not in {"completed", "skipped", "failed"}:
        raise RBridgeError("R stage result has an invalid terminal status")
    raw_results = payload.get("results")
    metadata = payload.get("metadata")
    if not isinstance(raw_results, list) or not all(isinstance(row, Mapping) for row in raw_results):
        raise RBridgeError("R stage result must contain a list of result objects")
    if not isinstance(metadata, Mapping):
        raise RBridgeError("R stage result must contain a metadata object")
    return RStageResult(
        status=str(status),
        return_code=0,
        command=(),
        stdout="",
        stderr="",
        results=tuple(dict(row) for row in raw_results),
        metadata=dict(metadata),
    )


def _normalize_command(command: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(command, str):
        normalized = (command,)
    else:
        normalized = tuple(str(item) for item in command)
    if not normalized or any(not item for item in normalized):
        raise RBridgeError("rscript_command must be a nonempty executable command")
    return normalized


def invoke_r_stage(
    contract: RStageContract,
    *,
    rscript_command: str | Sequence[str],
    r_entrypoint: str | Path,
    timeout_seconds: int,
    extra_environment: Mapping[str, str] | None = None,
) -> RStageResult:
    """Dispatch exactly one vanilla R process; no Python model fallback is possible here."""

    if not isinstance(timeout_seconds, int) or timeout_seconds < 1:
        raise RBridgeError("R stage timeout_seconds must be a positive integer")
    entrypoint = Path(r_entrypoint).resolve()
    if not entrypoint.is_file():
        raise RBridgeError(f"R stage entrypoint does not exist: {entrypoint}")
    command = (
        *_normalize_command(rscript_command),
        "--vanilla",
        str(entrypoint),
        "--contract",
        str(contract.contract_path),
    )
    environment = os.environ.copy()
    if extra_environment:
        environment.update({str(key): str(value) for key, value in extra_environment.items()})
    try:
        completed = subprocess.run(
            command,
            cwd=contract.stage_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return RStageResult(
            status="failed",
            return_code=-1,
            command=command,
            stdout="" if not isinstance(exc, subprocess.TimeoutExpired) else str(exc.stdout or ""),
            stderr=str(exc),
            results=(),
            metadata={"dispatch_error": type(exc).__name__},
        )
    if completed.returncode != 0:
        return RStageResult(
            status="failed",
            return_code=completed.returncode,
            command=command,
            stdout=completed.stdout,
            stderr=completed.stderr,
            results=(),
            metadata={"dispatch_error": "R_PROCESS_FAILED"},
        )
    validated = validate_r_stage_result(contract)
    return RStageResult(
        status=validated.status,
        return_code=completed.returncode,
        command=command,
        stdout=completed.stdout,
        stderr=completed.stderr,
        results=validated.results,
        metadata=validated.metadata,
    )


__all__ = [
    "RBridgeError",
    "RStageContract",
    "RStageResult",
    "invoke_r_stage",
    "validate_r_stage_result",
    "write_r_stage_contract",
]
