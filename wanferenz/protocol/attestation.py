import base64
import hashlib
import json
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import ed25519
from wanferenz.protocol.identity import (
    create_identity,
    restore_identity,
    public_identity,
    persist_identity,
)

SCHEMA = "wanferenz-receipt/1"


class AttestationFailure(Exception):
    pass


def _digest_bytes(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()


def _signed_bytes(attestation: dict) -> bytes:
    m = {
        entry_key: entry_value
        for entry_key, entry_value in attestation.items()
        if entry_key != "sig"
    }
    return json.dumps(
        m, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def receipt_body(attestation: dict) -> dict:
    return {
        entry_key: entry_value
        for entry_key, entry_value in attestation.items()
        if entry_key != "stage"
    }


class ActivationAttestor:
    def __init__(
        self,
        priv: ed25519.Ed25519PrivateKey,
        swarm_id: str,
        job_id: str,
        layer_start: int,
        layer_end: int,
        nonce: str | None = None,
    ):
        self.priv = priv
        self.meta = {
            "swarm_id": swarm_id,
            "job_id": job_id,
            "layer_start": layer_start,
            "layer_end": layer_end,
        }
        if nonce is not None:
            self.meta["nonce"] = nonce
        self._in = hashlib.sha256()
        self._out = hashlib.sha256()
        self.n = 0

    def observe(self, in_bytes: bytes, out_bytes: bytes) -> None:
        self._in.update(_digest_bytes(in_bytes))
        self._out.update(_digest_bytes(out_bytes))
        self.n += 1

    def finalize(self) -> dict:
        body = dict(
            self.meta,
            schema=SCHEMA,
            n_chunks=self.n,
            in_root=self._in.hexdigest(),
            out_root=self._out.hexdigest(),
            pubkey=base64.b64encode(self.priv.public_key().public_bytes_raw()).decode(),
        )
        body["sig"] = base64.b64encode(self.priv.sign(_signed_bytes(body))).decode()
        return body


def validate_attestation(attestation: dict, expected_pubkey: str | None = None) -> None:
    if attestation.get("schema") != SCHEMA:
        raise AttestationFailure(
            f"unknown receipt schema {attestation.get('schema')!r}"
        )
    public_identity = attestation.get("pubkey")
    sig_b64 = attestation.get("sig")
    if not public_identity or not sig_b64:
        raise AttestationFailure("receipt is unsigned")
    if expected_pubkey is not None and public_identity != expected_pubkey:
        raise AttestationFailure("receipt signer is not the node assigned this block")
    try:
        pub = ed25519.Ed25519PublicKey.from_public_bytes(
            base64.b64decode(public_identity)
        )
        pub.verify(base64.b64decode(sig_b64), _signed_bytes(attestation))
    except (InvalidSignature, ValueError, Exception) as failure:
        raise AttestationFailure(
            f"signature verification failed: {type(failure).__name__}"
        ) from failure


def ensure_identity(path: str) -> ed25519.Ed25519PrivateKey:
    import os

    if os.path.exists(path):
        return restore_identity(path)
    key = create_identity()
    persist_identity(key, path)
    try:
        os.chmod(path, 384)
    except OSError:
        pass
    return key


def validate_coverage(
    receipts: list[dict],
    layer_count: int,
    expected_by_signer: dict | None = None,
    expected_nonce: str | None = None,
    check_chain: bool = False,
) -> None:
    entries = []
    seen_pubkeys = set()
    for outcome in receipts:
        validate_attestation(outcome, None)
        if expected_nonce is not None and outcome.get("nonce") != expected_nonce:
            raise AttestationFailure(
                f"receipt nonce {outcome.get('nonce')!r} != job nonce (stale or replayed attestation)"
            )
        lo, hi = (outcome["layer_start"], outcome["layer_end"])
        if not 0 <= lo < hi <= layer_count:
            raise AttestationFailure(
                f"receipt block [{lo}:{hi}] outside [0:{layer_count}]"
            )
        if not isinstance(outcome.get("n_chunks"), int) or outcome["n_chunks"] <= 0:
            raise AttestationFailure(
                f"receipt for [{lo}:{hi}] attests {outcome.get('n_chunks')!r} chunks (zero-work attestation)"
            )
        pub = outcome["pubkey"]
        if pub in seen_pubkeys:
            raise AttestationFailure(f"duplicate signer {pub[:12]}..")
        seen_pubkeys.add(pub)
        if expected_by_signer is not None:
            expected = expected_by_signer.get(pub)
            if expected is None:
                raise AttestationFailure(
                    f"signer {pub[:12]}.. is not in the assignment map"
                )
            if tuple(expected) != (lo, hi):
                raise AttestationFailure(
                    f"signer {pub[:12]}.. attested [{lo}:{hi}], assigned {tuple(expected)}"
                )
        entries.append((lo, hi, outcome))
    if expected_by_signer is not None:
        missing = set(expected_by_signer) - seen_pubkeys
        if missing:
            raise AttestationFailure(
                f"assigned signer(s) produced no receipt: {sorted((child_process[:12] for child_process in missing))}"
            )
    entries.sort(key=lambda e: e[0])
    cursor = 0
    for lo, hi, _ in entries:
        if lo != cursor:
            raise AttestationFailure(
                f"layer coverage broken at {cursor}: next block starts {lo} (gap or overlap)"
            )
        cursor = hi
    if cursor != layer_count:
        raise AttestationFailure(
            f"layer coverage ends at {cursor}, expected {layer_count}"
        )
    if check_chain:
        for (lo_a, hi_a, ra), (lo_b, hi_b, rb) in zip(entries, entries[1:]):
            if ra["out_root"] != rb["in_root"]:
                raise AttestationFailure(
                    f"chain break: block [{lo_a}:{hi_a}] out_root {ra['out_root'][:12]} != block [{lo_b}:{hi_b}] in_root {rb['in_root'][:12]} — an attested output is not what the next stage attests it received (fabricated roots or a spliced attestation)"
                )
