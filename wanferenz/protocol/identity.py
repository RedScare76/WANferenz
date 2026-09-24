import base64
import os
import stat
from cryptography.hazmat.primitives.asymmetric import ed25519


class IdentityFailure(Exception):
    pass


def create_identity() -> ed25519.Ed25519PrivateKey:
    return ed25519.Ed25519PrivateKey.generate()


def persist_identity(priv: ed25519.Ed25519PrivateKey, path: str) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 384)
    with os.fdopen(descriptor, "w") as destination:
        destination.write(base64.b64encode(priv.private_bytes_raw()).decode())


def restore_identity(path: str) -> ed25519.Ed25519PrivateKey:
    permissions = os.stat(path).st_mode
    if stat.S_ISREG(permissions) and permissions & 63:
        raise IdentityFailure(
            f"publisher key {path} is group/world accessible (mode {stat.S_IMODE(permissions):o}) — chmod 600 it; anyone reading it can sign manifests nodes trust"
        )
    with open(path) as source:
        serialized = source.read().strip()
    return ed25519.Ed25519PrivateKey.from_private_bytes(base64.b64decode(serialized))


def public_identity(priv: ed25519.Ed25519PrivateKey) -> str:
    serialized = priv.public_key().public_bytes_raw()
    return base64.b64encode(serialized).decode()
