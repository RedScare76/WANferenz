from collections import deque
from itertools import islice
import json
import os
import socket
import struct

import numpy as np
import torch

FRAME_SIZE_LIMIT = int(os.environ.get("WANFERENZ_MAX_FRAME") or 256 * 1024 * 1024)
_WIRE_DTYPES = {
    str(dtype): dtype
    for dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.int64,
        torch.int32,
        torch.uint8,
        torch.bool,
    )
}
_DTYPE_WIDTH = {
    dtype: torch.empty(0, dtype=dtype).element_size() for dtype in _WIRE_DTYPES.values()
}
_IOV_CAP = 512


def _receive_exactly(sock: socket.socket, n: int) -> bytearray:
    storage = bytearray(n)
    remaining = memoryview(storage)
    while remaining:
        received = sock.recv_into(remaining, len(remaining))
        if not received:
            raise ConnectionError("peer closed the connection")
        remaining = remaining[received:]
    return storage


def _encode_segments(obj) -> list:
    tensors = []

    def describe(value):
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            tensors.append(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
            return {
                "__t__": len(tensors) - 1,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
            }
        if isinstance(value, dict):
            return {key: describe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return list(map(describe, value))
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(f"transport cannot encode {type(value).__name__}")

    metadata = json.dumps(describe(obj)).encode()
    segments = [struct.pack("!I", len(metadata)) + metadata]
    for tensor_bytes in tensors:
        segments.extend((struct.pack("!Q", len(tensor_bytes)), tensor_bytes))
    return segments


def _encode_frame(obj) -> bytes:
    return b"".join(_encode_segments(obj))


def _restore_tensor(description, tensors):
    dtype = _WIRE_DTYPES.get(description["dtype"])
    if dtype is None:
        raise ValueError(f"unknown tensor dtype {description['dtype']!r}")
    shape = description["shape"]
    storage = tensors[description["__t__"]]
    elements = 1
    for dimension in shape:
        if not isinstance(dimension, int) or dimension < 0:
            raise ValueError(f"bad tensor dim {dimension!r}")
        elements *= dimension
    required = elements * _DTYPE_WIDTH[dtype]
    if required != len(storage):
        raise ValueError(
            f"tensor {shape} needs {required} bytes, blob has {len(storage)}"
        )
    if elements == 0:
        return torch.empty(shape, dtype=dtype)
    copied = np.frombuffer(storage, dtype=np.uint8).copy()
    return torch.from_numpy(copied).view(dtype).reshape(shape)


def _decode_frame(buf):
    storage = memoryview(buf)
    metadata_length = struct.unpack_from("!I", storage, 0)[0]
    metadata = json.loads(bytes(storage[4 : 4 + metadata_length]))
    cursor = 4 + metadata_length
    tensors = []
    while cursor < len(storage):
        tensor_length = struct.unpack_from("!Q", storage, cursor)[0]
        cursor += 8
        tensors.append(storage[cursor : cursor + tensor_length])
        cursor += tensor_length

    def restore(value):
        if isinstance(value, dict):
            if "__t__" in value:
                return _restore_tensor(value, tensors)
            return {key: restore(item) for key, item in value.items()}
        return list(map(restore, value)) if isinstance(value, list) else value

    return restore(metadata)


def _write_segments(sock: socket.socket, parts: list) -> None:
    if not hasattr(sock, "sendmsg"):
        sock.sendall(b"".join(parts))
        return
    remaining = deque(memoryview(segment) for segment in parts if len(segment))
    while remaining:
        consumed = sock.sendmsg(list(islice(remaining, _IOV_CAP)))
        while remaining and consumed >= len(remaining[0]):
            consumed -= len(remaining.popleft())
        if consumed:
            remaining[0] = remaining[0][consumed:]


def send_frame(sock: socket.socket, obj) -> int:
    segments = _encode_segments(obj)
    payload_length = sum(map(len, segments))
    _write_segments(sock, [struct.pack("!Q", payload_length), *segments])
    return payload_length + 8


def receive_frame(sock: socket.socket):
    payload_length = struct.unpack("!Q", _receive_exactly(sock, 8))[0]
    if payload_length > FRAME_SIZE_LIMIT:
        raise ConnectionError(
            f"frame length {payload_length} exceeds FRAME_SIZE_LIMIT ({FRAME_SIZE_LIMIT})"
        )
    payload = _receive_exactly(sock, payload_length)
    try:
        return _decode_frame(payload)
    except (
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        RuntimeError,
        struct.error,
        json.JSONDecodeError,
    ) as failure:
        raise ConnectionError(
            f"malformed frame ({type(failure).__name__}: {str(failure)[:80]})"
        ) from failure


def key_from_env(var: str = "WANFERENZ_PSK") -> None:
    return None


def use_key(material) -> None:
    return None
