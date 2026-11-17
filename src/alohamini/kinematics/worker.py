"""Local JSON IPC for ROS clients whose Python/NumPy ABI differs from Pink.

The worker computes targets only. It has no Host connection or command owner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import uuid
from collections import OrderedDict


def _endpoint(value):
    if not value.startswith("ipc:///") or len(value) > 100:
        raise ValueError("IK workers require a local ipc:/// endpoint shorter than 101 characters")
    return value


class PinkWorkerClient:
    """Synchronous, deadline-bounded RPC; no automatic execution or solver fallback."""

    def __init__(self, endpoint, *, timeout_s=0.05):
        import zmq

        self.zmq = zmq
        self.endpoint = _endpoint(endpoint)
        if not 0 < timeout_s <= 10:
            raise ValueError("IK worker timeout must be in (0, 10] seconds")
        self.timeout_ms = max(1, round(timeout_s * 1000))
        self.context = zmq.Context()
        self.socket = None
        self.session = uuid.uuid4().hex
        self.sequence = 0

    def _disconnect(self):
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        # A timed-out solve may have changed the worker's velocity history even
        # though its result was never executed. Start a fresh history next time.
        self.session = uuid.uuid4().hex

    def request(self, operation, *, timeout_ms=None, **payload):
        if self.socket is None:
            self.socket = self.context.socket(self.zmq.REQ)
            self.socket.setsockopt(self.zmq.LINGER, 0)
            self.socket.setsockopt(self.zmq.MAXMSGSIZE, 1_000_000)
            self.socket.connect(self.endpoint)
        self.sequence += 1
        message = dict(
            protocol=1, sequence=self.sequence, session=self.session, op=operation, **payload
        )
        started = time.perf_counter()
        try:
            self.socket.send_json(message)
            if not self.socket.poll(self.timeout_ms if timeout_ms is None else timeout_ms):
                raise TimeoutError("Pink worker did not reply before the IK deadline")
            result = self.socket.recv_json()
            if result.get("protocol") != 1 or result.get("sequence") != self.sequence:
                raise ValueError("Invalid or stale IK worker response")
            if not result.get("ok"):
                raise ValueError(f"Pink worker: {result.get('error', 'unknown failure')}")
            result["rpc_elapsed_s"] = time.perf_counter() - started
            return result
        except self.zmq.ZMQError as exc:
            self._disconnect()
            raise RuntimeError(f"Pink worker IPC failed: {exc}") from exc
        except Exception:
            self._disconnect()
            raise

    def step(self, geometry, joints, target, dt, **settings):
        import numpy as np

        result = self.request(
            "step",
            chain=getattr(geometry, "frame", "arm"),
            geometry=geometry.parameters(),
            joints=np.asarray(joints).tolist(),
            target=np.asarray(target).tolist(),
            dt=dt,
            settings=settings,
        )
        candidate = np.asarray(result["joints"], dtype=float)
        if candidate.shape != np.shape(joints) or not np.isfinite(candidate).all():
            self._disconnect()
            raise ValueError("IK worker returned invalid joint positions")
        result["metrics"]["rpc_elapsed_s"] = result["rpc_elapsed_s"]
        return candidate, result["metrics"]

    def reset(self, geometry):
        if self.socket is not None:
            try:
                self.request("reset", chain=getattr(geometry, "frame", "arm"))
            except (TimeoutError, ValueError, self.zmq.ZMQError):
                self._disconnect()

    def close(self):
        self._disconnect()
        self.context.term()


class _Handler:
    def __init__(self):
        self.models = OrderedDict()

    def handle(self, message):
        from .arm import SerialArmKinematics

        if message.get("protocol") != 1:
            raise ValueError("Unsupported IK worker protocol")
        op = message["op"]
        if op == "health":
            import pink
            import pinocchio

            return dict(pink=pink.__version__, pinocchio=pinocchio.__version__)
        key = (str(message["session"]), str(message["chain"]))
        if op == "reset":
            self.models.pop(key, None)
            return {}
        if op != "step":
            raise ValueError("Unsupported IK worker operation")
        geometry = message["geometry"]
        digest = hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()
        entry = self.models.get(key)
        if entry is None or entry[0] != digest:
            entry = (digest, SerialArmKinematics.from_parameters(geometry))
            self.models[key] = entry
        self.models.move_to_end(key)
        while len(self.models) > 16:
            self.models.popitem(last=False)
        settings = dict(message.get("settings", {}))
        solver = entry[1].pink_solver(qp_solver=settings.pop("qp_solver", "daqp"))
        joints, metrics = solver.step(
            message["joints"], message["target"], message["dt"], **settings
        )
        return dict(joints=joints.tolist(), metrics=metrics)


def main(argv=None):
    import zmq

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True, type=_endpoint)
    args = parser.parse_args(argv)
    handler = _Handler()
    versions = handler.handle(dict(protocol=1, op="health"))
    os.umask(0o077)
    with zmq.Context() as context, context.socket(zmq.REP) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.MAXMSGSIZE, 1_000_000)
        socket.bind(args.endpoint)
        print(f"Pink IK ready: {versions}; endpoint={args.endpoint}", flush=True)
        try:
            while True:
                message = {}
                try:
                    message = socket.recv_json()
                    if not isinstance(message, dict):
                        raise ValueError("IK request must be a JSON object")
                    payload = handler.handle(message)
                    result = dict(ok=True, **payload)
                except (KeyError, TypeError, ValueError, RuntimeError, ImportError) as exc:
                    result = dict(ok=False, error=str(exc))
                socket.send_json(
                    dict(
                        protocol=1,
                        sequence=message.get("sequence") if isinstance(message, dict) else None,
                        **result,
                    )
                )
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
