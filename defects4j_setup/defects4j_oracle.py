"""Bridge from hdd_loc.py's Python oracle interface to a real Java program,
via a persistent JVM per checkout, and a differential (buggy vs golden)
oracle suited to semantic bugs that don't crash.

Nothing in hdd_loc.py changes to use this: HierarchicalDeltaDebugger only
ever needs ``Callable[[Any], bool]``. DifferentialOracle below is exactly
that -- an instance is callable and can be passed straight in as
``HierarchicalDeltaDebugger(oracle=differential_oracle)``.

Setup this module assumes you've already done, on your own machine (where
Defects4J and a JDK are installed -- this can't be done from a sandbox with
no Maven Central access):

    defects4j checkout -p Lang -v 1b -w /path/to/lang_1_buggy
    defects4j checkout -p Lang -v 1f -w /path/to/lang_1_fixed
    defects4j compile -w /path/to/lang_1_buggy
    defects4j compile -w /path/to/lang_1_fixed

Then, for THIS specific bug, write one small Java class implementing
BugAdapter (see BugAdapter.java) that calls the actual buggy method with
whatever input you're minimizing, and compile it (plus Harness.java)
against each checkout's own compiled classes:

    javac -cp /path/to/lang_1_buggy/target/classes:. Harness.java BugAdapter.java LangBugAdapter.java
    javac -cp /path/to/lang_1_fixed/target/classes:. Harness.java BugAdapter.java LangBugAdapter.java

(exact classes/ output directory depends on the project's build layout --
check with `defects4j export -p dir.bin.classes`).

Then:

    from defects4j_oracle import JavaHarnessClient, DifferentialOracle
    from hdd_loc import HierarchicalDeltaDebugger

    buggy = JavaHarnessClient(classpath="/path/to/lang_1_buggy/target/classes:.",
                               adapter_class="LangBugAdapter", port=45001)
    golden = JavaHarnessClient(classpath="/path/to/lang_1_fixed/target/classes:.",
                                adapter_class="LangBugAdapter", port=45002)
    buggy.start()
    golden.start()

    oracle = DifferentialOracle(buggy, golden)
    debugger = HierarchicalDeltaDebugger(oracle=oracle)
    result = debugger.reduce(original_input)   # original_input: your Python
                                                # structured representation
                                                # of the trigger test's data

    buggy.stop()
    golden.stop()
"""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# --------------------------------------------------------------------------
# Wire protocol (must match Harness.java exactly)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HarnessResponse:
    """One response from a Harness process: whether the adapter threw, and
    the payload (its return value, or the exception's class + message).
    """

    ok: bool
    payload: str


def _send_request(sock: socket.socket, payload: str) -> None:
    payload_bytes = payload.encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload_bytes)))
    sock.sendall(payload_bytes)


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Harness closed the connection unexpectedly")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_response(sock: socket.socket) -> HarnessResponse:
    status_byte = _recv_exactly(sock, 1)[0]
    (length,) = struct.unpack(">I", _recv_exactly(sock, 4))
    payload = _recv_exactly(sock, length).decode("utf-8")
    return HarnessResponse(ok=(status_byte == 0), payload=payload)


# --------------------------------------------------------------------------
# Persistent JVM client
# --------------------------------------------------------------------------


class JavaHarnessClient:
    """Manages one persistent `java Harness <adapter> <port>` process
    (one per checkout -- buggy or fixed) and talks to it over a local socket.
    """

    def __init__(
        self,
        classpath: str,
        adapter_class: str,
        port: int,
        java_bin: str = "java",
        harness_class: str = "Harness",
        startup_timeout: float = 30.0,
    ) -> None:
        self.classpath = classpath
        self.adapter_class = adapter_class
        self.port = port
        self.java_bin = java_bin
        self.harness_class = harness_class
        self.startup_timeout = startup_timeout
        self._process: Optional[subprocess.Popen] = None

    def start(self) -> None:
        if self._process is not None:
            return  # already started
        self._process = subprocess.Popen(
            [self.java_bin, "-cp", self.classpath, self.harness_class, self.adapter_class, str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._wait_for_ready()

    def _wait_for_ready(self) -> None:
        assert self._process is not None
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise RuntimeError(f"Harness process exited early:\n{stderr}")
            line = self._process.stdout.readline() if self._process.stdout else ""
            if line.strip() == "READY":
                return
        raise TimeoutError(f"Harness on port {self.port} did not report READY within {self.startup_timeout}s")

    def invoke(self, payload: str) -> HarnessResponse:
        # A fresh, short-lived connection per call -- the JVM process (and
        # whatever expensive state the adapter set up once) stays warm the
        # whole run; only this local TCP handshake is per-call, and that's
        # cheap on loopback. This also matches Harness.java's per-connection
        # accept() loop, so both sides agree on when a "session" ends.
        if self._process is None:
            raise RuntimeError("JavaHarnessClient not started -- call start() first")
        with socket.create_connection(("127.0.0.1", self.port), timeout=self.startup_timeout) as sock:
            _send_request(sock, payload)
            return _recv_response(sock)

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def __enter__(self) -> "JavaHarnessClient":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


# --------------------------------------------------------------------------
# Differential oracle
# --------------------------------------------------------------------------


class DifferentialOracle:
    """Callable oracle: True (still reproduces the bug) iff the buggy
    checkout's output differs from the golden/fixed checkout's output for
    the same candidate input -- catches semantic bugs (wrong output, no
    exception) as well as crash bugs (one side throws, the other doesn't).
    Pass an instance directly as HierarchicalDeltaDebugger(oracle=...);
    hdd_loc.py needs no changes to use this.
    """

    def __init__(
        self,
        buggy: JavaHarnessClient,
        golden: JavaHarnessClient,
        serialize: Callable[[Any], str] = json.dumps,
    ) -> None:
        self.buggy = buggy
        self.golden = golden
        self.serialize = serialize

    def __call__(self, candidate: Any) -> bool:
        payload = self.serialize(candidate)
        buggy_response = self.buggy.invoke(payload)
        golden_response = self.golden.invoke(payload)

        # Differ in outcome (one threw, the other didn't) OR differ in
        # payload (both returned/threw, but with different values/messages)
        # both count as "still reproduces the semantic divergence."
        return (buggy_response.ok != golden_response.ok) or (buggy_response.payload != golden_response.payload)


if __name__ == "__main__":
    # This demo needs a JDK to compile the toy adapters (SumAdapterBuggy /
    # SumAdapterFixed) and won't run in an environment with no javac -- it's
    # here to show the exact calling pattern for a real Defects4J bug.
    #
    #   javac Harness.java BugAdapter.java SumAdapterBuggy.java SumAdapterFixed.java
    #   python3 defects4j_oracle.py
    from hdd_loc import HierarchicalDeltaDebugger

    buggy = JavaHarnessClient(classpath=".", adapter_class="SumAdapter", port=45001)
    golden = JavaHarnessClient(classpath=".", adapter_class="SumAdapterFixed", port=45002)

    with buggy, golden:
        oracle = DifferentialOracle(
            buggy,
            golden,
            # SumAdapter takes a comma-separated string, not JSON -- so the
            # serializer here just renders the candidate list that way.
            serialize=lambda candidate: ",".join(str(v) for v in candidate),
        )

        original_input = [3, -1, 5, -2, 8]  # sum differs between buggy (16) and fixed (13)
        debugger = HierarchicalDeltaDebugger(oracle=oracle)
        result = debugger.reduce(original_input)

        print("Minimal input still causing semantic divergence:", result.minimal_failing_input)
        print(f"Total test cases: {len(result.test_records)}")