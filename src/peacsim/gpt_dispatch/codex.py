"""Codex app-server transport. The model receives text with no environment."""
import json
import os
from pathlib import Path
import select
import subprocess
from tempfile import TemporaryDirectory
from time import monotonic

from .decisions import INSTRUCTIONS, dumps


_DISABLED_FEATURES = (
    "apps", "plugins", "hooks", "memories", "shell_tool", "unified_exec", "view_image",
    "browser_use", "computer_use", "code_mode_host", "multi_agent_v2",
)
_TEXT_ITEMS = {"userMessage", "agentMessage", "reasoning", "contextCompaction"}


class CodexPolicy:
    """One ephemeral, environment-free Codex thread, using the installed CLI login."""

    method = "codex-app-server"

    def __init__(self, *, model, effort, schema, executable="codex", codex_home=None,
                 instructions=INSTRUCTIONS, require_same_thread=False):
        self.model = model
        self.effort = effort
        self.schema = schema
        self.instructions = instructions
        self.require_same_thread = require_same_thread
        self.executable = str(Path(executable).resolve()) if os.sep in executable else executable
        self.codex_home = Path(codex_home).resolve() if codex_home is not None else None
        self._proc = self._workspace = self._thread_id = None
        self._stderr = None
        self._buffer = bytearray()
        self._request_id = 0

    def close(self):
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait()
            self._proc.stdin.close()
            self._proc.stdout.close()
            self._proc = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        if self._workspace is not None:
            self._workspace.cleanup()
            self._workspace = None
        self._thread_id = None
        self._buffer.clear()

    @staticmethod
    def _remaining(deadline):
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Codex decision deadline exceeded")
        return remaining

    def _send(self, message, deadline):
        fd = self._proc.stdin.fileno()
        data = memoryview((dumps(message) + "\n").encode("utf-8"))
        while data:
            if not select.select([], [fd], [], self._remaining(deadline))[1]:
                raise TimeoutError("Codex decision deadline exceeded while sending input")
            try:
                written = os.write(fd, data)
            except BlockingIOError:
                continue
            data = data[written:]

    def _receive(self, deadline):
        fd = self._proc.stdout.fileno()
        while True:
            timeout = self._remaining(deadline)
            end = self._buffer.find(b"\n")
            if end >= 0:
                line = bytes(self._buffer[:end])
                del self._buffer[:end + 1]
                return json.loads(line)
            if not select.select([fd], [], [], timeout)[0]:
                raise TimeoutError("Codex decision deadline exceeded while awaiting output")
            chunk = os.read(fd, 65536)
            if not chunk:
                self._stderr.seek(max(0, self._stderr.seek(0, os.SEEK_END) - 1500))
                detail = self._stderr.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"Codex app-server exited before completing the request. {detail}")
            self._buffer.extend(chunk)

    def _request(self, method, params, deadline, on_event):
        self._request_id += 1
        request_id = self._request_id
        self._send(dict(id=request_id, method=method, params=params), deadline)
        while True:
            message = self._receive(deadline)
            if message.get("id") == request_id and "method" not in message:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message["result"]
            on_event(message)

    @staticmethod
    def _check_event(message):
        method = message.get("method")
        params = message.get("params", {})
        if method and "id" in message:
            raise PermissionError(f"Codex sent an unexpected server request: {method}")
        if method in ("item/started", "item/completed"):
            kind = params["item"]["type"]
            if kind not in _TEXT_ITEMS:
                raise PermissionError(f"Codex reported a non-text item: {kind}")
        if method == "model/rerouted":
            raise PermissionError("Codex attempted to reroute the requested model")

    def _start(self, deadline):
        self._workspace = TemporaryDirectory(prefix="flexdc-codex-")
        root = Path(self._workspace.name).resolve()
        cwd = root / "empty-workspace"
        cwd.mkdir()
        self._stderr = (root / "stderr.log").open("w+b")
        options = ["-c", 'web_search="disabled"', "-c", f"sqlite_home={dumps(str(root))}",
                   "-c", f"log_dir={dumps(str(root))}"]
        for feature in _DISABLED_FEATURES:
            options.extend(["--disable", feature])
        env = None if self.codex_home is None else {**os.environ, "CODEX_HOME": str(self.codex_home)}
        # Listing is local and launches no configured MCP processes. Check the
        # selected profile before starting a server that could initialize them.
        listed = subprocess.run(
            [self.executable, "mcp", "list", "--json", *options],
            cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=self._stderr,
            timeout=self._remaining(deadline),
        )
        if listed.returncode:
            raise RuntimeError("Codex could not enumerate configured MCP servers")
        enabled = [server["name"] for server in json.loads(listed.stdout) if server["enabled"]]
        if enabled:
            raise PermissionError(f"Use a Codex home without enabled MCP servers: {enabled}")
        self._proc = subprocess.Popen(
            [self.executable, "app-server", *options], cwd=cwd, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
        )
        os.set_blocking(self._proc.stdin.fileno(), False)
        self._request("initialize", dict(
            clientInfo=dict(name="flexdc_dispatch", title="FlexDC Dispatch", version="0.1.0"),
            capabilities=dict(experimentalApi=True),
        ), deadline, self._check_event)
        self._send(dict(method="initialized"), deadline)
        result = self._request("thread/start", dict(
            cwd=str(cwd), model=self.model, modelProvider="openai", allowProviderModelFallback=False,
            environments=[], runtimeWorkspaceRoots=[], dynamicTools=[], ephemeral=True,
            approvalPolicy="never", sandbox="read-only", baseInstructions=self.instructions,
        ), deadline, self._check_event)
        thread = result["thread"]
        if thread.get("environments") != [] or result.get("instructionSources") != []:
            raise PermissionError("Codex must confirm no environment or instruction files")
        if result.get("model") != self.model or result.get("modelProvider") != "openai":
            raise PermissionError("Codex did not select the requested OpenAI model")
        self._thread_id = thread["id"]
        mcp = self._request("mcpServerStatus/list", dict(threadId=self._thread_id),
                            deadline, self._check_event)
        if mcp.get("data") != [] or mcp.get("nextCursor"):
            names = [server.get("name") for server in mcp.get("data", ())]
            raise PermissionError(f"Codex thread reported MCP servers: {names}")

    def decide(self, request, *, timeout_seconds):
        deadline = monotonic() + timeout_seconds
        receipt = dict(model=self.model, raw_output=None, usage=None, thread_id=None, turn_id=None)
        events = []
        try:
            if self._thread_id is None:
                self._start(deadline)
            receipt["thread_id"] = self._thread_id
            receipt["isolation"] = dict(environments=[], mcp_servers=0)

            def on_event(message):
                self._check_event(message)
                events.append(message)

            params = dict(threadId=self._thread_id, environments=[],
                          input=[dict(type="text", text=dumps(request), text_elements=[])],
                          outputSchema=self.schema)
            if self.effort is not None:
                params["effort"] = self.effort
            result = self._request("turn/start", params, deadline, on_event)
            turn_id = receipt["turn_id"] = result["turn"]["id"]
            final_text = None
            while True:
                if events:
                    message = events.pop(0)
                else:
                    message = self._receive(deadline)
                    self._check_event(message)
                params = message.get("params", {})
                if params.get("threadId") != self._thread_id:
                    continue
                method = message.get("method")
                if method == "turn/completed" and params["turn"]["id"] == turn_id:
                    turn = params["turn"]
                    if turn["status"] != "completed":
                        return dict(**receipt, error=f"Codex turn {turn['status']}: {turn.get('error')}")
                    if final_text is None:
                        return dict(**receipt, error="Codex completed without a final assistant message")
                    receipt["raw_output"] = final_text
                    return receipt
                if params.get("turnId") != turn_id:
                    continue
                if method == "thread/tokenUsage/updated":
                    receipt["usage"] = params["tokenUsage"]["last"]
                if method == "item/completed" and params["item"]["type"] == "agentMessage":
                    item = params["item"]
                    if item.get("phase") in (None, "final_answer"):
                        final_text = receipt["raw_output"] = item["text"]
        except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
            lost_required_thread = self.require_same_thread and receipt["thread_id"] is not None
            self.close()
            return dict(**receipt, error=f"{type(exc).__name__}: {exc}",
                        retryable=not isinstance(exc, PermissionError) and not lost_required_thread)
