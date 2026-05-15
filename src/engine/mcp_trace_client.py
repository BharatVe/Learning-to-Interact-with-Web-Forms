import json
import os
import select
import shlex
import subprocess
import time
from typing import Any, Dict, List, Optional, Sequence, Union


class MCPClientError(RuntimeError):
    """Raised for MCP transport/protocol errors."""


class MCPClient:
    def __init__(
        self,
        command: Union[str, List[str]],
        timeout_ms: int = 5000,
        required_tools: Optional[Sequence[str]] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        if isinstance(command, str):
            self.command = shlex.split(command)
        else:
            self.command = list(command)
        if not self.command:
            raise MCPClientError("MCP command is empty")

        self.timeout_s = max(float(timeout_ms) / 1000.0, 0.1)
        self.required_tools = [tool for tool in (required_tools or []) if tool]
        self.env = dict(os.environ)
        if env:
            self.env.update(env)
        self._next_id = 1
        self._closed = False
        self.server_info: Dict[str, Any] = {}
        self.protocol_version: Optional[str] = None
        self.available_tools: List[str] = []
        self.tool_definitions: List[Dict[str, Any]] = []

        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self.env,
        )
        self._initialize()

    def _next_request_id(self) -> int:
        req_id = self._next_id
        self._next_id += 1
        return req_id

    def _read_stderr(self) -> str:
        if not self._proc.stderr:
            return ""
        try:
            if self._proc.poll() is not None:
                return self._proc.stderr.read() or ""
        except Exception:
            return ""
        return ""

    def _write_message(self, payload: Dict[str, Any]) -> None:
        if self._closed:
            raise MCPClientError("MCP client is closed")
        if not self._proc.stdin:
            raise MCPClientError("MCP stdin unavailable")
        raw = json.dumps(payload, ensure_ascii=True) + "\n"
        try:
            self._proc.stdin.write(raw)
            self._proc.stdin.flush()
        except BrokenPipeError as exc:
            stderr = self._read_stderr()
            raise MCPClientError(f"MCP server pipe closed. stderr={stderr}") from exc

    def _read_message(self, timeout_s: float) -> Dict[str, Any]:
        if self._closed:
            raise MCPClientError("MCP client is closed")
        if not self._proc.stdout:
            raise MCPClientError("MCP stdout unavailable")

        ready, _, _ = select.select([self._proc.stdout], [], [], timeout_s)
        if not ready:
            raise MCPClientError(f"MCP response timed out after {timeout_s:.2f}s")

        line = self._proc.stdout.readline()
        if not line:
            stderr = self._read_stderr()
            raise MCPClientError(f"MCP server exited unexpectedly. stderr={stderr}")

        try:
            message = json.loads(line)
        except Exception as exc:
            raise MCPClientError(f"Invalid MCP JSON: {line}") from exc
        if not isinstance(message, dict):
            raise MCPClientError("MCP response must be a JSON object")
        return message

    def _request(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        req_id = self._next_request_id()
        self._write_message(
            {
                "jsonrpc": "2.0",
                "id": req_id,
                "method": method,
                "params": params or {},
            }
        )
        started = time.monotonic()
        while True:
            remaining = self.timeout_s - (time.monotonic() - started)
            if remaining <= 0:
                raise MCPClientError(f"MCP request timed out: {method}")
            message = self._read_message(remaining)
            if message.get("id") != req_id:
                continue
            if "error" in message:
                raise MCPClientError(f"MCP error for {method}: {message['error']}")
            result = message.get("result")
            if not isinstance(result, dict):
                raise MCPClientError(f"MCP result for {method} must be an object")
            return result

    def _notify_initialized(self) -> None:
        self._write_message(
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            }
        )

    def _initialize(self) -> None:
        result = self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "thesis-runner", "version": "1.0.0"},
            },
        )
        self.protocol_version = str(result.get("protocolVersion", ""))
        server_info = result.get("serverInfo")
        self.server_info = server_info if isinstance(server_info, dict) else {}

        self._notify_initialized()
        tools_result = self._request("tools/list", {})
        tools = tools_result.get("tools")
        if not isinstance(tools, list):
            raise MCPClientError("MCP tools/list missing tools array")

        names: List[str] = []
        tool_defs: List[Dict[str, Any]] = []
        for tool in tools:
            if isinstance(tool, dict):
                name = tool.get("name")
                if isinstance(name, str):
                    names.append(name)
                    tool_defs.append(dict(tool))
        self.available_tools = sorted(set(names))
        self.tool_definitions = tool_defs
        missing = [tool for tool in self.required_tools if tool not in self.available_tools]
        if missing:
            missing_joined = ", ".join(missing)
            available = ", ".join(self.available_tools) or "(none)"
            raise MCPClientError(f"Missing MCP tools: {missing_joined}. Available: {available}")

    @staticmethod
    def _structured_payload_from_tool_result(result: Dict[str, Any]) -> Dict[str, Any]:
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured

        content = result.get("content")
        if isinstance(content, list) and content:
            text_fragments: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                json_payload = item.get("json")
                if isinstance(json_payload, dict):
                    return json_payload
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                text_fragments.append(text)
                try:
                    parsed = json.loads(text)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    return parsed
                return {"value": parsed}
            if text_fragments:
                merged = "\n".join(text_fragments)
                return {"text": merged, "content": text_fragments}
        return {}

    def call_tool(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        result = self._request(
            "tools/call",
            {"name": tool_name, "arguments": arguments or {}},
        )
        if result.get("isError"):
            raise MCPClientError(f"MCP tool call failed for '{tool_name}': {result}")
        payload = self._structured_payload_from_tool_result(result)
        payload["_tool_result"] = result
        return payload

    def record_action(self, payload: Dict[str, Any], tool_name: str = "record_action") -> Dict[str, Any]:
        parsed = self.call_tool(tool_name, payload)
        if parsed:
            return parsed
        raise MCPClientError("MCP tool returned no structured payload")

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": "mcp_server",
            "command": self.command,
            "protocol_version": self.protocol_version,
            "server_info": self.server_info,
            "available_tools": self.available_tools,
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.tool_definitions]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
        except Exception:
            pass
        try:
            if self._proc.poll() is None:
                self._proc.terminate()
                self._proc.wait(timeout=1.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass


class MCPTraceClient(MCPClient):
    def __init__(
        self,
        command: Union[str, List[str]],
        tool_name: str = "record_action",
        timeout_ms: int = 5000,
    ) -> None:
        self.tool_name = tool_name
        super().__init__(
            command=command,
            timeout_ms=timeout_ms,
            required_tools=[tool_name],
        )

    def record_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return super().record_action(payload, tool_name=self.tool_name)

    def summary(self) -> Dict[str, Any]:
        data = super().summary()
        data["tool_name"] = self.tool_name
        return data
