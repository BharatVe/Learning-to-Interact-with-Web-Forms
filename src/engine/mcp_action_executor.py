from typing import Any, Dict, Optional, Tuple

from engine.trace_contract import validate_action


class MCPActionExecutionError(RuntimeError):
    """Raised when an MCP-style action cannot be executed on the page."""


class MCPActionExecutor:
    """
    Execute browser interactions via MCP-style action names and arguments.
    This keeps runtime behavior aligned with the same action schema used in traces.
    """

    def __init__(
        self,
        page: Any,
        viewport: Dict[str, int],
        timeout_ms: int,
        type_delay_ms: int,
    ) -> None:
        self.page = page
        self.viewport = viewport
        self.timeout_ms = timeout_ms
        self.type_delay_ms = max(0, type_delay_ms)

    def _denorm(self, x_norm: Optional[int], y_norm: Optional[int]) -> Tuple[float, float]:
        if x_norm is None or y_norm is None:
            raise MCPActionExecutionError("missing_coords")
        width = max(int(self.viewport.get("width", 1)), 1)
        height = max(int(self.viewport.get("height", 1)), 1)
        x_px = float((x_norm / 999.0) * width)
        y_px = float((y_norm / 999.0) * height)
        return x_px, y_px

    def execute(self, name: str, args: Dict[str, Any], ok: bool = True) -> None:
        validate_action(name, args, ok=ok)
        if name == "hover_at":
            self._hover_at(args)
            return
        if name == "click_at":
            self._click_at(args)
            return
        if name == "type_text_at":
            self._type_text_at(args)
            return
        if name == "key_combination":
            self._key_combination(args)
            return
        if name == "scroll_document":
            self._scroll_document(args)
            return
        raise MCPActionExecutionError(f"unsupported_action: {name}")

    def _hover_at(self, args: Dict[str, Any]) -> None:
        x_px, y_px = self._denorm(args.get("x"), args.get("y"))
        self.page.mouse.move(x_px, y_px, steps=20)

    def _click_at(self, args: Dict[str, Any]) -> None:
        x_px, y_px = self._denorm(args.get("x"), args.get("y"))
        self.page.mouse.click(x_px, y_px)

    def _type_text_at(self, args: Dict[str, Any]) -> None:
        x_px, y_px = self._denorm(args.get("x"), args.get("y"))
        text = args.get("text")
        if not isinstance(text, str):
            raise MCPActionExecutionError("type_text_at_requires_text")
        self.page.mouse.click(x_px, y_px)
        self.page.keyboard.type(text, delay=self.type_delay_ms)

    def _key_combination(self, args: Dict[str, Any]) -> None:
        keys = args.get("keys")
        if not isinstance(keys, str) or not keys.strip():
            raise MCPActionExecutionError("invalid_keys")
        self.page.keyboard.press(keys)

    def _scroll_document(self, args: Dict[str, Any]) -> None:
        direction = args.get("direction")
        if direction not in {"up", "down"}:
            raise MCPActionExecutionError("invalid_scroll_direction")
        delta = 640 if direction == "down" else -640
        self.page.mouse.wheel(0, delta)
