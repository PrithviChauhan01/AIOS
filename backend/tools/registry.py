import asyncio

from tools.base import Tool
from tools.wikipedia import WikipediaTool
from tools.reminders import RemindersTool

# ── THE SHARED POOL — instantiated tools by name. Any teacher draws any tool. ──
_REGISTRY = {
    "wikipedia": WikipediaTool(),   # read-only
    "reminders": RemindersTool(),   # ACTION — writes state (first of its kind)
}


def get_tool(name: str) -> Tool | None:
    return _REGISTRY.get(name)


def is_action_tool(name: str) -> bool:
    """True if the named tool WRITES state. Actions will later be routed through a
    confirm gate before execution; read-only tools never need one."""
    tool = _REGISTRY.get(name)
    return bool(tool and tool.is_action)


def action_tools() -> list:
    """Names of all state-writing tools in the pool."""
    return [n for n, t in _REGISTRY.items() if t.is_action]


async def fetch_from(names: list, query: str) -> dict:
    """Run the named tools concurrently against one query. Returns {tool: results}.
    Unknown or failed tools contribute an empty list — never raises."""
    tools = [t for t in (get_tool(n) for n in names) if t is not None]
    fetched = await asyncio.gather(
        *(t.fetch(query) for t in tools), return_exceptions=True)

    combined = {}
    for tool, result in zip(tools, fetched):
        if isinstance(result, dict) and result.get("ok"):
            combined[tool.name] = result.get("results", [])
        else:
            combined[tool.name] = []
    return combined
