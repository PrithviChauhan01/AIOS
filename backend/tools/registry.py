import asyncio

from tools.base import Tool
from tools.wikipedia import WikipediaTool
from tools.search import SearchTool
from tools.places import PlacesTool
from tools.reminders import RemindersTool
from tools.jobs import JobsTool
from tools.documents import DocumentsTool
from tools.mailer import EmailTool
from tools.fitness import FitnessTool
from tools.leads import LeadsTool
from tools.study import StudyTool
from tools.habits import HabitsTool

# ── THE SHARED POOL — instantiated tools by name. Any teacher draws any tool. ──
_REGISTRY = {
    "wikipedia": WikipediaTool(),   # read-only
    "search": SearchTool(),         # read-only — live web search (Tavily)
    "places": PlacesTool(),         # read-only — real business directory (Google Places)
    "reminders": RemindersTool(),   # ACTION — writes state (first of its kind)
    "jobs": JobsTool(),             # ACTION — logs job applications
    "documents": DocumentsTool(),   # ACTION — general document store (tiered)
    "email": EmailTool(),           # ACTION — first EXTERNAL action, confirm-gated
    "fitness": FitnessTool(),       # ACTION — logs PRs/workouts to fitness_logs
    "leads": LeadsTool(),           # ACTION — persists prospecting rows (bulk)
    "study": StudyTool(),           # ACTION — logs study sessions
    "habits": HabitsTool(),         # ACTION — daily habit marks/checks
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


def format_pool_block(fetched: dict) -> str:
    """Format fetch_from output as a per-tool text block for injection into a book
    or cognition prompt. Tools that returned nothing are noted honestly so a model
    never assumes data it didn't get. Returns "" when NOTHING came back at all —
    callers then simply skip the block."""
    if not fetched or not any(fetched.values()):
        return ""
    parts = []
    for name, results in fetched.items():
        if results:
            lines = "\n".join(f"- {r}" for r in results)
            parts.append(f"[{name}]\n{lines}")
        else:
            parts.append(f"[{name}] returned no results")
    return "\n\n".join(parts)


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
