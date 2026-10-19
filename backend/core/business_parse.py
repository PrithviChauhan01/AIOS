"""Natural language → clients / projects / invoices commands (Slice B4).

Deterministic regexes, no LLM — the same contract as the pipeline parser in
agents/leadgen.py: a business write must never hinge on a model's reading, and a
line that matches none of these is ordinary chat. core.action_dispatch calls
parse_business_command from its gate and executes the result against
tools/business.py.

Commands understood:
  add client <name>[, contact <x>]
  add project <title> for <client>[, due <date>][, fee <amount>]
  mark project <title> as planned|active|review|done   /  move <title> project to review
  set deadline for <title> to <date>
  create invoice for <project|client>[ for <amount>][ due <date>]
  mark invoice <number|client> as sent|paid
  invoice <client|number> pdf
  what's due this week  /  show my clients  /  unpaid invoices
"""

import re

_STATUS = r"planned|active|review|done"
_DATE_KEYS = r"due(?:\s+on|\s+by)?|deadline|by"
# Clauses split on a comma (but not the thousands comma in "25,000") or before a
# keyword: due/deadline/by/fee/budget/amount, or "for|of <amount>".
_CLAUSE_SPLIT = re.compile(
    rf"\s*,(?!\d)\s*|\s+(?=(?:{_DATE_KEYS}|fee|budget|amount)\b"
    rf"|(?:for|of)\s+(?:rs\.?|inr|₹|\$|usd)?\s*\d)",
    re.IGNORECASE)

_AMOUNT_RE = re.compile(
    r"(?P<cur>rs\.?|inr|₹|\$|usd|eur|€)?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<mul>k|lakhs?|l)?\b"
    r"\s*(?P<cur2>inr|usd|eur|rupees)?", re.IGNORECASE)
_CURRENCY = {"rs": "INR", "rs.": "INR", "inr": "INR", "₹": "INR", "rupees": "INR",
             "$": "USD", "usd": "USD", "eur": "EUR", "€": "EUR"}

_P_DUE_WEEK = re.compile(
    r"\bwhat'?s\s+due\b|\bwhat\s+is\s+due\b|\bdue\s+this\s+week\b|\bdeadlines?\s+this\s+week\b",
    re.IGNORECASE)
_P_CLIENTS = re.compile(r"\b(?:show|list|view|who\s+are|what\s+are)\b[^.?]{0,12}\bclients\b",
                        re.IGNORECASE)
_P_UNPAID = re.compile(r"\b(?:unpaid|outstanding|overdue)\s+invoices?\b|\bwho\s+owes\s+me\b",
                       re.IGNORECASE)
_P_PDF = re.compile(
    r"^\s*(?:(?:make|generate|export|create)\s+(?:an?\s+|the\s+)?)?invoice\s+(?:pdf\s+(?:for|of)\s+)?"
    r"(?P<target>.+?)(?:\s+(?:invoice\s+)?(?:as\s+(?:a\s+)?)?pdf)?\s*$", re.IGNORECASE)
_P_ADD_CLIENT = re.compile(
    r"^\s*(?:please\s+)?(?:add|create|new)\s+(?:a\s+)?(?:new\s+)?client\s*:?\s+(?P<rest>.+?)\s*$",
    re.IGNORECASE)
_P_ADD_PROJECT = re.compile(
    r"^\s*(?:please\s+)?(?:add|create|new|start)\s+(?:a\s+)?(?:new\s+)?project\s*:?\s+"
    r"(?P<title>.+?)\s+for\s+(?P<rest>.+?)\s*$", re.IGNORECASE)
_P_PROJECT_STATUS = re.compile(
    rf"^\s*(?:please\s+)?(?:mark|move|set|update|put)\s+(?:the\s+)?(?:project\s+(?P<a>.+?)|(?P<b>.+?)\s+project)"
    rf"\s+(?:status\s+)?(?:as|to|in(?:to)?)\s+(?:status\s+)?(?P<status>{_STATUS})\s*\.?\s*$",
    re.IGNORECASE)
_P_DEADLINE = re.compile(
    r"^\s*(?:please\s+)?(?:set|change|move|push|update|extend)\s+(?:the\s+)?deadline\s+"
    r"(?:for|of|on)\s+(?:project\s+)?(?P<title>.+?)\s+to\s+(?P<when>.+?)\s*\.?\s*$", re.IGNORECASE)
_P_CREATE_INVOICE = re.compile(
    r"^\s*(?:please\s+)?(?:create|make|raise|generate|draft|issue|new)\s+(?:an?\s+|the\s+)?"
    r"invoice\s+(?:for|to)\s+(?P<rest>.+?)\s*\.?\s*$", re.IGNORECASE)
_P_MARK_INVOICE = re.compile(
    r"^\s*(?:please\s+)?mark\s+(?:the\s+)?(?:invoice\s+(?P<a>.+?)|(?P<b>.+?)\s+invoice)\s+"
    r"(?:as\s+)?(?P<status>sent|paid)\s*\.?\s*$", re.IGNORECASE)
_P_NUMBER_PAID = re.compile(r"^\s*(?P<a>INV-\d{4}-\d{3,})\s+(?:is\s+|was\s+|has\s+been\s+)?"
                            r"(?P<status>sent|paid)\s*\.?\s*$", re.IGNORECASE)


def parse_amount(text: str):
    """'50000' / 'Rs 50,000' / '₹1.5 lakh' / '$2k' → (amount, currency|None), else None."""
    m = _AMOUNT_RE.search(text or "")
    if not m:
        return None
    amount = float(m.group("num").replace(",", ""))
    mul = (m.group("mul") or "").lower()
    if mul == "k":
        amount *= 1_000
    elif mul.startswith("l"):
        amount *= 100_000
    cur = (m.group("cur") or m.group("cur2") or "").lower()
    return amount, _CURRENCY.get(cur)


def _clauses(rest: str):
    """Split '<head>, due Friday, fee 50000' into the head and keyed clauses."""
    parts = [p.strip() for p in _CLAUSE_SPLIT.split(rest or "") if p and p.strip()]
    head, opts = (parts[0] if parts else ""), {}
    for part in parts[1:]:
        low = part.lower()
        if re.match(rf"(?:{_DATE_KEYS})\b", low):
            opts["when"] = re.sub(rf"^(?:{_DATE_KEYS})\s*:?\s*(?:on\s+)?", "", part,
                                  flags=re.IGNORECASE).strip()
        elif re.match(r"(?:fee|budget|for|of|amount)\b", low):
            amt = parse_amount(part)
            if amt:
                opts["amount"], opts["currency"] = amt
        elif re.match(r"(?:contact|email|phone)\b", low):
            opts["contact"] = re.sub(r"^(?:contact|email|phone)\s*:?\s*", "", part,
                                     flags=re.IGNORECASE).strip()
        else:
            head = f"{head}, {part}" if not opts else head
    return head.strip(" .\"'“”"), opts


def _name(text: str) -> str:
    return re.sub(r"^(?:the|client|project)\s+", "", (text or "").strip(" .\"'“”"),
                  flags=re.IGNORECASE).strip()


def parse_business_command(message: str) -> dict | None:
    msg = (message or "").strip()
    if not msg:
        return None
    if _P_DUE_WEEK.search(msg):
        return {"action": "due_week"}
    if _P_UNPAID.search(msg):
        return {"action": "unpaid"}
    if _P_CLIENTS.search(msg):
        return {"action": "list_clients"}

    if re.search(r"\bpdf\b", msg, re.IGNORECASE):
        m = _P_PDF.search(msg)
        if m and _name(m.group("target")):
            return {"action": "invoice_pdf", "target": _name(m.group("target"))}

    m = _P_MARK_INVOICE.search(msg) or _P_NUMBER_PAID.search(msg)
    if m:
        target = _name(m.group("a") or m.groupdict().get("b") or "")
        if target:
            return {"action": "mark_invoice", "target": target,
                    "status": m.group("status").lower()}

    m = _P_CREATE_INVOICE.search(msg)
    if m:
        target, opts = _clauses(m.group("rest"))
        if _name(target):
            return {"action": "create_invoice", "target": _name(target),
                    "amount": opts.get("amount"), "currency": opts.get("currency"),
                    "due": opts.get("when")}

    m = _P_ADD_PROJECT.search(msg)
    if m:
        client, opts = _clauses(m.group("rest"))
        if _name(m.group("title")) and _name(client):
            return {"action": "add_project", "title": _name(m.group("title")),
                    "client": _name(client), "deadline": opts.get("when"),
                    "fee": opts.get("amount")}

    m = _P_ADD_CLIENT.search(msg)
    if m:
        name, opts = _clauses(m.group("rest"))
        if _name(name):
            return {"action": "add_client", "name": _name(name), "contact": opts.get("contact")}

    m = _P_PROJECT_STATUS.search(msg)
    if m:
        title = _name(m.group("a") or m.group("b"))
        if title:
            return {"action": "update_project", "title": title,
                    "status": m.group("status").lower(), "deadline": None}

    m = _P_DEADLINE.search(msg)
    if m and _name(m.group("title")):
        return {"action": "update_project", "title": _name(m.group("title")),
                "status": None, "deadline": m.group("when").strip()}
    return None
