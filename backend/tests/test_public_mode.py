"""Public-mode tests: disabled tools, 5/day rate limit, no persistence (+ boot guard, input cap).

Run from backend/:   venv\\Scripts\\python -m unittest discover -s tests -v

No network: triage, the local action extractor and every book call are patched.
"""

import asyncio
import os
import shutil
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from config import Config
from core import action_dispatch, public_session, rate_limit
from core import public_orchestrator as po

_VERDICT = {"sensitivity": "public", "complexity": "simple", "domain": "none",
            "loop_worthy": False, "domains": ["none"]}
_BOOK_REPLY = {"raw_text": "Paris, Sir.\n[mood: neutral]", "book_used": "groq",
               "tokens": 12, "tokens_in": 10, "tokens_out": 2}
_ACTION_TOOLS = ("reminders", "jobs", "documents", "email", "fitness", "leads",
                 "study", "habits")


class PublicModeCase(unittest.TestCase):
    """Public mode on, every store pointed at an empty temp dir, state reset."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aios_public_")
        self.sqlite = os.path.join(self.tmp, "public.db")
        self.chroma = os.path.join(self.tmp, "chroma")
        self.vault = os.path.join(self.tmp, "vault.db")
        patches = [
            mock.patch.object(Config, "PUBLIC_MODE", True),
            mock.patch.object(Config, "SQLITE_PATH", self.sqlite),
            mock.patch.object(Config, "CHROMA_PATH", self.chroma),
            mock.patch.object(Config, "VAULT_PATH", self.vault),
            mock.patch.object(po, "triage", return_value=dict(_VERDICT)),
            mock.patch.object(po, "select_book_for_tier", return_value=["groq"]),
            mock.patch.object(po, "select_fast_book", return_value=["groq"]),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.call_book = mock.AsyncMock(return_value=dict(_BOOK_REPLY))
        p = mock.patch.object(po, "call_book", self.call_book)
        p.start()
        self.addCleanup(p.stop)
        public_session.reset()
        rate_limit.reset()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def client(self) -> TestClient:
        from main import create_app
        c = TestClient(create_app())
        c.__enter__()
        self.addCleanup(c.__exit__, None, None, None)
        return c


class DisabledToolsTest(PublicModeCase):

    def test_registry_hides_every_writing_tool(self):
        from tools.registry import get_tool
        for name in _ACTION_TOOLS:
            self.assertIsNone(get_tool(name), name)
        for name in ("wikipedia", "search", "places"):  # read-only stays available
            self.assertIsNotNone(get_tool(name), name)

    def test_registry_unchanged_in_personal_mode(self):
        from tools.registry import get_tool
        with mock.patch.object(Config, "PUBLIC_MODE", False):
            for name in _ACTION_TOOLS:
                self.assertIsNotNone(get_tool(name), name)

    def test_run_action_refuses_every_tool_without_executing(self):
        runners = ["_run_reminders", "_run_jobs", "_run_documents", "_run_email",
                   "_run_fitness", "_run_leads", "_run_study", "_run_habits"]
        mocks = {}
        for r in runners:
            p = mock.patch.object(action_dispatch, r,
                                  side_effect=AssertionError(f"{r} executed"))
            mocks[r] = p.start()
            self.addCleanup(p.stop)
        for tool in _ACTION_TOOLS:
            res = action_dispatch.run_action(
                {"tool": tool, "action": "set", "session_id": "s", "raw_message": "x"})
            self.assertFalse(res["ok"], tool)
            self.assertEqual(res["error"], "disabled_public", tool)
            self.assertIn("public demo", res["ask"])
        for r, m in mocks.items():
            m.assert_not_called()

    def test_detect_action_returns_polite_refusal(self):
        with mock.patch.object(action_dispatch, "_extract",
                               return_value={"action": "set", "title": "stretch"}):
            action = action_dispatch.detect_action("remind me to stretch in 5 min", "s")
        self.assertEqual(action["tool"], "disabled")
        self.assertEqual(action["blocked"], "reminders")
        self.assertIn("can't set or manage reminders", action["ask"])
        self.assertNotIn("Sir", action["ask"])

    def test_email_and_documents_refused(self):
        cases = [
            ("_extract_email", "email this to bob@example.com", "email",
             {"tool": "email", "action": "draft"}),
            ("_extract_docs", "save this document", "documents",
             {"tool": "documents", "action": "save"}),
        ]
        for fn, msg, tool, extracted in cases:
            with mock.patch.object(action_dispatch, fn, return_value=dict(extracted)):
                action = action_dispatch.detect_action(msg, "s")
            self.assertEqual((action["tool"], action["blocked"]), ("disabled", tool))

    def test_vault_disabled(self):
        from core.vault import vault_store
        with self.assertRaises(RuntimeError):
            vault_store("hunter2")
        self.assertFalse(os.path.exists(self.vault))

    def test_brainstorm_and_personal_teachers_not_routable(self):
        for domain in ("brainstorm", "jobs", "leadgen", "life", "spirit"):
            self.assertNotIn(domain, po._PUBLIC_TEACHERS)

    def test_chat_refusal_end_to_end_makes_no_book_call(self):
        with mock.patch.object(action_dispatch, "_extract",
                               return_value={"action": "set", "title": "stretch"}):
            r = self.client().post("/chat", json={"message": "remind me to stretch"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("public demo", r.json()["response"])
        self.assertEqual(r.json()["provider_used"], "none")
        self.call_book.assert_not_called()

    def test_no_token_or_personal_routes(self):
        c = self.client()
        for path in ("/token", "/conversations/sessions", "/dashboard/overview",
                     "/traces", "/health/deep"):
            self.assertEqual(c.get(path).status_code, 404, path)
        self.assertEqual(c.get("/health").status_code, 200)


class RateLimitTest(PublicModeCase):

    def setUp(self):
        super().setUp()
        self.now = [1000.0]
        p = mock.patch.object(rate_limit, "_clock", lambda: self.now[0])
        p.start()
        self.addCleanup(p.stop)

    def test_ipv6_shares_quota_per_64(self):
        for i in range(5):
            self.assertTrue(rate_limit.check(f"2001:db8:1:2::{i + 1}")[0])
        self.assertFalse(rate_limit.check("2001:db8:1:2:ffff::9")[0])
        self.assertTrue(rate_limit.check("2001:db8:1:3::1")[0])  # different /64

    def test_five_per_day_with_remaining(self):
        for left in (4, 3, 2, 1, 0):
            allowed, remaining, _, _ = rate_limit.check("1.2.3.4")
            self.assertTrue(allowed)
            self.assertEqual(remaining, left)
        allowed, remaining, retry, msg = rate_limit.check("1.2.3.4")
        self.assertFalse(allowed)
        self.assertEqual(remaining, 0)
        self.assertGreater(retry, 0)
        self.assertIn("free AIOS messages", msg)
        self.assertTrue(rate_limit.check("5.6.7.8")[0])  # per IP
        self.now[0] += 24 * 60 * 60 + 1
        self.assertTrue(rate_limit.check("1.2.3.4")[0])

    def test_refused_requests_are_not_counted(self):
        for _ in range(5):
            rate_limit.check("1.2.3.4")
        self.now[0] += 60
        for _ in range(10):
            self.assertFalse(rate_limit.check("1.2.3.4")[0])
        # The wait is measured from the FIRST hit, not extended by the refusals.
        self.now[0] += 24 * 60 * 60 - 60 + 1
        self.assertTrue(rate_limit.check("1.2.3.4")[0])

    def test_endpoint_reports_remaining_then_429(self):
        c = self.client()
        self.assertEqual(c.get("/quota").json(), {"remaining": 5, "limit": 5})
        for left in (4, 3, 2, 1, 0):
            r = c.post("/chat", json={"message": "hi"})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["remaining"], left)
        r = c.post("/chat", json={"message": "hi"})
        self.assertEqual(r.status_code, 429)
        self.assertEqual(r.json()["remaining"], 0)
        self.assertIn("come back tomorrow", r.json()["detail"])
        self.assertIn("Retry-After", r.headers)
        self.assertEqual(c.get("/quota").json()["remaining"], 0)

    def test_invalid_input_does_not_spend_quota(self):
        c = self.client()
        c.post("/chat", json={"message": ""})
        c.post("/chat", json={"message": "x" * (Config.PUBLIC_MAX_INPUT_CHARS + 1)})
        self.assertEqual(c.get("/quota").json()["remaining"], 5)


class NoPersistTest(PublicModeCase):

    def _assert_nothing_on_disk(self):
        for path in (self.sqlite, self.chroma, self.vault):
            self.assertFalse(os.path.exists(path), path)

    def test_full_turn_writes_nothing(self):
        r = self.client().post("/chat", json={"message": "capital of France?",
                                              "trace": True})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["response"], "Paris.")  # ", Sir" scrubbed
        self.assertTrue(body["session_id"])
        self._assert_nothing_on_disk()

    def test_persistence_entry_points_are_noops(self):
        from core.memory import save_message
        from core.profile import store_fact, get_relevant_facts
        from core.extractor import extract_and_store
        from core.outcomes import log_outcome
        from core.trace import Trace
        save_message("s", "user", "hello")
        store_fact("likes tea")
        self.assertEqual(get_relevant_facts("tea"), [])
        self.assertIsNone(extract_and_store("hi", "hello"))
        log_outcome(domain="none", book_used="groq", ensemble=False,
                    self_check_passed=True, attempts=1, tokens=1, latency_ms=1,
                    complexity="simple", session_id="s")
        self.assertFalse(Trace("s").flush())
        self._assert_nothing_on_disk()

    def test_history_in_memory_and_continues_session(self):
        c = self.client()
        sid = c.post("/chat", json={"message": "capital of France?"}).json()["session_id"]
        r = c.post("/chat", json={"message": "and Spain?", "session_id": sid})
        self.assertEqual(r.json()["session_id"], sid)
        prompt = self.call_book.call_args_list[-1].args[0]
        self.assertIn("Visitor: capital of France?", prompt)
        self.assertEqual(len(public_session.get_history(sid)), 4)
        self._assert_nothing_on_disk()

    def test_unknown_session_id_is_replaced(self):
        r = self.client().post("/chat", json={"message": "hi", "session_id": "guess"})
        self.assertNotEqual(r.json()["session_id"], "guess")

    def test_session_expires_after_30_minutes_idle(self):
        now = [0.0]
        with mock.patch.object(public_session, "_clock", lambda: now[0]):
            sid = public_session.resolve(None)
            public_session.append_turn(sid, "hi", "hello")
            now[0] += 29 * 60
            self.assertEqual(public_session.resolve(sid), sid)  # touch resets idle
            now[0] += 29 * 60
            self.assertEqual(len(public_session.get_history(sid)), 2)
            now[0] += 30 * 60 + 1
            self.assertEqual(public_session.get_history(sid), [])
            self.assertNotEqual(public_session.resolve(sid), sid)


class PersonaAndInputTest(PublicModeCase):

    def test_prompt_is_public_persona_only(self):
        prompt = po._build_prompt({"message": "who are you?", "history": []}, None, False)
        self.assertIn("built by Prithvi", prompt)
        self.assertNotIn("Donna", prompt)  # personal SYSTEM_PROMPT never included

    def test_persona_echo_is_replaced(self):
        leaked = "Sure: " + po.PUBLIC_PERSONA.split("\n\n")[0]
        self.assertEqual(po._scrub(leaked), po._REFUSE_CONFIG)

    def test_input_cap(self):
        c = self.client()
        r = c.post("/chat", json={"message": "x" * (Config.PUBLIC_MAX_INPUT_CHARS + 1)})
        self.assertEqual(r.status_code, 400)
        self.assertIn("under 2000 characters", r.json()["detail"])
        self.call_book.assert_not_called()

    def test_trace_field_optional(self):
        c = self.client()
        self.assertNotIn("trace", c.post("/chat", json={"message": "q?"}).json())
        trace = c.post("/chat", json={"message": "q?", "trace": True}).json()["trace"]
        self.assertEqual(trace["triage"]["domain"], "none")
        self.assertEqual(trace["book"], "groq")
        self.assertIn("triage", [s["stage"] for s in trace["stages"]])
        self.assertTrue(all("ms" in s for s in trace["stages"]))


class StartupGuardTest(unittest.TestCase):

    def test_refuses_personal_default_paths(self):
        from config import _PERSONAL_SQLITE, _PERSONAL_CHROMA
        tmp = tempfile.gettempdir()
        with mock.patch.object(Config, "PUBLIC_MODE", True):
            for name, bad in (("SQLITE_PATH", _PERSONAL_SQLITE),
                              ("CHROMA_PATH", _PERSONAL_CHROMA),
                              ("VAULT_PATH", _PERSONAL_SQLITE)):
                good = {"SQLITE_PATH": os.path.join(tmp, "p.db"),
                        "CHROMA_PATH": os.path.join(tmp, "pc"),
                        "VAULT_PATH": os.path.join(tmp, "pv.db")}
                good[name] = bad
                with mock.patch.multiple(Config, **good):
                    with self.assertRaises(RuntimeError, msg=name):
                        Config.validate()

    def test_accepts_public_paths(self):
        tmp = tempfile.gettempdir()
        with mock.patch.multiple(Config, PUBLIC_MODE=True,
                                 SQLITE_PATH=os.path.join(tmp, "p.db"),
                                 CHROMA_PATH=os.path.join(tmp, "pc"),
                                 VAULT_PATH=os.path.join(tmp, "pv.db")):
            Config.validate()


if __name__ == "__main__":
    unittest.main()
