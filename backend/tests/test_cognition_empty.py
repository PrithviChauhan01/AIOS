"""Personal cognition never returns an empty reply (the public-mode fix, extended).

A reasoning model can spend a small output cap thinking and return no content. That
must count as a failed book (next book tried); if every book is empty the reply is an
honest fallback — on the action path one that says whether the action succeeded.

Run from backend/:   venv\\Scripts\\python -m unittest discover -s tests -v
"""

import asyncio
import unittest
from unittest import mock

from core import cognition


def _book(text, book):
    return {"raw_text": text, "book_used": book, "tokens": 10,
            "tokens_in": 5, "tokens_out": 5}


class CognitionEmptyReplyTest(unittest.TestCase):

    def setUp(self):
        patches = [
            mock.patch.object(cognition, "select_fast_book", return_value=["groq_fast", "groq"]),
            mock.patch.object(cognition, "select_book_for_tier", return_value=["groq_fast", "groq"]),
            mock.patch.object(cognition, "select_book", return_value=["groq_fast", "groq"]),
            mock.patch.object(cognition, "extract_and_store", return_value=None),
            mock.patch.object(cognition, "save_message", return_value=None),
            # No real memory store: a clean checkout has no Chroma collection.
            mock.patch.object(cognition, "get_relevant_facts", return_value=[]),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def run_pass(self, call_book, ctx_extra=None):
        ctx = {"message": "create invoice for Verve Media", "session_id": "t",
               "complexity": "trivial", "sensitivity": "public", "trace_id": "tr1",
               **(ctx_extra or {})}
        with mock.patch.object(cognition, "call_book", call_book):
            return asyncio.run(cognition.cognition_pass(ctx, "ACTION EXECUTED — done."))

    def test_empty_reply_falls_through_to_next_book(self):
        call = mock.AsyncMock(side_effect=[_book(None, "groq_fast"),
                                           _book("Invoice drafted, Sir.\n[mood: neutral]", "groq")])
        out = self.run_pass(call)
        self.assertEqual(out["response"], "Invoice drafted, Sir.")
        self.assertEqual(out["provider_used"], "groq")
        self.assertEqual(call.await_count, 2)

    def test_all_empty_on_successful_action_says_it_is_done(self):
        call = mock.AsyncMock(return_value=_book("[mood: neutral]", "groq_fast"))
        out = self.run_pass(call, {"action": "create_invoice",
                                   "action_result": {"ok": True, "trace_id": "tr1"}})
        self.assertTrue(out["response"].strip())
        self.assertIn("done", out["response"])

    def test_all_empty_on_failed_action_says_it_did_not_go_through(self):
        call = mock.AsyncMock(return_value=_book("", "groq_fast"))
        out = self.run_pass(call, {"action": "create_invoice",
                                   "action_result": {"ok": False, "trace_id": "tr1"}})
        self.assertIn("didn't go through", out["response"])

    def test_all_empty_without_action_is_never_blank(self):
        call = mock.AsyncMock(return_value=_book(None, "groq_fast"))
        self.assertTrue(self.run_pass(call)["response"].strip())


if __name__ == "__main__":
    unittest.main()
