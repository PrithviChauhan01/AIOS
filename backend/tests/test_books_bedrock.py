"""Book routing with Kimi K2.5 on Bedrock as the main book.

Order is bedrock_kimi → groq → cerebras → ollama; Bedrock is enabled only by a
startup probe; privacy (secret → ollama only) is untouched. Everything is mocked —
no AWS, no provider calls.

Run from backend/:   venv\\Scripts\\python -m unittest discover -s tests -v
"""

import asyncio
import contextlib
import io
import unittest
from unittest import mock

from botocore.exceptions import ClientError

from core import books
from core import provider_health as health

_ALL = {"bedrock_kimi", "groq", "groq_fast", "cerebras", "ollama"}


def _converse_reply(text, reasoning=None):
    content = [{"reasoningContent": {"reasoningText": {"text": reasoning}}}] if reasoning else []
    content.append({"text": text})
    return {"output": {"message": {"role": "assistant", "content": content}},
            "usage": {"inputTokens": 7, "outputTokens": 5, "totalTokens": 12}}


class _Clean(unittest.TestCase):

    def setUp(self):
        health._STATES.clear()
        self.addCleanup(health._STATES.clear)
        for p in (mock.patch.object(books, "_CONFIGURED", set(_ALL)),
                  mock.patch.dict(books._COOLDOWN, clear=True)):
            p.start()
            self.addCleanup(p.stop)


class OrderTest(_Clean):

    def test_tier_order_is_kimi_groq_cerebras_ollama(self):
        for tier in ("fast", "strong", "frontier"):
            self.assertEqual(books.select_book_for_tier(tier, "public"),
                             ["bedrock_kimi", "groq", "cerebras", "ollama"])

    def test_select_book_leads_with_main_chain_and_ends_with_ollama(self):
        order = books.select_book({"reasoning": "good"}, "public")
        self.assertEqual(order[:3], ["bedrock_kimi", "groq", "cerebras"])
        self.assertEqual(order[-1], "ollama")

    def test_disabled_bedrock_starts_at_groq(self):
        books._CONFIGURED.discard("bedrock_kimi")
        self.assertEqual(books.select_book_for_tier("strong", "public"),
                         ["groq", "cerebras", "ollama"])

    def test_privacy_unchanged_secret_is_ollama_only(self):
        self.assertEqual(books.select_book_for_tier("strong", "secret"), ["ollama"])
        self.assertEqual(books.select_book({"reasoning": "good"}, "secret"), ["ollama"])
        self.assertEqual(books.select_fast_book("secret"), ["ollama"])

    def test_ensemble_never_pairs_with_the_local_tail(self):
        self.assertEqual(books.select_ensemble_books("strong", "public"),
                         ["bedrock_kimi", "groq"])

    def test_open_bedrock_circuit_is_skipped_and_logged(self):
        health.record_fail("bedrock", 10, 403)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            order = books.select_book_for_tier("fast", "public")
        self.assertEqual(order[0], "groq")
        self.assertIn("skipping bedrock_kimi — bedrock circuit open", out.getvalue())


class ProbeTest(_Clean):

    def setUp(self):
        super().setUp()
        p = mock.patch.object(books, "_bedrock_probe_result", None)
        p.start()
        self.addCleanup(p.stop)

    def _probe(self, creds, converse=None):
        session = mock.Mock()
        session.get_credentials.return_value = creds
        out = io.StringIO()
        with mock.patch("boto3.Session", return_value=session), \
                mock.patch.object(books, "_bedrock_converse",
                                  converse or mock.Mock(return_value=_converse_reply("pong"))) as conv, \
                contextlib.redirect_stdout(out):
            first = books.probe_bedrock()
            second = books.probe_bedrock()  # idempotent: no second probe, no second log
        self.assertEqual(first, second)
        return first, out.getvalue(), conv

    def test_no_credentials_disables_without_calling_bedrock(self):
        ok, log, conv = self._probe(creds=None)
        self.assertFalse(ok)
        self.assertNotIn("bedrock_kimi", books._CONFIGURED)
        conv.assert_not_called()
        self.assertEqual(log.count("bedrock_kimi disabled"), 1)
        self.assertIn("no AWS credentials", log)

    def test_failed_probe_disables(self):
        boom = mock.Mock(side_effect=ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "no model access"},
             "ResponseMetadata": {"HTTPStatusCode": 403}}, "Converse"))
        ok, log, _ = self._probe(creds=object(), converse=boom)
        self.assertFalse(ok)
        self.assertNotIn("bedrock_kimi", books._CONFIGURED)
        self.assertEqual(log.count("bedrock_kimi disabled"), 1)
        self.assertIn("AccessDeniedException", log)

    def test_successful_probe_enables(self):
        books._CONFIGURED.discard("bedrock_kimi")
        ok, log, conv = self._probe(creds=object())
        self.assertTrue(ok)
        self.assertIn("bedrock_kimi", books._CONFIGURED)
        conv.assert_called_once()
        self.assertIn("bedrock_kimi enabled", log)
        # The probe is a diagnostic: it must not touch the circuit breaker.
        self.assertNotIn("bedrock", health._STATES)


class CallTest(_Clean):

    def test_converse_text_only_reasoning_dropped(self):
        conv = mock.Mock(return_value=_converse_reply("Good evening, Sir.", reasoning="hmm"))
        with mock.patch.object(books, "_bedrock_converse", conv):
            out = asyncio.run(books.call_book("hi", "bedrock_kimi", max_tokens=150))
        self.assertEqual(out["raw_text"], "Good evening, Sir.")
        self.assertEqual((out["tokens"], out["tokens_in"], out["tokens_out"]), (12, 7, 5))
        self.assertEqual(out["book_used"], "bedrock_kimi")
        # 150-token cap floored so reasoning can't starve the answer.
        self.assertEqual(conv.call_args.args[1], books._KIMI_MIN_TOKENS)
        self.assertEqual(health.circuit_of("bedrock"), health.CLOSED)

    def test_client_error_falls_back_to_groq_and_logs_why(self):
        denied = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "denied"},
                              "ResponseMetadata": {"HTTPStatusCode": 403}}, "Converse")
        groq = mock.Mock(return_value=("Sir.", 3, 1, 2))
        out = io.StringIO()

        async def chain():
            # The callers' loop (cognition / teachers): try each book, fall through.
            for book in books.select_book_for_tier("fast", "public"):
                try:
                    return await books.call_book("hi", book)
                except Exception:
                    continue

        with mock.patch.object(books, "_bedrock_converse", side_effect=denied), \
                mock.patch.dict(books._DISPATCH, {"groq": groq}), \
                contextlib.redirect_stdout(out):
            result = asyncio.run(chain())

        self.assertEqual(result["book_used"], "groq")
        log = out.getvalue()
        self.assertIn("bedrock_kimi failed (BedrockError: AccessDeniedException: denied)"
                      " → falling back to next book", log)
        self.assertIn("answered by groq", log)
        # 403 surfaced as a real status → the circuit opened on the first failure.
        self.assertEqual(health.circuit_of("bedrock"), health.OPEN)


if __name__ == "__main__":
    unittest.main()
