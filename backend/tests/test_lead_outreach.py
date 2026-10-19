"""Lead outreach (Slice B3): draft → mailer confirm gate → real send → lead + reminder.

Run from backend/:   venv\\Scripts\\python -m unittest discover -s tests -v

No network: composition (compose_with) and SMTP (_send_smtp) are patched.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import datetime
from unittest import mock

from config import Config
from agents.leadgen import parse_outreach_command, parse_pipeline_command
from core import action_dispatch
from tools import leads, mailer

_NOW = datetime(2026, 10, 19, 10, 0)  # a Monday → +3 days = 2026-10-22

_VERVE_RESEARCH = json.dumps({
    "email": "info@vervemedia.co.in", "website": "https://vervemedia.co.in/video-editing-services",
    "address": "not found", "phone": "not found",
    "source_urls": ["https://vervemedia.co.in/video-editing-services"],
    "snippet": "Verve Media offers video editing services in Mumbai for brands."})
_SWASHRI_RESEARCH = json.dumps({
    "email": "not found", "website": "https://swashrifilmstudio.com/video-editing",
    "address": "Ghatkopar West, Mumbai", "phone": "not found",
    "source_urls": ["https://swashrifilmstudio.com/video-editing"],
    "snippet": "At SwaShri Film Studio, Ghatkopar West, Mumbai, we offer video editing."})

_GOOD_DRAFT = ("Verve Media x Studio OS",
               "Hi Verve Media team,\n\nI saw you offer video editing services in Mumbai. "
               "I'm building Studio OS, software for creative studios. Open to a short call?"
               "\n\nPrithvi")


class OutreachCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aios_outreach_")
        self.db = os.path.join(self.tmp, "aios.db")
        for p in (mock.patch.object(Config, "SQLITE_PATH", self.db),
                  mock.patch.object(Config, "PUBLIC_MODE", False),
                  mock.patch.object(Config, "STUDIO_OS_PITCH",
                                    "Studio OS, software for creative studios"),
                  mock.patch.object(leads, "_now", return_value=_NOW)):
            p.start()
            self.addCleanup(p.stop)
        from db.sqlite_init import init_sqlite
        init_sqlite()
        conn = sqlite3.connect(self.db)
        conn.executemany(
            "INSERT INTO leads (studio_name, location, contact, research) VALUES (?, ?, ?, ?)",
            [("Verve Media", "Mumbai", "info@vervemedia.co.in · https://vervemedia.co.in",
              _VERVE_RESEARCH),
             ("SwaShri Film Studio", "Mumbai", "https://swashrifilmstudio.com/video-editing",
              _SWASHRI_RESEARCH)])
        conn.commit()
        conn.close()
        leads._SCHEMA_READY.discard(leads._db_path())
        mailer._PENDING.clear()
        action_dispatch._AWAITING_EMAIL.clear()
        action_dispatch._idem_store.clear()
        # A staged draft owns the session's next turn — never leak one into other tests.
        self.addCleanup(mailer._PENDING.clear)
        self.addCleanup(action_dispatch._idem_store.clear)
        self.compose = mock.patch("tools.mailer.compose_with", return_value=_GOOD_DRAFT)
        self.compose_mock = self.compose.start()
        self.addCleanup(self.compose.stop)
        self.smtp = mock.patch.object(
            mailer, "_send_smtp",
            side_effect=lambda to, subject, body: {"ok": True, "to": to, "subject": subject})
        self.smtp_mock = self.smtp.start()
        self.addCleanup(self.smtp.stop)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def turn(self, message):
        action = action_dispatch.detect_action(message, "s1")
        self.assertIsNotNone(action, message)
        res = action_dispatch.run_action(action)
        return action, res, action_dispatch.action_material(action, res)

    def lead(self, name):
        return next(r for r in leads.list_leads() if r["studio_name"] == name)

    def reminders(self):
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute("SELECT title, due_at FROM reminders").fetchall()
        finally:
            conn.close()


class ParserTest(unittest.TestCase):

    def test_outreach_and_follow_up_phrases(self):
        p = parse_outreach_command
        self.assertEqual(p("draft outreach to Verve Media"),
                         {"kind": "outreach", "name": "Verve Media"})
        self.assertEqual(p("Draft an outreach email to 8K."), {"kind": "outreach", "name": "8K"})
        self.assertEqual(p("write a cold email for SwaShri Film Studio")["kind"], "outreach")
        self.assertEqual(p("draft follow-up to Verve Media"),
                         {"kind": "follow_up", "name": "Verve Media"})
        self.assertEqual(p("draft a follow up email to 8K")["kind"], "follow_up")
        # not outreach, and the pipeline parser leaves draft commands alone
        self.assertIsNone(p("draft an email to my landlord"))
        self.assertIsNone(p("follow up with 8K tomorrow"))
        self.assertIsNone(parse_pipeline_command("draft follow-up to Verve Media"))


class DraftTest(OutreachCase):

    def test_outreach_draft_is_staged_not_sent(self):
        action, res, material = self.turn("draft outreach to Verve Media")
        self.assertEqual((action["tool"], action["action"]), ("email", "lead_draft"))
        self.assertTrue(res["ok"])
        self.assertEqual(res["to"], "info@vervemedia.co.in")
        self.assertEqual(res["subject"], "Verve Media x Studio OS")
        self.assertTrue(res["body"].rstrip().endswith("Prithvi"))
        self.assertTrue(material.startswith("ACTION STAGED (NOT SENT)"))
        self.smtp_mock.assert_not_called()
        self.assertEqual(mailer.get_pending("s1")["lead_id"], self.lead("Verve Media")["id"])
        self.assertEqual(self.lead("Verve Media")["stage"], "found")  # nothing updated yet

    def test_composer_gets_only_saved_facts_and_pitch(self):
        self.turn("draft outreach to Verve Media")
        system, user = self.compose_mock.call_args.args
        self.assertIn("Use ONLY the facts provided", system)
        self.assertIn("video editing services in Mumbai", user)   # the source snippet
        self.assertIn("Studio OS, software for creative studios", user)

    def test_invented_detail_falls_back_to_template(self):
        self.compose_mock.return_value = (
            "Hi", "Hi Verve,\n\nLoved your work for Nike — call me on +91 99999 00000 or see "
            "studio-os.io.\n\nPrithvi")
        _, res, _ = self.turn("draft outreach to Verve Media")
        self.assertNotIn("99999", res["body"])
        self.assertNotIn("studio-os.io", res["body"])
        self.assertIn("Studio OS, software for creative studios", res["body"])
        self.assertTrue(res["body"].rstrip().endswith("Prithvi"))

    def test_compose_unavailable_falls_back_and_is_signed(self):
        self.compose_mock.return_value = None
        _, res, _ = self.turn("draft outreach to SwaShri Film Studio")
        self.assertTrue(res["ok"])
        self.assertIn("SwaShri Film Studio", res["body"])
        self.assertTrue(res["body"].rstrip().endswith("Prithvi"))

    def test_unknown_lead_stages_nothing(self):
        _, res, material = self.turn("draft outreach to Nobody Studio")
        self.assertFalse(res["ok"])
        self.assertFalse(mailer.has_pending("s1"))
        self.assertTrue(material.startswith("ACTION NOT DONE"))


class MissingAddressTest(OutreachCase):

    def test_drafts_anyway_and_says_address_missing(self):
        _, res, material = self.turn("draft outreach to SwaShri Film Studio")
        self.assertTrue(res["ok"])
        self.assertTrue(res["missing_address"])
        self.assertIn("NO email address on file", material)
        self.assertIn("(missing — no email on file)", material)
        self.assertEqual(mailer.get_pending("s1")["to"], "")

    def test_send_without_address_is_refused_and_updates_nothing(self):
        self.turn("draft outreach to SwaShri Film Studio")
        _, res, material = self.turn("send it")
        self.assertFalse(res["ok"])
        self.assertEqual(res["error"], "no_recipient")
        self.smtp_mock.assert_not_called()
        self.assertEqual(self.lead("SwaShri Film Studio")["stage"], "found")
        self.assertTrue(material.startswith("ACTION NOT DONE"))

    def test_address_reply_fills_it_then_needs_explicit_send(self):
        self.turn("draft outreach to SwaShri Film Studio")
        action, res, _ = self.turn("use hello@swashrifilmstudio.com")
        self.assertEqual(action["action"], "set_to")
        self.assertEqual(res["to"], "hello@swashrifilmstudio.com")
        self.smtp_mock.assert_not_called()               # re-shown, not sent
        _, res, _ = self.turn("send it")
        self.assertTrue(res["sent"])
        self.smtp_mock.assert_called_once()
        self.assertEqual(self.smtp_mock.call_args.args[0], "hello@swashrifilmstudio.com")


class SingleRecipientTest(OutreachCase):

    def test_two_addresses_in_reply_are_refused(self):
        self.turn("draft outreach to SwaShri Film Studio")
        _, res, _ = self.turn("send to a@x.com, b@y.com")
        self.assertFalse(res["ok"])
        self.assertEqual(mailer.get_pending("s1")["to"], "")
        self.smtp_mock.assert_not_called()

    def test_confirm_send_refuses_multi_recipient_draft(self):
        mailer.stage("s1", {"to": "a@x.com;b@y.com", "subject": "s", "body": "b"})
        res = mailer.confirm_send("s1")
        self.assertFalse(res["ok"])
        self.assertEqual(res["error"], "multiple_recipients")
        self.smtp_mock.assert_not_called()


class SendTest(OutreachCase):

    def test_successful_send_updates_lead_and_sets_reminder(self):
        self.turn("draft outreach to Verve Media")
        _, res, material = self.turn("send it")
        self.assertTrue(res["sent"])
        self.smtp_mock.assert_called_once()
        lead = self.lead("Verve Media")
        self.assertEqual(lead["stage"], "contacted")
        self.assertEqual(lead["last_contact_at"], "2026-10-19 10:00:00")
        self.assertEqual(lead["next_follow_up_at"], "2026-10-22")
        self.assertIn("outreach sent: Verve Media x Studio OS", lead["notes"])
        self.assertEqual(self.reminders(), [
            ("Follow up with Verve Media (re: Verve Media x Studio OS)", "2026-10-22 10:00:00")])
        self.assertIn("is now 'contacted', next follow-up 2026-10-22", material)
        self.assertIn("reminder is set for 2026-10-22 10:00:00", material)
        self.assertFalse(mailer.has_pending("s1"))

    def test_failed_send_updates_nothing(self):
        self.smtp_mock.side_effect = lambda to, subject, body: {"ok": False, "error": "SMTP down"}
        self.turn("draft outreach to Verve Media")
        _, res, material = self.turn("send it")
        self.assertFalse(res["ok"])
        lead = self.lead("Verve Media")
        self.assertEqual(lead["stage"], "found")
        self.assertIsNone(lead["last_contact_at"])
        self.assertIsNone(lead["next_follow_up_at"])
        self.assertIsNone(lead["notes"])
        self.assertEqual(self.reminders(), [])
        self.assertTrue(mailer.has_pending("s1"))          # kept for retry/cancel
        self.assertTrue(material.startswith("ACTION NOT DONE"))

    def test_cancel_updates_nothing(self):
        self.turn("draft outreach to Verve Media")
        self.turn("cancel")
        self.smtp_mock.assert_not_called()
        self.assertIsNone(self.lead("Verve Media")["notes"])

    def test_lead_further_along_is_not_demoted(self):
        leads.set_stage("Verve Media", "demo")
        self.turn("draft outreach to Verve Media")
        self.turn("send it")
        self.assertEqual(self.lead("Verve Media")["stage"], "demo")

    def test_ordinary_email_send_touches_no_lead(self):
        mailer.stage("s1", {"to": "friend@example.com", "subject": "hi", "body": "b"})
        _, res, _ = self.turn("send it")
        self.assertTrue(res["sent"])
        self.assertNotIn("lead_update", res)
        self.assertEqual(self.reminders(), [])


class FollowUpTest(OutreachCase):

    def test_follow_up_needs_a_previous_email(self):
        _, res, material = self.turn("draft follow-up to Verve Media")
        self.assertFalse(res["ok"])
        self.assertEqual(res["needs"], "previous_email")
        self.assertFalse(mailer.has_pending("s1"))
        self.assertTrue(material.startswith("ACTION NOT DONE"))

    def test_follow_up_references_previous_email_and_records_send(self):
        self.turn("draft outreach to Verve Media")
        self.turn("send it")
        action_dispatch._idem_store.clear()
        self.compose_mock.return_value = (
            "Re: Verve Media x Studio OS",
            "Hi Verve Media team,\n\nFollowing up on my last note. Happy to chat.\n\nPrithvi")
        _, res, _ = self.turn("draft follow-up to Verve Media")
        self.assertEqual(res["subject"], "Re: Verve Media x Studio OS")
        _, user = self.compose_mock.call_args.args
        self.assertIn("Verve Media x Studio OS", user)       # previous subject from notes
        self.turn("send it")
        lead = self.lead("Verve Media")
        self.assertIn("follow-up sent: Re: Verve Media x Studio OS", lead["notes"])
        self.assertEqual(lead["stage"], "contacted")
        self.assertEqual(len(self.reminders()), 2)

    def test_follow_up_fallback_is_shorter_and_cites_subject(self):
        leads.record_send(self.lead("Verve Media")["id"], "Verve Media x Studio OS")
        self.compose_mock.return_value = None
        _, res, _ = self.turn("draft follow-up to Verve Media")
        self.assertEqual(res["subject"], "Re: Verve Media x Studio OS")
        self.assertIn('"Verve Media x Studio OS"', res["body"])


class PublicModeTest(OutreachCase):

    def test_public_mode_refuses(self):
        with mock.patch.object(Config, "PUBLIC_MODE", True):
            action = action_dispatch.detect_action("draft outreach to Verve Media", "s1")
            res = action_dispatch.run_action(action)
        self.assertEqual(action["tool"], "disabled")
        self.assertFalse(res["ok"])
        self.assertFalse(mailer.has_pending("s1"))
        self.compose_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
