"""Sales pipeline on the leads table (Slice B2): migration, each action, NL parsing,
action dispatch end to end, owner-only.

Run from backend/:   venv\\Scripts\\python -m unittest discover -s tests -v

No network: every command here is parsed deterministically and hits a temp SQLite db.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import date, datetime
from unittest import mock

from config import Config
from agents.leadgen import parse_pipeline_command
from core import action_dispatch
from tools import leads

_NOW = datetime(2026, 10, 19, 10, 0)  # a Monday

# The leads table exactly as it shipped before B2 — no pipeline columns.
_OLD_LEADS_DDL = """CREATE TABLE leads (
    id INTEGER PRIMARY KEY AUTOINCREMENT, studio_name TEXT NOT NULL, location TEXT,
    contact TEXT, research TEXT, outreach_sent INTEGER DEFAULT 0, response TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP, follow_up_at DATETIME)"""


class PipelineCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aios_pipeline_")
        self.db = os.path.join(self.tmp, "aios.db")
        conn = sqlite3.connect(self.db)
        conn.execute(_OLD_LEADS_DDL)
        conn.executemany(
            "INSERT INTO leads (studio_name, location, contact, research) VALUES (?, ?, ?, ?)",
            [("Verve Media", "Mumbai", "info@vervemedia.co.in", '{"source_urls": ["u1"]}'),
             ("8K Entertainment", "Mumbai", "+918693838888", '{"source_urls": ["u2"]}'),
             ("Brand Beavers", "Mumbai", "+91 9324159750", None)])
        conn.commit()
        conn.close()
        for p in (mock.patch.object(Config, "SQLITE_PATH", self.db),
                  mock.patch.object(Config, "PUBLIC_MODE", False),
                  mock.patch.object(leads, "_now", return_value=_NOW)):
            p.start()
            self.addCleanup(p.stop)
        leads._SCHEMA_READY.discard(leads._db_path())
        action_dispatch._idem_store.clear()
        from tools import mailer
        mailer._PENDING.clear()  # a staged draft would own this session's turns
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def lead(self, name):
        return next(r for r in leads.list_leads() if r["studio_name"] == name)


class MigrationTest(PipelineCase):

    def test_existing_rows_kept_and_default_to_found(self):
        rows = leads.list_leads()
        self.assertEqual(len(rows), 3)
        verve = self.lead("Verve Media")
        self.assertEqual(verve["contact"], "info@vervemedia.co.in")
        self.assertEqual(verve["research"], '{"source_urls": ["u1"]}')
        self.assertTrue(all(r["stage"] == "found" for r in rows))
        self.assertTrue(all(r["next_follow_up_at"] is None and r["notes"] is None for r in rows))

    def test_migration_is_idempotent(self):
        from db.sqlite_init import ensure_leads_schema
        conn = sqlite3.connect(self.db)
        ensure_leads_schema(conn)
        ensure_leads_schema(conn)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(leads)")]
        conn.close()
        for c in ("stage", "last_contact_at", "next_follow_up_at", "notes"):
            self.assertEqual(cols.count(c), 1)

    def test_new_leads_start_in_found(self):
        leads.save_leads([{"studio_name": "SwaShri Film Studio", "location": "Mumbai"}])
        self.assertEqual(self.lead("SwaShri Film Studio")["stage"], "found")


class ActionTest(PipelineCase):

    def test_contacted_auto_sets_follow_up_plus_3_days(self):
        res = leads.set_stage("Verve Media", "contacted")
        self.assertTrue(res["ok"])
        self.assertEqual(res["from_stage"], "found")
        self.assertEqual(res["lead"]["stage"], "contacted")
        self.assertEqual(res["lead"]["next_follow_up_at"], "2026-10-22")
        self.assertEqual(res["lead"]["last_contact_at"], "2026-10-19 10:00:00")

    def test_contacted_with_given_date_uses_it(self):
        res = leads.set_stage("Verve Media", "contacted", follow_up="Friday")
        self.assertEqual(res["lead"]["next_follow_up_at"], "2026-10-23")

    def test_won_and_lost_clear_follow_up(self):
        leads.set_stage("Verve Media", "contacted")
        self.assertIsNone(leads.set_stage("Verve Media", "won")["lead"]["next_follow_up_at"])
        leads.set_follow_up("8K", "tomorrow")
        self.assertIsNone(leads.set_stage("8K", "lost")["lead"]["next_follow_up_at"])

    def test_invalid_stage_and_unknown_or_ambiguous_lead_change_nothing(self):
        self.assertEqual(leads.set_stage("Verve Media", "pending")["needs"], "stage")
        self.assertIn("no lead called", leads.set_stage("Nobody Studio", "won")["ask"])
        leads.save_leads([{"studio_name": "Verve Media", "location": "Pune"}])
        res = leads.set_stage("Verve Media", "won")
        self.assertIn("more than one lead", res["ask"])
        self.assertTrue(all(r["stage"] == "found" for r in leads.list_leads()))

    def test_short_name_resolves(self):
        self.assertEqual(leads.add_note("8K", "x")["lead"]["studio_name"], "8K Entertainment")

    def test_notes_are_append_only_log(self):
        leads.add_note("8K", "asked for pricing")
        leads.set_stage("8K", "replied")
        leads.add_note("8K", "sent rate card")
        notes = self.lead("8K Entertainment")["notes"].splitlines()
        self.assertEqual(notes, [
            "[2026-10-19 10:00] asked for pricing",
            "[2026-10-19 10:00] stage: found → replied",
            "[2026-10-19 10:00] sent rate card",
        ])
        self.assertEqual(leads.add_note("8K", "  ")["needs"], "note")

    def test_set_follow_up(self):
        res = leads.set_follow_up("Brand Beavers", "in 5 days")
        self.assertEqual(res["lead"]["next_follow_up_at"], "2026-10-24")
        self.assertEqual(leads.set_follow_up("Brand Beavers", "blah blah")["needs"], "date")

    def test_follow_up_refused_on_closed_lead(self):
        leads.set_stage("Verve Media", "won")
        res = leads.set_follow_up("Verve Media", "tomorrow")
        self.assertFalse(res["ok"])
        self.assertIsNone(self.lead("Verve Media")["next_follow_up_at"])

    def test_list_by_stage(self):
        leads.set_stage("Verve Media", "contacted")
        leads.set_stage("8K", "demo")
        self.assertEqual([r["studio_name"] for r in leads.list_by_stage("contacted")],
                         ["Verve Media"])
        # full pipeline in stage order
        self.assertEqual([r["stage"] for r in leads.list_by_stage()],
                         ["found", "contacted", "demo"])

    def test_due_follow_ups_due_or_overdue_sorted_open_only(self):
        leads.set_follow_up("Verve Media", "2026-10-19")      # due today
        leads.set_follow_up("8K", "2026-10-15")               # overdue
        leads.set_follow_up("Brand Beavers", "2026-10-25")    # future
        leads.save_leads([{"studio_name": "Closed Co", "location": "Mumbai"}])
        leads.set_follow_up("Closed Co", "2026-10-10")
        leads.set_stage("Closed Co", "lost")                   # closed → excluded
        due = leads.due_follow_ups(today=date(2026, 10, 19))
        self.assertEqual([r["studio_name"] for r in due], ["8K Entertainment", "Verve Media"])


class ParserTest(unittest.TestCase):

    def test_spec_phrases(self):
        self.assertEqual(parse_pipeline_command("mark Verve Media as contacted"),
                         {"action": "stage", "name": "Verve Media", "stage": "contacted",
                          "when": None})
        self.assertEqual(parse_pipeline_command("add note to 8K: asked for pricing"),
                         {"action": "note", "name": "8K", "text": "asked for pricing"})
        self.assertEqual(parse_pipeline_command("who should I follow up with today?"),
                         {"action": "due"})
        self.assertEqual(parse_pipeline_command("show my pipeline"), {"action": "pipeline"})

    def test_variants(self):
        p = parse_pipeline_command
        self.assertEqual(p("Mark Verve Media as contacted, follow up on Friday")["when"], "Friday")
        self.assertEqual(p("mark 8K won")["stage"], "won")
        self.assertEqual(p("move V-Show Entertainment to demo")["name"], "V-Show Entertainment")
        self.assertEqual(p("set follow-up for Verve Media to next Monday"),
                         {"action": "follow_up", "name": "Verve Media", "when": "next Monday"})
        self.assertEqual(p("show contacted leads"), {"action": "list_stage", "stage": "contacted"})
        self.assertEqual(p("note for V-Show - wants a reel edit")["name"], "V-Show")

    def test_ordinary_chat_is_not_a_command(self):
        for msg in ("what is a sales pipeline?", "find 5 studios in Mumbai",
                    "how do I follow up after an interview?", "mark the email as read",
                    "set a reminder to call mom"):
            self.assertIsNone(parse_pipeline_command(msg), msg)


class DispatchTest(PipelineCase):
    """The real action-dispatch path: detect → run → material cognition confirms from."""

    def turn(self, message):
        action = action_dispatch.detect_action(message, "s1")
        self.assertIsNotNone(action, message)
        self.assertEqual(action["tool"], "leads")
        res = action_dispatch.run_action(action)
        return action, res, action_dispatch.action_material(action, res)

    def test_mark_contacted_end_to_end(self):
        _, res, material = self.turn("mark Verve Media as contacted")
        self.assertTrue(res["ok"])
        self.assertIn("Verve Media moved from found to contacted", material)
        self.assertIn("Next follow-up: 2026-10-22", material)
        self.assertEqual(self.lead("Verve Media")["stage"], "contacted")

    def test_add_note_end_to_end(self):
        _, _, material = self.turn("add note to 8K: asked for pricing")
        self.assertIn('note added to 8K Entertainment: "asked for pricing"', material)
        self.assertIn("asked for pricing", self.lead("8K Entertainment")["notes"])

    def test_follow_ups_today_end_to_end(self):
        self.turn("set follow-up for Brand Beavers to today")
        _, res, material = self.turn("who should I follow up with today?")
        self.assertEqual([r["studio_name"] for r in res["items"]], ["Brand Beavers"])
        self.assertIn("| Brand Beavers | found | — | 2026-10-19 |", material)

    def test_pipeline_table_end_to_end(self):
        self.turn("mark Verve Media as contacted")
        action, _, material = self.turn("show my pipeline")
        self.assertIn(action["action"], action_dispatch.LEADS_TABLE_VERBS)
        self.assertIn("| Studio | Stage | Last contact | Next follow-up |", material)
        self.assertIn("| Verve Media | contacted | 2026-10-19 | 2026-10-22 |", material)
        self.assertIn("| 8K Entertainment | found | — | — |", material)

    def test_unknown_lead_is_honest_not_done(self):
        _, res, material = self.turn("mark Nobody Studio as won")
        self.assertFalse(res["ok"])
        self.assertTrue(material.startswith("ACTION NOT DONE"))

    def test_double_submit_appends_note_once(self):
        self.turn("add note to 8K: asked for pricing")
        _, res, _ = self.turn("add note to 8K: asked for pricing")
        self.assertTrue(res.get("_idem_replay"))
        self.assertEqual(self.lead("8K Entertainment")["notes"].count("asked for pricing"), 1)

    def test_public_mode_refuses_and_writes_nothing(self):
        with mock.patch.object(Config, "PUBLIC_MODE", True):
            action = action_dispatch.detect_action("mark Verve Media as contacted", "s1")
            res = action_dispatch.run_action(action)
        self.assertEqual(action["tool"], "disabled")
        self.assertFalse(res["ok"])
        self.assertEqual(self.lead("Verve Media")["stage"], "found")


if __name__ == "__main__":
    unittest.main()
