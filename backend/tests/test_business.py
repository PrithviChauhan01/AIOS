"""Clients, projects, invoices (Slice B4): migration, every action, overdue, reminders,
invoice PDF, and the won-lead → client offer.

Run from backend/:   venv\\Scripts\\python -m unittest discover -s tests -v

No network: every command is parsed deterministically against a temp SQLite db.
"""

import os
import shutil
import sqlite3
import tempfile
import unittest
from datetime import date, datetime
from unittest import mock

from config import Config
from core import action_dispatch
from core.business_parse import parse_amount, parse_business_command
from tools import business, leads

_NOW = datetime(2026, 10, 19, 10, 0)  # a Monday


class BusinessCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aios_business_")
        self.db = os.path.join(self.tmp, "aios.db")
        self.now = mock.patch.object(business, "_now", return_value=_NOW)
        self.now.start()
        self.addCleanup(self.now.stop)
        for p in (mock.patch.object(Config, "SQLITE_PATH", self.db),
                  mock.patch.object(Config, "PUBLIC_MODE", False),
                  mock.patch.object(Config, "BUSINESS_NAME", "Prithvi Studio"),
                  mock.patch.object(Config, "BUSINESS_CONTACT", "prithvi@example.com"),
                  mock.patch.object(Config, "INVOICE_DUE_DAYS", 14),
                  mock.patch.object(business, "_PDF_DIR", os.path.join(self.tmp, "out")),
                  mock.patch.object(leads, "_now", return_value=_NOW)):
            p.start()
            self.addCleanup(p.stop)
        from db.sqlite_init import init_sqlite
        init_sqlite()
        conn = sqlite3.connect(self.db)
        conn.executemany(
            "INSERT INTO leads (studio_name, location, contact) VALUES (?, ?, ?)",
            [("Verve Media", "Mumbai", "info@vervemedia.co.in"),
             ("8K Entertainment", "Mumbai", "+918693838888")])
        conn.commit()
        conn.close()
        business._SCHEMA_READY.discard(business._db_path())
        leads._SCHEMA_READY.discard(leads._db_path())
        action_dispatch._idem_store.clear()
        action_dispatch._CLIENT_OFFER.clear()
        from tools import mailer
        mailer._PENDING.clear()
        self.addCleanup(action_dispatch._CLIENT_OFFER.clear)
        self.addCleanup(action_dispatch._idem_store.clear)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def turn(self, message, session="s1"):
        action = action_dispatch.detect_action(message, session)
        self.assertIsNotNone(action, message)
        res = action_dispatch.run_action(action)
        return action, res, action_dispatch.action_material(action, res)

    def reminders(self):
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute("SELECT title, due_at FROM reminders ORDER BY id").fetchall()
        finally:
            conn.close()

    def seed(self):
        business.add_client("Verve Media", "info@vervemedia.co.in")
        business.add_project("Verve Media", "Brand film", "2026-10-23", 50000)


class MigrationTest(unittest.TestCase):

    def test_existing_data_kept_and_tables_added(self):
        tmp = tempfile.mkdtemp(prefix="aios_business_mig_")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        db = os.path.join(tmp, "aios.db")
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE leads (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                     "studio_name TEXT NOT NULL, location TEXT, contact TEXT, research TEXT, "
                     "outreach_sent INTEGER DEFAULT 0, response TEXT, "
                     "created_at DATETIME DEFAULT CURRENT_TIMESTAMP, follow_up_at DATETIME)")
        conn.execute("CREATE TABLE reminders (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                     "title TEXT NOT NULL, due_at DATETIME NOT NULL, repeat TEXT, "
                     "done INTEGER DEFAULT 0, created_at DATETIME DEFAULT CURRENT_TIMESTAMP)")
        conn.execute("INSERT INTO leads (studio_name, contact) VALUES ('Old Lead', 'x@y.com')")
        conn.execute("INSERT INTO reminders (title, due_at) VALUES ('old', '2026-01-01 09:00:00')")
        conn.commit()
        conn.close()
        with mock.patch.object(Config, "SQLITE_PATH", db):
            from db.sqlite_init import init_sqlite
            init_sqlite()
            init_sqlite()  # idempotent
        conn = sqlite3.connect(db)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"clients", "projects", "invoices"} <= tables)
        self.assertEqual(conn.execute("SELECT studio_name, contact FROM leads").fetchall(),
                         [("Old Lead", "x@y.com")])
        self.assertEqual(conn.execute("SELECT title FROM reminders").fetchall(), [("old",)])
        conn.close()


class ParserTest(unittest.TestCase):

    def test_commands(self):
        p = parse_business_command
        self.assertEqual(p("add client Verve Media, contact info@vervemedia.co.in"),
                         {"action": "add_client", "name": "Verve Media",
                          "contact": "info@vervemedia.co.in"})
        self.assertEqual(p("add project Brand film for Verve Media, due 2026-11-20, fee 50000"),
                         {"action": "add_project", "title": "Brand film", "client": "Verve Media",
                          "deadline": "2026-11-20", "fee": 50000.0})
        self.assertEqual(p("move Brand film project to review")["status"], "review")
        self.assertEqual(p("set deadline for Brand film to next Friday")["deadline"], "next Friday")
        inv = p("create invoice for Verve Media for 50000 due in 14 days")
        self.assertEqual((inv["target"], inv["amount"], inv["due"]),
                         ("Verve Media", 50000.0, "in 14 days"))
        self.assertEqual(p("mark invoice INV-2026-001 as paid")["status"], "paid")
        self.assertEqual(p("INV-2026-002 is paid")["target"], "INV-2026-002")
        self.assertEqual(p("invoice Verve Media pdf"),
                         {"action": "invoice_pdf", "target": "Verve Media"})
        self.assertEqual(p("what's due this week?"), {"action": "due_week"})
        self.assertEqual(p("show my clients"), {"action": "list_clients"})
        self.assertEqual(p("unpaid invoices"), {"action": "unpaid"})

    def test_amounts(self):
        self.assertEqual(parse_amount("fee Rs 25,000"), (25000.0, "INR"))
        self.assertEqual(parse_amount("₹1.5 lakh"), (150000.0, "INR"))
        self.assertEqual(parse_amount("$2k"), (2000.0, "USD"))

    def test_chat_is_not_a_command(self):
        for msg in ("what is a client?", "how do I write an invoice?",
                    "mark Verve Media as contacted", "show my pipeline",
                    "draft outreach to Verve Media"):
            self.assertIsNone(parse_business_command(msg), msg)


class ClientTest(BusinessCase):

    def test_add_client_and_duplicate(self):
        _, res, material = self.turn("add client Verve Media, contact info@vervemedia.co.in")
        self.assertTrue(res["ok"])
        self.assertEqual(res["client"]["contact"], "info@vervemedia.co.in")
        self.assertIn("client Verve Media added", material)
        _, res, material = self.turn("add client verve media")
        self.assertFalse(res["ok"])
        self.assertTrue(material.startswith("ACTION NOT DONE"))

    def test_show_clients_table(self):
        self.seed()
        business.create_invoice("Brand film")
        _, res, material = self.turn("show my clients")
        self.assertEqual(len(res["items"]), 1)
        self.assertIn("| Verve Media | info@vervemedia.co.in | 1 | INR 50,000.00 |", material)


class ProjectTest(BusinessCase):

    def test_add_project_with_deadline_fee_and_reminder(self):
        business.add_client("Verve Media")
        _, res, material = self.turn(
            "add project Brand film for Verve Media, due 2026-10-30, fee 50000")
        p = res["project"]
        self.assertEqual((p["title"], p["status"], p["deadline"], p["fee"]),
                         ("Brand film", "planned", "2026-10-30", 50000.0))
        self.assertEqual(self.reminders(),
                         [("Deadline: Brand film (Verve Media)", "2026-10-30 10:00:00")])
        self.assertIn("A reminder is set for 2026-10-30 10:00:00", material)

    def test_title_containing_for_resolves_against_real_client(self):
        business.add_client("Verve Media")
        _, res, _ = self.turn("add project Content for socials for Verve Media, fee 20000")
        self.assertTrue(res["ok"])
        self.assertEqual((res["project"]["title"], res["project"]["client"]),
                         ("Content for socials", "Verve Media"))

    def test_title_resplit_never_guesses(self):
        # "X for Y for A for B": both "A for B" and "B" are real clients → ambiguous.
        business.add_client("A for B")
        business.add_client("B")
        res = business.add_project("Y for A for B", "X")
        self.assertFalse(res["ok"])
        self.assertIn("no client called", res["ask"])
        self.assertFalse(business.add_project("nobody for anyone", "Edit")["ok"])
        conn = sqlite3.connect(self.db)
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0], 0)
        conn.close()

    def test_unknown_client_changes_nothing(self):
        _, res, _ = self.turn("add project Brand film for Nobody Co")
        self.assertFalse(res["ok"])
        self.assertEqual(self.reminders(), [])

    def test_update_status_and_deadline(self):
        self.seed()
        _, res, material = self.turn("mark project Brand film as active")
        self.assertEqual(res["project"]["status"], "active")
        self.assertIn("status planned → active", material)
        _, res, _ = self.turn("set deadline for Brand film to 2026-11-05")
        self.assertEqual(res["project"]["deadline"], "2026-11-05")
        self.assertEqual(self.reminders()[-1],
                         ("Deadline: Brand film (Verve Media)", "2026-11-05 10:00:00"))
        self.assertFalse(business.update_project("Brand film", status="shipped")["ok"])

    def test_past_deadline_sets_no_reminder(self):
        business.add_client("Verve Media")
        res = business.add_project("Verve Media", "Old job", "2026-10-01")
        self.assertIsNone(res["reminder"])
        self.assertEqual(self.reminders(), [])


class InvoiceTest(BusinessCase):

    def test_create_invoice_numbering_defaults_and_reminder(self):
        self.seed()
        _, res, material = self.turn("create invoice for Verve Media")
        inv = res["invoice"]
        self.assertEqual(inv["number"], "INV-2026-001")
        self.assertEqual((inv["amount"], inv["currency"], inv["status"]), (50000.0, "INR", "draft"))
        self.assertEqual((inv["issued_at"], inv["due_at"]), ("2026-10-19", "2026-11-02"))
        self.assertIn("DRAFT", material)
        self.assertIn(("Invoice INV-2026-001 due — Verve Media (INR 50,000.00)",
                       "2026-11-02 10:00:00"), self.reminders())
        _, res, _ = self.turn("create invoice for Brand film for $2k due 2026-11-15")
        self.assertEqual((res["invoice"]["number"], res["invoice"]["amount"],
                          res["invoice"]["currency"], res["invoice"]["due_at"]),
                         ("INV-2026-002", 2000.0, "USD", "2026-11-15"))

    def test_client_with_several_projects_must_choose(self):
        self.seed()
        business.add_project("Verve Media", "Reel edits", fee=10000)
        res = business.create_invoice("Verve Media")
        self.assertFalse(res["ok"])
        self.assertIn("several projects", res["ask"])

    def test_no_amount_and_no_fee_asks(self):
        business.add_client("Verve Media")
        business.add_project("Verve Media", "Brand film")
        self.assertEqual(business.create_invoice("Brand film")["needs"], "amount")

    def test_mark_sent_then_paid(self):
        self.seed()
        business.create_invoice("Brand film")
        _, res, _ = self.turn("mark invoice INV-2026-001 as sent")
        self.assertEqual(res["invoice"]["status"], "sent")
        _, res, material = self.turn("mark Verve Media invoice paid")
        self.assertEqual(res["invoice"]["status"], "paid")
        self.assertIn("marked paid", material)
        self.assertEqual(business.unpaid_invoices(), [])

    def test_overdue_is_computed_automatically(self):
        self.seed()
        business.create_invoice("Brand film", due="2026-10-25")
        business.mark_invoice("INV-2026-001", "sent")
        with mock.patch.object(business, "_now", return_value=datetime(2026, 10, 27, 9, 0)):
            self.assertEqual(business.unpaid_invoices()[0]["status"], "overdue")
            business.mark_invoice("INV-2026-001", "paid")
            self.assertEqual(business.unpaid_invoices(), [])

    def test_sent_after_due_date_is_overdue(self):
        self.seed()
        business.create_invoice("Brand film", due="2026-10-25")
        with mock.patch.object(business, "_now", return_value=datetime(2026, 10, 30, 9, 0)):
            self.assertEqual(business.mark_invoice("INV-2026-001", "sent")["invoice"]["status"],
                             "overdue")

    def test_unpaid_list_sorted_by_due(self):
        self.seed()
        business.add_project("Verve Media", "Reel edits", fee=10000)
        business.create_invoice("Reel edits", due="2026-11-20")
        business.create_invoice("Brand film", due="2026-11-01")
        _, res, material = self.turn("unpaid invoices")
        self.assertEqual([i["number"] for i in res["items"]], ["INV-2026-002", "INV-2026-001"])
        self.assertIn("| INV-2026-002 | Verve Media | INR 50,000.00 | 2026-11-01 | draft |", material)

    def test_invoice_pdf(self):
        self.seed()
        business.create_invoice("Brand film")
        _, res, material = self.turn("invoice Verve Media pdf")
        self.assertTrue(res["ok"])
        self.assertTrue(os.path.exists(res["file_path"]))
        from pypdf import PdfReader
        text = " ".join(p.extract_text() for p in PdfReader(res["file_path"]).pages)
        for expected in ("Prithvi Studio", "Verve Media", "Brand film", "INR 50,000.00",
                         "INV-2026-001", "2026-11-02"):
            self.assertIn(expected, text)
        self.assertIn(res["file_path"], material)

    def test_invoice_pdf_escapes_ampersand(self):
        business.add_client("Dina Films & Media")
        business.add_project("Dina Films & Media", "Promo cut", fee=1000)
        business.create_invoice("Promo cut")
        res = business.invoice_pdf("Dina Films & Media")
        self.assertTrue(res["ok"])
        self.assertTrue(os.path.exists(res["file_path"]))


class DueThisWeekTest(BusinessCase):

    def test_deadlines_and_invoices_within_7_days_incl_late(self):
        business.add_client("Verve Media")
        business.add_project("Verve Media", "Brand film", "2026-10-23", 50000)   # in window
        business.add_project("Verve Media", "Late cut", "2026-10-15", 5000)      # already late
        business.add_project("Verve Media", "Far job", "2026-12-01", 5000)       # outside
        business.add_project("Verve Media", "Done job", "2026-10-21", 5000)
        business.update_project("Done job", status="done")                         # excluded
        business.create_invoice("Brand film", due="2026-10-24")                     # in window
        business.create_invoice("Far job", due="2026-11-30")                        # outside
        _, res, material = self.turn("what's due this week?")
        self.assertEqual([p["title"] for p in res["projects"]], ["Late cut", "Brand film"])
        self.assertEqual([i["number"] for i in res["invoices"]], ["INV-2026-001"])
        self.assertLess(material.index("2026-10-15"), material.index("2026-10-24"))
        self.assertNotIn("Far job", material)

    def test_nothing_due(self):
        _, _, material = self.turn("what's due this week")
        self.assertIn("nothing is due", material)


class WonToClientTest(BusinessCase):

    def test_won_offers_and_yes_creates_client_from_lead(self):
        _, res, material = self.turn("mark Verve Media as won")
        self.assertTrue(res["offer_client"])
        self.assertIn("shall I create a client from Verve Media", material)
        action, res, material = self.turn("yes")
        self.assertEqual((action["tool"], action["action"]), ("business", "client_from_lead"))
        c = res["client"]
        self.assertEqual((c["name"], c["contact"]), ("Verve Media", "info@vervemedia.co.in"))
        self.assertEqual(c["lead_id"], leads.find_lead("Verve Media")["lead"]["id"])
        self.assertIn("created from lead", material)
        self.assertEqual(len(business.list_clients()), 1)

    def test_no_declines_and_creates_nothing(self):
        self.turn("mark Verve Media as won")
        action, res, _ = self.turn("no")
        self.assertEqual(action["action"], "decline_offer")
        self.assertEqual(business.list_clients(), [])

    def test_offer_lasts_one_turn(self):
        self.turn("mark Verve Media as won")
        self.turn("show my pipeline")                  # anything else drops the offer
        self.assertIsNone(action_dispatch.detect_action("yes", "s1"))
        self.assertEqual(business.list_clients(), [])

    def test_no_offer_when_already_a_client_or_not_won(self):
        self.turn("mark Verve Media as won")
        self.turn("yes")
        leads.set_stage("Verve Media", "demo")
        action_dispatch._idem_store.clear()  # past the 60s double-submit window
        _, res, _ = self.turn("mark Verve Media as won")
        self.assertNotIn("offer_client", res)
        _, res, _ = self.turn("mark 8K as contacted")
        self.assertNotIn("offer_client", res)

    def test_offer_is_per_session(self):
        self.turn("mark Verve Media as won", session="s1")
        self.assertIsNone(action_dispatch.detect_action("yes", "other"))


class PublicModeTest(BusinessCase):

    def test_public_mode_refuses_and_writes_nothing(self):
        with mock.patch.object(Config, "PUBLIC_MODE", True):
            action = action_dispatch.detect_action("add client Verve Media", "s1")
            res = action_dispatch.run_action(action)
        self.assertEqual(action["tool"], "disabled")
        self.assertFalse(res["ok"])
        self.assertEqual(business.list_clients(), [])


if __name__ == "__main__":
    unittest.main()
