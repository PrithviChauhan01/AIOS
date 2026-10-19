"""Leadgen web prospecting: Tavily-first, book only structures, code verifies, leads saved.

Run from backend/:   venv\\Scripts\\python -m unittest discover -s tests -v

No network: Tavily and every book call are patched.
"""

import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest import mock

from config import Config
from agents import leadgen as lg
from tools import leads as leads_tool
from tools import tavily

_SOURCES = [
    {"url": "https://swashrifilmstudio.com/video-editing", "title": "Video Editing",
     "content": "At SwaShri Film Studio, Ghatkopar West, Mumbai, we offer video editing. "
                "Call +91 98200 12345 or mail hello@swashrifilmstudio.com"},
    {"url": "https://www.justdial.com/Mumbai/Video-Editing-Studios",
     "title": "Top Video Editing Studios in Mumbai",
     "content": "Popular Video Editing Studios in Mumbai · V-Show Entertainment · Dina Films & Media"},
    {"url": "https://d7leadfinder.com/leads/1", "title": "Video Editing in Mumbai",
     "content": "| Corporate Video Mumbai | 9198929xxxx7 | mail@s6c80au.biz Register to View |"},
]


def _book_json(rows):
    return {"raw_text": json.dumps(rows), "book_used": "groq", "tokens": 10,
            "tokens_in": 5, "tokens_out": 5}


_CANDIDATES = [
    # real, fully sourced
    {"studio_name": "SwaShri Film Studio", "address": "Ghatkopar West, Mumbai",
     "phone": "+91 98200 12345", "email": "hello@swashrifilmstudio.com",
     "website": "swashrifilmstudio.com",
     "source_url": "https://swashrifilmstudio.com/video-editing"},
    # listed on a directory — the book invents contacts that are not in the text
    {"studio_name": "V-Show Entertainment", "address": "Andheri, Mumbai",
     "phone": "+91 99999 00000", "email": "info@vshow.in", "website": "vshow.in",
     "source_url": "https://www.justdial.com/Mumbai/Video-Editing-Studios"},
    # masked contacts on a lead-list site
    {"studio_name": "Corporate Video Mumbai", "address": "not found",
     "phone": "9198929xxxx7", "email": "mail@s6c80au.biz", "website": "not found",
     "source_url": "https://d7leadfinder.com/leads/1"},
    # invented business, cites a real URL that never names it
    {"studio_name": "Phantom Edits", "address": "Bandra", "phone": "not found",
     "email": "not found", "website": "not found",
     "source_url": "https://swashrifilmstudio.com/video-editing"},
    # invented source URL
    {"studio_name": "Dina Films & Media", "address": "not found", "phone": "not found",
     "email": "not found", "website": "not found", "source_url": "https://made.up/page"},
]


class LeadgenWebCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="aios_leadgen_")
        self.db = os.path.join(self.tmp, "aios.db")
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE leads (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                     "studio_name TEXT NOT NULL, location TEXT, contact TEXT, research TEXT, "
                     "outreach_sent INTEGER DEFAULT 0, response TEXT, "
                     "created_at DATETIME DEFAULT CURRENT_TIMESTAMP, follow_up_at DATETIME)")
        conn.commit()
        conn.close()
        self.search = mock.AsyncMock(return_value=list(_SOURCES))
        self.extract = mock.AsyncMock(return_value={"url": "", "content": ""})
        self.book = mock.AsyncMock(return_value=_book_json(_CANDIDATES))
        patches = [
            mock.patch.object(Config, "SQLITE_PATH", self.db),
            mock.patch.object(Config, "PUBLIC_MODE", False),
            mock.patch.object(tavily, "search", self.search),
            mock.patch.object(tavily, "extract", self.extract),
            mock.patch.object(lg, "select_book_for_tier", return_value=["groq"]),
            mock.patch.object(lg.LeadgenTeacher, "_traced_book",
                              lambda _self, prompt, book, ctx, **kw: self.book(prompt, book)),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_teacher(self, message, **ctx):
        ctx = {"message": message, "session_id": "t", "sensitivity": "public", **ctx}
        return asyncio.run(lg.LeadgenTeacher().run(ctx))

    def saved(self):
        return leads_tool.list_leads()

    def test_find_studios_uses_tavily_and_verifies_every_field(self):
        out = self.run_teacher("find 5 video editing studios in Mumbai")
        self.assertTrue(self.search.await_count >= 1)
        text = out["raw_text"]
        self.assertTrue(text.startswith(lg._WEB_PREFIX))
        # real lead keeps its sourced fields and link
        self.assertIn("SwaShri Film Studio", text)
        self.assertIn("+91 98200 12345", text)
        self.assertIn("hello@swashrifilmstudio.com", text)
        self.assertIn("Source: https://swashrifilmstudio.com/video-editing", text)
        # directory-listed lead survives, but its invented contacts do not
        self.assertIn("V-Show Entertainment", text)
        for fake in ("+91 99999 00000", "info@vshow.in", "vshow.in", "Andheri"):
            self.assertNotIn(fake, text)
        # masked values are never completed
        self.assertNotIn("9198929", text)
        self.assertNotIn("s6c80au", text)
        # invented business / invented source URL are dropped
        self.assertNotIn("Phantom Edits", text)
        self.assertNotIn("made.up", text)
        self.assertIn("Only 3 of the 5 requested", text)

    def test_every_saved_lead_has_source_urls_and_no_unverified_contact(self):
        self.run_teacher("find 5 video editing studios in Mumbai")
        rows = self.saved()
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertEqual(r["location"], "Mumbai")
            research = json.loads(r["research"])
            self.assertTrue(research["source_urls"])
            self.assertEqual(research["found_via"], "tavily")
        vshow = next(r for r in rows if r["studio_name"] == "V-Show Entertainment")
        self.assertEqual(vshow["contact"], "not found")
        self.assertEqual(json.loads(vshow["research"])["email"], "not found")

    def test_rerun_skips_duplicates_by_name_and_location(self):
        self.run_teacher("find 5 video editing studios in Mumbai")
        out = self.run_teacher("find 5 video editing studios in Mumbai")
        self.assertEqual(len(self.saved()), 3)
        self.assertIn("3 already on file were skipped", out["raw_text"])
        # same name, different city → a different lead
        leads_tool.save_leads([{"studio_name": "SwaShri Film Studio", "location": "Pune"}])
        self.assertEqual(len(self.saved()), 4)

    def test_enrich_reads_own_site_only_for_literal_contacts(self):
        no_contact = [dict(_CANDIDATES[0], phone="not found", email="not found")]
        self.book.return_value = _book_json(no_contact)
        self.search.return_value = [dict(_SOURCES[0], content="SwaShri Film Studio, Mumbai")]
        self.extract.return_value = {
            "url": "https://swashrifilmstudio.com",
            "content": "Reach us: +91 22 4019 0400 · studio@swashrifilmstudio.com · "
                       "scam@e795f9fte.in"}
        out = self.run_teacher("find 1 video editing studio in Mumbai")
        self.assertIn("+91 22 4019 0400", out["raw_text"])
        self.assertIn("studio@swashrifilmstudio.com", out["raw_text"])
        self.assertIn("https://swashrifilmstudio.com", self.saved()[0]["research"])

    def test_garbled_phone_replaced_by_source_rendering(self):
        t = lg.LeadgenTeacher()
        text = "Call +918693838888) 03 for bookings · landline (91) 22 40190400"
        self.assertEqual(t._verify_phone("+918693838888) 03", text), "+918693838888")
        self.assertEqual(t._verify_phone("+91 22 4019 0400", text), "22 40190400")
        self.assertEqual(t._verify_phone("+91 99999 00000", text), "not found")

    def test_secret_never_reaches_tavily(self):
        with mock.patch.object(lg.LeadgenTeacher, "_run_places_prospecting",
                               mock.AsyncMock(return_value={"raw_text": "places"})):
            out = self.run_teacher("find 5 studios in Mumbai", sensitivity="secret")
        self.search.assert_not_awaited()
        self.assertEqual(out["raw_text"], "places")

    def test_tavily_down_falls_back_to_places(self):
        self.search.side_effect = tavily.TavilyKeyMissing("no key")
        with mock.patch.object(lg.LeadgenTeacher, "_run_places_prospecting",
                               mock.AsyncMock(return_value={"raw_text": "places"})) as places:
            out = self.run_teacher("find 5 studios in Mumbai")
        places.assert_awaited_once()
        self.assertEqual(out["raw_text"], "places")
        self.assertEqual(self.saved(), [])

    def test_public_mode_does_not_save(self):
        with mock.patch.object(Config, "PUBLIC_MODE", True):
            out = self.run_teacher("find 5 video editing studios in Mumbai")
        self.assertIn("NOT saved", out["raw_text"])
        self.assertEqual(self.saved(), [])

    def test_routing(self):
        t = lg.LeadgenTeacher()
        self.assertEqual(t._select_mode("research video editing studios in Mumbai"), "prospect")
        self.assertEqual(t._select_mode("find 5 video editing studios in Mumbai"), "prospect")
        self.assertEqual(t._select_mode("research Luma Labs"), "single")
        self.assertEqual(t._lead_count("find 5 studios in Mumbai"), 5)
        self.assertEqual(t._location_of("find 5 video editing studios in Mumbai"), "Mumbai")


if __name__ == "__main__":
    unittest.main()
