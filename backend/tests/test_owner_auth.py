"""Owner login tests: bcrypt + TOTP verification, lockout, POST /token, 12h JWT.

Run from backend/:   venv\\Scripts\\python -m unittest discover -s tests -t . -v
"""

import datetime
import unittest
from unittest import mock

import bcrypt
import jwt
import pyotp
from fastapi.testclient import TestClient

from config import Config
from core import owner_auth

PASSWORD = "correct horse battery"
SECRET = pyotp.random_base32()
HASH = bcrypt.hashpw(PASSWORD.encode(), bcrypt.gensalt(rounds=4)).decode()
T0 = 1_700_000_010.0  # 10s into a 30s TOTP step


def code_at(t: float) -> str:
    return pyotp.TOTP(SECRET).at(t)


class OwnerAuthCase(unittest.TestCase):

    def setUp(self):
        self.now = [T0]
        for p in (mock.patch.object(Config, "PUBLIC_MODE", False),
                  mock.patch.object(Config, "OWNER_PASSWORD_HASH", HASH),
                  mock.patch.object(Config, "OWNER_TOTP_SECRET", SECRET),
                  mock.patch.object(owner_auth, "_clock", lambda: self.now[0])):
            p.start()
            self.addCleanup(p.stop)
        owner_auth.reset()

    def ok(self, ip="1.1.1.1", password=PASSWORD, code=None):
        return owner_auth.attempt(ip, password, code_at(self.now[0]) if code is None else code)


class TotpTest(OwnerAuthCase):

    def test_valid_password_and_code(self):
        self.assertEqual(self.ok(), "ok")

    def test_wrong_code_rejected(self):
        bad = "000000" if code_at(T0) != "000000" else "111111"
        self.assertEqual(self.ok(code=bad), "invalid")

    def test_malformed_codes_rejected(self):
        for bad in ("", "12345", "1234567", "abcdef", " 12 34 "):
            owner_auth.reset()
            self.assertEqual(self.ok(code=bad), "invalid", repr(bad))

    def test_one_step_clock_drift_allowed_two_not(self):
        self.assertEqual(self.ok(code=code_at(T0 - 30)), "ok")
        owner_auth.reset()
        self.assertEqual(self.ok(code=code_at(T0 - 60)), "invalid")
        owner_auth.reset()
        self.assertEqual(self.ok(code=code_at(T0 + 60)), "invalid")

    def test_code_cannot_be_replayed(self):
        code = code_at(T0)
        self.assertEqual(self.ok(code=code), "ok")
        self.assertEqual(self.ok(code=code), "invalid")
        self.now[0] += 30
        self.assertEqual(self.ok(), "ok")  # next step's code still works

    def test_wrong_password_with_valid_code(self):
        self.assertEqual(self.ok(password="nope"), "invalid")

    def test_unconfigured_owner_refuses_everything(self):
        with mock.patch.object(Config, "OWNER_PASSWORD_HASH", ""):
            self.assertEqual(self.ok(), "invalid")
        owner_auth.reset()
        with mock.patch.object(Config, "OWNER_TOTP_SECRET", ""):
            self.assertEqual(self.ok(), "invalid")

    def test_both_factors_always_evaluated(self):
        with mock.patch.object(owner_auth, "_check_totp",
                               wraps=owner_auth._check_totp) as totp:
            self.ok(password="nope")
        totp.assert_called_once()


class LockoutTest(OwnerAuthCase):

    def test_three_failures_lock_ip_for_15_minutes(self):
        for _ in range(3):
            self.assertEqual(self.ok(password="nope"), "invalid")
        self.assertEqual(self.ok(), "locked")  # even correct credentials
        self.now[0] += 15 * 60 - 1
        self.assertEqual(self.ok(), "locked")
        self.now[0] += 2
        self.assertEqual(self.ok(), "ok")

    def test_lock_is_per_ip_only(self):
        for _ in range(3):
            self.ok(ip="1.1.1.1", password="nope")
        self.assertEqual(self.ok(ip="1.1.1.1"), "locked")
        self.assertEqual(self.ok(ip="2.2.2.2"), "ok")  # other IPs unaffected

    def test_failures_spread_across_ips_never_lock(self):
        for ip in ("1.1.1.1", "2.2.2.2", "3.3.3.3", "4.4.4.4"):
            self.assertEqual(self.ok(ip=ip, password="nope"), "invalid")
        self.assertEqual(self.ok(ip="5.5.5.5"), "ok")

    def test_locked_attempt_does_not_check_credentials(self):
        for _ in range(3):
            self.ok(password="nope")
        with mock.patch.object(owner_auth, "_check_password") as pw:
            self.assertEqual(self.ok(), "locked")
        pw.assert_not_called()

    def test_failures_outside_window_expire(self):
        self.ok(password="nope")
        self.ok(password="nope")
        self.now[0] += 15 * 60 + 1
        self.assertEqual(self.ok(password="nope"), "invalid")
        self.assertEqual(self.ok(), "ok")

    def test_success_resets_counter(self):
        self.ok(password="nope")
        self.ok(password="nope")
        self.assertEqual(self.ok(), "ok")
        self.now[0] += 30
        self.ok(password="nope")
        self.ok(password="nope")
        self.assertEqual(self.ok(), "ok")


class TokenEndpointTest(OwnerAuthCase):

    def setUp(self):
        super().setUp()
        from main import create_app
        self.client = TestClient(create_app())  # no lifespan: no DB/scheduler start

    def post(self, password=PASSWORD, code=None):
        return self.client.post("/token", json={
            "password": password, "code": code_at(self.now[0]) if code is None else code})

    def test_success_issues_12h_jwt(self):
        r = self.post()
        self.assertEqual(r.status_code, 200)
        claims = jwt.decode(r.json()["token"], Config.JWT_SECRET, algorithms=["HS256"])
        ttl = claims["exp"] - claims["iat"]
        self.assertEqual(ttl, int(datetime.timedelta(hours=12).total_seconds()))

    def test_generic_error_for_either_factor(self):
        bad_pw = self.post(password="nope")
        owner_auth.reset()
        bad_code = self.post(code="000000" if code_at(T0) != "000000" else "111111")
        self.assertEqual(bad_pw.status_code, 401)
        self.assertEqual(bad_pw.json(), bad_code.json())
        self.assertEqual(bad_pw.json(), {"detail": "Invalid credentials."})

    def test_lockout_returns_429(self):
        for _ in range(3):
            self.post(password="nope")
        r = self.post()
        self.assertEqual(r.status_code, 429)
        self.assertNotIn("token", r.json())

    def test_get_token_no_longer_exists(self):
        self.assertEqual(self.client.get("/token").status_code, 405)

    def test_oversized_fields_rejected_before_hashing(self):
        r = self.client.post("/token", json={"password": "x" * 5000, "code": "123456"})
        self.assertEqual(r.status_code, 422)


class RequireAuthTest(unittest.TestCase):
    """Only 2FA-minted, short-lived tokens open personal routes."""

    def check(self, token):
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials
        from core.auth import require_auth
        try:
            require_auth(HTTPAuthorizationCredentials(scheme="Bearer", credentials=token))
            return True
        except HTTPException:
            return False

    def mint(self, **claims):
        now = datetime.datetime.now(datetime.timezone.utc)
        payload = {"user": "prithvi", "iat": now, "exp": now + datetime.timedelta(hours=1)}
        payload.update(claims)
        return jwt.encode({k: v for k, v in payload.items() if v is not None},
                          Config.JWT_SECRET, algorithm="HS256")

    def test_login_token_accepted(self):
        from core.auth import generate_token
        self.assertTrue(self.check(generate_token()))

    def test_legacy_365_day_token_rejected(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        self.assertFalse(self.check(self.mint(exp=now + datetime.timedelta(days=365))))

    def test_overlong_lifetime_rejected_even_with_scope(self):
        now = datetime.datetime.now(datetime.timezone.utc)
        self.assertFalse(self.check(self.mint(scope="owner-2fa",
                                              exp=now + datetime.timedelta(days=30))))

    def test_missing_iat_rejected(self):
        self.assertFalse(self.check(self.mint(scope="owner-2fa", iat=None)))

    def test_alg_none_and_wrong_secret_rejected(self):
        payload = {"scope": "owner-2fa", "iat": 0, "exp": 2**31}
        self.assertFalse(self.check(jwt.encode(payload, None, algorithm="none")))
        self.assertFalse(self.check(jwt.encode(payload, "x" * 40, algorithm="HS256")))


if __name__ == "__main__":
    unittest.main()
