"""One-time owner credential setup for POST /token.

Run from backend/:   venv\\Scripts\\python scripts\\setup_owner.py

Prompts for a password (twice, hidden), then writes into backend/.env:
  * OWNER_PASSWORD_HASH — bcrypt hash of the password
  * OWNER_TOTP_SECRET   — a fresh base32 TOTP secret
replacing those lines if present, else appending them. Neither value is printed;
the only output is a terminal QR code to scan with Google Authenticator.

Re-running issues a NEW secret, which invalidates the authenticator entry from any
previous run. Restart the backend afterwards so it picks up the new values.
"""

import getpass
import os
import re
import sys
import tempfile
from pathlib import Path

try:
    import bcrypt
    import pyotp
    import qrcode
    from dotenv import dotenv_values
except ImportError as e:
    sys.exit(f"{e}. Run with the backend venv:  venv\\Scripts\\python scripts\\setup_owner.py")

MIN_PASSWORD_LEN = 10
# The .env that config.py's load_dotenv() reads: the one beside config.py.
BACKEND_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BACKEND_DIR / ".env"
_KEY_LINE = re.compile(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=")


def write_env(path: Path, values: dict[str, str]) -> None:
    """Set each KEY in `values` in the .env at `path`: an existing `KEY=` (or
    `export KEY=`) line is replaced in place, a missing key is appended. Other
    lines and their line endings are kept. Written atomically, so an interrupted
    run can't leave a truncated .env."""
    lines = []
    if path.exists():
        with open(path, encoding="utf-8", newline="") as f:  # newline="": keep CRLF
            lines = f.read().splitlines(keepends=True)
    eol = "\r\n" if any(l.endswith("\r\n") for l in lines) else "\n"
    pending = dict(values)
    for i, line in enumerate(lines):
        m = _KEY_LINE.match(line)
        if m and m.group(1) in values:
            # Single-quoted: the bcrypt hash contains '$', which dotenv would expand.
            lines[i] = f"{m.group(1)}='{values[m.group(1)]}'{eol}"
            pending.pop(m.group(1), None)
    if pending and lines and not lines[-1].endswith(("\n", "\r")):
        lines[-1] += eol
    lines += [f"{k}='{v}'{eol}" for k, v in pending.items()]

    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".env.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def main() -> int:
    if not (BACKEND_DIR / "config.py").is_file():
        print(f"config.py not found in {BACKEND_DIR}; refusing to guess where .env is.",
              file=sys.stderr)
        return 1
    # The QR is drawn with Unicode block characters; a legacy Windows console
    # (cp1252) can't encode them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    password = getpass.getpass("Owner password: ")
    if len(password) < MIN_PASSWORD_LEN:
        print(f"Password must be at least {MIN_PASSWORD_LEN} characters.", file=sys.stderr)
        return 1
    if len(password.encode("utf-8")) > 72:
        print("Password must be at most 72 bytes (bcrypt limit).", file=sys.stderr)
        return 1
    if getpass.getpass("Confirm password: ") != password:
        print("Passwords do not match.", file=sys.stderr)
        return 1

    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode()
    secret = pyotp.random_base32()
    uri = pyotp.TOTP(secret).provisioning_uri(name="owner", issuer_name="AIOS")

    values = {"OWNER_PASSWORD_HASH": pw_hash, "OWNER_TOTP_SECRET": secret}
    try:
        write_env(ENV_PATH, values)
    except OSError as e:
        print(f"Could not write {ENV_PATH}: {e}", file=sys.stderr)
        return 1
    # Read back the way config.py will, so a silent mismatch can't go unnoticed.
    saved = dotenv_values(ENV_PATH)
    if any(saved.get(k) != v for k, v in values.items()):
        print(f"Wrote {ENV_PATH} but the values did not read back. Check the file.",
              file=sys.stderr)
        return 1

    print("\n── Scan with Google Authenticator ──")
    qr = qrcode.QRCode(border=1)
    qr.add_data(uri)
    qr.make(fit=True)
    qr.print_ascii(invert=True)
    print("\nRestart the backend to apply. Don't share this QR: it encodes the secret.")
    print(f"Saved to {ENV_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
