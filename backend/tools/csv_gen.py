import os
import re
import csv
import time


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return s or "dossier"


def _clean(text: str) -> str:
    """Strip markdown bold/italic markers and swap glyphs that render badly."""
    text = text.replace("**", "").replace("__", "").replace("*", "")
    text = text.replace("₹", "Rs. ").replace("•", "-")
    return text.strip()


def _is_header(line: str) -> bool:
    if ":" not in line:
        return False
    label = line.split(":", 1)[0].strip()
    if not label or len(label.split()) > 5:
        return False
    return label == label.upper() or label.istitle()


def make_csv(title: str, body_text: str, out_dir: str = "outputs") -> str:
    """Flatten a structured dossier into [Section, Detail] rows. Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{_slug(title)}_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    rows = []
    section = ""
    for raw in body_text.splitlines():
        line = _clean(raw)
        if not line:
            continue
        if _is_header(line):
            label, _, inline = line.partition(":")
            section = label.strip()
            if inline.strip():
                rows.append([section, inline.strip()])
        elif line[0] in "-*•":
            rows.append([section, line.lstrip("-*• ").strip()])
        else:
            rows.append([section, line])

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Section", "Detail"])
        writer.writerows(rows)

    return path