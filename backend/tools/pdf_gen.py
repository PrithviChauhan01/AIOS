import os
import re
import time

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable


def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return s or "dossier"


def _clean(text: str) -> str:
    """Strip markdown bold/italic markers and swap glyphs reportlab can't render."""
    text = text.replace("**", "").replace("__", "").replace("*", "")
    text = text.replace("₹", "Rs. ").replace("•", "-")
    return text.strip()


def _is_header(line: str) -> bool:
    """A section header is a short line whose label (before any ':') is all-caps
    or Title Case — e.g. 'SERVICES:', 'Fit Signals:'."""
    if ":" not in line:
        return False
    label = line.split(":", 1)[0].strip()
    if not label or len(label.split()) > 5:
        return False
    return label == label.upper() or label.istitle()


def make_pdf(title: str, body_text: str, out_dir: str = "outputs") -> str:
    """Render a structured dossier (SECTION: / - item lines) into a clean PDF.
    Returns the saved file path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{_slug(title)}_{time.strftime('%Y%m%d_%H%M%S')}.pdf")

    styles = getSampleStyleSheet()
    bullet = styles["Normal"].clone("Bullet")
    bullet.leftIndent = 18

    doc = SimpleDocTemplate(path, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch)

    story = [
        Paragraph(_clean(title), styles["Title"]),
        HRFlowable(width="100%", thickness=1, spaceBefore=2, spaceAfter=6),
        Paragraph(f"Generated {time.strftime('%Y-%m-%d %H:%M')}", styles["Normal"]),
        Spacer(1, 14),
    ]

    first_section = True
    for raw in body_text.splitlines():
        line = _clean(raw)
        if not line:
            continue

        if _is_header(line):
            label, _, inline = line.partition(":")
            if not first_section:
                story.append(Spacer(1, 10))
            first_section = False
            story.append(Paragraph(label.strip(), styles["Heading2"]))
            if inline.strip():
                story.append(Paragraph(inline.strip(), styles["Normal"]))
        elif line[0] in "-*•":
            story.append(Paragraph(line.lstrip("-*• ").strip(), bullet))
        else:
            story.append(Paragraph(line, styles["Normal"]))

    doc.build(story)
    return path