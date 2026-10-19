import os
import re
import time
from xml.sax.saxutils import escape as xml_escape

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


def make_invoice_pdf(number: str, business: str, business_contact: str, client: str,
                     client_contact: str, item: str, amount: float, currency: str,
                     issued: str, due: str, status: str = "", out_dir: str = "outputs") -> str:
    """One-line-item invoice: from / bill-to, number + dates, the line item and total.
    Amounts are printed with the currency CODE (INR 50,000.00) — the default fonts
    have no ₹ glyph. Returns the saved file path."""
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{number.lower().replace('-', '_')}_{_slug(client)}.pdf")
    styles = getSampleStyleSheet()
    total = f"{currency} {amount:,.2f}"
    doc = SimpleDocTemplate(path, leftMargin=0.9 * inch, rightMargin=0.9 * inch,
                            topMargin=0.9 * inch, bottomMargin=0.9 * inch)

    def esc(text):
        # Paragraph text is XML: "Dina Films & Media" must not break the build.
        return xml_escape(_clean(str(text or "")))

    def para(text, style="Normal"):
        return Paragraph(esc(text).replace("\n", "<br/>"), styles[style])

    def block(label, *lines):
        return Paragraph(f"<b>{label}</b><br/>" + "<br/>".join(esc(l) for l in lines if l),
                         styles["Normal"])

    meta = Table([["Invoice", number], ["Issued", issued], ["Due", due]]
                 + ([["Status", status.upper()]] if status else []),
                 colWidths=[1.0 * inch, 1.8 * inch])
    meta.setStyle(TableStyle([("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                              ("ALIGN", (1, 0), (1, -1), "RIGHT")]))
    parties = Table([[block("From", business, business_contact),
                      block("Bill to", client, client_contact)]],
                    colWidths=[3.3 * inch, 3.3 * inch])
    parties.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    lines = Table([["Description", "Amount"], [para(item), total], ["Total", total]],
                  colWidths=[4.9 * inch, 1.7 * inch])
    lines.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEABOVE", (0, -1), (-1, -1), 0.8, colors.black),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    doc.build([
        Paragraph("INVOICE", styles["Title"]),
        HRFlowable(width="100%", thickness=1, spaceBefore=2, spaceAfter=10),
        meta, Spacer(1, 16), parties, Spacer(1, 22), lines, Spacer(1, 18),
        para(f"Payment due by {due}. Please quote {number} with your payment."),
    ])
    return path