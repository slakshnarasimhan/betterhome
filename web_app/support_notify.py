"""Build PDF/Excel for a generated recommendation and email support.

The customer-facing PDF/Excel buttons run in the browser, so support would
otherwise never see the files unless the user downloaded them. After each
successful generation we write both files on the server and send them to
contact@betterhomeapp.com.

Render env (set in the dashboard; never commit secrets):

  SUPPORT_EMAIL     default contact@betterhomeapp.com

  Preferred on Render (HTTP, not SMTP):
    SENDGRID_API_KEY
    SENDGRID_FROM     a verified sender, e.g. noreply@betterhomeapp.com

  SMTP fallback:
    SMTP_HOST, SMTP_PORT (587), SMTP_USER, SMTP_PASSWORD, SMTP_FROM

If neither mail path is configured, generation still succeeds and we log a skip.
HTML, PDF, and Excel are also committed to GitHub in the same step (see github_commit.py).
"""
from __future__ import annotations

import base64
import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from typing import Any, Dict, Iterable, List, Optional, Tuple
from xml.sax.saxutils import escape

import requests
from dotenv import load_dotenv
from openpyxl import Workbook
from openpyxl.styles import Font
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from github_commit import commit_generated_files

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DEFAULT_SUPPORT_EMAIL = "contact@betterhomeapp.com"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def support_email_address() -> str:
    return (os.getenv("SUPPORT_EMAIL") or DEFAULT_SUPPORT_EMAIL).strip() or DEFAULT_SUPPORT_EMAIL


def mail_configured() -> bool:
    if os.getenv("SENDGRID_API_KEY") and os.getenv("SENDGRID_FROM"):
        return True
    return bool(os.getenv("SMTP_HOST") and os.getenv("SMTP_FROM"))


def _product_price(product: Dict[str, Any]) -> float:
    for key in ("bh_price", "better_home_price", "price", "retail_price", "mrp_price"):
        raw = product.get(key)
        if raw in (None, ""):
            continue
        try:
            return float(str(raw).replace(",", "").replace("₹", "").strip())
        except (TypeError, ValueError):
            continue
    return 0.0


def _product_name(product: Dict[str, Any]) -> str:
    brand = str(product.get("brand") or "").strip()
    title = str(product.get("title") or product.get("model") or product.get("name") or "").strip()
    if brand and title.lower().startswith(brand.lower()):
        return title
    return " ".join(part for part in (brand, title) if part).strip() or "Product"


def _room_label(room: str) -> str:
    return str(room or "").replace("_", " ").title() or "Room"


def flatten_recommendation_rows(recommendations: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(recommendations, dict):
        return rows

    for room, categories in recommendations.items():
        if not isinstance(categories, dict):
            continue
        for category, options in categories.items():
            if isinstance(options, dict):
                for nested_category, nested_options in options.items():
                    if not isinstance(nested_options, list):
                        continue
                    for product in nested_options:
                        if isinstance(product, dict):
                            rows.append(_row_from_product(room, nested_category, product))
            elif isinstance(options, list):
                for product in options:
                    if isinstance(product, dict):
                        rows.append(_row_from_product(room, category, product))
    return rows


def _row_from_product(room: str, category: str, product: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "room": _room_label(room),
        "category": str(category or "").replace("_", " ").title(),
        "name": _product_name(product),
        "sku": str(product.get("sku") or ""),
        "price": _product_price(product),
    }


def client_fields(user_data: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    data = user_data or {}
    demographics = data.get("demographics") if isinstance(data.get("demographics"), dict) else {}
    bedrooms = data.get("num_bedrooms")
    if bedrooms in (None, ""):
        bedrooms = demographics.get("bedrooms")
    return [
        ("Name", str(data.get("name") or "Not provided")),
        ("Email", str(data.get("email") or "Not provided")),
        ("Phone", str(data.get("mobile") or data.get("phone") or "Not provided")),
        ("Address", str(data.get("address") or "Not provided")),
        ("City", str(data.get("city") or "Not provided")),
        ("Budget", f"Rs. {float(data.get('total_budget') or 0):,.0f}"),
        ("Bedrooms", str(bedrooms if bedrooms not in (None, "") else "Not provided")),
    ]


def write_recommendations_excel(path: str, user_data: Dict[str, Any], recommendations: Dict[str, Any]) -> str:
    rows = flatten_recommendation_rows(recommendations)
    workbook = Workbook()
    products = workbook.active
    products.title = "Recommendations"
    products.append(["Room", "Category", "Product", "SKU", "Price"])
    for cell in products[1]:
        cell.font = Font(bold=True)
    total = 0.0
    for row in rows:
        products.append([row["room"], row["category"], row["name"], row["sku"], row["price"]])
        total += float(row["price"] or 0)
    products.append(["", "", "TOTAL", "", total])

    client = workbook.create_sheet("Client")
    client.append(["Field", "Value"])
    for cell in client[1]:
        cell.font = Font(bold=True)
    for label, value in client_fields(user_data):
        client.append([label, value])

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    workbook.save(path)
    return path


def write_recommendations_pdf(path: str, user_data: Dict[str, Any], recommendations: Dict[str, Any]) -> str:
    rows = flatten_recommendation_rows(recommendations)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    doc = SimpleDocTemplate(path, pagesize=letter, leftMargin=0.6 * inch, rightMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("ExportTitle", parent=styles["Title"], fontSize=16, spaceAfter=8)
    heading = ParagraphStyle("ExportHeading", parent=styles["Heading2"], fontSize=12, spaceBefore=12, spaceAfter=6)
    body = ParagraphStyle("ExportBody", parent=styles["Normal"], fontSize=9, leading=12)

    story: List[Any] = [
        Paragraph("BetterHome Recommendations", title),
        Paragraph(escape(datetime.now().strftime("Generated %d %B %Y %H:%M")), body),
        Spacer(1, 10),
        Paragraph("Client Information", heading),
    ]
    client_table = Table(
        [[Paragraph(f"<b>{escape(label)}</b>", body), Paragraph(escape(value), body)] for label, value in client_fields(user_data)],
        colWidths=[1.6 * inch, 5.2 * inch],
    )
    client_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dddddd")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(client_table)
    story.append(Paragraph("Recommended Products", heading))

    table_data = [[
        Paragraph("<b>Room</b>", body),
        Paragraph("<b>Category</b>", body),
        Paragraph("<b>Product</b>", body),
        Paragraph("<b>Price</b>", body),
    ]]
    total = 0.0
    for row in rows:
        total += float(row["price"] or 0)
        table_data.append([
            Paragraph(escape(row["room"]), body),
            Paragraph(escape(row["category"]), body),
            Paragraph(escape(row["name"]), body),
            Paragraph(escape(f"Rs. {row['price']:,.0f}"), body),
        ])
    table_data.append(["", "", Paragraph("<b>TOTAL</b>", body), Paragraph(f"<b>Rs. {total:,.0f}</b>", body)])
    products_table = Table(table_data, colWidths=[1.3 * inch, 1.4 * inch, 3.0 * inch, 1.1 * inch])
    products_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f4f6f9")]),
    ]))
    story.append(products_table)
    doc.build(story)
    return path


def _paths_for_html(html_path: str) -> Tuple[str, str]:
    abs_html = os.path.abspath(html_path)
    dest_dir = os.path.dirname(abs_html) or "."
    stem = os.path.splitext(os.path.basename(abs_html))[0]
    return (
        os.path.join(dest_dir, f"{stem}_recommendations.pdf"),
        os.path.join(dest_dir, f"{stem}_recommendations.xlsx"),
    )


def _read_attachment(path: str) -> Tuple[str, bytes]:
    with open(path, "rb") as handle:
        return os.path.basename(path), handle.read()


def _send_via_sendgrid(subject: str, body: str, attachments: Iterable[str]) -> bool:
    api_key = os.getenv("SENDGRID_API_KEY") or ""
    sender = (os.getenv("SENDGRID_FROM") or "").strip()
    if not api_key or not sender:
        return False
    payload_attachments = []
    for path in attachments:
        filename, raw = _read_attachment(path)
        mime = "application/pdf" if filename.lower().endswith(".pdf") else XLSX_MIME
        payload_attachments.append({
            "content": base64.b64encode(raw).decode("ascii"),
            "type": mime,
            "filename": filename,
            "disposition": "attachment",
        })
    response = requests.post(
        "https://api.sendgrid.com/v3/mail/send",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "personalizations": [{"to": [{"email": support_email_address()}]}],
            "from": {"email": sender, "name": "BetterHome"},
            "subject": subject,
            "content": [{"type": "text/plain", "value": body}],
            "attachments": payload_attachments,
        },
        timeout=45,
    )
    if response.status_code in (200, 202):
        print(f"Support email sent via SendGrid to {support_email_address()}")
        return True
    print(f"SendGrid email failed ({response.status_code}): {response.text[:500]}")
    return False


def _send_via_smtp(subject: str, body: str, attachments: Iterable[str]) -> bool:
    host = (os.getenv("SMTP_HOST") or "").strip()
    sender = (os.getenv("SMTP_FROM") or os.getenv("SMTP_USER") or "").strip()
    if not host or not sender:
        return False
    port = int(os.getenv("SMTP_PORT") or "587")
    user = os.getenv("SMTP_USER") or ""
    password = os.getenv("SMTP_PASSWORD") or ""
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = support_email_address()
    message.set_content(body)
    for path in attachments:
        filename, raw = _read_attachment(path)
        mime = "application/pdf" if filename.lower().endswith(".pdf") else XLSX_MIME
        maintype, _, subtype = mime.partition("/")
        message.add_attachment(raw, maintype=maintype, subtype=subtype, filename=filename)

    with smtplib.SMTP(host, port, timeout=45) as smtp:
        smtp.ehlo()
        if os.getenv("SMTP_STARTTLS", "true").lower() not in ("0", "false", "no"):
            smtp.starttls()
            smtp.ehlo()
        if user:
            smtp.login(user, password)
        smtp.send_message(message)
    print(f"Support email sent via SMTP to {support_email_address()}")
    return True


def send_support_email(subject: str, body: str, attachments: List[str]) -> bool:
    existing = [path for path in attachments if path and os.path.isfile(path)]
    if not existing:
        print("Support email skipped: no PDF/Excel files to attach")
        return False
    if not mail_configured():
        print("Support email skipped: set SENDGRID_API_KEY + SENDGRID_FROM, or SMTP_HOST + SMTP_FROM")
        return False
    try:
        if os.getenv("SENDGRID_API_KEY") and os.getenv("SENDGRID_FROM"):
            if _send_via_sendgrid(subject, body, existing):
                return True
        if os.getenv("SMTP_HOST") and os.getenv("SMTP_FROM"):
            return _send_via_smtp(subject, body, existing)
        print("Support email skipped: mail provider responded unsuccessfully or is incomplete")
        return False
    except Exception as exc:
        print(f"Warning: support email failed: {exc}")
        return False


def export_and_email_recommendations(
    user_data: Optional[Dict[str, Any]],
    recommendations: Optional[Dict[str, Any]],
    html_path: str,
) -> bool:
    """Write PDF + Excel next to the generated HTML and email them to support."""
    try:
        if not html_path:
            return False
        pdf_path, xlsx_path = _paths_for_html(html_path)
        write_recommendations_pdf(pdf_path, user_data or {}, recommendations or {})
        write_recommendations_excel(xlsx_path, user_data or {}, recommendations or {})
        print(f"Wrote support exports: {pdf_path} , {xlsx_path}")
        phone = (user_data or {}).get("mobile") or (user_data or {}).get("phone")
        commit_generated_files([html_path, pdf_path, xlsx_path], phone=phone)
        fields = dict(client_fields(user_data or {}))
        subject = f"New recommendation: {fields.get('Name', 'Customer')} ({fields.get('Phone', '')})"
        body = (
            "A new BetterHome recommendation was generated. "
            "PDF and Excel are attached so support can follow up even if the customer did not download them.\n\n"
            + "\n".join(f"{label}: {value}" for label, value in client_fields(user_data or {}))
            + f"\n\nSource HTML: {os.path.basename(html_path)}\n"
        )
        return send_support_email(subject, body, [pdf_path, xlsx_path])
    except Exception as exc:
        print(f"Warning: support export/email failed: {exc}")
        return False
