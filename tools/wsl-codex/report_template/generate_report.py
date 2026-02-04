import json
import os
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics import renderPM
from reportlab.graphics.shapes import Drawing, Rect, String

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CONFIG_PATH = os.path.join(BASE_DIR, "report_config.json")
CONTENT_PATH = os.path.join(BASE_DIR, "content.md")
OUTPUT_PATH = os.path.join(os.path.dirname(BASE_DIR), "report_template.pdf")


def ensure_assets():
    os.makedirs(ASSETS_DIR, exist_ok=True)

    logo_path = os.path.join(ASSETS_DIR, "logo_placeholder.png")
    if not os.path.exists(logo_path):
        d = Drawing(240, 120)
        d.add(Rect(0, 0, 240, 120, strokeColor=colors.grey, fillColor=colors.whitesmoke))
        d.add(String(120, 60, "LOGO", textAnchor="middle", fontSize=24, fillColor=colors.grey))
        renderPM.drawToFile(d, logo_path, fmt="PNG")

    fig_path = os.path.join(ASSETS_DIR, "figure_placeholder.png")
    if not os.path.exists(fig_path):
        d = Drawing(600, 320)
        d.add(Rect(0, 0, 600, 320, strokeColor=colors.grey, fillColor=colors.whitesmoke))
        d.add(String(300, 160, "FIGURE PLACEHOLDER", textAnchor="middle", fontSize=18, fillColor=colors.grey))
        renderPM.drawToFile(d, fig_path, fmt="PNG")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_content():
    with open(CONTENT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip().splitlines()


def build_styles():
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="TitleLarge",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=34,
            textColor=colors.black,
            spaceAfter=10,
        )
    )

    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Title"],
            fontName="Helvetica",
            fontSize=16,
            leading=20,
            textColor=colors.grey,
            spaceAfter=20,
        )
    )

    styles.add(
        ParagraphStyle(
            name="Heading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.black,
            spaceBefore=12,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=16,
            textColor=colors.black,
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=9, textColor=colors.grey
        )
    )

    return styles


def header_footer(canvas, doc, title, version, date_str):
    canvas.saveState()
    width, height = A4

    header_y = height - 1.5 * cm
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, header_y, title)
    canvas.drawRightString(width - 2 * cm, header_y, f"{version} | {date_str}")

    footer_y = 1.2 * cm
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawCentredString(width / 2, footer_y, str(doc.page))

    canvas.restoreState()


def build_cover_story(config, styles):
    story = []
    story.append(Spacer(1, 6 * cm))
    story.append(Paragraph(config["title"], styles["TitleLarge"]))
    story.append(Paragraph(config["subtitle"], styles["Subtitle"]))

    authors = ", ".join(config.get("authors", []))
    meta_lines = [
        authors,
        config.get("affiliation", ""),
        f"Date: {config.get('date', '')}",
        f"Version: {config.get('version', '')}",
    ]
    for line in meta_lines:
        if line:
            story.append(Paragraph(line, styles["Body"]))

    return story


def build_body_story(config, styles):
    content_lines = load_content()
    story = []

    for line in content_lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith("# "):
            heading = line[2:]
            story.append(Paragraph(heading, styles["Heading"]))
            continue
        if line == "[TABLE:sample]":
            story.append(Spacer(1, 6))
            table_data = [
                ["Metric", "Value", "Notes"],
                ["Accuracy", "92%", "Validation set"],
                ["Precision", "0.88", "Macro average"],
                ["Recall", "0.91", "Macro average"],
            ]
            tbl = Table(table_data, colWidths=[5 * cm, 4 * cm, 7 * cm])
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                        ("TOPPADDING", (0, 0), (-1, 0), 6),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 12))
            continue
        if line == "[FIGURE:sample]":
            story.append(Spacer(1, 6))
            fig_path = os.path.join(ASSETS_DIR, "figure_placeholder.png")
            img = Image(fig_path, width=14 * cm, height=7.5 * cm)
            img.hAlign = "LEFT"
            story.append(img)
            story.append(Paragraph("Figure 1. Sample figure placeholder.", styles["Small"]))
            story.append(Spacer(1, 12))
            continue

        if line.startswith("- "):
            story.append(Paragraph(line[2:], styles["Body"]))
        else:
            story.append(Paragraph(line, styles["Body"]))

    return story


def build_pdf():
    ensure_assets()
    config = load_config()
    styles = build_styles()

    doc = BaseDocTemplate(
        OUTPUT_PATH,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2 * cm,
        title=config.get("title", "Report"),
        author=", ".join(config.get("authors", [])),
    )

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height - 0.8 * cm, id="normal")

    def on_page(canvas, doc_obj):
        header_footer(canvas, doc_obj, config.get("title", "Report"), config.get("version", ""), config.get("date", ""))

    page_template = PageTemplate(id="body", frames=[frame], onPage=on_page)
    doc.addPageTemplates([page_template])

    story = []

    # Cover page (no header/footer)
    cover_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="cover")

    def cover_on_page(canvas, doc_obj):
        width, height = A4
        logo_path = os.path.join(ASSETS_DIR, "logo_placeholder.png")
        logo = ImageReader(logo_path)
        canvas.drawImage(logo, width - 4 * cm, height - 3 * cm, width=2.2 * cm, height=1.1 * cm, mask="auto")

    cover_template = PageTemplate(id="cover", frames=[cover_frame], onPage=cover_on_page)
    doc.addPageTemplates([cover_template])

    story.extend(build_cover_story(config, styles))
    story.append(PageBreak())

    story.extend(build_body_story(config, styles))

    doc.build(story)


if __name__ == "__main__":
    build_pdf()
