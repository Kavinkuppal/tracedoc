"""Shared PDF pieces: bundled fonts, colors, text styles, safe text, a deterministic canvas.

Output depends only on the report's content. ReportLab runs in invariant mode with page
compression off, the creation date is the latest run's end time (not the clock), and the
fonts (Inter 4.1, SIL Open Font License) ship inside the package, so no machine fonts are
used. Text is never set below MIN_FONT_SIZE.
"""

from __future__ import annotations

import math
import unicodedata
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib.colors import Color, HexColor
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import TableStyle

from ...__about__ import __version__

FONTS_DIR = Path(__file__).resolve().parent.parent / "fonts"
REGULAR, SEMIBOLD = "Inter", "Inter-SemiBold"
_FONT_FILES = {REGULAR: "Inter-Regular.ttf", SEMIBOLD: "Inter-SemiBold.ttf"}

PAGE_WIDTH, PAGE_HEIGHT = LETTER
MARGIN_X = 40.0
MARGIN_TOP = 38.0
MARGIN_BOTTOM = 34.0
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X
MIN_FONT_SIZE = 9.0

INK = HexColor("#1F2328")
MUTED = HexColor("#59636E")
RULE = HexColor("#D1D9E0")
PANEL = HexColor("#F6F8FA")
WHITE = HexColor("#FFFFFF")
SEVERITY_COLORS = {"high": HexColor("#B42318"), "medium": HexColor("#B54708"), "low": MUTED}
# Agent map shading: one hue, light to dark, by share of runs with a serious problem.
RAMP_LOW = HexColor("#FDE8E4")
RAMP_HIGH = HexColor("#8F1D13")


def ramp(share: float) -> Color:
    """Fill for a share from 0 to 1 (0 is the neutral panel color)."""
    if share <= 0:
        return PANEL
    t = min(1.0, share)
    return Color(
        RAMP_LOW.red + (RAMP_HIGH.red - RAMP_LOW.red) * t,
        RAMP_LOW.green + (RAMP_HIGH.green - RAMP_LOW.green) * t,
        RAMP_LOW.blue + (RAMP_HIGH.blue - RAMP_LOW.blue) * t,
    )


def ink_on(fill: Color) -> Color:
    """Text color that stays readable on `fill`."""
    luminance = 0.2126 * fill.red + 0.7152 * fill.green + 0.0722 * fill.blue
    return WHITE if luminance < 0.5 else INK


@cache
def register_fonts() -> None:
    for name, file in _FONT_FILES.items():
        pdfmetrics.registerFont(TTFont(name, str(FONTS_DIR / file)))
    pdfmetrics.registerFontFamily(
        REGULAR, normal=REGULAR, bold=SEMIBOLD, italic=REGULAR, boldItalic=SEMIBOLD
    )


@cache
def _covered() -> frozenset[int]:
    register_fonts()
    font = pdfmetrics.getFont(REGULAR)
    return frozenset(font.face.charToGlyph)  # type: ignore[attr-defined]


def safe(text: Any) -> str:
    """Text the bundled font can draw: control characters become spaces, and characters the
    font lacks (for example CJK or emoji) become '?'."""
    covered = _covered()
    out = []
    for ch in str(text):
        if ch in "\n\t\r":
            out.append(" ")
        elif unicodedata.category(ch) in ("Cc", "Cf", "Cs", "Co", "Cn"):
            continue
        elif ord(ch) in covered:
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


def fit_middle(text: Any, font: str, size: float, width: float) -> str:
    """`text` shortened in the middle ("very_long_ag…name_2") so it fits `width` points.
    Names that share a long prefix stay distinguishable by their ends."""
    text = safe(text)
    full = stringWidth(text, font, size)
    if full <= width:
        return text
    n = min(len(text) - 1, int(len(text) * width / full) + 2)  # start near the answer
    while n > 0:
        head = text[: math.ceil(n * 0.6)]
        tail = text[len(text) - (n - len(head)) :] if n > len(head) else ""
        candidate = head + "…" + tail
        if stringWidth(candidate, font, size) <= width:
            return candidate
        n -= 1
    return "…"


def fit_lines(text: Any, font: str, size: float, width: float, lines: int = 2) -> str:
    """`text` shortened in the middle so it takes at most `lines` lines of `width` points."""
    return fit_middle(text, font, size, width * lines * 0.9)


def markup(text: Any) -> str:
    """Safe text, escaped for a Paragraph (which reads a small XML markup)."""
    return escape(safe(text))


def style(
    name: str,
    size: float = 9.5,
    *,
    font: str = REGULAR,
    leading: float | None = None,
    color: Color = INK,
    **extra: Any,
) -> ParagraphStyle:
    assert size >= MIN_FONT_SIZE, f"{name}: {size} pt is below the {MIN_FONT_SIZE} pt minimum"
    register_fonts()
    return ParagraphStyle(
        name,
        fontName=font,
        fontSize=size,
        leading=leading or round(size * 1.3, 1),
        textColor=color,
        bulletFontName=REGULAR,
        bulletFontSize=size,
        **extra,
    )


def table_style(commands: list[tuple[Any, ...]]) -> TableStyle:
    """A TableStyle whose cell font is Inter at 9 pt (ReportLab's default is Helvetica 10 pt,
    which it sets on every cell even when the cell holds a Paragraph)."""
    register_fonts()
    return TableStyle(
        [
            ("FONTNAME", (0, 0), (-1, -1), REGULAR),
            ("FONTSIZE", (0, 0), (-1, -1), MIN_FONT_SIZE),
            ("LEADING", (0, 0), (-1, -1), MIN_FONT_SIZE * 1.3),
            *commands,
        ]
    )


class Styles:
    """Every text style the PDFs use."""

    def __init__(self) -> None:
        self.title = style("title", 20, font=SEMIBOLD, leading=24, keepWithNext=1)
        self.h2 = style("h2", 13, font=SEMIBOLD, leading=16, keepWithNext=1)
        self.h3 = style("h3", 11, font=SEMIBOLD, leading=14, keepWithNext=1)
        self.verdict = style("verdict", 12.5, font=SEMIBOLD, leading=16)
        self.body = style("body", 9.5, leading=12.5)
        self.body_semibold = style("body_semibold", 9.5, font=SEMIBOLD, leading=12.5)
        self.bullet = style("bullet", 9.5, leading=12.5, leftIndent=11, bulletIndent=1)
        self.intro = style("intro", 9, leading=11.5, color=MUTED, keepWithNext=1, spaceAfter=8)
        self.section = style("section", 20, font=SEMIBOLD, leading=24, keepWithNext=1, spaceAfter=6)
        self.group = style(
            "group", 13, font=SEMIBOLD, leading=16, keepWithNext=1, spaceBefore=6, spaceAfter=4
        )
        self.indented_muted = style("indented_muted", 9, leading=11.5, color=MUTED, leftIndent=12)
        self.label = style("label", 9, leading=11.5, color=MUTED)
        self.small = style("small", 9, leading=11.5)
        self.small_muted = style("small_muted", 9, leading=11.5, color=MUTED)
        self.small_right = style("small_right", 9, leading=11.5, color=MUTED, alignment=2)
        self.tile_value = style("tile_value", 17, font=SEMIBOLD, leading=20)
        self.card_title = style("card_title", 11, font=SEMIBOLD, leading=14)
        self.cell = style("cell", 9, leading=11.5)
        self.cell_head = style("cell_head", 9, font=SEMIBOLD, leading=11.5, color=MUTED)
        self.mono = style("mono", 9, leading=11.5, color=INK)

    def all(self) -> list[ParagraphStyle]:
        return [v for v in vars(self).values() if isinstance(v, ParagraphStyle)]


def pdf_date(created: datetime) -> str:
    return created.strftime("D:%Y%m%d%H%M%S+00'00'")


def deterministic_canvas(
    created: datetime, title: str, subject: str, total_pages: int | None
) -> type[Canvas]:
    """A Canvas class with fixed metadata that writes "Page n of N" on every page.

    With total_pages=None nothing is written in the footer: that is the measuring pass of a
    document whose length isn't known yet (details.pdf is rendered twice).
    """
    stamp = pdf_date(created)

    class ReportCanvas(Canvas):
        pages_drawn = 0

        def __init__(self, filename: Any, *args: Any, **kwargs: Any) -> None:
            register_fonts()
            kwargs["invariant"] = 1
            kwargs["pageCompression"] = 0
            kwargs["initialFontName"] = REGULAR  # not Helvetica: only the bundled fonts appear
            kwargs["initialFontSize"] = MIN_FONT_SIZE
            kwargs["initialLeading"] = MIN_FONT_SIZE * 1.3
            kwargs.setdefault("pagesize", LETTER)
            super().__init__(filename, *args, **kwargs)
            self.setDateFormatter(lambda *_: stamp)
            self.setTitle(title)
            self.setSubject(subject)
            self.setAuthor("tracedoc")
            self.setCreator(f"tracedoc {__version__}")

        def showPage(self) -> None:
            if total_pages is not None:
                self._page_number(total_pages)
            ReportCanvas.pages_drawn = self.getPageNumber()
            super().showPage()

        def _page_number(self, total: int) -> None:
            from ...catalog import label

            self.saveState()
            self.setFont(REGULAR, MIN_FONT_SIZE)
            self.setFillColor(MUTED)
            self.drawRightString(
                PAGE_WIDTH - MARGIN_X,
                MARGIN_BOTTOM - 16,
                safe(label("page_of", page=self.getPageNumber(), pages=total)),
            )
            self.restoreState()

    return ReportCanvas
