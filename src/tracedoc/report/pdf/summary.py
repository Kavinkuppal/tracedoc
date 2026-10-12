"""summary.pdf: exactly two pages, with a fixed layout.

Page 1, "What to fix": header, verdict, five scorecard tiles, and up to three fix cards.
Page 2, "Where and how often": agent map, agent table, other findings, what's working, footer.

Every block is measured before it is drawn, and nothing flows: the canvas is drawn by hand,
so there is no way to spill onto a third page. When something doesn't fit, rows are dropped
(the third card, agent rows, other-finding rows) and the page says where the rest is in
details.pdf. Text is never set below 9 pt.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reportlab.platypus import Flowable, Paragraph, Table

from ...catalog import label, units
from ..view import Card, ReportView, fmt_cost, fmt_pct, fmt_time
from .agent_map import AgentMap
from .base import (
    CONTENT_WIDTH,
    MARGIN_BOTTOM,
    MARGIN_TOP,
    MARGIN_X,
    MUTED,
    PAGE_HEIGHT,
    PANEL,
    REGULAR,
    RULE,
    SEVERITY_COLORS,
    Styles,
    deterministic_canvas,
    fit_lines,
    markup,
    table_style,
)

FOOTER_SPACE = 16.0
CARD_PAD = 7.0
CARD_LABEL_WIDTH = 86.0
TILE_GAP = 8.0
SECTION_GAP = 12.0


class OverflowError_(RuntimeError):
    """A block could not be placed even after dropping every optional row (a tracedoc bug)."""


@dataclass
class Cursor:
    """Top-down placement on one page, measuring before drawing."""

    canv: object
    y: float = PAGE_HEIGHT - MARGIN_TOP
    bottom: float = MARGIN_BOTTOM + FOOTER_SPACE

    @property
    def room(self) -> float:
        return self.y - self.bottom

    def place(self, flowable: Flowable, gap_after: float = 0.0, x: float = MARGIN_X) -> float:
        _, h = flowable.wrap(CONTENT_WIDTH, self.room)
        if h > self.room + 0.01:
            raise OverflowError_(
                f"{type(flowable).__name__} needs {h:.0f} pt, {self.room:.0f} left"
            )
        flowable.drawOn(self.canv, x, self.y - h)
        self.y -= h + gap_after
        return h

    def skip(self, h: float) -> None:
        self.y -= h


def height_of(flowable: Flowable, width: float = CONTENT_WIDTH) -> float:
    return flowable.wrap(width, 10_000)[1]


# ---- page 1 ----------------------------------------------------------------------------------


def _header(view: ReportView, s: Styles) -> Table:
    h = view.header

    def cell(key: str, value: str) -> list[Paragraph]:
        return [Paragraph(markup(label(key)), s.label), Paragraph(markup(value), s.body)]

    rows = [
        [
            cell("header_system", h.system),
            cell("header_dates", h.dates),
            cell("header_runs", h.runs),
        ],
        [cell("header_models", h.models), "", cell("header_cost", h.cost)],
    ]
    table = Table(
        rows, colWidths=[CONTENT_WIDTH * 0.42, CONTENT_WIDTH * 0.25, CONTENT_WIDTH * 0.33]
    )
    table.setStyle(
        table_style(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("SPAN", (0, 1), (1, 1)),
            ]
        )
    )
    return table


class Tiles(Flowable):
    """Five scorecard tiles in one row, all as tall as the tallest."""

    def __init__(self, view: ReportView, s: Styles) -> None:
        super().__init__()
        self.view, self.s = view, s
        self.tile_w = (CONTENT_WIDTH - TILE_GAP * (len(view.tiles) - 1)) / len(view.tiles)
        self.parts = [
            (
                Paragraph(markup(t.value), s.tile_value),
                Paragraph(markup(t.label), s.label),
                Paragraph(markup(t.detail), s.label) if t.detail else None,
            )
            for t in view.tiles
        ]
        inner = self.tile_w - 16
        self.height = (
            max(sum(p.wrap(inner, 10_000)[1] for p in part if p is not None) for part in self.parts)
            + 18
        )

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        return CONTENT_WIDTH, self.height

    def draw(self) -> None:
        c = self.canv
        for i, (value, name, detail) in enumerate(self.parts):
            x = i * (self.tile_w + TILE_GAP)
            c.saveState()
            c.setFillColor(PANEL)
            c.setStrokeColor(RULE)
            c.roundRect(x, 0, self.tile_w, self.height, 6, stroke=1, fill=1)
            c.restoreState()
            y = self.height - 9
            for p in (value, name, detail):
                if p is None:
                    continue
                _, h = p.wrap(self.tile_w - 16, 10_000)
                p.drawOn(c, x + 8, y - h)
                y -= h + (2 if p is value else 0)


class CardBox(Flowable):
    """One fix: title and confidence, then labeled rows, in a box with a severity bar."""

    def __init__(self, card: Card, page_of: Callable[[str], int], s: Styles) -> None:
        super().__init__()
        self.card = card
        self.color = SEVERITY_COLORS.get(card.severity, MUTED)
        inner = CONTENT_WIDTH - 2 * CARD_PAD - 4
        self.title = Paragraph(f"{card.rank}.&nbsp;&nbsp;{markup(card.title)}", s.card_title)
        self.confidence = Paragraph(
            f"{markup(label('card_confidence'))}: {markup(card.confidence)}", s.small_right
        )
        evidence = label("evidence_ref", id=card.finding_id, page=page_of(card.finding_id))
        rows = [
            ("card_where", card.where),
            ("card_what_happened", card.what_happened),
            ("card_why", card.why),
            ("card_change", card.what_to_change),
            ("card_evidence", evidence),
        ]
        self.table = Table(
            [
                [Paragraph(markup(label(k)), s.label), Paragraph(markup(v), s.small)]
                for k, v in rows
            ],
            colWidths=[CARD_LABEL_WIDTH, inner - CARD_LABEL_WIDTH],
        )
        self.table.setStyle(
            table_style(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 1),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
                ]
            )
        )
        self.inner = inner
        conf_w = 200.0
        self.conf_w = conf_w
        self.title_h = max(
            self.title.wrap(inner - conf_w - 8, 10_000)[1],
            self.confidence.wrap(conf_w, 10_000)[1],
        )
        self.table_h = self.table.wrap(inner, 10_000)[1]
        self.height = CARD_PAD + self.title_h + 4 + self.table_h + CARD_PAD

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        return CONTENT_WIDTH, self.height

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        c.setStrokeColor(RULE)
        c.roundRect(0, 0, CONTENT_WIDTH, self.height, 6, stroke=1, fill=0)
        c.setFillColor(self.color)
        c.rect(0, 0, 4, self.height, stroke=0, fill=1)
        c.restoreState()
        x = 4 + CARD_PAD
        top = self.height - CARD_PAD
        self.title.wrap(self.inner - self.conf_w - 8, 10_000)
        self.title.drawOn(c, x, top - self.title_h + (self.title_h - self.title.height))
        self.confidence.wrap(self.conf_w, 10_000)
        self.confidence.drawOn(c, x + self.inner - self.conf_w, top - self.confidence.height)
        self.table.wrap(self.inner, 10_000)
        self.table.drawOn(c, x, top - self.title_h - 4 - self.table_h)


def _page_one(cursor: Cursor, view: ReportView, s: Styles, page_of: Callable[[str], int]) -> None:
    cursor.place(Paragraph(markup(label("summary_title")), s.title), 6)
    cursor.place(_header(view, s), 4)
    cursor.place(Paragraph(markup(view.verdict), s.verdict), SECTION_GAP - 2)
    cursor.place(Tiles(view, s), SECTION_GAP)
    cursor.place(Paragraph(markup(label("top_fixes_title")), s.h2), 6)
    if not view.cards:
        cursor.place(Paragraph(markup(label("top_fixes_none")), s.body))
        return
    if view.fewer_fixes:
        note = label("fixes_fewer", problems=units("problems", len(view.cards)))
        cursor.place(Paragraph(markup(note), s.small_muted), 6)
    boxes = [CardBox(card, page_of, s) for card in view.cards]
    more = Paragraph(markup(label("more_fixes")), s.small_muted)
    shown = len(boxes)
    while shown > 1:  # drop whole cards from the end until the rest fit
        need = sum(b.height for b in boxes[:shown]) + 6 * (shown - 1)
        if shown < len(boxes):
            need += 6 + height_of(more)
        if need <= cursor.room:
            break
        shown -= 1
    for i, box in enumerate(boxes[:shown]):
        cursor.place(box, 6 if i < shown - 1 else 0)
    if shown < len(boxes):
        cursor.skip(6)
        cursor.place(more)


# ---- page 2 ----------------------------------------------------------------------------------


def _agent_table(view: ReportView, s: Styles, rows: int) -> Table:
    head = [
        label("col_agent"),
        label("col_runs"),
        label("col_problem_share"),
        label("col_median_cost"),
        label("col_median_time"),
        label("col_common_problem"),
    ]
    data = [[Paragraph(markup(h), s.cell_head) for h in head]]
    name_width = CONTENT_WIDTH * 0.24 - 8
    for a in view.agents[:rows]:
        values = [
            fit_lines(a.name, REGULAR, s.cell.fontSize, name_width),
            f"{a.runs:,}",
            label("runs_share", n=a.problem_runs, pct=fmt_pct(a.problem_share)),
            fmt_cost(a.median_cost),
            fmt_time(a.median_time),
            a.common_problem,
        ]
        data.append([Paragraph(markup(v), s.cell) for v in values])
    widths = [0.24, 0.08, 0.17, 0.12, 0.12, 0.27]
    table = Table(data, colWidths=[CONTENT_WIDTH * w for w in widths], repeatRows=1)
    table.setStyle(_grid_style())
    return table


def _other_table(view: ReportView, s: Styles, rows: int) -> Table:
    head = [label("col_problem"), label("col_agent"), label("col_runs_affected")]
    data = [[Paragraph(markup(h), s.cell_head) for h in head]]
    for r in view.other_rows[:rows]:
        agent = fit_lines(r.agent, REGULAR, s.cell.fontSize, CONTENT_WIDTH * 0.3 - 8)
        data.append([Paragraph(markup(v), s.cell) for v in (r.problem, agent, r.runs)])
    table = Table(data, colWidths=[CONTENT_WIDTH * w for w in (0.5, 0.3, 0.2)])
    table.setStyle(_grid_style())
    return table


def _grid_style() -> Any:
    return table_style(
        [
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ("LINEBELOW", (0, 0), (-1, 0), 0.75, RULE),
            ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE),
        ]
    )


def _page_two(cursor: Cursor, view: ReportView, s: Styles) -> None:
    cursor.place(Paragraph(markup(label("summary_page2_title")), s.title), 8)
    footer = Paragraph(markup(label("footer")), s.small_muted)

    def build(map_agents: int, agent_rows: int, other_rows: int, working: int) -> list[Flowable]:
        out: list[Flowable] = [Paragraph(markup(label("agent_map_title")), s.h3)]
        if view.agents:
            out.append(
                AgentMap(view.agents, view.edges, view.entry_agent, CONTENT_WIDTH, s, map_agents)
            )
        out.append(Paragraph(markup(label("agent_table_title")), s.h3))
        if view.agents:
            out.append(_agent_table(view, s, agent_rows))
            hidden_agents = len(view.agents) - agent_rows
            if hidden_agents > 0:
                out.append(Paragraph(markup(label("agents_more", n=hidden_agents)), s.small_muted))
        else:
            out.append(Paragraph(markup(label("agents_none")), s.body))
        out.append(Paragraph(markup(label("other_title")), s.h3))
        if other_rows:
            out.append(_other_table(view, s, other_rows))
        else:
            out.append(Paragraph(markup(label("other_none")), s.body))
        hidden = view.other_hidden + len(view.other_rows) - other_rows
        if hidden > 0:
            out.append(Paragraph(markup(label("other_more", n=hidden)), s.small_muted))
        out.append(Paragraph(markup(label("working_title")), s.h3))
        lines = view.working[:working]
        if lines:
            out += [Paragraph(markup(line), s.bullet, bulletText="•") for line in lines]
        else:
            out.append(Paragraph(markup(label("working_none")), s.body))
        return out

    def total(blocks: list[Flowable]) -> float:
        return sum(height_of(b) + _gap_after(b, s) for b in blocks)

    limits = {
        "map": min(len(view.agents), 8),
        "agents": len(view.agents),
        "other": len(view.other_rows),
        "working": len(view.working),
    }
    room = cursor.room - height_of(footer) - 6
    # Drop rows until everything fits: agent rows first (down to 3), then other findings
    # (down to 3), then more agent rows, other rows, map boxes, and finally the working lines.
    order = [("agents", 3), ("other", 3), ("agents", 1), ("other", 0), ("map", 2), ("working", 0)]
    blocks = build(limits["map"], limits["agents"], limits["other"], limits["working"])
    for key, floor in order:
        while total(blocks) > room and limits[key] > floor:
            limits[key] -= 1
            blocks = build(limits["map"], limits["agents"], limits["other"], limits["working"])
    for b in blocks:
        cursor.place(b, _gap_after(b, s))
    cursor.y = cursor.bottom + height_of(footer)  # pin the footer to the bottom
    cursor.place(footer)


def _gap_after(block: Flowable, s: Styles) -> float:
    if isinstance(block, Paragraph) and block.style is s.h3:
        return 4.0
    if isinstance(block, Paragraph) and block.style is s.bullet:
        return 2.0
    return SECTION_GAP - 2


def write_summary(view: ReportView, path: Path) -> None:
    s = Styles()
    canvas_class = deterministic_canvas(
        view.created, label("summary_title"), view.header.system, total_pages=2
    )
    canv = canvas_class(str(path))

    def page_of(finding_id: str) -> int:
        return view.finding_pages[finding_id]

    _page_one(Cursor(canv), view, s, page_of)
    canv.showPage()
    _page_two(Cursor(canv), view, s)
    canv.showPage()
    canv.save()
