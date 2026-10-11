"""details.pdf: the supporting document, with fixed sections and no page limit.

1. How to read this report (fixed text from the catalog, plus what the AI judge skipped).
2. Every finding, one block each, grouped into problems in ranking order.
3. Run index, and every agent.
4. Check catalog (fixed text).

Rendered twice: the first pass counts the pages (for "Page n of N") and records the page each
finding starts on; the second writes the file. Both passes lay out the same content, so the
page numbers summary.pdf cites are the ones in the file.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
)

from ...catalog import catalog, label
from ..view import FindingView, ProblemView, ReportView, fmt_cost, fmt_pct, fmt_time, runs_of
from .base import (
    CONTENT_WIDTH,
    MARGIN_BOTTOM,
    MARGIN_TOP,
    MARGIN_X,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    RULE,
    SEVERITY_COLORS,
    Styles,
    deterministic_canvas,
    markup,
    table_style,
)

LABEL_WIDTH = 104.0


class Story:
    """The flowables of a document. Headings added with `lead` are kept on the page of what
    follows them: they join the next block (a KeepTogether), or else chain to the next
    paragraph or table through keepWithNext. (ReportLab can't chain a heading to a
    KeepTogether by itself.)"""

    def __init__(self) -> None:
        self.items: list[Flowable] = []
        self.pending: list[Flowable] = []

    def lead(self, *flowables: Flowable) -> None:
        self.pending += flowables

    def block(self, *flowables: Flowable) -> None:
        self.items.append(KeepTogether([*self.pending, *flowables]))
        self.pending = []

    def add(self, *flowables: Flowable) -> None:
        self.items += [*self.pending, *flowables]
        self.pending = []


class Anchor(Flowable):
    """Zero-size marker: records the page it lands on, and optionally a PDF bookmark."""

    def __init__(
        self, key: str, pages: dict[str, int], outline: tuple[str, int] | None = None
    ) -> None:
        super().__init__()
        self.key, self.pages, self.outline = key, pages, outline
        self.keepWithNext = 1  # travel with the heading that follows to its page

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        return 0.0, 0.0

    def draw(self) -> None:
        self.pages[self.key] = self.canv.getPageNumber()
        if self.outline:
            title, level = self.outline
            self.canv.bookmarkPage(self.key)
            self.canv.addOutlineEntry(title, self.key, level=level, closed=level > 0)


def _p(text: str, style: Any) -> Paragraph:
    return Paragraph(markup(text), style)


def _bullets(lines: list[str], s: Styles) -> list[Flowable]:
    return [Paragraph(markup(line), s.bullet, bulletText="•") for line in lines]


def _fields(rows: list[tuple[str, Any]], s: Styles) -> Table:
    data = []
    for key, value in rows:
        cell = value if isinstance(value, (Paragraph, Table, list)) else _p(str(value), s.small)
        data.append([_p(label(key), s.label), cell])
    table = Table(data, colWidths=[LABEL_WIDTH, CONTENT_WIDTH - LABEL_WIDTH])
    table.setStyle(
        table_style(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 1.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5),
            ]
        )
    )
    return table


def _tools(f: FindingView, s: Styles) -> list[Paragraph] | Paragraph:
    if not f.tools:
        return _p(label("no_tools"), s.small)
    out = []
    for i, t in enumerate(f.tools, 1):
        out.append(_p(label("tool_line", n=i, tool=t.name, args=t.args), s.small))
        key = "tool_error" if t.failed else "tool_result"
        out.append(_p(label(key, text=t.result), s.indented_muted))
    return out


def _finding(f: FindingView, s: Styles, pages: dict[str, int]) -> list[Flowable]:
    badges = label("finding_badges", severity=f.severity, confidence=f.confidence, method=f.method)
    heading = Paragraph(
        markup(label("finding_heading", id=f.id, title=f.title)),
        s.h3,
    )
    color = SEVERITY_COLORS.get(f.finding.severity)
    badge_style = s.small_muted.clone("badge", textColor=color) if color else s.small_muted
    rows: list[tuple[str, Any]] = [
        ("f_where", f.where),
        ("f_run", label("run_label", id=f.run_id, request=f.request)),
        ("f_what_happened", f.what_happened),
        ("f_tools", _tools(f, s)),
        ("f_output", f.output),
    ]
    if f.error:
        rows.append(("f_error", f.error))
    if f.quote:
        rows.append(("f_quote", f"“{f.quote}”"))
    change = f.what_to_change + (f"  ({f.change_kind})" if f.change_kind else "")
    rows += [
        ("f_change", change),
        ("f_id", f"{f.id} · {label('f_ref')} {f.reference}"),
    ]
    return [
        Anchor(f.id, pages),
        heading,
        _p(badges, badge_style),
        _p(f.looks_for, s.small_muted),
        Spacer(0, 3),
        _fields(rows, s),
        Spacer(0, 10),
    ]


def _problem(story: Story, p: ProblemView, total: int, s: Styles, pages: dict[str, int]) -> None:
    heading = label("group_heading", title=p.title, agent=p.agent, runs=runs_of(p.runs, total))
    story.lead(_p(heading, s.group))
    for f in p.findings:
        story.block(*_finding(f, s, pages))


def _table(head: list[str], rows: list[list[str]], widths: list[float], s: Styles) -> Table:
    data = [[_p(h, s.cell_head) for h in head]]
    data += [[_p(v, s.cell) for v in row] for row in rows]
    table = Table(data, colWidths=[CONTENT_WIDTH * w for w in widths], repeatRows=1)
    table.setStyle(
        table_style(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("LINEBELOW", (0, 0), (-1, 0), 0.75, RULE),
                ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE),
            ]
        )
    )
    return table


def _section(story: Story, key: str, pages: dict[str, int], s: Styles) -> None:
    title = label(key)
    story.lead(Anchor(key, pages, (title, 0)), _p(title, s.section))


def _story(view: ReportView, s: Styles, pages: dict[str, int]) -> list[Flowable]:
    how = catalog()["how_to_read"]
    total = view.run_count
    h = view.header
    story = Story()
    story.add(
        _p(label("details_title"), s.title),
        _p(label("details_subtitle"), s.small_muted),
        Spacer(0, 6),
        _fields(
            [
                ("header_system", h.system),
                ("header_dates", h.dates),
                ("header_runs", h.runs),
                ("header_models", h.models),
                ("header_cost", h.cost),
            ],
            s,
        ),
        Spacer(0, 14),
    )

    # 1. How to read this report
    _section(story, "section1", pages, s)
    story.add(_p(how["intro"], s.body), Spacer(0, 6))
    for title_key, body_key in (
        ("terms_title", "terms"),
        ("scorecard_title", "scorecard"),
        ("serious_title", "serious"),
        ("finding_title", "finding"),
        ("confidence_title", "confidence"),
        ("ranking_title", "ranking"),
        ("fixes_title", "fixes"),
        ("cost_title", "cost"),
    ):
        body = how[body_key]
        story.lead(_p(how[title_key], s.h3))
        story.add(*(_bullets(body, s) if isinstance(body, list) else [_p(body, s.body)]))
        story.add(Spacer(0, 6))
    story.lead(_p(how["caps_title"], s.h3))
    story.add(*_bullets(view.caps, s), Spacer(0, 14))

    # 2. Every finding
    _section(story, "section2", pages, s)
    if not view.main_problems and not view.low_problems:
        story.add(_p(label("no_findings"), s.body))
    for title_key, intro_key, problems in (
        ("section2_main", "section2_main_intro", view.main_problems),
        ("section2_low", "section2_low_intro", view.low_problems),
    ):
        if not problems:
            continue
        story.lead(_p(label(title_key), s.h2), _p(label(intro_key), s.intro))
        for p in problems:
            _problem(story, p, total, s, pages)
    story.add(Spacer(0, 8))

    # 3. Run index, and every agent
    _section(story, "section3", pages, s)
    story.add(
        _table(
            [
                label("idx_run"),
                label("idx_request"),
                label("idx_outcome"),
                label("idx_cost"),
                label("idx_time"),
                label("idx_steps"),
                label("idx_findings"),
            ],
            [[r.id, r.request, r.outcome, r.cost, r.time, r.steps, r.findings] for r in view.runs],
            [0.06, 0.39, 0.13, 0.09, 0.08, 0.07, 0.18],
            s,
        ),
        Spacer(0, 12),
    )
    story.lead(_p(label("agents_full_title"), s.h2), _p(label("agents_all_intro"), s.intro))
    if view.agents:
        story.add(
            _table(
                [
                    label("col_agent"),
                    label("col_runs"),
                    label("col_problem_share"),
                    label("col_median_cost"),
                    label("col_median_time"),
                    label("col_common_problem"),
                ],
                [
                    [
                        a.name,
                        f"{a.runs:,}",
                        label("runs_share", n=a.problem_runs, pct=fmt_pct(a.problem_share)),
                        fmt_cost(a.median_cost),
                        fmt_time(a.median_time),
                        a.common_problem,
                    ]
                    for a in view.agents
                ],
                [0.24, 0.08, 0.17, 0.12, 0.12, 0.27],
                s,
            )
        )
    else:
        story.add(_p(label("agents_none"), s.body))
    story.add(Spacer(0, 14))

    # 4. Check catalog
    _section(story, "section4", pages, s)
    for entry in view.checks:
        story.block(
            _p(entry.title, s.h3),
            _fields(
                [
                    ("cat_how", entry.how),
                    ("cat_looks_for", entry.looks_for),
                    ("cat_why", entry.why),
                ],
                s,
            ),
            Spacer(0, 6),
        )
    story.lead(_p(label("cat_signal"), s.h2))
    for entry in view.signals:
        story.block(
            _p(entry.title, s.h3), _fields([("cat_looks_for", entry.looks_for)], s), Spacer(0, 6)
        )
    story.add()
    return story.items


def _build(view: ReportView, target: Any, total_pages: int | None) -> tuple[int, dict[str, int]]:
    s = Styles()
    pages: dict[str, int] = {}
    canvas_class = deterministic_canvas(
        view.created, label("details_title"), view.header.system, total_pages
    )
    frame = Frame(
        MARGIN_X,
        MARGIN_BOTTOM + 8,
        CONTENT_WIDTH,
        PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM - 8,
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
        id="body",
    )
    doc = BaseDocTemplate(
        target,
        pagesize=(PAGE_WIDTH, PAGE_HEIGHT),
        pageTemplates=[PageTemplate(id="page", frames=[frame])],
        title=label("details_title"),
        author="tracedoc",
        invariant=1,
        pageCompression=0,
    )
    doc.build(_story(view, s, pages), canvasmaker=canvas_class)
    return canvas_class.pages_drawn, pages


def write_details(view: ReportView, path: Path) -> None:
    """Write details.pdf, and record in `view` the page each finding starts on."""
    total, _ = _build(view, BytesIO(), None)
    total_again, pages = _build(view, str(path), total)
    if total_again != total:  # the footer never changes the layout, so this can't happen
        raise RuntimeError("details.pdf page count changed between passes")
    view.finding_pages = {f.id: pages[f.id] for f in view.findings}
