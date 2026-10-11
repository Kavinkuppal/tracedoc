"""The agent map: agents as boxes, handoffs as arrows, drawn as vector graphics.

Layout is deterministic. Agents form connected groups (one per system when several are
analyzed together). Within a group, agents are placed in columns by their handoff distance
from the agent that usually starts the run; groups sit side by side when they fit, else one
above the other; a chain too long for the page falls back to a grid. At most MAX_AGENTS boxes
are drawn (the agents in the most runs), plus one box counting the rest.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass

from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import Flowable, Paragraph

from ...catalog import label
from ..view import AgentRow, fmt_pct
from .base import (
    INK,
    MIN_FONT_SIZE,
    MUTED,
    PANEL,
    REGULAR,
    RULE,
    SEMIBOLD,
    WHITE,
    Styles,
    fit_middle,
    ink_on,
    markup,
    ramp,
    safe,
)

MAX_AGENTS = 8
MAX_COLUMNS = 4
BOX_HEIGHT = 34.0
MAX_BOX_WIDTH = 150.0
ROW_GAP = 22.0
COMPONENT_GAP = 30.0
LEGEND_HEIGHT = 30.0
NAME_SIZE = 9.5
SHARE_SIZE = MIN_FONT_SIZE


@dataclass
class Box:
    name: str
    x: float
    y: float
    """Bottom-left corner, in the flowable's coordinates."""
    w: float
    h: float
    share: float | None
    """Share of runs with a serious problem; None for the "more agents" box."""

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2


def _components(names: list[str], edges: list[tuple[str, str, int]]) -> list[list[str]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    for a, b, _ in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)
    seen: set[str] = set()
    groups = []
    for name in names:  # names arrive in display order: most runs first
        if name in seen:
            continue
        group, queue = [], deque([name])
        seen.add(name)
        while queue:
            n = queue.popleft()
            group.append(n)
            for m in sorted(neighbors[n]):
                if m not in seen:
                    seen.add(m)
                    queue.append(m)
        groups.append(group)
    return groups


def _layers(
    group: list[str], edges: list[tuple[str, str, int]], start: str, order: dict[str, int]
) -> list[list[str]]:
    """Columns by handoff distance from `start`, following arrows (then any link)."""
    out: dict[str, list[str]] = defaultdict(list)
    undirected: dict[str, list[str]] = defaultdict(list)
    for a, b, _ in edges:
        out[a].append(b)
        undirected[a].append(b)
        undirected[b].append(a)
    depth = {start: 0}
    for adjacency in (out, undirected):
        queue = deque(sorted(depth, key=lambda n: (depth[n], order[n])))
        while queue:
            n = queue.popleft()
            for m in sorted(adjacency[n], key=lambda m: order[m]):
                if m in order and m in group and m not in depth:
                    depth[m] = depth[n] + 1
                    queue.append(m)
    for n in group:
        depth.setdefault(n, 0)
    columns: dict[int, list[str]] = defaultdict(list)
    for n in sorted(group, key=lambda n: (depth[n], order[n])):
        columns[depth[n]].append(n)
    return [columns[d] for d in sorted(columns)]


class AgentMap(Flowable):
    """The map plus its legend, sized to `width`."""

    def __init__(
        self,
        agents: list[AgentRow],
        edges: list[tuple[str, str, int]],
        entry: str | None,
        width: float,
        styles: Styles,
        max_agents: int = MAX_AGENTS,
    ) -> None:
        super().__init__()
        self.width = width
        self.styles = styles
        shown = agents[:max_agents]
        self.hidden = len(agents) - len(shown)
        self.share = {a.name: a.serious_share for a in shown}
        names = [a.name for a in shown]
        order = {n: i for i, n in enumerate(names)}
        self.edges = [(a, b, n) for a, b, n in edges if a in order and b in order]
        self.boxes = self._layout(names, order, entry)
        self.height = max((b.y + b.h for b in self.boxes), default=LEGEND_HEIGHT)

    # ---- layout ------------------------------------------------------------------------------

    def _layout(self, names: list[str], order: dict[str, int], entry: str | None) -> list[Box]:
        if not names:
            return []
        groups = _components(names, self.edges)
        group_layers = []
        for group in groups:
            start = entry if entry in group else group[0]
            group_layers.append(_layers(group, self.edges, start, order))
        columns_side_by_side = sum(len(g) for g in group_layers)
        if self.hidden:
            columns_side_by_side += 1
        if columns_side_by_side <= MAX_COLUMNS:
            return self._place_side_by_side(group_layers)
        if max(len(g) for g in group_layers) <= MAX_COLUMNS:
            return self._place_stacked(group_layers)
        return self._place_grid([n for g in group_layers for layer in g for n in layer])

    def _box_width(self, columns: int, gaps: float) -> float:
        return min(MAX_BOX_WIDTH, (self.width - gaps) / max(1, columns))

    def _column_gap(self, columns: int) -> float:
        return 48.0 if columns <= 3 else 34.0

    def _place_side_by_side(self, group_layers: list[list[list[str]]]) -> list[Box]:
        columns = sum(len(g) for g in group_layers) + (1 if self.hidden else 0)
        gap = self._column_gap(columns)
        group_gaps = COMPONENT_GAP * (len(group_layers) - 1 + (1 if self.hidden else 0))
        inner_gaps = gap * sum(len(g) - 1 for g in group_layers)
        w = self._box_width(columns, group_gaps + inner_gaps)
        rows = max([len(layer) for g in group_layers for layer in g] + [1])
        height = rows * BOX_HEIGHT + (rows - 1) * ROW_GAP
        used = columns * w + group_gaps + inner_gaps
        x = (self.width - used) / 2
        boxes = []
        for g in group_layers:
            for i, layer in enumerate(g):
                boxes += self._column(layer, x, height, w)
                x += w + (gap if i < len(g) - 1 else COMPONENT_GAP)
        if self.hidden:
            boxes.append(self._more_box(x, height / 2 - BOX_HEIGHT / 2, w))
        return self._lift(boxes)

    def _place_stacked(self, group_layers: list[list[list[str]]]) -> list[Box]:
        columns = max(len(g) for g in group_layers)
        gap = self._column_gap(columns)
        w = self._box_width(columns, gap * (columns - 1))
        boxes: list[Box] = []
        y_top = 0.0
        for g in group_layers:
            rows = max(len(layer) for layer in g)
            height = rows * BOX_HEIGHT + (rows - 1) * ROW_GAP
            used = len(g) * w + gap * (len(g) - 1)
            x = (self.width - used) / 2
            for layer in g:
                boxes += self._column(layer, x, height, w, y_offset=-y_top - height)
                x += w + gap
            y_top += height + ROW_GAP + 6
        if self.hidden:
            boxes.append(self._more_box((self.width - w) / 2, -y_top - BOX_HEIGHT, w))
        return self._lift(boxes)

    def _place_grid(self, names: list[str]) -> list[Box]:
        columns = MAX_COLUMNS
        gap = self._column_gap(columns)
        w = self._box_width(columns, gap * (columns - 1))
        items: list[str | None] = [*names, *([None] if self.hidden else [])]
        boxes = []
        for i, name in enumerate(items):
            row, col = divmod(i, columns)
            if row % 2:  # snake: odd rows run right to left, so a chain never jumps back
                col = columns - 1 - col
            x = col * (w + gap) + (self.width - columns * w - gap * (columns - 1)) / 2
            y = -(row + 1) * (BOX_HEIGHT + ROW_GAP)
            if name is None:
                boxes.append(self._more_box(x, y, w))
            else:
                boxes.append(Box(name, x, y, w, BOX_HEIGHT, self.share[name]))
        return self._lift(boxes)

    def _column(
        self, layer: list[str], x: float, height: float, w: float, y_offset: float = 0.0
    ) -> list[Box]:
        used = len(layer) * BOX_HEIGHT + (len(layer) - 1) * ROW_GAP
        top = y_offset + height - (height - used) / 2
        return [
            Box(n, x, top - (i + 1) * BOX_HEIGHT - i * ROW_GAP, w, BOX_HEIGHT, self.share[n])
            for i, n in enumerate(layer)
        ]

    def _more_box(self, x: float, y: float, w: float) -> Box:
        return Box(label("agent_map_more", n=self.hidden), x, y, w, BOX_HEIGHT, None)

    @staticmethod
    def _lift(boxes: list[Box]) -> list[Box]:
        """Shift so the lowest box sits just above the legend."""
        if not boxes:
            return boxes
        low = min(b.y for b in boxes)
        for b in boxes:
            b.y = b.y - low + LEGEND_HEIGHT
        return boxes

    # ---- drawing -----------------------------------------------------------------------------

    def wrap(self, avail_width: float, avail_height: float) -> tuple[float, float]:
        return self.width, self.height

    def draw(self) -> None:
        by_name = {b.name: b for b in self.boxes if b.share is not None}
        pairs = {(a, b) for a, b, _ in self.edges}
        lines = []
        for a, b, n in self.edges:
            two_way = (b, a) in pairs
            lines.append((self._arrow(by_name[a], by_name[b], offset=4.0 if two_way else 0.0), n))
        for box in self.boxes:
            self._box(box)
        taken = [(b.x - 2, b.y - 2, b.x + b.w + 2, b.y + b.h + 2) for b in self.boxes]
        for (x1, y1, x2, y2), n in lines:
            self._count(x1, y1, x2, y2, n, taken)
        self._legend()

    def _box(self, box: Box) -> None:
        c = self.canv
        c.saveState()
        if box.share is None:
            c.setStrokeColor(RULE)
            c.setDash(3, 2)
            c.setFillColor(WHITE)
            c.roundRect(box.x, box.y, box.w, box.h, 5, stroke=1, fill=1)
            c.setFillColor(MUTED)
            c.setFont(REGULAR, SHARE_SIZE)
            c.drawCentredString(
                box.cx, box.cy - 3, fit_middle(box.name, REGULAR, SHARE_SIZE, box.w - 10)
            )
            c.restoreState()
            return
        fill = ramp(box.share)
        c.setFillColor(fill)
        c.setStrokeColor(RULE if box.share == 0 else fill)
        c.roundRect(box.x, box.y, box.w, box.h, 5, stroke=1, fill=1)
        text = ink_on(fill)
        c.setFillColor(text)
        c.setFont(SEMIBOLD, NAME_SIZE)
        c.drawCentredString(
            box.cx, box.y + box.h - 14, fit_middle(box.name, SEMIBOLD, NAME_SIZE, box.w - 10)
        )
        c.setFont(REGULAR, SHARE_SIZE)
        c.drawCentredString(
            box.cx, box.y + 7, safe(label("agent_map_serious", pct=fmt_pct(box.share)))
        )
        c.restoreState()

    def _arrow(self, a: Box, b: Box, offset: float) -> tuple[float, float, float, float]:
        """Draw the arrow from a to b; return its end points."""
        dx, dy = b.cx - a.cx, b.cy - a.cy
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length * offset, dx / length * offset  # shift two-way pairs apart
        x1, y1 = _edge_point(a, b.cx + nx, b.cy + ny, nx, ny)
        x2, y2 = _edge_point(b, a.cx + nx, a.cy + ny, nx, ny)
        c = self.canv
        c.saveState()
        c.setStrokeColor(MUTED)
        c.setFillColor(MUTED)
        c.setLineWidth(1)
        ux, uy = (x2 - x1), (y2 - y1)
        seg = math.hypot(ux, uy) or 1.0
        ux, uy = ux / seg, uy / seg
        head = 6.0
        c.line(x1, y1, x2 - ux * head, y2 - uy * head)
        path = c.beginPath()
        path.moveTo(x2, y2)
        path.lineTo(x2 - ux * head - uy * 3, y2 - uy * head + ux * 3)
        path.lineTo(x2 - ux * head + uy * 3, y2 - uy * head - ux * 3)
        path.close()
        c.drawPath(path, stroke=0, fill=1)
        c.restoreState()
        return x1, y1, x2, y2

    def _count(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        count: int,
        taken: list[tuple[float, float, float, float]],
    ) -> None:
        """Write the handoff count on its arrow, at the first spot along the line (middle
        first, then toward either end) that overlaps no box and no other count."""
        text = safe(label("handoff_count", n=count))
        tw = stringWidth(text, REGULAR, MIN_FONT_SIZE) + 4
        th = 11.0

        def rect_at(t: float) -> tuple[float, float, float, float]:
            mx, my = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
            return mx - tw / 2, my - th / 2, mx + tw / 2, my + th / 2

        def overlap(r: tuple[float, float, float, float]) -> float:
            return sum(
                max(0.0, min(r[2], o[2]) - max(r[0], o[0]))
                * max(0.0, min(r[3], o[3]) - max(r[1], o[1]))
                for o in taken
            )

        candidates = [rect_at(t) for t in (0.5, 0.35, 0.65, 0.22, 0.78)]
        best = min(candidates, key=lambda r: (overlap(r) > 0, overlap(r)))  # first clear spot
        taken.append(best)
        c = self.canv
        c.saveState()
        c.setFillColor(WHITE)
        c.rect(best[0], best[1], tw, th, stroke=0, fill=1)
        c.setFillColor(INK)
        c.setFont(REGULAR, MIN_FONT_SIZE)
        c.drawCentredString((best[0] + best[2]) / 2, best[1] + 2.5, text)
        c.restoreState()

    def _legend(self) -> None:
        c = self.canv
        steps, step_w, y = 5, 14.0, 8.0
        c.saveState()
        c.setFont(REGULAR, MIN_FONT_SIZE)
        c.setFillColor(MUTED)
        c.drawString(0, y + 1, safe(label("agent_map_scale_low")))
        x0 = stringWidth(safe(label("agent_map_scale_low")), REGULAR, MIN_FONT_SIZE) + 4
        for i in range(steps):
            c.setFillColor(ramp((i + 1) / steps) if i else PANEL)
            c.setStrokeColor(RULE)
            c.rect(x0 + i * step_w, y, step_w, 9, stroke=1 if i == 0 else 0, fill=1)
        c.setFillColor(MUTED)
        end = x0 + steps * step_w + 4
        high = safe(label("agent_map_scale_high"))
        c.drawString(end, y + 1, high)
        c.restoreState()
        legend_x = end + stringWidth(high, REGULAR, MIN_FONT_SIZE) + 14
        para = Paragraph(markup(label("agent_map_legend")), self.styles.small_muted)
        _, h = para.wrap(self.width - legend_x, LEGEND_HEIGHT)
        para.drawOn(c, legend_x, max(0.0, y + 10 - h))


def _edge_point(box: Box, tx: float, ty: float, nx: float, ny: float) -> tuple[float, float]:
    """Where the line from the box's (shifted) center toward (tx, ty) leaves the box."""
    cx, cy = box.cx + nx, box.cy + ny
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    half_w, half_h = box.w / 2 + 2, box.h / 2 + 2
    scale = min(
        half_w / abs(dx) if dx else math.inf,
        half_h / abs(dy) if dy else math.inf,
    )
    return cx + dx * scale, cy + dy * scale
