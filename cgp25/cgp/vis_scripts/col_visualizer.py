#!/usr/bin/env python3
"""Visualize CGP "Active nodes by column" logs.

The expected input format is lines like:

    123: Active nodes by column:8, 10, 6, 5, 3, 4, 4, 3, 2, 1,

This script reads the generation number and the per-column counts, then writes
an SVG line chart where each column is a differently colored line.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

LINE_RE = re.compile(r"^(\d+)\s*:\s*Active nodes by column:\s*(.*)$")


def parse_column_log(lines: Iterable[str]) -> Tuple[List[int], List[List[int]]]:
    generations: List[int] = []
    rows: List[List[int]] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        match = LINE_RE.match(line)
        if not match:
            continue

        generation = int(match.group(1))
        values_text = match.group(2).strip()
        values = [int(value.strip()) for value in values_text.split(",") if value.strip()]

        if not values:
            continue

        generations.append(generation)
        rows.append(values)

    if not rows:
        raise ValueError("No 'Active nodes by column' rows were found in the input file.")

    column_count = len(rows[0])
    for index, row in enumerate(rows):
        if len(row) != column_count:
            raise ValueError(
                f"Row {index + 1} has {len(row)} columns, expected {column_count}."
            )

    return generations, rows


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def hex_color(red: int, green: int, blue: int) -> str:
    return f"#{red:02x}{green:02x}{blue:02x}"


def viridis_like_color(value: float) -> str:
    value = clamp(value, 0.0, 1.0)
    stops = [
        (0.0, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.5, (33, 145, 140)),
        (0.75, (94, 201, 97)),
        (1.0, (253, 231, 37)),
    ]

    for index in range(len(stops) - 1):
        left_pos, left_color = stops[index]
        right_pos, right_color = stops[index + 1]
        if left_pos <= value <= right_pos:
            span = right_pos - left_pos or 1.0
            ratio = (value - left_pos) / span
            red = round(left_color[0] + (right_color[0] - left_color[0]) * ratio)
            green = round(left_color[1] + (right_color[1] - left_color[1]) * ratio)
            blue = round(left_color[2] + (right_color[2] - left_color[2]) * ratio)
            return hex_color(red, green, blue)

    return hex_color(*stops[-1][1])


def svg_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "normal") -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'font-family="DejaVu Sans, Arial, sans-serif" text-anchor="{anchor}" '
        f'font-weight="{weight}" fill="#1f2937">{svg_escape(text)}</text>'
    )


def line_plot_svg(
    generations: Sequence[int],
    rows: Sequence[Sequence[int]],
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
) -> str:
    margin_left = 52.0
    margin_right = 18.0
    margin_top = 28.0
    margin_bottom = 34.0
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    min_value = 0
    max_value = max(max(row) for row in rows)
    if max_value == min_value:
        max_value = min_value + 1

    min_generation = min(generations)
    max_generation = max(generations)
    generation_span = max_generation - min_generation or 1

    column_count = len(rows[0])
    colors = [
        "#0f766e",
        "#b45309",
        "#7c3aed",
        "#dc2626",
        "#2563eb",
        "#15803d",
        "#db2777",
        "#4b5563",
        "#ca8a04",
        "#0891b2",
    ]

    parts = [
        f'<g transform="translate({x:.2f}, {y:.2f})">',
        f'<rect x="0" y="0" width="{width:.2f}" height="{height:.2f}" rx="16" fill="#ffffff" stroke="#d1d5db"/>',
        svg_text(margin_left, 18, title, size=16, weight="700"),
        f'<line x1="{margin_left:.2f}" y1="{margin_top:.2f}" x2="{margin_left:.2f}" y2="{margin_top + plot_height:.2f}" stroke="#9ca3af"/>',
        f'<line x1="{margin_left:.2f}" y1="{margin_top + plot_height:.2f}" x2="{margin_left + plot_width:.2f}" y2="{margin_top + plot_height:.2f}" stroke="#9ca3af"/>',
    ]

    parts.append(svg_text(margin_left + plot_width / 2, margin_top + plot_height + 32, "Generation", size=12, anchor="middle", weight="700"))
    parts.append(svg_text(16, margin_top + plot_height / 2, "Amount", size=12, anchor="middle", weight="700"))

    y_ticks = 5
    for index in range(y_ticks + 1):
        ratio = index / y_ticks
        value = max_value - (max_value - min_value) * ratio
        py = margin_top + plot_height * ratio
        parts.append(f'<line x1="{margin_left:.2f}" y1="{py:.2f}" x2="{margin_left + plot_width:.2f}" y2="{py:.2f}" stroke="#e5e7eb"/>')
        parts.append(svg_text(margin_left - 8, py + 4, f"{int(round(value))}", size=11, anchor="end"))

    tick_count = min(8, len(generations))
    if tick_count > 1:
        for index in range(tick_count):
            position = round(index * (len(generations) - 1) / (tick_count - 1))
            generation = generations[position]
            px = margin_left + plot_width * ((generation - min_generation) / generation_span)
            parts.append(f'<line x1="{px:.2f}" y1="{margin_top + plot_height:.2f}" x2="{px:.2f}" y2="{margin_top + plot_height + 5:.2f}" stroke="#9ca3af"/>')
            parts.append(svg_text(px, margin_top + plot_height + 18, str(generation), size=11, anchor="middle"))

    for column_index in range(column_count):
        points: List[str] = []
        for generation, row in zip(generations, rows):
            px = margin_left + plot_width * ((generation - min_generation) / generation_span)
            py = margin_top + plot_height * (1.0 - ((row[column_index] - min_value) / (max_value - min_value)))
            points.append(f"{px:.2f},{py:.2f}")
        color = colors[column_index % len(colors)]
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>'
        )

    legend_x = margin_left + plot_width - 4
    legend_y = margin_top + 10
    for column_index in range(column_count):
        color = colors[column_index % len(colors)]
        row_y = legend_y + column_index * 17
        parts.append(f'<rect x="{legend_x - 84:.2f}" y="{row_y - 10:.2f}" width="10" height="10" fill="{color}"/>')
        parts.append(svg_text(legend_x - 68, row_y - 1, f"Col {column_index + 1}", size=11))

    parts.append("</g>")
    return "".join(parts)


def build_svg(generations: Sequence[int], rows: Sequence[Sequence[int]], mode: str, title: str) -> str:
    width = 1220
    height = 520
    body = line_plot_svg(generations, rows, 24, 24, width - 48, height - 48, title)

    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#f8fafc"/>\n'
        f'{body}\n'
        '</svg>\n'
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CGP active-node column logs.")
    parser.add_argument(
        "input",
        nargs="?",
        default="col.txt",
        help="Path to the log file produced by the CGP run.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="col_visualization.svg",
        help="Output SVG file. Defaults to col_visualization.svg.",
    )
    parser.add_argument(
        "--title",
        default="CGP Active Nodes by Column",
        help="Figure title prefix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    generations, rows = parse_column_log(input_path.read_text().splitlines())
    svg = build_svg(generations, rows, "line", args.title)
    output_path.write_text(svg, encoding="utf-8")

    print(f"Wrote {output_path} from {len(rows)} rows and {len(rows[0])} columns.")


if __name__ == "__main__":
    main()
