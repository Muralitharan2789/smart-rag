import fitz  # this is pymupdf's import name
from docx import Document
from pathlib import Path


def parse_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF, converting tables to markdown syntax.
    Two detection passes are used:
      1. pymupdf's built-in find_tables() — catches tables with visible
         ruling/borders in the PDF's structure.
      2. A position-based heuristic fallback — catches tables that are only
         whitespace/column-aligned, with no visible borders at all (common
         in reports, financial statements, and government-style documents).
    Everything is reassembled in top-to-bottom reading order using each
    block's vertical position on the page.
    """
    doc = fitz.open(file_path)
    output_parts = []

    for page in doc:
        tables = page.find_tables()
        table_bboxes = [fitz.Rect(t.bbox) for t in tables]

        page_blocks = []  # list of (y0, content) — sorted into reading order at the end

        # Pass 1: real, ruled tables
        for t in tables:
            page_blocks.append((t.bbox[1], _table_to_markdown(t.extract())))

        # Pass 2: heuristic detection on whatever wasn't already claimed by a real table
        heuristic_tables, plain_lines = _detect_heuristic_tables(page, table_bboxes)
        for y0, md in heuristic_tables:
            page_blocks.append((y0, md))

        for y0, text in _merge_text_lines(plain_lines):
            page_blocks.append((y0, text))

        page_blocks.sort(key=lambda b: b[0])
        for _, content in page_blocks:
            if content.strip():
                output_parts.append(content)

    doc.close()
    return "\n\n".join(output_parts)


def _detect_heuristic_tables(page, exclude_bboxes, min_columns=2, min_rows=2):
    """
    Fallback table detector for tables with no visible borders — groups words
    into visual lines by y-position, then into columns by large horizontal
    gaps, and flags runs of consecutive lines with a consistent column count
    as a table.

    Tunable constants below (gap_threshold, y_tolerance, min_rows) are
    heuristics, not guarantees — if you test on more documents and see
    false positives (ordinary short lines mistaken for a table) or false
    negatives (a real borderless table still missed), these are the values
    to adjust first.
    """
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)

    # Drop any word that falls inside an already-detected real table
    filtered = [
        w for w in words
        if not any(fitz.Rect(w[0], w[1], w[2], w[3]).intersects(tb) for tb in exclude_bboxes)
    ]

    # Group words into visual lines (words sharing roughly the same y-position)
    filtered.sort(key=lambda w: (round(w[1]), w[0]))
    y_tolerance = 3
    lines, current_line, current_y = [], [], None
    for w in filtered:
        y = w[1]
        if current_y is None or abs(y - current_y) <= y_tolerance:
            current_line.append(w)
            current_y = current_y if current_y is not None else y
        else:
            lines.append(current_line)
            current_line, current_y = [w], y
    if current_line:
        lines.append(current_line)

    # Within each line, split into "cells" wherever there's a large horizontal gap
    gap_threshold = 15  # points
    line_cells = []
    for line in lines:
        line_sorted = sorted(line, key=lambda w: w[0])
        cells, current_cell, last_x1 = [], [], None
        for w in line_sorted:
            x0 = w[0]
            if last_x1 is not None and (x0 - last_x1) > gap_threshold:
                cells.append(" ".join(current_cell))
                current_cell = []
            current_cell.append(w[4])
            last_x1 = w[2]
        if current_cell:
            cells.append(" ".join(current_cell))
        line_cells.append((line[0][1], cells))

    # Find runs of consecutive lines with a stable, multi-column cell count
    heuristic_tables = []
    used_indices = set()
    i = 0
    while i < len(line_cells):
        y0, cells = line_cells[i]
        if len(cells) >= min_columns:
            run = [line_cells[i]]
            j = i + 1
            while j < len(line_cells) and len(line_cells[j][1]) == len(cells):
                run.append(line_cells[j])
                j += 1
            if len(run) >= min_rows:
                rows = [r[1] for r in run]
                heuristic_tables.append((run[0][0], _table_to_markdown(rows)))
                used_indices.update(range(i, j))
                i = j
                continue
        i += 1

    remaining_lines = [
        (y0, " ".join(cells))
        for idx, (y0, cells) in enumerate(line_cells)
        if idx not in used_indices
    ]

    return heuristic_tables, remaining_lines


def _merge_text_lines(lines, y_gap_threshold=20):
    """Merges consecutive single-line entries back into paragraph-sized text blocks."""
    if not lines:
        return []
    lines_sorted = sorted(lines, key=lambda l: l[0])
    merged = []
    current_y0 = lines_sorted[0][0]
    current_parts = [lines_sorted[0][1]]
    last_y = lines_sorted[0][0]
    for y0, text in lines_sorted[1:]:
        if y0 - last_y <= y_gap_threshold:
            current_parts.append(text)
        else:
            merged.append((current_y0, " ".join(current_parts)))
            current_y0 = y0
            current_parts = [text]
        last_y = y0
    merged.append((current_y0, " ".join(current_parts)))
    return merged


def parse_docx(file_path: str) -> str:
    """Extract text from a DOCX, converting Word tables to markdown syntax."""
    doc = Document(file_path)
    output_parts = []

    for element in doc.element.body:
        if element.tag.endswith("}p"):  # paragraph
            para_text = element.text
            if para_text and para_text.strip():
                output_parts.append(para_text.strip())
        elif element.tag.endswith("}tbl"):  # table
            table_data = _docx_table_to_rows(element, doc)
            output_parts.append(_table_to_markdown(table_data))

    return "\n\n".join(output_parts)


def _docx_table_to_rows(tbl_element, doc) -> list[list[str]]:
    from docx.table import Table
    table = Table(tbl_element, doc)
    return [[cell.text.strip() for cell in row.cells] for row in table.rows]


def _table_to_markdown(rows: list[list[str]]) -> str:
    """Convert a list-of-rows table into markdown table syntax."""
    if not rows or not rows[0]:
        return ""

    clean_rows = [[str(cell) if cell is not None else "" for cell in row] for row in rows]
    header = clean_rows[0]
    separator = ["---"] * len(header)
    body = clean_rows[1:]

    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def parse_document(file_path: str) -> str:
    """Entry point: dispatches to the right parser based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")