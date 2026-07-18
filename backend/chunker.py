import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_type: str  # "text" or "table"


def _split_into_blocks(text: str) -> list[tuple[str, str]]:
    """
    Splits raw text into a list of (block_type, block_content) tuples.
    A block is either a contiguous markdown table, or a paragraph of plain text.
    """
    lines = text.split("\n")
    blocks = []
    current_text_lines = []
    current_table_lines = []
    in_table = False

    def flush_text():
        nonlocal current_text_lines
        if current_text_lines:
            joined = "\n".join(current_text_lines).strip()
            if joined:
                blocks.append(("text", joined))
            current_text_lines = []

    def flush_table():
        nonlocal current_table_lines
        if current_table_lines:
            blocks.append(("table", "\n".join(current_table_lines)))
            current_table_lines = []

    table_line_pattern = re.compile(r"^\s*\|.*\|\s*$")

    for line in lines:
        is_table_line = bool(table_line_pattern.match(line))
        if is_table_line:
            if not in_table:
                flush_text()
                in_table = True
            current_table_lines.append(line)
        else:
            if in_table:
                flush_table()
                in_table = False
            current_text_lines.append(line)

    flush_text()
    flush_table()
    return blocks


def _split_long_text_block(text: str, max_chunk_size: int) -> list[str]:
    """
    Splits an oversized plain-text block into smaller, non-overlapping pieces,
    respecting paragraph boundaries where possible. Overlap between final
    chunks is handled once, by chunk_document's merge loop — not here, to
    avoid double-applying it.
    """
    if len(text) <= max_chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    pieces = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) <= max_chunk_size:
            current += ("\n\n" if current else "") + para
        else:
            if current.strip():
                pieces.append(current.strip())
            current = para

            # A single paragraph might exceed max_chunk_size on its own —
            # hard split, no overlap added here.
            while len(current) > max_chunk_size:
                pieces.append(current[:max_chunk_size])
                current = current[max_chunk_size:]

    if current.strip():
        pieces.append(current.strip())

    return pieces

def chunk_document(text: str, max_chunk_size: int = 800, overlap: int = 100) -> list[Chunk]:
    """
    Groups blocks into chunks up to max_chunk_size characters.
    Tables are always kept whole as a single chunk, even if larger than max_chunk_size —
    this is the table-aware guarantee. Text blocks are grouped together up to the size
    limit, with a small overlap carried forward for retrieval continuity.
    """
    blocks = _split_into_blocks(text)
    chunks: list[Chunk] = []

    current_text = ""

    for block_type, content in blocks:
        if block_type == "table":
            # Flush whatever text chunk we were building first
            if current_text.strip():
                chunks.append(Chunk(text=current_text.strip(), chunk_type="text"))
                current_text = ""
            # Table becomes its own chunk, whole, no matter its size
            chunks.append(Chunk(text=content, chunk_type="table"))
            continue

        # block_type == "text"
        # First, split this block itself if it's oversized on its own —
        
        # fixes long uninterrupted speaker turns in a transcript, which
        # have no internal structure to split on otherwise.
        pieces = _split_long_text_block(content, max_chunk_size)

        for piece in pieces:
            if len(current_text) + len(piece) <= max_chunk_size:
                current_text += ("\n\n" if current_text else "") + piece
            else:
                if current_text.strip():
                    chunks.append(Chunk(text=current_text.strip(), chunk_type="text"))
                overlap_text = current_text[-overlap:] if overlap and current_text else ""
                current_text = (overlap_text + "\n\n" + piece).strip()

    if current_text.strip():
        chunks.append(Chunk(text=current_text.strip(), chunk_type="text"))

    return chunks