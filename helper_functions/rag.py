from pypdf import PdfReader
import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from helper_functions.chunk_class import Chunk

# global vars
SECTION_RE = re.compile(r"^\s*(\d+)\s+[A-Z].+")
SUBSECTION_RE = re.compile(r"^\s*([a-z])\s+[A-Z].+")

# keywords for definition detection to be used for BM25 boosting and merging
DEF_TITLE_KEYWORDS = [
    "definition", "definitions", "define", "meaning", "glossary",
    "diagnostic criteria", "staging", "stage", "classification"
]

DEF_CONTENT_KEYWORDS = [
    "means", "refers to", "shall mean", "is defined as",
    "diagnosed", "diagnosis", "criteria", "confirmed by a specialist",
    "must meet all of the following"
]

MEDICAL_HINT_KEYWORDS = [
    # moderate list, enough to catch some medical contexts
    "carcinoma", "metastasis", "tumour", "tumor", "malignant",
    "chemotherapy", "radiotherapy", "mri", "ct scan", "biopsy",
    "coronary", "angiography", "renal", "neurological",
    "parkinson", "alzheimer", "copd", "stroke"
]


# extract and read pdfs
def extract_pages(_pdf_path):
    reader = PdfReader(_pdf_path)
    texts = []
    for p in reader.pages:
        texts.append(p.extract_text() or "")
    return texts


# flatten pages into lines with page numbers using enumerate
def flatten_pages_to_lines(_pages):
    lines = []
    for page_num, text in enumerate(_pages, start=1):
        # split by line breaks into dictionary objects
        for line in text.splitlines():
            lines.append({"page": page_num, "line": line.rstrip()})
    return lines


# common grouping function for sections and subsections
def grouping(_lines, _pattern, _default_title, _final_end_page=None, _initial_start_page=None):
    blocks = []
    current = {
        "title": _default_title,
        "lines": [],
        "start_page": _initial_start_page or _lines[0]["page"],
    }

    for entry in _lines:
        page = entry["page"]
        text_line = entry["line"]

        if _pattern.match(text_line):
            if current["lines"]:
                current["end_page"] = page
                blocks.append(current)
            current = {
                "title": text_line.strip(),
                "lines": [],
                "start_page": page,
            }
        else:
            current["lines"].append(entry)

    current["end_page"] = current.get(
        "end_page",
        _final_end_page if _final_end_page is not None else current["start_page"]
    )
    blocks.append(current)
    return blocks


# group lines into sections
def group_sections(_lines):
    return grouping(_lines, SECTION_RE, "introduction")


# group lines into subsections
def group_subsections(_section):
    return grouping(
        _section["lines"],
        SUBSECTION_RE,
        "main",
        _final_end_page=_section.get("end_page"),
        _initial_start_page=_section.get("start_page")
    )

# convert chunks to paragraphs
def subsection_to_paragraphs(_subsection):
    paragraphs = []
    current = []

    for entry in _subsection["lines"]:
        line = entry["line"].strip()

        if line == "":
            if current:
                paragraphs.append("\n".join(current))
                current = []
        else:
            current.append(line)

    if current:
        paragraphs.append("\n".join(current))
    return paragraphs


# use langchain to further chunk long paragraphs
def chunk_with_langchain(_text, _size=600, _overlap=120):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_size,
        chunk_overlap=_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(_text)


# check for keywords indicating definition-like chunks
def looks_like_definition_chunk(_chunk):
    title = f"{_chunk.section} {_chunk.subsection}".lower()
    text = _chunk.text.lower()

    # title keyword hit
    if any(each_keyword in title for each_keyword in DEF_TITLE_KEYWORDS):
        return True

    # content keyword hit
    if any(each_keyword in text for each_keyword in DEF_CONTENT_KEYWORDS):
        return True

    # medical hits
    med_hits = sum(1 for each_keyword in MEDICAL_HINT_KEYWORDS if each_keyword in text)
    if med_hits >= 2:
        return True

    # bullet density heuristic
    bullet_count = _chunk.text.count("•")
    if bullet_count >= 5:
        return True

    # long multi-page definition blocks
    if (_chunk.page_end - _chunk.page_start) >= 2 and any(each_keyword in text for each_keyword in ["stage", "criteria", "means"]):
        return True

    return False


def merge_definition_chunks(_chunks):
    """
    Merge all definition-like chunks for a policy into one big chunk.
    Keep all other chunks unchanged.
    """
    if not _chunks:
        return _chunks

    policy_id = _chunks[0].policy_id
    def_chunks = [each_chunk for each_chunk in _chunks if looks_like_definition_chunk(each_chunk)]
    other_chunks = [each_chunk for each_chunk in _chunks if each_chunk not in def_chunks]

    if not def_chunks:
        return _chunks

    # combine all definitions into one text
    merged_text_parts = []
    page_start = min(each_chunk.page_start for each_chunk in def_chunks)
    page_end = max(each_chunk.page_end for each_chunk in def_chunks)

    # keep order by chunk_index so text flows
    for c in sorted(def_chunks, key=lambda x: x.chunk_index):
        merged_text_parts.append(
            f"[{c.section} | {c.subsection} | pages {c.page_start}-{c.page_end}]\n{c.text}"
        )

    merged_text = "\n\n---\n\n".join(merged_text_parts)

    # split merged definitions to avoid exceeding embedding limits
    splitter = RecursiveCharacterTextSplitter(
        # token limit
        chunk_size=4000,
        chunk_overlap=200,
        length_function=len
    )
    split_defs = splitter.split_text(merged_text)

    # convert each piece into its own chunk
    new_chunks = []
    offset = max(each_chunk.chunk_index for each_chunk in _chunks) + 1

    for index, piece in enumerate(split_defs, start=1):
        new_chunks.append(
            Chunk(
                policy_id=policy_id,
                section="definitions_merged",
                subsection=f"merged_part_{index}",
                page_start=page_start,
                page_end=page_end,
                chunk_index=offset + index,
                text=piece
            )
        )

    return other_chunks + new_chunks

# main hybrid chunking function
def hybrid_chunk_pdf(_pdf_path):
    policy_id = Path(_pdf_path).stem

    pages = extract_pages(_pdf_path)
    lines = flatten_pages_to_lines(pages)
    sections = group_sections(lines)

    chunks = []
    chunk_index = 0

    for each_sec in sections:
        sec_title = each_sec["title"]
        subsections = group_subsections(each_sec)

        for each_subsec in subsections:
            sub_title = each_subsec["title"]
            paragraphs = subsection_to_paragraphs(each_subsec)

            if not paragraphs:
                continue

            header = f"{sec_title}\n{sub_title}\n"
            long_text = header + "\n\n".join(paragraphs)
            lang_chunks = chunk_with_langchain(long_text)

            for each_chunk in lang_chunks:
                chunk_index += 1
                chunks.append(
                    Chunk(
                        policy_id=policy_id,
                        section=sec_title,
                        subsection=sub_title,
                        page_start=each_subsec["start_page"],
                        page_end=each_subsec["end_page"],
                        chunk_index=chunk_index,
                        text=each_chunk
                    )
                )
    # merge definition chunks per policy
    chunks = merge_definition_chunks(chunks)
    return chunks

