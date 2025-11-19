import re  # used for matching headings and article patterns in text

"""
This module provides utilities for splitting long legal/regulatory documents
into structured sections based on:
- Major document hierarchy levels (PART, CHAPTER, TITLE, SECTION, ANNEX)
- Individual article headings (e.g., "Article 1", "Article 2", ...)

The output of `split_into_section` is a list of dicts, each representing a
section or article with extracted text, heading, and optional subheadings.

Intended use:
This function is typically called before chunking and embedding, as part of
a RAG pipeline.
"""

# Regular expression to detect headings such as PART, CHAPTER, TITLE, SECTION, ANNEX
HEADING_REGEX = re.compile(
    r"(?:^|\n)(?P<heading>(?:PART|CHAPTER|TITLE|SECTION|ANNEX)\s+[\w\dIVXLCDM\-\.]+[^\n]*)",
    re.IGNORECASE
)
HEADING_REGEX.__doc__ = """
Regular expression for capturing structural document headings such as:

    PART I
    CHAPTER 2
    TITLE IV — GENERAL PROVISIONS
    SECTION 3. Definitions
    ANNEX A

The regex captures:
- Start of a line (^ or newline)
- One of the heading labels
- A sequence that may include numbers, roman numerals, hyphens, or periods
"""

# Regular expression to detect article headings like "Article 1", "Article 2", etc.
ARTICLE_HEADING_REGEX = re.compile(
    r"(?:(?:^|\n)\s*)(?P<article>Article\s+(?P<number>\d{1,3}))\s*(?:\n|$)",
    flags=re.IGNORECASE
)
ARTICLE_HEADING_REGEX.__doc__ = """
Regular expression for detecting article-level headings such as:

    Article 1
    Article 2
    Article 15

The captured group 'number' is used to determine sequencing and section boundaries.
"""


def split_into_section(text):
    """
    Split a full-document string into structured sections and articles.

    This function performs a hierarchical decomposition of text using:
    - Structural headings: PART, CHAPTER, TITLE, SECTION, ANNEX
    - Article headings: Article 1, Article 2, ...

    The function maintains a hierarchy dictionary to build fully-qualified headings
    for each article, such as:

        PART I > CHAPTER 2 > Article 5

    Args:
        text (str):
            The full raw text extracted from a PDF or other document.

    Returns:
        list[dict]:
            A list of sections, where each section has the structure:
                {
                    "heading": str,        # full hierarchical heading
                    "text": str,           # text belonging to this section/article
                    "sub_heading": list    # specific headings (e.g., ["Article 3"])
                }

    Notes:
        - Preamble text before the first detected heading is captured as a "PREAMBLE" section.
        - Articles are split only if the next detected article follows the numeric sequence.
        - Structural headings update the currently active hierarchy.
    """

    sections = []  # list to store the split sections
    hierarchy = {  # active document structure levels
        "PART": None,
        "CHAPTER": None,
        "TITLE": None,
        "SECTION": None,
        "ANNEX": None
    }

    matches = []  # store both structure and article matches

    # 1. Collect structural headings
    for m in HEADING_REGEX.finditer(text):
        matches.append({
            "type": "structure",
            "match": m,
            "start": m.start(),
            "end": m.end()
        })

    # 2. Collect article headings
    for m in ARTICLE_HEADING_REGEX.finditer(text):
        matches.append({
            "type": "article",
            "match": m,
            "start": m.start(),
            "end": m.end(),
            "article_number": int(m.group("number"))
        })

    # Sort all matches by their appearance in the text
    matches.sort(key=lambda x: x["start"])

    # Capture any text before the first heading as "PREAMBLE"
    if matches and matches[0]["start"] > 0:
        preamble = text[:matches[0]["start"]].strip()
        if preamble:
            sections.append({
                "heading": "PREAMBLE",
                "text": preamble,
                "sub_heading": []
            })

    # 3. Process all matched headings in order
    i = 0
    while i < len(matches):
        m = matches[i]
        match_obj = m["match"]
        start = m["end"]

        # Determine where the next section starts
        end = len(text)
        for j in range(i + 1, len(matches)):
            if matches[j]["type"] == "article":
                # Split only if the next article follows numerically (Article N+1)
                if matches[j]["article_number"] == m.get("article_number", -999) + 1:
                    end = matches[j]["start"]
                    break
            elif matches[j]["type"] == "structure":
                end = matches[j]["start"]
                break

        content = text[start:end].strip()

        # Handling structural headings (PART / CHAPTER / TITLE / SECTION / ANNEX)
        if m["type"] == "structure":
            heading_text = match_obj.group("heading").strip()

            # Update hierarchy
            for level in hierarchy:
                if heading_text.upper().startswith(level):
                    hierarchy[level] = heading_text
                    # Clear lower levels after this level
                    for k in list(hierarchy.keys())[list(hierarchy).index(level) + 1:]:
                        hierarchy[k] = None
                    break

        # Handling article sections
        elif m["type"] == "article":
            heading_text = match_obj.group("article").strip()

            # Build full heading path
            full_heading = " > ".join([v for v in hierarchy.values() if v]) + f" > {heading_text}"

            # Save article section
            sections.append({
                "heading": full_heading.strip(),
                "text": content,
                "sub_heading": [heading_text]
            })

        i += 1

    return sections
