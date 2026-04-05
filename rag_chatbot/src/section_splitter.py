import re  # used for matching headings and article patterns in text

"""
Utilities for splitting long legal or regulatory documents into structured
sections based on major headings and article numbers. The main function
`split_into_section` returns a list of structured sections for downstream
chunking and embedding.
"""

# Structural heading patterns (PART, CHAPTER, TITLE, SECTION, ANNEX)
HEADING_REGEX = re.compile(
    r"(?:^|\n)(?P<heading>(?:PART|CHAPTER|TITLE|SECTION|ANNEX)\s+[\w\dIVXLCDM\-\.]+[^\n]*)",
    re.IGNORECASE
)
HEADING_REGEX.__doc__ = """Regex for capturing structural headings like PART I, CHAPTER 2, TITLE IV, SECTION 3, ANNEX A."""

# Article heading pattern: "Article 1", "Article 2", etc.
ARTICLE_HEADING_REGEX = re.compile(
    r"(?:(?:^|\n)\s*)(?P<article>Article\s+(?P<number>\d{1,3}))\s*(?:\n|$)",
    flags=re.IGNORECASE
)
ARTICLE_HEADING_REGEX.__doc__ = """Regex for capturing article headings such as 'Article 1' or 'Article 15'."""


def split_into_section(text):
    """Split a document into structured sections using headings and article numbers."""
    sections = []
    hierarchy = {
        "PART": None,
        "CHAPTER": None,
        "TITLE": None,
        "SECTION": None,
        "ANNEX": None
    }

    matches = []

    # Structural headings
    for m in HEADING_REGEX.finditer(text):
        matches.append({
            "type": "structure",
            "match": m,
            "start": m.start(),
            "end": m.end()
        })

    # Article headings
    for m in ARTICLE_HEADING_REGEX.finditer(text):
        matches.append({
            "type": "article",
            "match": m,
            "start": m.start(),
            "end": m.end(),
            "article_number": int(m.group("number"))
        })

    matches.sort(key=lambda x: x["start"])

    # Preamble before first heading
    if matches and matches[0]["start"] > 0:
        preamble = text[:matches[0]["start"]].strip()
        if preamble:
            sections.append({
                "heading": "PREAMBLE",
                "text": preamble,
                "sub_heading": []
            })

    i = 0
    while i < len(matches):
        m = matches[i]
        match_obj = m["match"]
        start = m["end"]

        # Determine end of this section
        end = len(text)
        for j in range(i + 1, len(matches)):
            if matches[j]["type"] == "article":
                if matches[j]["article_number"] == m.get("article_number", -999) + 1:
                    end = matches[j]["start"]
                    break
            elif matches[j]["type"] == "structure":
                end = matches[j]["start"]
                break

        content = text[start:end].strip()

        # Structural heading: update hierarchy
        if m["type"] == "structure":
            heading_text = match_obj.group("heading").strip()

            for level in hierarchy:
                if heading_text.upper().startswith(level):
                    hierarchy[level] = heading_text
                    # Clear lower levels
                    for k in list(hierarchy.keys())[list(hierarchy).index(level) + 1:]:
                        hierarchy[k] = None
                    break

        # Article heading: create a section
        elif m["type"] == "article":
            heading_text = match_obj.group("article").strip()
            full_heading = " > ".join([v for v in hierarchy.values() if v]) + f" > {heading_text}"

            sections.append({
                "heading": full_heading.strip(),
                "text": content,
                "sub_heading": [heading_text]
            })

        i += 1

    return sections
