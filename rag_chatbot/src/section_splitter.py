import re

HEADING_REGEX = re.compile(
    r"(?:^|\n)(?P<heading>(?:PART|CHAPTER|TITLE|SECTION|ANNEX)\s+[\w\dIVXLCDM\-\.]+[^\n]*)",
    re.IGNORECASE
)

ARTICLE_HEADING_REGEX = re.compile(
    r"(?:(?:^|\n)\s*)(?P<article>Article\s+(?P<number>\d{1,3}))\s*(?:\n|$)",
    flags=re.IGNORECASE
)

def split_into_section(text):
    sections = []
    hierarchy = {
        "PART": None,
        "CHAPTER": None,
        "TITLE": None,
        "SECTION": None,
        "ANNEX": None
    }

    matches = []
    for m in HEADING_REGEX.finditer(text):
        matches.append({
            "type": "structure",
            "match": m,
            "start": m.start(),
            "end": m.end()
        })
    for m in ARTICLE_HEADING_REGEX.finditer(text):
        matches.append({
            "type": "article",
            "match": m,
            "start": m.start(),
            "end": m.end(),
            "article_number": int(m.group("number"))
        })

    matches.sort(key=lambda x: x["start"])

    # 1. Preamble
    if matches and matches[0]["start"] > 0:
        preamble = text[:matches[0]["start"]].strip()
        if preamble:
            sections.append({
                "heading": "PREAMBLE",
                "text": preamble,
                "sub_heading": []
            })

    # 2. Loop through matches and build sections
    i = 0
    while i < len(matches):
        m = matches[i]
        match_obj = m["match"]
        start = m["end"]

        # Determine where to end this section
        end = len(text)
        for j in range(i + 1, len(matches)):
            if matches[j]["type"] == "article":
                # Only split if it's exactly the next article
                if matches[j]["article_number"] == m.get("article_number", -999) + 1:
                    end = matches[j]["start"]
                    break
            elif matches[j]["type"] == "structure":
                end = matches[j]["start"]
                break

        content = text[start:end].strip()

        if m["type"] == "structure":
            heading_text = match_obj.group("heading").strip()
            for level in hierarchy:
                if heading_text.upper().startswith(level):
                    hierarchy[level] = heading_text
                    # Clear lower hierarchy
                    for k in list(hierarchy.keys())[list(hierarchy).index(level) + 1:]:
                        hierarchy[k] = None
                    break

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
