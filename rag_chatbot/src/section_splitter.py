import re  # used for matching headings and article patterns in text

# Regular expression to detect headings like PART, CHAPTER, TITLE, SECTION, or ANNEX
HEADING_REGEX = re.compile(
    r"(?:^|\n)(?P<heading>(?:PART|CHAPTER|TITLE|SECTION|ANNEX)\s+[\w\dIVXLCDM\-\.]+[^\n]*)",
    re.IGNORECASE
)

# Regular expression to detect articles like "Article 1", "Article 2", etc.
ARTICLE_HEADING_REGEX = re.compile(
    r"(?:(?:^|\n)\s*)(?P<article>Article\s+(?P<number>\d{1,3}))\s*(?:\n|$)",
    flags=re.IGNORECASE
)

def split_into_section(text):
    sections = []  # list to store the split sections
    hierarchy = {  # track document structure levels
        "PART": None,
        "CHAPTER": None,
        "TITLE": None,
        "SECTION": None,
        "ANNEX": None
    }

    matches = []  # store both structure and article matches

    # find structure headings (PART, CHAPTER, etc.)
    for m in HEADING_REGEX.finditer(text):
        matches.append({
            "type": "structure",
            "match": m,
            "start": m.start(),
            "end": m.end()
        })

    # find article headings (Article 1, Article 2, etc.)
    for m in ARTICLE_HEADING_REGEX.finditer(text):
        matches.append({
            "type": "article",
            "match": m,
            "start": m.start(),
            "end": m.end(),
            "article_number": int(m.group("number"))
        })

    # sort matches in order as they appear in text
    matches.sort(key=lambda x: x["start"])

    # 1. Handle preamble text before the first heading
    if matches and matches[0]["start"] > 0:
        preamble = text[:matches[0]["start"]].strip()
        if preamble:
            sections.append({
                "heading": "PREAMBLE",
                "text": preamble,
                "sub_heading": []
            })

    # 2. Loop through matches and create sections
    i = 0
    while i < len(matches):
        m = matches[i]
        match_obj = m["match"]
        start = m["end"]

        # find where the next section or article starts
        end = len(text)
        for j in range(i + 1, len(matches)):
            if matches[j]["type"] == "article":
                # split only if next article follows in sequence
                if matches[j]["article_number"] == m.get("article_number", -999) + 1:
                    end = matches[j]["start"]
                    break
            elif matches[j]["type"] == "structure":
                end = matches[j]["start"]
                break

        content = text[start:end].strip()  # extract section content

        if m["type"] == "structure":
            heading_text = match_obj.group("heading").strip()
            for level in hierarchy:
                if heading_text.upper().startswith(level):
                    hierarchy[level] = heading_text
                    # clear all lower levels after the current one
                    for k in list(hierarchy.keys())[list(hierarchy).index(level) + 1:]:
                        hierarchy[k] = None
                    break

        elif m["type"] == "article":
            heading_text = match_obj.group("article").strip()
            # build full heading based on the current hierarchy
            full_heading = " > ".join([v for v in hierarchy.values() if v]) + f" > {heading_text}"

            # save this article section
            sections.append({
                "heading": full_heading.strip(),
                "text": content,
                "sub_heading": [heading_text]
            })

        i += 1  # move to next match

    return sections  # return all extracted sections

