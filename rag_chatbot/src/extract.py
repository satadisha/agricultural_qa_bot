import fitz  # used to open and read PDF files
import re    # used for finding and fixing text patterns


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF and fix spaced-out uppercase words."""
    doc = fitz.open(pdf_path)
    raw_text = "\n".join(page.get_text() for page in doc)

    cleaned_text = re.sub(
        r'\b(?:[A-Z]\s){2,}[A-Z]\b',
        lambda m: m.group(0).replace(" ", ""),
        raw_text
    )

    return cleaned_text






