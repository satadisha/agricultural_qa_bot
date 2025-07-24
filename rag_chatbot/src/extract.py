import fitz
import re

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    raw_text = "\n".join(page.get_text() for page in doc)
    # Fix overly spaced words like: "D E C I D E D" -> "DECIDED"
    cleaned_text = re.sub(r'\b(?:[A-Z]\s){2,}[A-Z]\b', lambda m: m.group(0).replace(" ", ""), raw_text)

    return cleaned_text
    # return "\n".join(page.get_text() for page in doc)




