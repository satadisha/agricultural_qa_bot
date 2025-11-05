import fitz  # used to open and read PDF files
import re    # used for finding and fixing text patterns

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)  # open the PDF file
    raw_text = "\n".join(page.get_text() for page in doc)  # get text from each page and join with new lines
    
    # fix words with spaces between letters (like "D E C I D E D" → "DECIDED")
    cleaned_text = re.sub(
        r'\b(?:[A-Z]\s){2,}[A-Z]\b',       # find spaced-out uppercase words
        lambda m: m.group(0).replace(" ", ""),  # remove spaces between letters
        raw_text
    )

    return cleaned_text  # return the cleaned text





