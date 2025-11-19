import fitz  # used to open and read PDF files
import re    # used for finding and fixing text patterns

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file and clean spaced-out uppercase words.

    This function:
      - Opens the PDF using PyMuPDF (fitz)
      - Reads text from each page and concatenates it with newline separators
      - Detects words where letters are separated by spaces (e.g., "D E C I D E D")
      - Removes the spaces between the letters to fix them (→ "DECIDED")

    Parameters:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Cleaned text extracted from the PDF.
    """
    
    doc = fitz.open(pdf_path)  # open the PDF file
    raw_text = "\n".join(page.get_text() for page in doc)  # extract text from each page
    
    # fix words with spaces between letters (e.g., "D E C I D E D" → "DECIDED")
    cleaned_text = re.sub(
        r'\b(?:[A-Z]\s){2,}[A-Z]\b',               # detect spaced-out uppercase words
        lambda m: m.group(0).replace(" ", ""),     # remove internal spaces
        raw_text
    )

    return cleaned_text  # return the cleaned text





