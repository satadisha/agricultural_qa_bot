import sys
from pathlib import Path

# Add the parent directory to the Python path so we can import from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.extract import extract_text_from_pdf

def test_pdf_extraction(pdf_path: str):
    pdf = Path(pdf_path)

    if not pdf.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    text = extract_text_from_pdf(pdf)

    print(f"\n📄 Extracted {len(text)} characters from: {pdf.name}")
    print("\n🔍 First 500 characters:\n")
    print(text[:500])


if __name__ == "__main__":
    # Adjust path relative to project root (not test/)
    test_pdf_extraction("data/belarus/belarus_2014.pdf")



# python test/test_extract.py