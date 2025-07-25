import sys
from pathlib import Path

# Add project root to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.section_splitter import split_into_section
from src.extract import extract_text_from_pdf


def test_split_pdf(pdf_path: str):
    pdf = Path(pdf_path)

    if not pdf.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    # 1. Extract raw text
    text = extract_text_from_pdf(pdf)
    print(f"\n📄 Extracted {len(text)} characters from: {pdf.name}")

    # 2. Split into sections
    sections = split_into_section(text)
    print(f"🧩 Total sections found: {len(sections)}")

    # 3. Show details for each section (limit to first 10)
    for idx, section in enumerate(sections[:10]):
        heading = section['heading']
        content = section['text']
        char_count = len(content)

        print(f"\n--- Section {idx + 1} ---")
        print(f"📌 Heading: {heading}")
        print(f"✏️  Characters in section: {char_count}")
        print(f"📄 Content preview:\n{content[:200]}...")

if __name__ == "__main__":
    test_split_pdf("data/italy.pdf")



#. python test/test_splitter.py
