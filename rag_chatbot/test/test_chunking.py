import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.extract import extract_text_from_pdf
from src.section_splitter import split_into_section
from src.chunking import chunk_text, count_tokens


def test_chunking(pdf_path: str, max_tokens: int = 512, sentence_overlap: int = 2):
    pdf = Path(pdf_path)

    if not pdf.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    # 1. Extract
    text = extract_text_from_pdf(pdf)
    print(f"\n📄 Extracted {len(text)} characters from: {pdf.name}")

    # 2. Split into sections
    sections = split_into_section(text)
    print(f"🧩 Total sections: {len(sections)}")

    # 3. Chunk each section
    total_chunks = 0
    for idx, section in enumerate(sections[:]):  # Limit to first 10 sections for display
        heading = section["heading"]
        section_text = section["text"]
        chunks = chunk_text(
            section_text,
            heading,
            max_tokens=max_tokens,
            sentence_overlap=sentence_overlap
        )

        print(f"\n--- Section {idx+1} ---")
        print(f"📌 Heading: {heading}")
        print(f"✏️  Section length: {len(section_text)} characters")
        print(f"📦 Chunks: {len(chunks)}")

        for i, chunk in enumerate(chunks[:2]):  # Show only first 2 chunks
            char_len = len(chunk['text'])
            token_len = count_tokens(chunk['text'])
            print(f"   • Chunk {i+1}: {char_len} characters | {token_len} tokens")
            print(f"     Text preview:\n{chunk['text'][:300]}...\n")

        total_chunks += len(chunks)

    print(f"\n✅ Total chunks from all sections: {total_chunks}")


if __name__ == "__main__":
    # Adjust max_tokens or sentence_overlap as needed
    test_chunking("data/belarus/belarus_2014.pdf", max_tokens=512, sentence_overlap=2)




# python test/test_chunking.py