import re

def landing_ai_full_section_splitter(doc_text):
    """
    Split a Landing AI Markdown-style document into sectional chunks WITHOUT removing any text.
    Preserves everything: summary, timestamps, IDs, URLs, bullets, marginalia, etc.
    
    Args:
        doc_text (str): Full Landing AI output text.
    
    Returns:
        List[dict]: Each dict contains 'section_title' and 'chunk_text'.
    """
    chunks = []

    # Step 1: Capture everything before the first heading as 'Pre-section'
    pre_heading_match = re.match(r"^(.*?)(?=\n# |\n[A-Z][a-z]+ :|\Z)", doc_text, flags=re.S)
    if pre_heading_match:
        intro_text = pre_heading_match.group(1).strip()
        if intro_text:
            chunks.append({
                "section_title": "Pre-section",
                "chunk_text": intro_text
            })

    # Step 2: Split by Markdown headings (# ...) OR capitalized labels ending with ':' (e.g., "Summary :", "Design Elements :")
    pattern = r"(^# .+?$|^[A-Z][A-Za-z0-9 &-]+ :)"
    matches = list(re.finditer(pattern, doc_text, flags=re.M))
    
    for i, match in enumerate(matches):
        start_idx = match.end()
        section_title = match.group().strip().strip("# ").strip()
        
        # Determine end of chunk
        if i + 1 < len(matches):
            end_idx = matches[i + 1].start()
        else:
            end_idx = len(doc_text)
        
        chunk_text = doc_text[start_idx:end_idx].strip()
        chunks.append({
            "section_title": section_title,
            "chunk_text": chunk_text
        })

    return chunks


# ----------------------
# Example usage
# ----------------------
md_file_path = "doc2/Clean Energy Market Analysis in the US.extraction.md"

with open(md_file_path, "r", encoding="utf-8") as f:
    doc_text = f.read()

chunks = landing_ai_full_section_splitter(doc_text)
for idx, chunk in enumerate(chunks, 1):
    print(f"--- Chunk {idx} ---")
    print("Section:", chunk["section_title"])
    print(chunk["chunk_text"])
    print()
