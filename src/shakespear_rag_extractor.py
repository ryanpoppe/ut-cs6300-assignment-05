import requests
from bs4 import BeautifulSoup
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urljoin

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

BASE_URL = "http://shakespeare.mit.edu/"
INDEX_URL = BASE_URL

DB_DIR = Path("./shakespeare_db")
COLLECTION_NAME = "shakespeare_speeches"
EMBEDDER = SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L6-v2")

act_re = re.compile(r"^ACT\s+([IVXLC]+)", re.IGNORECASE)
scene_re = re.compile(r"^SCENE\s+([IVXLC]+)\.?\s*(.*)$", re.IGNORECASE)
line_num_re = re.compile(r"(\d+\.\d+\.\d+)")

def clean_text(t):
    return re.sub(r"\s+", " ", t.strip())

# def extract_play_links():
#     """Scrape MIT Shakespeare homepage for play links (excluding footer tables)."""
#     print("Fetching play index...")
#     r = requests.get(BASE_URL)
#     r.raise_for_status()
#     soup = BeautifulSoup(r.text, "html.parser")

#     # Find the main plays table by looking for one containing <h2> tags
#     main_table = None
#     for table in soup.find_all("table"):
#         if table.find("h2"):
#             main_table = table
#             break

#     if not main_table:
#         raise RuntimeError("Could not find main play listing table.")

#     plays = {}
#     for a in main_table.find_all("a", href=True):
#         href = a["href"].strip()
#         text = a.get_text(strip=True)

#         # Skip poems and other non-play items
#         if "Poetry" in href or "sonnet" in href.lower():
#             continue

#         # Make full URL
#         full_url = urljoin(BASE_URL, href)

#         # Normalize play name
#         play_name = text or href.strip("/").split("/")[-1].replace("_", " ").title()
#         play_name = play_name.replace("Index.Html", "").strip()

#         plays[play_name] = full_url

#     print(f"✅ Found {len(plays)} plays.")
#     for name, url in list(plays.items())[:5]:
#         print(f"  • {name}: {url}")

#     return plays

def extract_play_paths() -> dict:
    """Get local file paths for each play."""
    # return a dict with name and path
    # the plays are in the plays directory with the name of the play as the file name
    plays_dir = os.path.join(os.path.dirname(__file__), "plays")
    plays = {}
    for filename in os.listdir(plays_dir):
        if filename.endswith(".html"):
            play_name = filename.replace("_", " ").replace(".html", "").title()
            play_path = os.path.join(plays_dir, filename)
            plays[play_name] = play_path

    return plays

def extract_context(tag, direction="before"):
    """Look near a speech for a stage direction or nearby text."""
    step = tag.find_previous if direction == "before" else tag.find_next
    for _ in range(3):
        candidate = step(["i", "blockquote"])
        if not candidate:
            break
        if candidate.name == "i":
            return clean_text(candidate.get_text())
        elif candidate.name == "blockquote":
            speaker_tag = candidate.find_previous("a", attrs={"name": re.compile("^speech")})
            if speaker_tag and speaker_tag.find("b"):
                spkr = speaker_tag.find("b").get_text(strip=True)
                block_text = clean_text(candidate.get_text(" "))
                return f"{spkr}: {block_text}"
    return None

def parse_play(play_name, play_path):
    """Extract speeches from a single play."""

    print(f"Loading {play_name} -> {play_path}")
    with open(play_path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")

    speeches = []
    current_act = None
    current_scene = None
    processed_blockquotes = set()

    for tag in soup.find_all(True):
        if tag.name.lower() == "h3":
            txt = tag.get_text(strip=True)
            if act_re.match(txt):
                current_act = act_re.match(txt).group(1)
                continue
            elif scene_re.match(txt):
                current_scene = scene_re.match(txt).group(1)
                continue

        # Detect speech anchors
        if tag.name == "a" and tag.has_attr("name") and tag.find("b"):
            speaker = tag.find("b").get_text(strip=True)
            block = tag.find_next_sibling("blockquote")
            if not block:
                continue

            blockquote_id = id(block)
            if blockquote_id in processed_blockquotes:
                continue
            processed_blockquotes.add(blockquote_id)

            lines, line_nums = [], []
            for a in block.find_all("a"):
                line_text = a.get_text(" ", strip=True)
                if line_text:
                    lines.append(line_text)
                name_attr = a.get("name")
                if name_attr and line_num_re.match(name_attr):
                    line_nums.append(name_attr)

            text = clean_text(" ".join(lines))
            if not text:
                continue

            line_start = line_nums[0] if line_nums else None
            line_end = line_nums[-1] if line_nums else None
            speech_id = f"{play_name.replace(' ', '_').lower()}_{current_act}_{current_scene}_{line_start}_{line_end}"

            context_before = extract_context(tag, "before")
            context_after = extract_context(block, "after")

            speeches.append({
                "id": speech_id,
                "play": play_name,
                "act": current_act,
                "scene": current_scene,
                "speaker": speaker,
                "line_start": line_start,
                "line_end": line_end,
                "text": text,
                "context_before": context_before,
                "context_after": context_after,
                "embedding_vector": None
            })

    print(f"  -> {len(speeches)} speeches extracted.")
    return speeches

def build_context_windows(speeches, window_size=3):
    """Merge speeches into overlapping context windows."""
    windows = []
    n = len(speeches)
    for i, s in enumerate(speeches):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        merged_text = " ".join(sp["text"] for sp in speeches[start:end])
        merged_speakers = [sp["speaker"] for sp in speeches[start:end]]
        window_id = f"{s['id']}_window"

        windows.append({
            "id": window_id,
            "play": s["play"],
            "act": s["act"],
            "scene": s["scene"],
            "speakers": merged_speakers,
            "text": merged_text,
            "center_speech_id": s["id"],
            "embedding_vector": None
        })
    return windows

def main():
    load_plays_start_time = time.time()
    plays = extract_play_paths()
    print(f"Found {len(plays)} candidate plays.")
    all_speeches = []
    all_windows = []

    for name, path in plays.items():
        try:
            speeches = parse_play(name, path)
            all_speeches.extend(speeches)
            all_windows.extend(build_context_windows(speeches))
            time.sleep(1)
        except Exception as e:
            print(f"Error parsing {name}: {e}")

    print(f"Total speeches: {len(all_speeches)}")
    print(f"Total windows: {len(all_windows)}")

    with open("shakespeare_speeches.jsonl", "w", encoding="utf-8") as f:
        for s in all_speeches:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open("shakespeare_windows.jsonl", "w", encoding="utf-8") as f:
        for w in all_windows:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")

    print("Extraction complete. Files saved:")
    print("  -> shakespeare_speeches.jsonl")
    print("  -> shakespeare_windows.jsonl")
    load_plays_end_time = time.time()
    print(f"Time to parse plays: {load_plays_end_time - load_plays_start_time:.2f} seconds")
    create_db_start_time = time.time()
    load_speeches_to_chroma("shakespeare_speeches.jsonl")
    create_db_end_time = time.time()
    print(f"⏱️  Time to create ChromaDB: {create_db_end_time - create_db_start_time:.2f} seconds")

def load_speeches_to_chroma(jsonl_path="shakespeare_speeches.jsonl"):
    """Load speeches from JSONL into ChromaDB."""
    print(f"Loading speeches from {jsonl_path} into ChromaDB...")
    
    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print("  → Deleted existing collection")
    except:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDER,
        metadata={"description": "Shakespeare speeches with context"}
    )
    
    speeches = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            speeches.append(json.loads(line))
    
    batch_size = 100
    for i in range(0, len(speeches), batch_size):
        batch = speeches[i:i+batch_size]
        
        documents = []
        metadatas = []
        ids = []
        
        for s in batch:
            doc_text = s["text"]
            if s.get("context_before"):
                doc_text = f"[Context before: {s['context_before']}] {doc_text}"
            if s.get("context_after"):
                doc_text = f"{doc_text} [Context after: {s['context_after']}]"
            
            documents.append(doc_text)
            ids.append(s["id"])
            metadatas.append({
                "play": s["play"],
                "act": s["act"] or "",
                "scene": s["scene"] or "",
                "speaker": s["speaker"],
                "line_start": s["line_start"] or "",
                "line_end": s["line_end"] or "",
            })
        
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"  → Added batch {i//batch_size + 1}/{(len(speeches)-1)//batch_size + 1}")
    
    print(f"✅ Loaded {len(speeches)} speeches into ChromaDB")
    return collection

def query_shakespeare(query_text, n_results=5):
    """Query the Shakespeare RAG system."""
    client = chromadb.PersistentClient(path=str(DB_DIR))
    
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=EMBEDDER
        )
    except:
        print("Collection not found. Run load_speeches_to_chroma() first.")
        return None
    
    results = collection.query(
        query_texts=[query_text],
        n_results=n_results
    )
    
    return results

def format_results(results):
    """Format query results for display."""
    if not results or not results["documents"][0]:
        return "No results found."
    
    output = []
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        output.append(f"\n--- Result {i+1} (distance: {distance:.4f}) ---")
        output.append(f"Play: {metadata['play']}")
        output.append(f"Act {metadata['act']}, Scene {metadata['scene']}")
        output.append(f"Speaker: {metadata['speaker']}")
        output.append(f"Lines: {metadata['line_start']} - {metadata['line_end']}")
        output.append(f"\n{doc}")
        output.append("-" * 50)
    
    return "\n".join(output)

if __name__ == "__main__":
    main()
