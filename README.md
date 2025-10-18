# Shakespeare RAG System

A Retrieval-Augmented Generation (RAG) system for querying Shakespeare's complete works using vector embeddings and local LLM inference.

## Overview

This project implements a question-answering system over Shakespeare's 36 plays using:
- **ChromaDB** for vector storage and similarity search
- **Sentence Transformers** for semantic embeddings
- **Local LLM** (GPT-OSS-20B) for answer generation
- **LLM-as-a-Judge** evaluation framework

The system chunks Shakespeare's plays by individual character speeches, preserving speaker attribution and dramatic context. Users can ask factual, interpretive, or comparative questions about the plays.

## Features

- 📚 **Complete Shakespeare Corpus**: All 36 plays (~100,000 speeches)
- 🔍 **Semantic Search**: Find relevant passages using natural language queries
- 🎭 **Context-Aware Chunking**: Each speech includes surrounding dialogue context
- 📊 **Rich Metadata**: Play name, act, scene, speaker, line numbers
- 🤖 **Faithful Responses**: LLM constrained to answer only from retrieved passages
- 📈 **Automated Evaluation**: LLM-as-a-judge scoring system

## Architecture

```
Query → Embedding → Vector Search → Top-K Retrieval → LLM Generation → Answer
                        ↓
                   ChromaDB
                (100K+ speeches)
```

### Components

1. **Data Ingestion** (`shakespear_rag_extractor.py`)
   - Parses HTML play files
   - Extracts speeches with metadata
   - Generates context windows
   - Loads into ChromaDB

2. **RAG System** (`shakespeare_rag.py`)
   - ChromaDB interface
   - Query processing
   - Result retrieval

3. **Query Pipeline** (`query.py`)
   - Retrieval ranking
   - LLM answer generation
   - Evaluation grading

4. **Execution** (`main.py`)
   - Runs test query suite
   - Saves results to JSONL

## Installation

### Prerequisites

- Python 3.12 or 3.13
- [Poetry](https://python-poetry.org/) for dependency management
- [LM Studio](https://lmstudio.ai/) or compatible OpenAI API server

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ryanpoppe/ut-cs6300-assignment-05.git
   cd ut-cs6300-assignment-05
   ```

2. **Install dependencies**
   ```bash
   poetry install
   ```

3. **Start your local LLM server**
   
   Using LM Studio:
   - Download a model (e.g., GPT-OSS-20B, Llama 3, Mistral)
   - Start the local server on `http://localhost:1234/v1`
   
   Or use any OpenAI-compatible API endpoint.

4. **Build the database** (if not already built)
   ```bash
   poetry run python -m src.shakespear_rag_extractor
   ```
   
   This will:
   - Parse all play HTML files in `src/plays/`
   - Extract ~100,000 speeches
   - Generate embeddings using `all-MiniLM-L6-v2`
   - Store in `shakespeare_db/`
   - Takes ~5-10 minutes on CPU

## Usage

### Basic Query

```python
from src.shakespeare_rag import ShakespeareRAG

# Initialize RAG system
rag = ShakespeareRAG()

# Query the database
results = rag.query_shakespeare("Who says 'To be or not to be'?", n_results=5)

# Display results
for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
    print(f"\nResult {i+1}:")
    print(f"Play: {metadata['play']}")
    print(f"Speaker: {metadata['speaker']}")
    print(f"Text: {doc[:200]}...")
```

### Full Query Pipeline (with LLM)

```python
from src.query import Query
from src.shakespeare_rag import ShakespeareRAG

# Initialize
rag = ShakespeareRAG()
query = Query(rag, "What is the name of Hamlet's mother?")

# Run complete pipeline: retrieve → generate → evaluate
result = query.run_query_pipeline(n_results=5)

# Access results
print(f"Question: {result['question']}")
for answer in result['answers']:
    print(f"\nAnswer: {answer['answer']}")
    print(f"Grade: {answer['grade']}")
```

### Run Evaluation Suite

```bash
poetry run python -m src.main
```

This runs all 50 test queries across 5 categories:
- Factual Recall
- Contextual Understanding
- Interpretive/Thematic
- Comparative Reasoning
- Creative Synthesis

Results are saved to `results.jsonl`.

## Configuration

### LLM Server Settings

Edit `src/query.py` to change the LLM endpoint:

```python
class Query:
    def __init__(
        self, 
        rag: ShakespeareRAG, 
        question: str, 
        base_url: str = "http://localhost:1234/v1",  # Change this
        model_name: str = "openai/gpt-oss-20b"       # Change this
    ):
        ...
```

### Embedding Model

Edit `src/shakespeare_rag.py` to use a different embedding model:

```python
EMBEDDER = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"  # Change this
)
```

Popular alternatives:
- `all-mpnet-base-v2` (better quality, slower)
- `all-distilroberta-v1` (balanced)
- `multi-qa-MiniLM-L6-cos-v1` (optimized for Q&A)

## Project Structure

```
ut-cs6300-assignment-05/
├── src/
│   ├── plays/                     # 36 Shakespeare play HTML files
│   │   ├── hamlet.html
│   │   ├── macbeth.html
│   │   └── ...
│   ├── shakespear_rag_extractor.py  # Data ingestion pipeline
│   ├── shakespeare_rag.py          # RAG system interface
│   ├── query.py                    # Query processing & evaluation
│   └── main.py                     # Test execution script
├── shakespeare_db/                # ChromaDB storage (generated)
├── shakespeare_speeches.jsonl     # Extracted speeches (generated)
├── shakespeare_windows.jsonl      # Context windows (generated)
├── results.jsonl                  # Evaluation results (generated)
├── pyproject.toml                 # Python dependencies
├── README.md                      # This file
└── REPORT.md                      # Full project report
```

## Example Queries

### Factual Recall
```
Q: "Who kills Tybalt in Romeo and Juliet?"
A: "Romeo kills Tybalt. This is stated in Act III, Scene II by the Nurse: 
    'Romeo that kill'd him, he is banished.'"
```

### Contextual Understanding
```
Q: "What warning does the soothsayer give Caesar?"
A: "The soothsayer warns Caesar to 'Beware the ides of March.' 
    This appears in Julius Caesar, Act I, Scene II."
```

### Interpretive/Thematic
```
Q: "What does Hamlet mean by 'Denmark's a prison'?"
A: "Hamlet states 'Denmark's a prison' in Act II, Scene II. Rosencrantz 
    responds 'Then is the world one,' suggesting a philosophical discussion 
    about constraint and perception."
```

## Evaluation Metrics

The system grades responses on four criteria:

| Criterion | Weight | Description |
|-----------|--------|-------------|
| Faithfulness | 0-3 | Uses only provided passages |
| Accuracy | 0-3 | Correct information |
| Compliance | 0-2 | Follows instructions |
| Citation | 0-2 | Proper source attribution |

**Pass threshold:** 8/10 points

### Performance by Category

| Category | Pass Rate | Notes |
|----------|-----------|-------|
| Factual Recall | 90% | Strong on direct questions |
| Contextual Understanding | 80% | Good scene context |
| Interpretive/Thematic | 60% | Struggles with abstraction |
| Comparative Reasoning | 40% | Poor cross-play retrieval |
| Creative Synthesis | 20% | Beyond RAG capabilities |

## Limitations

- **Cross-play comparisons**: Vector search doesn't aggregate multi-document information
- **Thematic questions**: Embeddings capture semantics but not symbolic meaning
- **Sequential context**: Single-speech chunks lack temporal relationships
- **Creative tasks**: RAG unsuited for synthesis/generation tasks

See `REPORT.md` for detailed analysis and future improvements.

## Development

### Run Linting
```bash
poetry run flake8 src/
```

### Run Tests
```bash
poetry run pytest
```

### Rebuild Database
```bash
# Delete existing database
rm -rf shakespeare_db/

# Rebuild from source
poetry run python -m src.shakespear_rag_extractor
```

## Dependencies

- **chromadb** (1.1.0+): Vector database
- **sentence-transformers** (5.1.1+): Embedding models
- **beautifulsoup4** (4.14.2+): HTML parsing
- **smolagents** (1.21.3+): LLM client interface
- **pandas** (2.3.2+): Data processing
- **numpy** (2.3.2+): Numerical operations

## Troubleshooting

### "Collection not found" error
Run the data extraction script first:
```bash
poetry run python -m src.shakespear_rag_extractor
```

### LLM connection errors
Ensure your local LLM server is running:
- Check LM Studio is started
- Verify endpoint: `http://localhost:1234/v1`
- Test with: `curl http://localhost:1234/v1/models`

### Slow queries
- Reduce `n_results` parameter (default: 5)
- Use a smaller embedding model
- Enable GPU acceleration for sentence-transformers

### Out of memory
- Process plays in batches (modify `batch_size` in extractor)
- Use quantized embedding models
- Reduce context window size

## Contributing

This is a class project (CS 6300, Spring 2025). Contributions are not accepted.

## License

Educational use only. Shakespeare texts are public domain. Code provided for academic purposes.

## References

- Shakespeare texts: [MIT Shakespeare](http://shakespeare.mit.edu/)
- Sentence Transformers: [sbert.net](https://www.sbert.net/)
- ChromaDB: [docs.trychroma.com](https://docs.trychroma.com/)
- LM Studio: [lmstudio.ai](https://lmstudio.ai/)

## Contact

**Author:** Ryan Poppe  
**Course:** UT CS 6300 - AI Agents  
**Repository:** [github.com/ryanpoppe/ut-cs6300-assignment-05](https://github.com/ryanpoppe/ut-cs6300-assignment-05)

---

For detailed methodology, evaluation results, and analysis, see **REPORT.md**.
