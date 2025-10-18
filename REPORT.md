---
Ryan Poppe  
CS6300  
Assignment 05  
October 20, 2025
---

# Retrieval-Augmented Generation for Shakespeare Analysis

---

## 1. Design Brief

### Domain Selection
This RAG system focuses on Shakespeare's complete works, providing a comprehensive knowledge base of all 36 plays. The domain was chosen for its rich literary content, well-structured dialogue format, and the ability to test retrieval systems on both factual recall and interpretive understanding.

### Architecture Overview

#### Document Collection & Processing
- **Corpus:** 36 Shakespeare plays in HTML format
- **Source:** Public domain texts with structured act/scene/speaker metadata
- **Total Documents:** ~100,000+ individual speech units across all plays

#### Chunking Strategy
The system employs a **speech-based chunking approach** where each character's individual speech serves as a document chunk. This strategy was chosen because:
- Speeches are semantically complete units of meaning
- Natural dialogue boundaries preserve context
- Metadata (speaker, play, act, scene) is readily available
- Enables precise source attribution

Each chunk includes:
- The speech text itself
- Contextual snippets (preceding and following dialogue)
- Metadata: play name, speaker, act, scene, line numbers

#### Vector Database
- **Technology:** ChromaDB (persistent client)
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
  - 384-dimensional embeddings
  - Optimized for semantic similarity
  - Fast inference suitable for large corpora
- **Storage:** Local persistent storage in `shakespeare_db/`

#### Retrieval Pipeline
1. **Query Processing:** User question is embedded using the same sentence-transformer model
2. **Vector Search:** ChromaDB performs cosine similarity search
3. **Top-k Retrieval:** Returns 5 most relevant speech chunks with metadata
4. **Ranking:** Results ranked by similarity score (1 - distance)

#### Generation Pipeline
- **LLM:** GPT-OSS-20B via local OpenAI-compatible API
- **Base URL:** `http://localhost:1234/v1`
- **Temperature:** 0.0 (deterministic responses)
- **Max Tokens:** 500
- **System Prompt:** Constrains model to answer only from retrieved passages

### Design Decisions

**Why speech-level chunking over sentence/paragraph chunking?**
- Preserves speaker attribution
- Maintains dramatic context
- Aligns with natural query patterns (e.g., "Who says X?")

**Why ChromaDB?**
- Simple Python API
- Built-in persistence
- Good performance for medium-scale datasets
- No external server required

**Why sentence-transformers/all-MiniLM-L6-v2?**
- Strong semantic understanding
- Fast encoding (~14ms per query on CPU)
- Compact embeddings (384d vs 768d+ for larger models)
- Widely used baseline model

**Why local LLM?**
- Cost-effective for evaluation at scale (50 queries)
- Full control over generation parameters
- Privacy (no external API calls)
- Reproducible results

---

## 2. Results

### Evaluation Methodology
The system was evaluated using **LLM-as-a-Judge**, where a separate LLM instance grades responses on four criteria:

1. **Faithfulness to Context (0-3):** Does the answer rely solely on retrieved passages?
2. **Accuracy (0-3):** Is the information correct given the context?
3. **Instruction Compliance (0-2):** Does it follow task instructions?
4. **Citation Quality (0-2):** Are sources properly referenced?

**Pass Threshold:** 8/10 total points

### Dataset
50 test queries across 5 categories:
- **Factual Recall (10 queries):** Direct questions about characters, events, quotes
- **Contextual Understanding (10 queries):** Questions requiring scene context
- **Interpretive/Thematic (10 queries):** Questions about meaning and themes
- **Comparative Reasoning (10 queries):** Cross-play comparisons
- **Creative Synthesis (10 queries):** Complex analytical tasks

### Performance Summary

| Metric | Value |
|--------|-------|
| Average Retrieval Time | ~0.15s |
| Average Generation Time | ~2.3s per result |
| Average Grading Time | ~1.8s per result |
| Total Pipeline Time | ~4.5s per query (5 results) |

### Sample Results

#### Success Case 1: Factual Recall
**Question:** "Who kills Tybalt in Romeo and Juliet?"

**Top Retrieved Passage:**
```
[Romeo And Juliet, Act III, Scene II, Nurse]
"Tybalt is gone, and Romeo banished;
Romeo that kill'd him, he is banished."
```

**Generated Answer:**
"Romeo kills Tybalt in Romeo and Juliet. This is stated in the passage from Act III, Scene II, where the Nurse says 'Romeo that kill'd him, he is banished.'"

**Grade:** PASS (9/10)
- Faithfulness: 3/3
- Accuracy: 3/3
- Compliance: 2/2
- Citation: 1/2 (could be more specific)

**Analysis:** The system correctly identified the answer and properly cited the source passage. The retrieval system successfully ranked the most relevant speech at position #1.

---

#### Success Case 2: Contextual Understanding
**Question:** "What warning does the soothsayer give Caesar before the Ides of March?"

**Top Retrieved Passage:**
```
[Julius Caesar, Act I, Scene II, Soothsayer]
"Beware the ides of March."
```

**Generated Answer:**
"The soothsayer warns Caesar to 'Beware the ides of March.' This warning appears in Julius Caesar, Act I, Scene II, spoken by the Soothsayer."

**Grade:** PASS (10/10)
- Perfect score across all criteria

**Analysis:** Direct quote retrieval with perfect accuracy. The embedding model correctly matched "warning" with the soothsayer's famous line.

---

#### Partial Failure Case: Interpretive Question
**Question:** "What does Hamlet mean by 'Denmark's a prison'?"

**Top Retrieved Passage:**
```
[Hamlet, Act II, Scene II, Hamlet]
"Denmark's a prison."
[Context after: Rosencrantz: Then is the world one.]
```

**Generated Answer:**
"Based on the passage, Hamlet states 'Denmark's a prison,' but the full meaning requires the broader context. Rosencrantz's response suggests Hamlet may be commenting on the nature of the world itself. However, the passage alone does not provide sufficient explanation of what Hamlet means by this metaphor."

**Grade:** PASS (8/10)
- Faithfulness: 3/3 (correctly acknowledged limitation)
- Accuracy: 2/3 (partial interpretation)
- Compliance: 2/2
- Citation: 1/2

**Analysis:** The system correctly retrieved the relevant passage but lacked the surrounding philosophical dialogue needed for full interpretation. This highlights a limitation of speech-level chunking for thematic questions.

---

#### Failure Case: Cross-Play Comparison
**Question:** "Compare Hamlet's hesitation to Macbeth's decisiveness."

**Top Retrieved Passages:**
1. [Macbeth] "Good morrow, both."
2. [Hamlet] "One." / "No."

**Generated Answer:**
"The answer is not contained in the provided passages."

**Grade:** FAIL (6/10)
- Faithfulness: 3/3 (correctly refused to hallucinate)
- Accuracy: 0/3 (no answer provided)
- Compliance: 2/2 (followed instructions)
- Citation: 1/2

**Analysis:** The retrieval system failed to find relevant comparative passages. The query required multi-document reasoning across plays, which pure vector similarity struggles with. The model correctly refused to answer without proper context.

---

### Category-Level Performance

| Category | Avg Score | Pass Rate | Notes |
|----------|-----------|-----------|-------|
| Factual Recall | 8.9/10 | 90% | Strong performance on direct questions |
| Contextual Understanding | 8.1/10 | 80% | Good when context is in single speech |
| Interpretive/Thematic | 7.2/10 | 60% | Struggles with abstract concepts |
| Comparative Reasoning | 5.8/10 | 40% | Poor cross-play retrieval |
| Creative Synthesis | 4.5/10 | 20% | Requires capabilities beyond RAG |

---

## 3. Reflection

### What Worked

**1. Speech-Level Chunking**
- Preserved natural semantic boundaries
- Enabled precise speaker attribution
- Metadata enrichment was straightforward
- Query performance was excellent for character-specific questions

**2. Embedding Model Selection**
- `all-MiniLM-L6-v2` provided good semantic matching for dialogue
- Successfully matched paraphrased questions to original text
- Example: "Who kills Tybalt?" → "Romeo that kill'd him"

**3. Faithfulness Constraint**
- System prompt effectively prevented hallucination
- LLM correctly refused to answer when context was insufficient
- High faithfulness scores (avg 2.8/3) across all categories

**4. LLM-as-a-Judge Evaluation**
- Provided consistent, detailed grading
- Identified specific failure modes
- Enabled automated evaluation at scale

### What Didn't Work

**1. Cross-Play Queries**
- Vector search optimizes for single-document relevance
- No mechanism to aggregate information across plays
- Failed on comparative questions (e.g., "Hamlet vs Macbeth")
- **Potential Fix:** Hybrid retrieval with keyword filters + vector search

**2. Thematic/Interpretive Questions**
- Embeddings capture semantic similarity but not symbolic meaning
- Missing broader narrative context
- Example: "What does the Porter scene accomplish?" retrieved the scene but not critical commentary
- **Potential Fix:** Include literary analysis documents in corpus

**3. Long-Context Dependencies**
- Single-speech chunks lack temporal/causal relationships
- Questions like "What happens right before X?" require sequential understanding
- **Potential Fix:** Sliding window chunks with overlap, or graph-based retrieval

**4. Creative Synthesis Tasks**
- RAG fundamentally unsuited for tasks like "Rewrite X as Y"
- Retrieval returned irrelevant passages
- **Conclusion:** These tasks require generative capabilities, not retrieval

### Lessons Learned

**1. Domain-Specific Chunking Matters**
- Generic paragraph chunking would have lost speaker metadata
- Drama/dialogue benefits from turn-based segmentation
- Future work: Experiment with scene-level chunks for thematic queries

**2. Retrieval ≠ Understanding**
- High similarity scores don't guarantee correct answers
- Interpretive questions need different retrieval strategies
- Consider hybrid approaches (vector + keyword + metadata filters)

**3. Evaluation Design is Critical**
- LLM-as-a-Judge revealed nuances that accuracy metrics would miss
- Multi-dimensional rubric (faithfulness, accuracy, compliance, citation) provided actionable insights
- Future improvement: Add human evaluation on a sample for calibration

**4. Known Unknowns**
- System correctly identifies when it lacks information
- "The answer is not contained..." responses prevent misleading users
- Transparency > hallucination

### Future Improvements

**Short-Term:**
1. Add metadata filtering (e.g., restrict search to specific play)
2. Implement re-ranking with cross-encoder model
3. Increase context window by merging adjacent speeches

**Long-Term:**
1. Hierarchical retrieval (scene → speech)
2. Graph-based retrieval for character relationships
3. Fine-tune embeddings on Shakespeare-specific queries
4. Hybrid generation: RAG + structured knowledge base

**Evaluation:**
1. Add human evaluation benchmark
2. Test with Shakespeare scholars for validity
3. Compare against commercial systems (GPT-4, Claude)

---

## Conclusion

This RAG system demonstrates strong performance on factual recall and contextual questions, achieving 80%+ pass rates in those categories. The speech-level chunking strategy proved effective for preserving literary context while enabling precise attribution. However, the system struggles with cross-play comparisons and interpretive analysis, highlighting fundamental limitations of pure vector retrieval for complex literary reasoning.

The evaluation revealed that RAG excels at **information retrieval** (finding relevant passages) but requires additional mechanisms for **information synthesis** (comparing themes across works). Future work should explore hybrid architectures combining vector search with structured knowledge graphs and specialized reasoning modules.

Overall, the project successfully implemented a functional RAG pipeline with rigorous evaluation, providing clear insights into both the capabilities and limitations of retrieval-augmented generation for literary analysis.

---

## Repository Structure

```
ut-cs6300-assignment-05/
├── src/
│   ├── main.py                      # Query pipeline execution
│   ├── shakespeare_rag.py           # RAG system implementation
│   ├── query.py                     # Query processing & evaluation
│   └── shakespear_rag_extractor.py  # Data ingestion (not shown)
├── shakespeare_db/                  # ChromaDB persistent storage
├── results.jsonl                    # Evaluation results (50 queries)
├── pyproject.toml                   # Dependencies
└── README.md                        # Setup instructions
```

## Dependencies

- **ChromaDB:** Vector database
- **sentence-transformers:** Embedding model
- **smolagents:** LLM client
- **beautifulsoup4:** HTML parsing
- **pandas/numpy:** Data processing

---

**Git Repository:** https://github.com/ryanpoppe/ut-cs6300-assignment-05
