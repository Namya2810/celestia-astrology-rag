# Retrieval evaluation

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` | FAISS exact cosine search | k = 3 | 60 test questions (12 signs x 5: 3 name the sign, 2 are reverse lookups that only describe it).

| Chunking | Retrieval | Chunks | Hit@1 | Hit@3 | MRR@3 | Field@3 | Hit@1 named | Hit@1 reverse |
|---|---|---|---|---|---|---|---|---|
| Whole sign (current) | similarity | 12 | 70% | 80% | 0.74 | - | 100% | 25% |
| Whole sign (current) | mmr | 12 | 70% | 82% | 0.76 | - | 100% | 25% |
| Field-level | similarity | 180 | 90% | 98% | 0.94 | 97% | 97% | 79% |
| Field-level | mmr | 180 | 90% | 97% | 0.93 | 92% | 97% | 79% |
| Fixed 300 chars / 50 overlap | similarity | 92 | 73% | 90% | 0.82 | - | 97% | 38% |
| Fixed 300 chars / 50 overlap | mmr | 92 | 73% | 82% | 0.77 | - | 97% | 38% |
| Fixed 800 chars / 100 overlap | similarity | 33 | 65% | 80% | 0.71 | - | 97% | 17% |
| Fixed 800 chars / 100 overlap | mmr | 33 | 65% | 70% | 0.67 | - | 97% | 17% |
