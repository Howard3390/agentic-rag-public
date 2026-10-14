"""Read-only LinearRAG retrieval worker for the repository's Python 3.9 environment.

Input and output are one JSON object on stdin/stdout. The MCP server runs in the
main environment and launches this worker in the separate LinearRAG environment.
"""

import json
import os
import re
import sys
import types
from collections import defaultdict
from contextlib import redirect_stdout
from pathlib import Path


class CachedEmbeddingStore:
    """Read existing LinearRAG Parquet rows in batches and mmap their vectors."""

    def __init__(self, parquet_path, cache_dir):
        import numpy as np
        import pyarrow.parquet as pq

        self.hash_ids = []
        self.texts = []
        parquet = pq.ParquetFile(str(parquet_path))
        for batch in parquet.iter_batches(batch_size=2048, columns=["hash_id", "text"]):
            self.hash_ids.extend(batch.column(0).to_pylist())
            self.texts.extend(batch.column(1).to_pylist())
        self.hash_id_to_text = dict(zip(self.hash_ids, self.texts))
        self.hash_id_to_idx = {value: index for index, value in enumerate(self.hash_ids)}
        self.text_to_hash_id = dict(zip(self.texts, self.hash_ids))

        cache_dir.mkdir(parents=True, exist_ok=True)
        vector_path = cache_dir / (parquet_path.stem + ".npy")
        manifest_path = cache_dir / (parquet_path.stem + ".json")
        source_stat = parquet_path.stat()
        manifest = {"size": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns, "rows": len(self.hash_ids)}
        valid = False
        if vector_path.is_file() and manifest_path.is_file():
            try:
                valid = json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
            except (OSError, ValueError):
                pass
        if not valid:
            temporary = vector_path.with_suffix(".tmp")
            vector_batches = parquet.iter_batches(batch_size=128, columns=["embedding"])
            first = next(vector_batches, None)
            if first is None:
                raise ValueError("Empty LinearRAG embedding index: " + str(parquet_path))
            first_array = np.asarray(first.column(0).to_pylist(), dtype=np.float32)
            if first_array.ndim != 2:
                raise ValueError("Invalid LinearRAG embedding shape")
            vectors = np.lib.format.open_memmap(
                str(temporary), mode="w+", dtype="float32",
                shape=(len(self.hash_ids), first_array.shape[1]),
            )
            cursor = len(first_array)
            vectors[:cursor] = first_array
            for batch in vector_batches:
                values = np.asarray(batch.column(0).to_pylist(), dtype=np.float32)
                vectors[cursor:cursor + len(values)] = values
                cursor += len(values)
            if cursor != len(self.hash_ids):
                raise ValueError("LinearRAG embedding row count mismatch")
            vectors.flush()
            del vectors
            os.replace(str(temporary), str(vector_path))
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.embeddings = np.load(str(vector_path), mmap_mode="r")

    def get_hash_id_to_text(self):
        return self.hash_id_to_text


class IndexEntityNER:
    """Use indexed entity names and proper-name spans without loading the large transformer."""

    def __init__(self, entity_texts):
        self.entity_texts = [text.lower().strip() for text in entity_texts if len(text.strip()) >= 3]

    def question_ner(self, question):
        proper_names = []
        tokens = re.findall(r"[\w'-]+", question)
        skip = {"what", "which", "who", "when", "where", "why", "how", "in", "is", "are", "was", "were", "the", "a", "an"}
        connectors = {"of", "the", "and", "in", "on", "for"}
        index = 0
        while index < len(tokens):
            token = tokens[index]
            leading_the = token.lower() == "the" and index + 1 < len(tokens) and tokens[index + 1][:1].isupper()
            if not token[:1].isupper() or (token.lower() in skip and not leading_the):
                index += 1
                continue
            phrase = [token]
            cursor = index + 1
            while cursor < len(tokens):
                next_token = tokens[cursor]
                if next_token[:1].isupper():
                    phrase.append(next_token)
                    cursor += 1
                elif next_token.lower() in connectors and cursor + 1 < len(tokens) and tokens[cursor + 1][:1].isupper():
                    phrase.extend([next_token, tokens[cursor + 1]])
                    cursor += 2
                else:
                    break
            candidate = " ".join(phrase).lower()
            if len(candidate) >= 4:
                proper_names.append(candidate)
            index = cursor
        if proper_names:
            return proper_names[:4]

        lowered = question.lower()
        candidates = []
        for entity in self.entity_texts:
            position = lowered.find(entity)
            if position >= 0:
                end = position + len(entity)
                if (position == 0 or not lowered[position - 1].isalnum()) and (
                    end == len(lowered) or not lowered[end].isalnum()
                ):
                    candidates.append(entity)
        candidates.sort(key=len, reverse=True)
        selected = []
        for candidate in candidates:
            if not any(candidate in other for other in selected):
                selected.append(candidate)
            if len(selected) >= 4:
                break
        return selected


def search(request):
    repo = Path(request["repo_root"]).resolve()
    dataset = request.get("dataset", "hotpotqa")
    if not dataset.replace("_", "").isalnum():
        raise ValueError("Invalid dataset name")
    query = request["query"].strip()
    top_k = int(request.get("top_k", 10))
    if not query or top_k < 1:
        raise ValueError("query must be nonempty and top_k positive")

    cache_dir = repo / "import" / dataset
    required = [
        cache_dir / "passage_embedding.parquet",
        cache_dir / "entity_embedding.parquet",
        cache_dir / "sentence_embedding.parquet",
        cache_dir / "ner_results.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("LinearRAG index files are missing: " + ", ".join(missing))

    sys.path.insert(0, str(repo))
    from src.utils import load_env_file

    load_env_file(str(repo / ".env"))
    model_name = os.environ.get("EMBEDDING_MODEL")
    if not model_name:
        raise ValueError("LinearRAG EMBEDDING_MODEL is missing")

    import igraph as ig

    # This worker uses the original CPU graph-search methods, but bypasses the
    # constructor and never calls the torch/vectorized, pandas/EmbeddingStore,
    # or spaCy paths. Avoid importing those large optional dependencies when
    # loading the upstream class; with a tight Windows commit limit they can
    # exhaust memory before the index has been read.
    ner_module = types.ModuleType("src.ner")
    store_module = types.ModuleType("src.embedding_store")
    torch_module = types.ModuleType("torch")

    class UnusedSpacyNER:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("The retrieval worker must use IndexEntityNER")

    ner_module.SpacyNER = UnusedSpacyNER
    class UnusedEmbeddingStore:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("The retrieval worker must use CachedEmbeddingStore")

    store_module.EmbeddingStore = UnusedEmbeddingStore
    replacements = {
        "src.ner": ner_module,
        "src.embedding_store": store_module,
        "torch": torch_module,
    }
    previous_modules = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        from src.LinearRAG import LinearRAG
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    from src.config import LinearRAGConfig
    from src.embedding_api import APIEmbeddingModel

    config = LinearRAGConfig(
        dataset_name=dataset,
        embedding_model=APIEmbeddingModel(model_name),
        spacy_model=request.get("spacy_model", "en_core_web_trf"),
        working_dir=str(repo / "import"),
        retrieval_top_k=top_k,
        max_iterations=int(request.get("max_iterations", 3)),
        top_k_sentence=int(request.get("top_k_sentence", 3)),
        passage_ratio=float(request.get("passage_ratio", 2.0)),
        iteration_threshold=float(request.get("iteration_threshold", 0.4)),
        use_vectorized_retrieval=False,
    )
    rag = LinearRAG.__new__(LinearRAG)
    rag.config = config
    rag.dataset_name = dataset
    rag.device = "cpu"
    rag.passage_embedding_store = CachedEmbeddingStore(
        cache_dir / "passage_embedding.parquet", Path(request["cache_dir"])
    )
    rag.entity_embedding_store = CachedEmbeddingStore(
        cache_dir / "entity_embedding.parquet", Path(request["cache_dir"])
    )
    rag.sentence_embedding_store = CachedEmbeddingStore(
        cache_dir / "sentence_embedding.parquet", Path(request["cache_dir"])
    )
    rag.spacy_ner = IndexEntityNER(rag.entity_embedding_store.texts)
    rag.graph = ig.Graph(directed=False)

    # Rebuild the graph in memory from existing NER and embeddings. The upstream
    # index() writes cache files and calls embedding/NER for any misses.
    with (cache_dir / "ner_results.json").open(encoding="utf-8") as handle:
        ner = json.load(handle)
    passage_to_entities = ner["passage_hash_id_to_entities"]
    sentence_to_entities = ner["sentence_to_entities"]
    _, _, passage_to_entities, rag.entity_to_sentence, rag.sentence_to_entity = (
        rag.extract_nodes_and_edges(passage_to_entities, sentence_to_entities)
    )
    rag.entity_hash_id_to_sentence_hash_ids = {
        rag.entity_embedding_store.text_to_hash_id[entity]: [
            rag.sentence_embedding_store.text_to_hash_id[sentence]
            for sentence in sentences
        ]
        for entity, sentences in rag.entity_to_sentence.items()
    }
    rag.sentence_hash_id_to_entity_hash_ids = {
        rag.sentence_embedding_store.text_to_hash_id[sentence]: [
            rag.entity_embedding_store.text_to_hash_id[entity]
            for entity in entities
        ]
        for sentence, entities in rag.sentence_to_entity.items()
    }
    rag.node_to_node_stats = defaultdict(dict)
    rag.add_entity_to_passage_edges(passage_to_entities)
    rag.add_adjacent_passage_edges()
    rag.augment_graph()

    # Reuse the original seed, graph walk, PPR and dense fallback methods.
    # Avoid LinearRAG.retrieve(), which copies all mmap vectors into RAM.
    rag.entity_hash_ids = rag.entity_embedding_store.hash_ids
    rag.entity_embeddings = rag.entity_embedding_store.embeddings
    rag.passage_hash_ids = rag.passage_embedding_store.hash_ids
    rag.passage_embeddings = rag.passage_embedding_store.embeddings
    rag.sentence_hash_ids = rag.sentence_embedding_store.hash_ids
    rag.sentence_embeddings = rag.sentence_embedding_store.embeddings
    rag.vertex_idx_to_node_name = {vertex.index: vertex["name"] for vertex in rag.graph.vs}
    question_embedding = config.embedding_model.encode(
        query, normalize_embeddings=True, show_progress_bar=False, batch_size=config.batch_size
    )
    seed_indices, seed_entities, seed_hash_ids, seed_scores = rag.get_seed_entities(query)
    if seed_entities:
        ranked_hash_ids, ranked_scores = rag.graph_search_with_seed_entities(
            query, question_embedding, seed_indices, seed_entities, seed_hash_ids, seed_scores
        )
        selected_hash_ids = ranked_hash_ids[:top_k]
        selected_scores = ranked_scores[:top_k]
        selected_passages = [rag.passage_embedding_store.hash_id_to_text[value] for value in selected_hash_ids]
    else:
        ranked_indices, ranked_scores = rag.dense_passage_retrieval(question_embedding)
        selected_indices = ranked_indices[:top_k]
        selected_scores = ranked_scores[:top_k]
        selected_passages = [rag.passage_embedding_store.texts[index] for index in selected_indices]

    source_path = repo / "dataset" / dataset / "chunks.json"
    passages = []
    for text, score in zip(selected_passages, selected_scores):
        source_id = rag.passage_embedding_store.text_to_hash_id[text]
        prefix = text.split(":", 1)[0]
        chunk_index = int(prefix) if prefix.isdigit() else None
        passages.append({
            "content": text,
            "source_id": source_id,
            "source_ref": str(source_path),
            "chunk_index": chunk_index,
            "score": float(score),
        })
    return {"status": "success", "dataset": dataset, "passages": passages}


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        request = json.load(sys.stdin)
        with redirect_stdout(sys.stderr):
            output = search(request)
        print(json.dumps(output, ensure_ascii=False))
    except Exception as exc:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"status": "error", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        sys.exit(1)
