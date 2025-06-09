import os
import json
import time
from datetime import datetime
import logging
import re
import threading
import numpy as np

# Conditional imports for optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    SentenceTransformer, faiss = None, None
try:
    import sqlite3
except ImportError:
    sqlite3 = None
try:
    from datasets import load_dataset, Dataset
except ImportError:
    load_dataset, Dataset = None, None

log = logging.getLogger(__name__)

# --- Configuration ---
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "RAM").upper()
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "app_data/ai_memory.db")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_MEMORY_DATASET_REPO = os.getenv("HF_MEMORY_DATASET_REPO")
HF_RULES_DATASET_REPO = os.getenv("HF_RULES_DATASET_REPO")

# --- Globals for RAG ---
_embedder, _dimension = None, 384
_faiss_memory_index, _memory_items_list = None, []
_faiss_rules_index, _rules_items_list = None, []
_initialized, _init_lock = False, threading.Lock()

def _get_sqlite_connection():
    if not sqlite3: raise ImportError("sqlite3 module is required for SQLite backend.")
    db_dir = os.path.dirname(SQLITE_DB_PATH)
    if db_dir: os.makedirs(db_dir, exist_ok=True)
    return sqlite3.connect(SQLITE_DB_PATH, timeout=10)

def _init_sqlite_tables():
    try:
        with _get_sqlite_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS memories (id INTEGER PRIMARY KEY, memory_json TEXT NOT NULL UNIQUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            cursor.execute("CREATE TABLE IF NOT EXISTS rules (id INTEGER PRIMARY KEY, rule_text TEXT NOT NULL UNIQUE, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
            conn.commit()
    except Exception as e:
        log.error(f"SQLite table initialization error: {e}", exc_info=True)

def initialize_memory_system():
    global _initialized, _embedder, _dimension, _faiss_memory_index, _memory_items_list, _faiss_rules_index, _rules_items_list
    with _init_lock:
        if _initialized: return
        log.info(f"Initializing memory system with backend: {STORAGE_BACKEND}")
        if not SentenceTransformer or not faiss:
            log.critical("SentenceTransformers or FAISS not installed. Semantic search is unavailable.")
            return
        try:
            _embedder = SentenceTransformer('all-MiniLM-L6-v2', cache_folder="./sentence_transformer_cache")
            _dimension = _embedder.get_sentence_embedding_dimension()
        except Exception as e:
            log.critical(f"Failed to load SentenceTransformer model: {e}", exc_info=True)
            return

        if STORAGE_BACKEND == "SQLITE": _init_sqlite_tables()
        
        # Load Memories and Rules from backend
        _memory_items_list = _load_data_from_backend("memory")
        _rules_items_list = _load_data_from_backend("rule")
        
        # Build FAISS indices
        _faiss_memory_index = _build_faiss_index(_memory_items_list, "memory")
        log.info(f"Loaded {len(_memory_items_list)} memories and built FAISS index.")
        _rules_items_list = sorted(list(set(_rules_items_list))) # Ensure unique before indexing
        _faiss_rules_index = _build_faiss_index(_rules_items_list, "rule")
        log.info(f"Loaded {len(_rules_items_list)} rules and built FAISS index.")
        
        _initialized = True

def _load_data_from_backend(item_type: str) -> list:
    """Loads data for either 'memory' or 'rule' from the configured backend."""
    col_name = "memory_json" if item_type == "memory" else "rule_text"
    repo_name = HF_MEMORY_DATASET_REPO if item_type == "memory" else HF_RULES_DATASET_REPO
    
    if STORAGE_BACKEND == "SQLITE" and sqlite3:
        try:
            with _get_sqlite_connection() as conn:
                return [row[0] for row in conn.execute(f"SELECT {col_name} FROM {item_type}s ORDER BY created_at")]
        except Exception as e:
            log.error(f"Error loading {item_type}s from SQLite: {e}")
    elif STORAGE_BACKEND == "HF_DATASET" and HF_TOKEN and load_dataset and repo_name:
        try:
            dataset = load_dataset(repo_name, token=HF_TOKEN, trust_remote_code=True)
            if "train" in dataset and col_name in dataset["train"].column_names:
                return dataset["train"][col_name]
        except Exception as e:
            log.error(f"Error loading {item_type}s from HF Dataset {repo_name}: {e}")
    return []

def _build_faiss_index(items_list: list, item_type: str):
    """Helper to build a FAISS index from a list of strings or JSON strings."""
    index = faiss.IndexFlatL2(_dimension)
    if not items_list: return index
    
    texts_to_embed = []
    if item_type == "memory":
        for mem_json_str in items_list:
            try:
                mem_obj = json.loads(mem_json_str)
                texts_to_embed.append(f"User: {mem_obj.get('user_input', '')}\nAI: {mem_obj.get('bot_response', '')}\nTakeaway: {mem_obj.get('metrics', {}).get('takeaway', 'N/A')}")
            except (json.JSONDecodeError, TypeError): continue
    else: # Rules are just strings
        texts_to_embed = items_list

    if texts_to_embed:
        embeddings = _embedder.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=False)
        if embeddings.ndim == 2 and embeddings.shape[1] == _dimension:
            index.add(embeddings.astype(np.float32))
    return index

def _persist_data(item_list: list, repo_name: str, col_name: str):
    """Pushes a list of data to a Hugging Face Dataset."""
    if STORAGE_BACKEND == "HF_DATASET" and HF_TOKEN and repo_name and Dataset:
        try:
            log.info(f"Pushing {len(item_list)} items to HF Hub: {repo_name}")
            Dataset.from_dict({col_name: list(item_list)}).push_to_hub(repo_name, token=HF_TOKEN, private=True)
        except Exception as e:
            log.error(f"Failed to push to HF Dataset {repo_name}: {e}")

def add_memory_entry(user_input: str, metrics: dict, bot_response: str):
    if not _initialized: initialize_memory_system()
    memory_obj = {"user_input": user_input, "metrics": metrics, "bot_response": bot_response, "timestamp": datetime.utcnow().isoformat()}
    memory_json_str = json.dumps(memory_obj)
    
    text_to_embed = f"User: {user_input}\nAI: {bot_response}\nTakeaway: {metrics.get('takeaway', 'N/A')}"
    embedding = _embedder.encode([text_to_embed], convert_to_numpy=True).astype(np.float32)
    
    _faiss_memory_index.add(embedding)
    _memory_items_list.append(memory_json_str)
    
    if STORAGE_BACKEND == "SQLITE":
        with _get_sqlite_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO memories (memory_json) VALUES (?)", (memory_json_str,))
            conn.commit()
    _persist_data(_memory_items_list, HF_MEMORY_DATASET_REPO, "memory_json")

def retrieve_memories_semantic(query: str, k: int = 3) -> list[dict]:
    if not _initialized or _faiss_memory_index.ntotal == 0: return []
    query_embedding = _embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = _faiss_memory_index.search(query_embedding, min(k, _faiss_memory_index.ntotal))
    return [json.loads(_memory_items_list[i]) for i in indices[0] if 0 <= i < len(_memory_items_list)]

def add_rule_entry(rule_text: str):
    if not _initialized: initialize_memory_system()
    rule_text = rule_text.strip()
    if not rule_text or rule_text in _rules_items_list: return

    embedding = _embedder.encode([rule_text], convert_to_numpy=True).astype(np.float32)
    _faiss_rules_index.add(embedding)
    _rules_items_list.append(rule_text)
    _rules_items_list.sort()
    
    if STORAGE_BACKEND == "SQLITE":
        with _get_sqlite_connection() as conn:
            conn.execute("INSERT OR IGNORE INTO rules (rule_text) VALUES (?)", (rule_text,))
            conn.commit()
    _persist_data(_rules_items_list, HF_RULES_DATASET_REPO, "rule_text")

def retrieve_rules_semantic(query: str, k: int = 5) -> list[str]:
    if not _initialized or _faiss_rules_index.ntotal == 0: return []
    query_embedding = _embedder.encode([query], convert_to_numpy=True).astype(np.float32)
    _, indices = _faiss_rules_index.search(query_embedding, min(k, _faiss_rules_index.ntotal))
    return [_rules_items_list[i] for i in indices[0] if 0 <= i < len(_rules_items_list)]

def remove_rule_entry(rule_text_to_delete: str):
    global _faiss_rules_index
    if not _initialized or rule_text_to_delete not in _rules_items_list: return

    _rules_items_list.remove(rule_text_to_delete)
    _faiss_rules_index = _build_faiss_index(_rules_items_list, "rule") # Rebuild index

    if STORAGE_BACKEND == "SQLITE":
        with _get_sqlite_connection() as conn:
            conn.execute("DELETE FROM rules WHERE rule_text = ?", (rule_text_to_delete,))
            conn.commit()
    _persist_data(_rules_items_list, HF_RULES_DATASET_REPO, "rule_text")

def get_all_rules_cached() -> list[str]: return list(_rules_items_list)
def get_all_memories_cached() -> list[dict]: return [json.loads(m) for m in _memory_items_list]

def clear_all_memory_data_backend():
    global _memory_items_list, _faiss_memory_index
    if STORAGE_BACKEND == "SQLITE":
        with _get_sqlite_connection() as conn: conn.execute("DELETE FROM memories"); conn.commit()
    _memory_items_list = []
    if _faiss_memory_index: _faiss_memory_index.reset()
    _persist_data([], HF_MEMORY_DATASET_REPO, "memory_json")

def clear_all_rules_data_backend():
    global _rules_items_list, _faiss_rules_index
    if STORAGE_BACKEND == "SQLITE":
        with _get_sqlite_connection() as conn: conn.execute("DELETE FROM rules"); conn.commit()
    _rules_items_list = []
    if _faiss_rules_index: _faiss_rules_index.reset()
    _persist_data([], HF_RULES_DATASET_REPO, "rule_text")

def load_rules_from_file(filepath: str):
    if not os.path.exists(filepath): return
    with open(filepath, 'r', encoding='utf-8') as f:
        rules = re.split(r'\n\s*---\s*\n', f.read())
        for rule in rules:
            if rule.strip(): add_rule_entry(rule.strip())

def load_memories_from_file(filepath: str):
    if not os.path.exists(filepath): return
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    mem = json.loads(line)
                    if all(k in mem for k in ["user_input", "bot_response", "metrics"]):
                        add_memory_entry(mem["user_input"], mem["metrics"], mem["bot_response"])
                except json.JSONDecodeError: continue
