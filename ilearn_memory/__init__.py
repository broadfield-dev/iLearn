import logging
from .storage import (
    initialize_memory_system,
    add_memory_entry,
    retrieve_memories_semantic,
    get_all_memories_cached,
    clear_all_memory_data_backend,
    add_rule_entry,
    retrieve_rules_semantic,
    remove_rule_entry,
    get_all_rules_cached,
    clear_all_rules_data_backend,
    load_memories_from_file,
    load_rules_from_file,
)
from .learning import generate_rule_updates

# Configure a logger for the library
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# Suppress overly verbose logs from third-party libraries
for lib_name in ["urllib3", "huggingface_hub", "sentence_transformers", "faiss", "datasets"]:
    if logging.getLogger(lib_name):
        logging.getLogger(lib_name).setLevel(logging.WARNING)

__all__ = [
    "initialize_memory_system", "add_memory_entry", "retrieve_memories_semantic", 
    "get_all_memories_cached", "clear_all_memory_data_backend", "add_rule_entry", 
    "retrieve_rules_semantic", "remove_rule_entry", "get_all_rules_cached", 
    "clear_all_rules_data_backend", "generate_rule_updates",
    "load_memories_from_file", "load_rules_from_file"
]
