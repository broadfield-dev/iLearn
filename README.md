# iLearn-Memory: A Python Library for Self-Improving AI

**iLearn-Memory** is a Python library designed to provide the core components for building autonomous, self-improving AI agents. It encapsulates a dual-memory system, a reflective learning loop, and a flexible multi-provider LLM interface, allowing developers to focus on agent orchestration and tool development.

The library is built on the **MCP (Memory, Consciousness, Personality)** loop, a continuous cycle of action, reflection, and self-improvement.

## ✨ Core Philosophy: The MCP Loop

#### 🧠 Memory (M)
The foundation of the agent's knowledge. This library provides a **dual-memory system** that separates raw experience from abstracted wisdom:
*   **Episodic Memories**: Verbatim records of interactions (user queries, AI responses, and takeaways). These represent the agent's direct experience.
*   **Semantic Knowledge (Rules)**: Distilled insights and guiding principles that form the agent's identity and operational logic. These represent the agent's learned wisdom.

This memory system is **pluggable and persistent**, capable of using volatile RAM, a local SQLite database, or a cloud-based Hugging Face Dataset for robust storage. All knowledge is indexed using `FAISS` for fast, semantic retrieval.

#### 🤔 Consciousness (C)
This is the agent's **deliberative and reflective process**. The library provides the core function for the reflection stage:
*   **Post-Interaction Reflection**: After an interaction is complete, an agent can use the `generate_rule_updates` function to analyze its performance. It uses a "teacher" LLM to curate its knowledge base, proposing additions or updates to its core `Rules`. The orchestration of when and how to call this is left to the developer.

#### 🎭 Personality (P)
The agent's `Personality` is the active embodiment of its `Rules`. These rules are not just static instructions; they are the dynamic, evolving guidelines that define an agent's identity, behavior, and response style. The library supports rule types such as:
*   **`CORE_RULE`**: Defines the fundamental identity.
*   **`RESPONSE_PRINCIPLE`**: Guides the style and content of responses.
*   **`BEHAVIORAL_ADJUSTMENT`**: Fine-tunes behavior based on specific feedback.
*   **`GENERAL_LEARNING`**: Stores factual information or learned preferences.

The **MCP Loop** connects these components: An agent acts, records the outcome (**Memory**), and then uses this library to reflect on it and update its core beliefs (**Personality**), ensuring it becomes more capable over time.

---

## 🚀 Key Features

*   **Self-Improving Knowledge Base**: Provides functions to analyze interactions and refine an agent's guiding `Rules`, enabling it to learn from successes and failures.
*   **Long-Term Semantic Memory**: Utilizes `FAISS` and `sentence-transformers` to recall past interactions (`Memories`) and `Rules` with semantic search, ensuring context and continuity.
*   **Pluggable Storage Backends**: Configure the library to store data in-memory (RAM), a local `SQLite` database, or a private `Hugging Face Dataset` repository.
*   **Flexible Multi-Provider LLM Integration**: A unified API handler in `llm.py` allows seamless integration with multiple LLM providers (Groq, OpenRouter, OpenAI, Google).
*   **Dynamic Model Configuration**: The `models.json` file makes it easy to add or change models without altering library code.
*   **Extensive Configuration**: Highly configurable through environment variables (`.env` file), controlling the storage backend and API keys.

---

## 🛠️ Library Architecture (File Breakdown)

*   `ilearn_memory/storage.py`
    *   **Role**: The heart of the AI's knowledge base. It handles the storage, retrieval, and management of both **Memories** (experiences) and **Rules** (personality). It implements the semantic search (`FAISS`) and pluggable storage backends (RAM, SQLite, HF Dataset).
*   `ilearn_memory/learning.py`
    *   **Role**: Implements the reflective part of the learning loop. Its `generate_rule_updates` function uses an LLM to analyze an interaction and propose structured updates to the agent's `Rules`.
*   `ilearn_memory/llm.py`
    *   **Role**: A versatile, multi-provider LLM API handler. It abstracts the complexities of calling different model APIs (e.g., OpenAI-compatible vs. Google Gemini) into a single, standardized `call_model_stream` function.
*   `ilearn_memory/models.json`
    *   **Role**: A configuration file that maps user-friendly model names to their specific API identifiers for each provider. This is central to the multi-provider API integration.
*   `setup.py` & `requirements.txt`
    *   **Role**: Standard Python package definition and dependency list for installing the library.

---

## ⚙️ Installation & Configuration

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/broadfield-dev/iLearn-memory.git
    cd iLearn-memory
    ```

2.  **Install Dependencies**:
    It's highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```
    To make the library available to other projects on your system, install it in editable mode:
    ```bash
    pip install -e .
    ```

3.  **Create `.env` File**:
    Create a file named `.env` in your project's root directory and add your configuration.
    
## API KEYS (Fill in at least one provider you intend to use)
```python
    GROQ_API_KEY="gsk_..."
    OPENAI_API_KEY="sk_..."
    GOOGLE_API_KEY="AIza..."
    # Required for the HF_DATASET backend
    HF_TOKEN="hf_..."
```
## STORAGE CONFIGURATION
```python
    #Options: RAM, SQLITE, HF_DATASET
    STORAGE_BACKEND="SQLITE" 
    SQLITE_DB_PATH="data/ai_memory.db" 
    #Optional: For HF_DATASET backend
    #HF_MEMORY_DATASET_REPO="your-hf-username/memories-repo"
    #HF_RULES_DATASET_REPO="your-hf-username/rules-repo"
```
---
## 🐍 Usage Example

Below is an example script (`example.py`) demonstrating how to use the `ilearn-memory` library to build a simple agent that learns from an interaction.

```python
import asyncio
import os
from dotenv import load_dotenv

# Import the library's functions
import ilearn_memory
from ilearn_memory.llm import call_model_stream

async def main():
    # Load .env file for API keys and storage config
    load_dotenv()
    print("--- Initializing Memory System ---")
    ilearn_memory.initialize_memory_system()

    # Clear previous data for a clean run
    ilearn_memory.clear_all_rules_data_backend()
    ilearn_memory.clear_all_memory_data_backend()

    # 1. Add some initial guiding rules
    print("\n--- Adding Initial Rules ---")
    initial_rules = [
        "[CORE_RULE|1.0] My name is iLearn. I am a helpful assistant.",
        "[RESPONSE_PRINCIPLE|0.9] I should be concise and to the point."
    ]
    for rule in initial_rules:
        ilearn_memory.add_rule_entry(rule)
    print(f"Initial rules in KB: {ilearn_memory.get_all_rules_cached()}")

    # 2. Simulate a user interaction
    user_query = "What is the capital of France, and can you tell me a bit about it?"
    print(f"\n--- Simulating Interaction ---")
    print(f"User Query: {user_query}")

    # Use the library's LLM handler to get a response
    # A real agent would have more sophisticated logic for prompt construction
    relevant_rules = ilearn_memory.retrieve_rules_semantic(user_query, k=2)
    prompt_context = "\n".join(relevant_rules)
    
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Follow these rules:\n{prompt_context}"},
        {"role": "user", "content": user_query}
    ]
    
    print("Agent Response: ", end="")
    bot_response = ""
    # Using a provider and model for this example (ensure key is in .env)
    provider = os.getenv("PROVIDER", "groq")
    model = os.getenv("MODEL", "Llama 3 8B (Groq)")

    async for chunk in call_model_stream(provider, model, messages):
        print(chunk, end="", flush=True)
        bot_response += chunk
    print()

    # 3. Store the interaction as a 'memory'

    interaction_summary = f"User asked '{user_query}', AI responded '{bot_response}'"
    metrics = {"takeaway": "Provided a direct factual answer but was a bit wordy."}
    ilearn_memory.add_memory_entry(user_query, metrics, bot_response)
    print("\n--- Stored Interaction in Episodic Memory ---")
    print(f"Total memories stored: {len(ilearn_memory.get_all_memories_cached())}")

    # 4. Trigger the reflective learning loop
    print("\n--- Triggering Reflective Learning ---")
    # This process would typically run in the background in a real agent
    proposed_updates = await ilearn_memory.generate_rule_updates(
        interaction_summary=interaction_summary,
        relevant_rules=relevant_rules,
        provider=provider, # A more powerful "teacher" model could be used here
        model_display_name=model
    )

    # 5. Apply the learned updates to the Rules database
    if proposed_updates:
        print(f"\n--- Applying {len(proposed_updates)} Learned Updates ---")
        for op in proposed_updates:
            print(f"Action: {op['action']}, Insight: {op['insight']}")
            if op['action'] == 'update' and op.get('old_insight_to_replace'):
                ilearn_memory.remove_rule_entry(op['old_insight_to_replace'])
                ilearn_memory.add_rule_entry(op['insight'])
            elif op['action'] == 'add':
                ilearn_memory.add_rule_entry(op['insight'])
    else:
        print("\n--- No new rules were generated from this interaction. ---")

    print("\n--- Final State of Rules ---")
    print(f"Final rules in KB: {ilearn_memory.get_all_rules_cached()}")

if __name__ == "__main__":
    # Ensure you have a .env file with an API key (e.g., GROQ_API_KEY)
    asyncio.run(main())

```
