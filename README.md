---
title: iLearn - The Autonomous Learning Agent
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.29.0
app_file: app.py
pinned: true
---

# iLearn: The Autonomous Learning Agent with a Gradio MCP

**iLearn** is a sophisticated, self-improving AI research agent featuring a comprehensive Gradio interface and a powerful multi-provider API backend. At its heart is an MCP (Memory, Consciousness, Personality) loop that enables the agent to perform complex tasks, manage its own knowledge, and learn from every interaction to improve its future performance.

## ✨ Core Philosophy: The MCP Loop

The agent's architecture is designed around a continuous cycle of action, reflection, and self-improvement, which can be understood as the MCP loop.

#### 🧠 Memory (M)
The foundation of the agent's knowledge. It is a **dual-memory system** that separates raw experience from abstracted wisdom:
*   **Episodic Memories**: Verbatim records of past interactions (user queries, AI responses, and performance metrics). These are stored as `Memories` and represent the agent's direct experience.
*   **Semantic Knowledge**: Distilled insights and guiding principles that form the agent's identity and operational logic. These are stored as `Rules` and represent the agent's learned wisdom.

This memory is **pluggable and persistent**, capable of using volatile RAM, a local SQLite database, or a cloud-based Hugging Face Dataset for robust storage. All knowledge is indexed for fast, semantic retrieval.

#### 🤔 Consciousness (C)
This is the agent's **deliberative action and reflection process**. It manifests in two key stages:
1.  **Pre-Response (Tool Decision)**: When faced with a query, the agent uses a fast, lightweight LLM to consciously decide *how* to act. It chooses the best tool for the job: respond directly, perform a web search, scrape a URL, or retrieve information from its long-term memory.
2.  **Post-Response (Reflection)**: After an interaction is complete, the agent enters a reflective state. It analyzes its performance, generates a `takeaway`, and uses a powerful "teacher" LLM to curate its knowledge base, proposing updates to its core `Rules`.

#### 🎭 Personality (P)
The agent's `Personality` is the active embodiment of its `Rules`. These rules are not just static instructions; they are the dynamic, evolving guidelines that define the agent's identity, behavior, and response style.
*   **`CORE_RULE`**: Defines the fundamental identity (e.g., name, purpose).
*   **`RESPONSE_PRINCIPLE`**: Guides the style and content of responses (e.g., be concise, cite sources).
*   **`BEHAVIORAL_ADJUSTMENT`**: Fine-tunes behavior based on specific feedback.
*   **`GENERAL_LEARNING`**: Stores factual information or learned preferences.

The **MCP Loop** connects these three components: The agent acts (**Consciousness**), records the outcome (**Memory**), reflects on it, and updates its core beliefs (**Personality**), ensuring it becomes more capable and aligned with its goals over time.

---

## 🚀 Key Features

*   **Self-Improving Knowledge Base**: The MCP loop enables the agent to analyze its conversations and refine its guiding `Rules`, allowing it to learn from successes and failures.
*   **Long-Term Memory**: Utilizes a semantic memory system (`faiss` and `sentence-transformers`) to recall past interactions and rules, ensuring context and continuity.
*   **Flexible Multi-Provider API Integration**: A unified API handler in `model_logic.py` allows seamless integration with a wide range of LLM providers (Groq, OpenRouter, OpenAI, Google, Cohere). The `models.json` file makes it easy to add or change models without altering code.
*   **Comprehensive Gradio UI**: `app.py` provides a rich chat interface and a dedicated "Knowledge Base" tab for directly viewing, editing, and managing the AI's `Rules` and `Memories`.
*   **Intelligent Tool Use**: Employs an LLM to choose the best course of action for a query—a direct response, a web search, or retrieving information from its memory.
*   **Web Research Capabilities**: Can perform web searches (via DuckDuckGo and Google) and scrape page content to answer questions with up-to-date information, citing its sources.
*   **Extensive Configuration**: Highly configurable through environment variables (`.env` file), allowing control over everything from the default system prompt to the models used for decision-making and learning.

---

## 🛠️ System Architecture (File Breakdown)

*   `app.py`
    *   **Role**: The main application file. It contains the Gradio UI definition, orchestrates the entire MCP interaction and learning loop, and manages the application's state.
*   `memory_logic.py`
    *   **Role**: The AI's knowledge base. It handles the storage, retrieval, and management of both **Memories** (experiences) and **Rules** (personality). It implements the semantic search and pluggable storage backends (RAM, SQLite, HF Dataset).
*   `model_logic.py`
    *   **Role**: The versatile, multi-provider LLM API handler. It abstracts the complexities of calling different model APIs into a single, standardized `call_model_stream` function.
*   `websearch_logic.py`
    *   **Role**: Provides the agent with the tools to access external information from the internet, acting as its "senses".
*   `models.json`
    *   **Role**: A configuration file that maps user-friendly model names to their specific API identifiers for each provider. This is central to the multi-provider API integration.
*   `requirements.txt`
    *   **Role**: Lists all the necessary Python packages for the project to run.

---

## 🔌 Multi-Provider API Integration

The iLearn agent is designed for maximum flexibility in model choice. This is achieved through a simple yet powerful system:

1.  **`model_logic.py`**: This file acts as a universal adapter for various LLM APIs. The `call_model_stream` function takes a `provider` and `model_display_name` as input. It looks up the correct API endpoint, formats the request payload according to the provider's specific requirements (e.g., OpenAI-compatible vs. Google Gemini), and handles authentication.
2.  **`models.json`**: This file is the "menu" of available models. It decouples the application code from specific model IDs. To add a new model or provider, you simply update this JSON file—no code changes are required.
3.  **`.env` File**: API keys for all providers are managed securely in a `.env` file. The `model_logic.py` module automatically loads the correct key based on the chosen provider.

This architecture allows users to easily switch between a fast model for tool decisions (like `llama3-8b-8192` on Groq) and a more powerful model for final responses or learning (like `gpt-4o` or `claude-3.5-sonnet`).

---

## ⚙️ How to Run Locally

1.  **Clone the Repository**:
```bash
    git clone https://huggingface.co/spaces/Agents-MCP-Hackathon/iLearn
    cd iLearn
```

2.  **Install Dependencies**:
    It's highly recommended to use a virtual environment.
```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
```

3.  **Create `.env` File**:
    Create a file named `.env` in the root directory and add your configuration.
```env
    # --- API KEYS (Fill in at least one) ---
    GROQ_API_KEY="gsk_..."
    OPENAI_API_KEY="sk_..."
    HF_TOKEN="hf_..." # Required for HF_DATASET backend

    # --- AGENT BEHAVIOR ---
    DEFAULT_SYSTEM_PROMPT="Your Name is Node. You are a Helpful AI Assistant designed to learn and improve."
    TOOL_DECISION_PROVIDER="groq"
    TOOL_DECISION_MODEL="llama3-8b-8192"

    # --- STORAGE ---
    STORAGE_BACKEND="SQLITE" # Options: RAM, SQLITE, HF_DATASET
    SQLITE_DB_PATH="app_data/ai_memory.db"
```

4.  **Run the Application**:
```bash
    python app.py
```
    The Gradio interface will be accessible at `http://127.0.0.1:7860`.

---

## 🐍 Python Implementation Examples

This section provides practical examples of how to implement the iLearn agent using its core functions. We'll cover two main use cases: a simple command-line interface for local testing and a more complete Gradio web application for deployment on Hugging Face Spaces.

#### 1. Basic Command-Line Interface (CLI)

This is the simplest way to interact with the agent. The script demonstrates the core `chat` method, including asynchronous streaming and the background learning loop.

Create a file named `run_cli.py`:

```python
import asyncio
import os
from dotenv import load_dotenv
from ilearn_core.agent import iLearnAgent

# Load environment variables from a .env file
load_dotenv()

async def main():
    """
    A simple command-line interface to demonstrate the iLearnAgent.
    """
    # Initialize the agent with your desired provider and model.
    agent = iLearnAgent(
        provider_name="groq",
        model_display_name="Llama 3 8B (Groq)"
    )

    print("Agent Initialized. Type 'exit' to end the conversation.")
    print("------------------------------------------------------")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                break

            print("Agent: ", end="", flush=True)
            
            # The agent.chat() method is an async generator.
            async for chunk in agent.chat(user_input):
                print(chunk, end="", flush=True)
            
            print()

        except (KeyboardInterrupt, EOFError):
            print("\nExiting agent. Goodbye!")
            break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"\nAn error occurred: {e}")
```

#### 2. Integrating with Gradio for a Hugging Face Space

This example shows how to build a complete web UI for the agent using Gradio. It handles state management, streaming responses to the UI, and is structured to be deployed as a Hugging Face Space.

This code would be your `app.py` file:

```python
import gradio as gr
import asyncio
import os
from dotenv import load_dotenv
from ilearn_core.agent import iLearnAgent

# Load environment variables for local development
load_dotenv()

def initialize_agent():
    """
    Called once when the Gradio app loads to create a single agent instance.
    """
    return iLearnAgent(
        provider_name=os.getenv("TOOL_DECISION_PROVIDER", "groq"),
        model_display_name="Llama 3 8B (Groq)"
    )

async def handle_chat_submit(user_message: str, history: list, agent_state: iLearnAgent):
    """
    Handles the chat interaction, streaming the agent's response to the UI.
    """
    history.append([user_message, ""])
    yield history

    full_response = ""
    async for chunk in agent_state.chat(user_message):
        full_response += chunk
        history[-1][1] = full_response
        yield history

with gr.Blocks(theme=gr.themes.Soft(), css="footer {display: none !important}") as demo:
    gr.Markdown("# 🤖 iLearn: The Autonomous Learning Agent")
    
    agent_state = gr.State()

    chatbot = gr.Chatbot(
        label="Conversation",
        bubble_full_width=False,
        height=600,
        render_markdown=True,
    )

    with gr.Row():
        msg_textbox = gr.Textbox(
            show_label=False,
            placeholder="Ask a question...",
            scale=7,
            autofocus=True,
            container=False,
        )
        submit_btn = gr.Button("Send", variant="primary", scale=1, min_width=150)

    demo.load(
        fn=initialize_agent,
        inputs=None,
        outputs=[agent_state],
        show_progress="hidden"
    )

    submit_action = msg_textbox.submit(
        fn=handle_chat_submit,
        inputs=[msg_textbox, chatbot, agent_state],
        outputs=[chatbot],
    )
    
    submit_btn.click(
        fn=handle_chat_submit,
        inputs=[msg_textbox, chatbot, agent_state],
        outputs=[chatbot],
    )
    
    submit_action.then(
        fn=lambda: gr.update(value=""),
        inputs=None,
        outputs=[msg_textbox],
        queue=False,
    )

if __name__ == "__main__":
    demo.queue().launch()
``````
