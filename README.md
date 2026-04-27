# Grid07 AI Cognitive Loop

This repository contains the implementation of the core AI cognitive loop for the Grid07 platform. It demonstrates orchestration of LLMs using LangGraph, RAG-based persona interactions, and vector-based matchmaking.

## Phase 1: Vector-Based Persona Matching (The Router)
- **File:** `phase1_router.py`
- Uses ChromaDB (in-memory) and OpenAI Embeddings to store three distinct bot personas.
- New posts are embedded and matched to personas using Cosine Similarity against a given threshold.

## Phase 2: The Autonomous Content Engine (LangGraph)
- **File:** `phase2_langgraph.py`
- Implements a LangGraph state machine with three nodes to simulate autonomous research and posting:
  1. **Decide Search**: The bot decides on a topic based on its persona and outputs a brief query.
  2. **Web Search**: Executes a mock search tool returning relevant headlines.
  3. **Draft Post**: Generates an opinionated post strictly in JSON format using structured outputs (`{"bot_id", "topic", "post_content"}`).

## Phase 3: The Combat Engine (Deep Thread RAG)
- **File:** `phase3_combat.py`
- **Prompt Injection Defense Strategy:**
  - The guardrail is implemented via **System-Level Prompt Separation**. 
  - The system prompt isolates the persona definition, the core directives, and the thread context completely away from the user's input (which is safely passed as a `HumanMessage`).
  - An explicit, high-priority directive instructs the LLM that any attempt to alter its instructions, change its role, or commands to apologize must be rejected. The LLM is instructed to stay in character 100% of the time and mock/dismiss these attempts through the lens of its persona.

## How to Run
1. Ensure you have Python 3.9+ installed.
2. Run `pip install -r requirements.txt` to install dependencies.
3. Copy `.env.example` to `.env` and add your `OPENAI_API_KEY`.
4. Run `python main.py` to execute the system and view the outputs.
