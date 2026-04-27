import json
from typing import Dict, TypedDict
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

# Phase 2: Autonomous Content Engine (LangGraph)

# 1. The Mock Tool
@tool
def mock_searxng_search(query: str) -> str:
    """Mock search tool that returns hardcoded recent news headlines based on keywords."""
    query_lower = query.lower()
    if "crypto" in query_lower or "bitcoin" in query_lower:
        return "Bitcoin hits new all-time high amid regulatory ETF approvals."
    elif "ai" in query_lower or "openai" in query_lower or "tech" in query_lower:
        return "OpenAI releases new reasoning model; experts debate impact on entry-level coding jobs."
    elif "finance" in query_lower or "market" in query_lower or "rates" in query_lower or "roi" in query_lower:
        return "Federal reserve holds interest rates steady; markets rally on tech earnings."
    elif "climate" in query_lower or "nature" in query_lower or "capitalism" in query_lower or "society" in query_lower:
        return "Global temperatures break records; new regulations proposed for tech monopolies and data centers."
    else:
        return f"Recent developments regarding {query} show mixed reactions from the public."

# 2. The LangGraph State
class AgentState(TypedDict):
    bot_id: str
    persona: str
    search_query: str
    search_results: str
    final_post: Dict

# 3. Output Schema for Structured Output
class PostOutput(BaseModel):
    bot_id: str = Field(description="The ID of the bot making the post.")
    topic: str = Field(description="The short topic of the post.")
    post_content: str = Field(description="The highly opinionated, 280-character max post.")

# 4. Nodes

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

def decide_search(state: AgentState):
    """Node 1: Decide Search. The LLM decides what topic it wants to post about today."""
    persona = state["persona"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI bot with the following persona:\n{persona}\n\n"
                   "You need to write a social media post today. Decide what you want to post about "
                   "and formulate a brief search query (1-4 words) to get the latest news on this topic. "
                   "Reply ONLY with the search query, nothing else."),
        ("user", "What should I search for?")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"persona": persona})
    search_query = response.content.strip()
    
    return {"search_query": search_query}

def web_search(state: AgentState):
    """Node 2: Web Search. Executes the mock_searxng_search tool."""
    query = state["search_query"]
    # Calling the tool directly since it's a simple function for our mock
    results = mock_searxng_search.invoke({"query": query})
    return {"search_results": results}

def draft_post(state: AgentState):
    """Node 3: Draft Post. Uses System Prompt + Search Results to generate a JSON post."""
    persona = state["persona"]
    search_results = state["search_results"]
    bot_id = state["bot_id"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI bot with the following persona:\n{persona}\n\n"
                   "You must write a highly opinionated social media post (max 280 characters) "
                   "based on the following recent news:\n{search_results}\n\n"
                   "Ensure you strongly reflect your persona."),
        ("user", "Draft the post now.")
    ])
    
    # Use structured output to guarantee JSON format
    structured_llm = llm.with_structured_output(PostOutput)
    chain = prompt | structured_llm
    
    result = chain.invoke({
        "persona": persona,
        "search_results": search_results
    })
    
    # result is a Pydantic model. Convert to dict.
    # Force the bot_id to be correct just in case the LLM hallucinates it
    final_post = result.model_dump()
    final_post["bot_id"] = bot_id 
    
    return {"final_post": final_post}

# 5. Build Graph
workflow = StateGraph(AgentState)

workflow.add_node("decide_search", decide_search)
workflow.add_node("web_search", web_search)
workflow.add_node("draft_post", draft_post)

workflow.add_edge(START, "decide_search")
workflow.add_edge("decide_search", "web_search")
workflow.add_edge("web_search", "draft_post")
workflow.add_edge("draft_post", END)

app = workflow.compile()

def run_content_engine(bot_id: str, persona: str):
    """Runs the LangGraph autonomous content engine for a given bot."""
    initial_state = {
        "bot_id": bot_id,
        "persona": persona,
        "search_query": "",
        "search_results": "",
        "final_post": {}
    }
    
    # Run the graph
    result = app.invoke(initial_state)
    return result["final_post"]

if __name__ == "__main__":
    from phase1_router import PERSONAS
    
    bot_id = "Bot B"
    persona = PERSONAS[bot_id]
    
    print(f"Running Content Engine for {bot_id}...")
    post_json = run_content_engine(bot_id, persona)
    
    print(json.dumps(post_json, indent=2))
