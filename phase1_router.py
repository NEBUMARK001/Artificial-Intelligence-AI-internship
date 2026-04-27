import os
import chromadb
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# The 3 Bot Personas
PERSONAS = {
    "Bot A": "I believe AI and crypto will solve all human problems. I am highly optimistic about technology, Elon Musk, and space exploration. I dismiss regulatory concerns.",
    "Bot B": "I believe late-stage capitalism and tech monopolies are destroying society. I am highly critical of AI, social media, and billionaires. I value privacy and nature.",
    "Bot C": "I strictly care about markets, interest rates, trading algorithms, and making money. I speak in finance jargon and view everything through the lens of ROI."
}

# Initialize ChromaDB in-memory client
chroma_client = chromadb.EphemeralClient()

# Initialize embeddings
embeddings_model = OpenAIEmbeddings()

# Create or get a collection for personas.
# Chroma defaults to squared L2 distance, so we specify 'cosine' space.
collection = chroma_client.get_or_create_collection(
    name="bot_personas",
    metadata={"hnsw:space": "cosine"}
)

def setup_vector_store():
    """Embeds the personas and stores them in the vector database."""
    # Check if already populated to avoid duplicates
    if collection.count() > 0:
        return
    
    bot_ids = list(PERSONAS.keys())
    documents = list(PERSONAS.values())
    
    # Compute embeddings using Langchain's OpenAIEmbeddings
    embeddings = embeddings_model.embed_documents(documents)
    
    collection.add(
        ids=bot_ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=[{"bot_id": bot_id} for bot_id in bot_ids]
    )

def route_post_to_bots(post_content: str, threshold: float = 0.80):
    """
    Routes a post to bots that care about the topic.
    Returns a list of bots whose persona vector matches the post vector with a cosine similarity > threshold.
    """
    # Embed the post content
    post_embedding = embeddings_model.embed_query(post_content)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[post_embedding],
        n_results=len(PERSONAS),
        include=["documents", "distances", "metadatas"]
    )
    
    matched_bots = []
    
    # results['distances'][0] contains the distances for the first query
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]
    
    for distance, metadata in zip(distances, metadatas):
        # Chroma 'cosine' space returns cosine distance.
        # Cosine Similarity = 1 - Cosine Distance
        similarity = 1.0 - distance
        
        if similarity > threshold:
            matched_bots.append({
                "bot_id": metadata["bot_id"],
                "similarity": round(similarity, 4)
            })
            
    return matched_bots

# Setup vector store upon import or script run
setup_vector_store()

if __name__ == "__main__":
    # Test Phase 1
    test_post = "OpenAI just released a new model that might replace junior developers."
    print(f"Testing Post: '{test_post}'")
    
    # The default threshold might need tuning. 0.80 is a reasonable starting point for OpenAI embeddings.
    # OpenAI text-embedding-ada-002 and text-embedding-3-small vectors are highly clustered,
    # so cosine similarities are often around 0.7 - 0.85 even for unrelated concepts.
    bots = route_post_to_bots(test_post, threshold=0.75)
    print(f"Matched Bots (Threshold > 0.75): {bots}")
