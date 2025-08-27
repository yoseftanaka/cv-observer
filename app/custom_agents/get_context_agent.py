from dotenv import load_dotenv
from agents import function_tool, Agent
from chroma.chroma_processor import ChromaProcessor

load_dotenv()

chroma_processor = ChromaProcessor("cv_data")


@function_tool
def get_context(prompt: str, collection_name: str = "cv_data", chunk_size: int = 3):
    """
    Get context from vector database for relevant information.
    """
    query_result = chroma_processor.query(prompt, collection_name, chunk_size, None)
    return query_result.get("documents")[0]


SYSTEM_PROMPT = """
You are an assistant agent for answering question about CV information

RULES:
- Always use get_context to get relevant data based on user query
- If there is no relevant data, reply with "The CV you provided didn't have the queried information"
- If there is relevant data, use the data to answer the question
"""

agent = Agent(
    name="Assistant Agent",
    instructions=SYSTEM_PROMPT,
    model="gpt-4o",
    tools=[get_context],
)
