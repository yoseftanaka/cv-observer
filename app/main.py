import asyncio
from agents import Runner, Agent, function_tool
from ocr.ocr_processor import OCRProcessor
from chroma.chroma_processor import ChromaProcessor
from custom_agents.bullet_point_agent import BulletPointAgent

ocr = OCRProcessor()
chroma = ChromaProcessor("cv_data")
bullet_agent = BulletPointAgent()


@function_tool
def process_cv():
    """Extracts OCR text from cv.pdf."""
    return ocr.process_pdf().model_dump()


@function_tool
def extract_bullet_points(markdown: str):
    """Convert markdown text into flat bullet points."""
    return bullet_agent.generate_bullet_point(markdown).choices[0].message.content


@function_tool
def store_in_chroma(document: str, doc_id: str):
    """Store extracted bullet points into Chroma for retrieval."""
    collection = chroma.get_collection("cv_data")
    collection.add(
        documents=[document],
        metadatas=[{"document_id": doc_id}],
        ids=[doc_id],
    )
    return f"Stored doc {doc_id}"


@function_tool
def query_chroma(prompt: str, chunk_size: int = 3):
    """Query Chroma for relevant CV info."""
    result = chroma.query(prompt, "cv_data", chunk_size, None)
    return result.get("documents")[0]


assistant_agent = Agent(
    name="CV Assistant",
    instructions="""
    You are an autonomous assistant that can:
    - Process CVs with OCR
    - Extract bullet points
    - Store/retrieve information from Chroma
    - Answer user questions
    Decide which tools to use depending on the request.
    """,
    model="gpt-4o",
    tools=[process_cv, extract_bullet_points, store_in_chroma, query_chroma],
)


async def main():
    messages = []
    while True:
        user_input = input("Prompt: ")
        messages.append({"role": "user", "content": user_input})
        runner = await Runner.run(starting_agent=assistant_agent, input=messages)
        messages = runner.to_input_list()
        print(runner.last_agent.name)
        print(runner.final_output)
        print("======" * 20)


if __name__ == "__main__":
    asyncio.run(main())
