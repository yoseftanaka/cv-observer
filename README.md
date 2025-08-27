# README

## Setup
- To sync:
`uv sync`

- To run application:
`uv run app/main.py`

## Where is RAG?
- CV.pdf is saved as asset and read by OCR processor
- embedding is created

## Where is Agentic AI?
- assistant_agent decide by itself which tools to use, I just give it an order of instruction via prompt (Chain of Thoughts). It contains 4 tools.

## Warning Note!
- If you restart the app repeatedly, your files may be cluttered inside data folder, as avery time the app is launched, something is created inside data folder.

## Additional Note:
- The CV read is inside /app/ocr/cvs. Feel free to change the CV.