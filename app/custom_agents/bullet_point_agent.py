import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class BulletPointAgent:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate_bullet_point(self, markdown):
        SYSTEM_PROMPT = """
            You are data extractor
            You need to extract bullet points from the markdown

            OUTPUT FORMAT:
            - [point_1]
            - [point_2]

            IMPORTANT:
            - Flat bullet points. No nested bullet point.
            - Each bullet point should be short and concise.
            - Do not add headings
            - Do not add explanation
            - Always include numbers and date if present
        """

        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Convert the following markdown to bullet points: {markdown}",
                },
            ],
        )
        return response
