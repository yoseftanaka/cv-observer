import os
from dotenv import load_dotenv
from mistralai import Mistral

load_dotenv()


class OCRProcessor:
    def __init__(self):
        self.api_key = os.getenv("MISTRAL_API_KEY")
        self.client = Mistral(api_key=self.api_key)

    def process_pdf(self):
        uploaded = self.client.files.upload(
            file={"file_name": "cv.pdf", "content": open("app/ocr/cvs/cv.pdf", "rb")},
            purpose="ocr",
        )

        signed = self.client.files.get_signed_url(file_id=uploaded.id)

        ocr_resp = self.client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed.url,
            },
        )

        return ocr_resp
