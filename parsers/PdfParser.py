from parsers.ParserInterface import ParserInterface
import pdfplumber


class PdfParser(ParserInterface):
    """
    Parser for PDF files.
    Implements the `parse` method to extract text from PDF documents.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def parse(self) -> str:
        text = ""
        try:
            with pdfplumber.open(self.file_path) as pdf:
                page_count = len(pdf.pages)
                if page_count > 0:
                    for page in pdf.pages:
                        text += page.extract_text() or ""
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF file: {e}")
