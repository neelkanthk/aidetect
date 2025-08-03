from parsers.ParserInterface import ParserInterface
from pathlib import Path
import docx


class DocxParser(ParserInterface):
    """
    Parser for DOCX files.
    Implements the `parse` method to extract text from DOCX documents.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def parse(self) -> str:
        if self.file_path.suffix.lower() != '.docx':
            raise ValueError("Only .docx files are supported")

        try:
            doc = docx.Document(self.file_path)
            # Extract text, preserving paragraphs and removing empty lines
            text = "\n".join([
                para.text.strip()
                for para in doc.paragraphs
                if para.text.strip()
            ])
            return text

        except Exception as e:
            raise Exception(f"Error parsing DOCX file: {str(e)}")
