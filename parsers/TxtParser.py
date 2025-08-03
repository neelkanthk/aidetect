from parsers.ParserInterface import ParserInterface
from pathlib import Path


class TxtParser(ParserInterface):
    """
    Parser for TXT files.
    Implements the `parse` method to extract text from TXT documents.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def parse(self) -> str:
        try:
            return Path(self.file_path).read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to different encoding if UTF-8 fails
            return Path(self.file_path).read_text(encoding='latin-1')
        except Exception as e:
            raise Exception(f"Error reading file: {e}")
