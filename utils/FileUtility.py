from pathlib import Path
import magic
import os


class FileUtility:
    @staticmethod
    def get_file_extension(file_path: str) -> str:
        """Returns the file extension of the given file path."""
        return os.path.splitext(file_path)[1].lower()

    @staticmethod
    def is_supported_file_type(file_path: str) -> bool:
        """Checks if the file type is supported."""
        supported_extensions = [".txt", ".pdf", ".docx"]
        return FileUtility.get_file_extension(file_path) in supported_extensions

    @staticmethod
    def get_file_mime(file_path: str) -> str:
        """Returns the MIME type of the file."""
        mime = magic.Magic(mime=True)
        return mime.from_file(file_path)
