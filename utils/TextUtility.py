import re
import unicodedata


class TextUtility:
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Normalizes and cleans text extracted from various file formats (PDF, TXT, DOCX, MD).

        Args:
            text (str): The input text to be cleaned

        Returns:
            str: Cleaned and normalized text

        Note:
            - Replaces multiple whitespace characters with a single space
            - Removes leading and trailing whitespace
            - Normalizes unicode characters
            - Removes special characters and control characters
            - Fixes common PDF extraction artifacts
            - Removes markdown formatting
            - Standardizes quotes and apostrophes
        """

        # Handle None or empty string
        if not text:
            return ""
            # Normalize unicode characters
        text = unicodedata.normalize('NFKC', text)

        # Replace common PDF artifacts
        text = re.sub(r'\f', ' ', text)
        text = re.sub(r'[˗‐‑‒–—―]', '-', text)  # Various hyphens/dashes

        # Remove markdown formatting
        text = re.sub(r'[#*`_~\[\]()]', '', text)
        text = re.sub(r'\n>{1,}', ' ', text)

        # Standardize quotes and apostrophes
        text = re.sub(r'[''′`]', "'", text)
        text = re.sub(r'[""″]', '"', text)

        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable())

        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()  # Remove leading/trailing whitespace
