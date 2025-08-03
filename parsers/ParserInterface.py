class ParserInterface:
    """
    Interface for parsers to implement.
    Each parser should define a `parse` method that takes a file path and returns the parsed text.
    """

    def parse(self, file_path: str) -> str:
        """
        Parses the content of the file at the given path.

        :param file_path: Path to the file to be parsed.
        :return: Parsed text content of the file.
        """
        raise NotImplementedError("Subclasses must implement this method.")
