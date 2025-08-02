import argparse
from rich import print
import json
import os
import magic
from parsers.txt_parser import parse_txt
from parsers.pdf_parser import parse_pdf
from parsers.docx_parser import parse_docx
from parsers.md_parser import parse_md
from utils.cleaner import clean_text
from detectors.lexical_diversity import lex_diversity


def parse_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    mime = magic.Magic(mime=True)
    mime_type = mime.from_file(file_path)

    if mime_type == "text/plain" and ext in [".txt"]:
        return parse_txt(file_path)
    elif mime_type == "application/pdf" and ext == ".pdf":
        return parse_pdf(file_path)
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" and ext in [".docx", ".doc"]:
        return parse_docx(file_path)
    elif mime_type == "text/markdown" and ext in [".md"]:
        return parse_md(file_path)
    else:
        raise ValueError("Unsupported file format.")


def main():
    parser = argparse.ArgumentParser(description="Detect AI-generated content from files.")
    parser.add_argument("filepath", help="Path to the input file")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    print("[bold blue]🧠 AI Content Detector CLI[/bold blue]")

    try:
        # Parse the input file based on its type
        raw_text = parse_file(args.filepath)

        # Clean and preprocess the text
        text = clean_text(raw_text)

        # Lexical Diversity analysis stats
        lexical_diversity_stats = lex_diversity(text)

        print(lexical_diversity_stats)
    except Exception as e:
        print(f"[bold red]Error:[/bold red] {str(e)}")
        return


if __name__ == "__main__":
    main()
