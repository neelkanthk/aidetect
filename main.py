import argparse
from rich import print
import json
import os
import magic
from utils.cleaner import clean_text
from parsers.PdfParser import PdfParser
from parsers.TxtParser import TxtParser
from parsers.DocxParser import DocxParser
from utils.FileUtility import FileUtility
from utils.TextUtility import TextUtility
from detectors.Perplexity import Perplexity
from detectors.LexicalDiversity import LexicalDiversity
from detectors.Perplexity import Perplexity


def parse_file(file_path):
    """    Parses the input file based on its type and returns the raw text.   """

    ext = FileUtility.get_file_extension(file_path)
    mime_type = FileUtility.get_file_mime(file_path)

    if mime_type == "text/plain" and ext in [".txt"]:
        return TxtParser(file_path).parse()
    elif mime_type == "application/pdf" and ext == ".pdf":
        return PdfParser(file_path).parse()
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" and ext in [".docx", ".doc"]:
        return DocxParser(file_path).parse()
    else:
        raise ValueError("Unsupported file format.")


def main():
    parser = argparse.ArgumentParser(description="Detect AI-generated content from files.")
    parser.add_argument("filepath", help="Path to the input file")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")

    args = parser.parse_args()

    lexical_diversity_detector = LexicalDiversity()
    perplexity_detector = Perplexity()

    print("[bold blue]🧠 AI Detect - An AI Content Detector Tool[/bold blue]")

    try:
        # Parse the input file based on its type
        raw_text = parse_file(args.filepath)

        # Clean and preprocess the text
        text = TextUtility.clean_text(raw_text)

        # Lexical Diversity analysis stats
        lexical_diversity_stats = LexicalDiversity(text).calculate()

        # Estimate perplexity of the text
        perplexity = Perplexity(text, model_id="Qwen/Qwen2.5-0.5B").calculate()

        print("[bold green]Analysis Complete![/bold green]")
        if args.json:
            output = {
                "lexical_diversity": lexical_diversity_stats,
                "perplexity": perplexity
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"Lexical Diversity Stats: {lexical_diversity_stats}")
            print(f"Perplexity: {perplexity}")

    except Exception as e:
        print(f"[bold red]Error:[/bold red] {str(e)}")
        # print(e.with_traceback(tback=True))
        return


if __name__ == "__main__":
    main()
