import numpy as np
import re
from lexicalrichness import LexicalRichness


class LexicalDiversity:
    def __init__(self, cleaned_text: str = ""):
        self.cleaned_text = cleaned_text

    def calculate_lexical_diversity_metrics(self, words: list) -> dict:
        """
        Calculates the lexical diversity metrics of a given text based on the diversity of words used.
        Lexical diversity is a measure of how many unique words are used in a text compared to the total number of words.
        It helps analyze the richness and variety of vocabulary in text.

        Args:
            words (list): List of words in the text

        Returns:
            dict: Dictionary containing different lexical diversity metrics
        """
        text = " ".join(words)  # Convert to text

        lex = LexicalRichness(text)

        return {
            "basic_ttr": lex.ttr,  # Type-Token Ratio
            "root_ttr": lex.rttr,  # Root Type-Token Ratio
            "maas_index": lex.Maas,  # Maas's D
            "yule_k": float(lex.yulek)  # Yule's K
        }

    def extract_sentences(self, text: str) -> list:
        """
        Splits the input text into sentences using punctuation marks (., !, ?).
        Returns a list of cleaned sentences.
        """
        sentences = re.split(r'[.!?]', text)

        # Create a new list of valid sentences without punctuation
        cleaned_sentences = []
        for sentence in sentences:
            cleaned = sentence.strip()
            if cleaned:
                # Append the cleaned sentence to the list
                cleaned_sentences.append(cleaned)

        # Return the list of cleaned sentences
        return cleaned_sentences

    def calculate_avg_sentence_length(self, sentences: list) -> float:
        """
        Calculate the average number of words per sentence.
        """
        if not sentences:
            return 0.0

        sentence_lengths = []
        for sentence in sentences:
            words = sentence.split()  # extract words from the sentence
            words_count = len(words)  # count the number of words
            sentence_lengths.append(words_count)  # store word count in each sentence

        # Calculate the average length of sentences
        total_length = sum(sentence_lengths)
        num_sentences = len(sentence_lengths)
        average_length = total_length / num_sentences

        return average_length

    def calculate(self) -> dict:
        """
            Analyzes lexical diversity properties of text
            to help detect AI-generated content.
        """

        sentences = self.extract_sentences(self.cleaned_text)

        avg_sentence_len = self.calculate_avg_sentence_length(sentences)

        words = self.cleaned_text.split()
        # Calculate lexical diversity
        lexical_diversity = self.calculate_lexical_diversity_metrics(words)
        return {
            "total_words": len(words),
            "unique_words": len(set(words)),
            "avg_sentence_length": avg_sentence_len,
            "lexical_diversity": lexical_diversity
        }
