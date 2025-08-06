from evaluate import load
import time
import torch


class Perplexity:
    """Class to evaluate the perplexity of a given text."""

    def __init__(self, text: str = "", model_id: str = "Qwen/Qwen2.5-0.5B"):
        self.text = text
        self.model_id = model_id

    def calculate(self) -> dict:
        """
        Evaluate the perplexity of a given text using a language model.
        Perplexity is a measure of how "surprised" or "confused" a language model is by the text it processes.

        Low Perplexity -> AI
        High Perplexity -> HUman
        Args:
            text (str): The input text to evaluate.
            model_id (str): The model ID to use for perplexity evaluation.
        Returns:
            dict: A dictionary containing the perplexity and model ID.
        """

        try:
            perplexity_metric = load("perplexity", module_type="metric")

            # Use GPU if available
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            results = perplexity_metric.compute(
                predictions=[self.text],
                model_id=self.model_id,
                device=device
            )
            perplexity = results['perplexities'][0]
            return {
                "perplexity": perplexity,
                "model_id": self.model_id
            }

        except Exception as e:
            print(f"[bold red]Error:[/bold red] {str(e)}")
            # ToDo: Fallback to manual calculation if evaluate library is not available
