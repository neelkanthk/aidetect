from evaluate import load
import time
import torch


def evaluate_perplexity(text):
    """
    Evaluate the perplexity of a given text using a language model.
    Args:
        text (str): The input text to evaluate.
    float: The estimated perplexity of the input text.
    """
    start = time.time()
    try:
        perplexity_metric = load("perplexity", module_type="metric")

        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        results = perplexity_metric.compute(
            predictions=[text],
            model_id="Qwen/Qwen2.5-0.5B",
            device=device
        )
        end = time.time()
        print(f"Total runtime of the program is {end - start} seconds")
        return results['perplexities'][0]

    except Exception as e:
        print(f"[bold red]Error:[/bold red] {str(e)}")
        # ToDo: Fallback to manual calculation if evaluate library is not available
