from arc.config import all_configs
from arc.external.architects import (
    preprocess_model_tokenizer_formatter,
    get_and_fix_peft_weights,
    fix_dtypes,
)
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from loguru import logger
from unsloth import FastLanguageModel
import peft
import numpy as np

fine_tuning_config = next(
    config for config in all_configs if config.name == "architects"
)


@dataclass
class SolutionCandidate:
    """Data class to store candidate responses and their probabilities."""

    text: str
    probability: float
    token_probabilities: List[float]


class SolutionGenerator:
    def __init__(self, peft_checkpoint_path: str, device: str = "cuda"):
        """Initialize the inference module.
        Args:
            checkpoint_path: Path to the model checkpoint
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = (
            device if torch.cuda.is_available() and device == "cuda" else "cpu"
        )
        self.model = None
        self.tokenizer = None
        self.peft_checkpoint_path = peft_checkpoint_path

    def load_model(self) -> None:
        """Load the fine-tuned model from checkpoint."""
        try:
            logger.info(f"Loading model from {self.peft_checkpoint_path}")

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=fine_tuning_config.model_config.model,
                dtype=fine_tuning_config.model_config.model_dtype,
                load_in_4bit=fine_tuning_config.model_config.load_in_4bit,
            )

            model, tokenizer, formatter = preprocess_model_tokenizer_formatter(
                model, tokenizer
            )

            model = peft.PeftModel.from_pretrained(
                model=model,
                model_id=self.peft_checkpoint_path,
                device_map="cuda",
            )

            weight_set_result = peft.set_peft_model_state_dict(
                model,
                get_and_fix_peft_weights(self.peft_checkpoint_path),
            )
            assert (
                not weight_set_result.unexpected_keys
            ), "error loading weights - some keys not available in model"

            self.model = fix_dtypes(model.merge_and_unload())
            self.tokenizer = tokenizer
            self.formatter = formatter

            FastLanguageModel.for_inference(self.model)

            logger.info("Model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def generate_full_response(
        self, prompt: str, max_new_tokens: int = 10000
    ) -> Tuple[str, float]:
        """Generate a complete response for the given prompt with its probability.

        Args:
            prompt: Input prompt text
            max_length: Maximum length of the generated response

        Returns:
            Tuple of (generated_text, probability)
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer not initialized")

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate with logits
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    output_scores=True,
                    return_dict_in_generate=True,
                )

            # Get generated text
            input_length = inputs["input_ids"].shape[1]
            response = self.tokenizer.decode(
                outputs.sequences[0][input_length:], skip_special_tokens=True
            )

            # Calculate sequence probability
            sequence_probs = []
            for logits in outputs.scores:
                probs = F.softmax(logits, dim=-1)
                # Get probability of selected token
                token_prob = probs[0, torch.argmax(probs[0])].item()
                sequence_probs.append(token_prob)

            total_probability = np.exp(np.sum(np.log(sequence_probs)))

            return response, total_probability

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def get_next_token_distribution(self, prompt: str) -> Dict[str, float]:
        """Get distribution of probabilities for all possible next tokens.

        Args:
            prompt: Input prompt text

        Returns:
            Dictionary mapping token text to probability
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer not initialized")

        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Get model output
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)

            # Convert to dictionary of token: probability
            token_prob_dict = {}
            for idx, value in enumerate(probs[0]):
                token_prob_dict[self.tokenizer.decode([idx])] = value.item()

            return token_prob_dict

        except Exception as e:
            logger.error(f"Error getting token distribution: {str(e)}")
            raise


# def get_candidate_responses(
#         self,
#         prompt: str,
#         probability_threshold: float = 0.1,
#         max_depth: int = 10,
#         max_candidates: int = 100,
#         format_checker: callable = None
#     ) -> List[CandidateResponse]:
#         """Generate candidate responses using depth-first search with probability threshold.

#         Args:
#             prompt: Input prompt text
#             probability_threshold: Minimum cumulative probability threshold
#             max_depth: Maximum depth for token generation
#             max_candidates: Maximum number of candidates to return
#             format_checker: Optional function to validate response format

#         Returns:
#             List of CandidateResponse objects
#         """
#         def dfs(
#             current_text: str,
#             current_prob: float,
#             token_probs: List[float],
#             depth: int,
#             candidates: List[CandidateResponse]
#         ) -> None:
#             if depth >= max_depth or len(candidates) >= max_candidates:
#                 return

#             # Get next token distribution
#             next_tokens = self.get_next_token_distribution(current_text)

#             for token, prob in next_tokens.items():
#                 new_prob = current_prob * prob
#                 if new_prob < probability_threshold:
#                     continue

#                 new_text = current_text + token
#                 new_token_probs = token_probs + [prob]

#                 # Check if sequence ends with EOS token
#                 if token == self.tokenizer.eos_token:
#                     # If format checker provided, validate the response
#                     if format_checker is None or format_checker(new_text):
#                         candidates.append(CandidateResponse(
#                             text=new_text,
#                             probability=new_prob,
#                             token_probabilities=new_token_probs
#                         ))
#                 else:
#                     # Continue DFS
#                     dfs(new_text, new_prob, new_token_probs, depth + 1, candidates)

#         try:
#             candidates = []
#             dfs(prompt, 1.0, [], 0, candidates)

#             # Sort by probability
#             candidates.sort(key=lambda x: x.probability, reverse=True)

#             return candidates[:max_candidates]

#         except Exception as e:
#             logger.error(f"Error generating candidate responses: {str(e)}")
#             raise

# # Example usage
# if __name__ == "__main__":
#     # Initialize inference module
#     inference = ModelInference("checkpoint-30000")

#     # Load model and tokenizer
#     inference.load_model()
#     inference.load_tokenizer()

#     # Example format checker
#     def example_format_checker(text: str) -> bool:
#         # Implement your format validation logic here
#         return True

#     # Generate response
#     prompt = "Your prompt here"
#     response, prob = inference.generate_response(prompt)
#     print(f"Generated response: {response}")
#     print(f"Response probability: {prob}")

#     # Get next token distribution
#     next_tokens = inference.get_next_token_distribution(prompt)
#     print(f"Next token distribution: {next_tokens}")

#     # Get candidate responses
#     candidates = inference.get_candidate_responses(
#         prompt,
#         format_checker=example_format_checker
#     )
#     print(f"Found {len(candidates)} candidate responses")
