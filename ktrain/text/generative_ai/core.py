from transformers import pipeline
import torch
from ...torch_base import TorchBase
from typing import Optional
import warnings


class GenerativeAI(TorchBase):
    """
    interface to Transformer-based generative AI models like GPT*
    """

    def __init__(
        self,
        model_name: str = "nlpcloud/instruct-gpt-j-fp16",
        device: Optional[str] = None,
    ):
        """
        ```
        interface to GenerativeAI models using the transformers library

        Args:
          model_name(str): name of the model.  Currently, only the nlpcloud/instruct-gpt-j-fp16
          device(str): device to use ("cpu" for CPU, "cuda" for GPU, "cuda:0" for first GPU, "cuda:1" for second GPU ,etc.):
        ```
        """

        super().__init__(device=device)
        self.device_id = self.device_to_id()
        if self.device_id < 0:
            self.generator = pipeline(model=model_name, device=self.device_id)
        else:
            self.generator = pipeline(
                model=model_name, torch_dtype=torch.float16, device=self.device_id
            )

    def execute(self, prompt: str, max_new_tokens: int = 512, **kwargs):
        """
        ```
        Issue a prompt to the model.  The default model is an instruction-fine-tuned model based on GPT-J.
        This means that you should always construct your prompt in the form of an instruction.
        In addition to max_new_tokens, additonal parmeters can be supplied that will be fed directly to the model.
        Examples include min_new_tokens and max_time.

        Example:

        model = GenerativeAI()
        prompt = "Tell me whether the following sentence is positive,  negative, or neutral in sentiment.\\nThe reactivity of  your team has been amazing, thanks!\\n"
        model.prompt(prompt)


        Args:
          prompt(str): prompt to supply to model
          max_new_tokens(int):  The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        Returns:
          str: generated text
        ```
        """
        prompt = prompt.strip() + "\n"
        result = self.generator(prompt, max_new_tokens=512, **kwargs)
        result = result[0]["generated_text"]
        if result.startswith(prompt):
            result = result.replace(prompt, "")
        if not result:
            warnings.warn(
                "No output was generated. The model is sensitive to where you please newlines. Try adding or removing embedded newlines in the prompt. Follow the example notebook for tips."
            )

        return result.replace("\\n", "\n")
