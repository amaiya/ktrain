from transformers import pipeline, GenerationConfig
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
        max_new_tokens: int = 512,
        do_sample: bool = True,
        **kwargs
    ):
        """
        ```
        Interface to GenerativeAI models using the transformers library.
        Extra kwargs are supplied directly to the generate method of the model.

        Args:
          model_name(str): name of the model.  Currently, only the nlpcloud/instruct-gpt-j-fp16
          device(str): device to use ("cpu" for CPU, "cuda" for GPU, "cuda:0" for first GPU, "cuda:1" for second GPU ,etc.):
          max_new_tokens(int):  The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
          do_sample(bool):  If True, use sampling instead of the default greedy decoding.
        ```
        """

        super().__init__(device=device)
        self.device_id = self.device_to_id()
        self.config = GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=do_sample, **kwargs
        )
        if self.device_id < 0:
            self.generator = pipeline(
                model=model_name, device=self.device_id, generation_config=self.config
            )
        else:
            self.generator = pipeline(
                model=model_name,
                torch_dtype=torch.float16,
                device=self.device_id,
                generation_config=self.config,
            )
        self.generator.model.generation_config.pad_token_id = (
            self.generator.model.generation_config.eos_token_id
        )

    def execute(self, prompt: str):
        """
        ```
        Issue a prompt to the model.  The default model is an instruction-fine-tuned model based on GPT-J.
        This means that you should always construct your prompt in the form of an instruction.


        Example:

        model = GenerativeAI()
        prompt = "Tell me whether the following sentence is positive,  negative, or neutral in sentiment.\\nThe reactivity of  your team has been amazing, thanks!\\n"
        model.prompt(prompt)


        Args:
          prompt(str): prompt to supply to model
        Returns:
          str: generated text
        ```
        """
        prompt = prompt.strip() + "\n"
        result = self.generator(prompt)
        result = result[0]["generated_text"]
        if result.startswith(prompt):
            result = result.replace(prompt, "")
        if not result:
            warnings.warn(
                "No output was generated. The model is sensitive to where you please newlines. Try adding or removing embedded newlines in the prompt. Follow the example notebook for tips."
            )

        return result.replace("\\n", "\n")
