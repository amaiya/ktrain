from ... import imports as I
from ...torch_base import TorchBase


class ImageCaptioner(TorchBase):
    """
    interface to Image Captioner
    """

    def __init__(self, model_name="ydshieh/vit-gpt2-coco-en", device=None):
        """
        ```
        ImageCaptioner constructor

        Args:
          model_name(str): name of  image captioning model
          device(str): device to use (e.g., 'cuda', 'cpu')
        ```
        """
        if not I.PIL_INSTALLED:
            raise Exception(
                "PIL is not installed. Please install with: pip install pillow>=9.0.1"
            )

        super().__init__(
            device=device, quantize=False, min_transformers_version="4.12.3"
        )
        self.model_name = model_name
        from transformers import (
            AutoTokenizer,
            VisionEncoderDecoderModel,
            ViTFeatureExtractor,
        )

        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name).to(
            self.torch_device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.extractor = ViTFeatureExtractor.from_pretrained(self.model_name)

    def caption(self, images):
        """
        ```
        Performs image captioning

        This method supports a single image or a list of images. If the input is an image, the return
        type is a string. If text is a list, a list of strings is returned
        Args:
            images: image|list
        Returns:
            list of captions
        ```
        """
        # Convert single element to list
        values = [images] if not isinstance(images, list) else images

        # Open images if file strings
        values = [
            I.Image.open(image) if isinstance(image, str) else image for image in values
        ]

        # Feature extraction
        pixels = self.extractor(images=values, return_tensors="pt").pixel_values
        pixels = pixels.to(self.torch_device)

        # Run model
        import torch

        with torch.no_grad():
            outputs = self.model.generate(
                pixels, max_length=16, num_beams=4, return_dict_in_generate=True
            ).sequences

        # Tokenize outputs into text results
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        captions = [caption.strip() for caption in captions]

        # Return single element if single element passed in
        return captions[0] if not isinstance(images, list) else captions
