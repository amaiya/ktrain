from transformers import pipeline

from ... import imports as I
from ...torch_base import TorchBase


class ObjectDetector(TorchBase):
    """
    interface to Image Captioner
    """

    def __init__(self, device=None, classification=False, threshold=0.9):
        """
        ```
        Object detection constructor

        Args:
          device(str): device to use (e.g., 'cuda', 'cpu')
          threshold(float):  threshold for object detection
          classification(bool): If True, simpy do image classification
        ```
        """
        if not I.PIL_INSTALLED:
            raise Exception(
                "PIL is not installed. Please install with: pip install pillow>=9.0.1"
            )

        super().__init__(
            device=device, quantize=False, min_transformers_version="4.12.3"
        )
        self.pipeline = pipeline(
            "image-classification" if classification else "object-detection",
            device=self.device_to_id(),
        )
        self.threshold = threshold
        self.classification = classification

    def detect(self, images, flatten=False, workers=0):
        """
        ```
        Performs object detection

        This method supports a single image or a list of images. If the input is an image, the return
        type is a string. If text is a list, a list of strings is returned
        Args:
            images: image|list
            flatten: flatten output to a list of objects
            workers: number of concurrent workers to use for processing data, defaults to None
        Returns:
            list of (label, score)

        ```
        """
        # Convert single element to list
        values = [images] if not isinstance(images, list) else images

        # Open images if file strings
        values = [
            I.Image.open(image) if isinstance(image, str) else image for image in values
        ]

        # Run pipeline
        results = (
            self.pipeline(values, num_workers=workers)
            if self.classification
            else self.pipeline(values, threshold=self.threshold, num_workers=workers)
        )

        # Build list of (id, score)
        outputs = []
        for result in results:
            # Convert to (label, score) tuples
            result = [
                (x["label"], x["score"]) for x in result if x["score"] > self.threshold
            ]

            # Sort by score descending
            result = sorted(result, key=lambda x: x[1], reverse=True)

            # Deduplicate labels
            unique = set()
            elements = []
            for label, score in result:
                if label not in unique:
                    elements.append(label if flatten else (label, score))
                    unique.add(label)

            outputs.append(elements)

        # Return single element if single element passed in
        return outputs[0] if not isinstance(images, list) else outputs
