import warnings


class TorchBase:
    """
    Utility methods for working pretrained Torch models
    """

    def __init__(self, device, quantize=False, min_transformers_version=None):
        if min_transformers_version is not None:
            import transformers
            from packaging import version

            if version.parse(transformers.__version__) < version.parse(
                min_transformers_version
            ):
                raise Exception(
                    f"This feature requires transformers>={min_transformers_version}. "
                    + "It is usually safe for you to manually upgrade transformers even if ktrain installed a lower version."
                )
        try:
            import torch
        except (ImportError, OSError):
            raise Exception(
                "This capability in ktrain requires PyTorch to be installed. Please install for your environment: "
                + "https://pytorch.org/get-started/locally/"
            )
        self.quantize = quantize
        self.torch_device = device
        if self.torch_device is None:
            self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    def quantize_model(self, model):
        """
        quantize a model
        """
        import torch

        if self.torch_device == "cpu":
            return torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        elif self.torch_device != "cpu":
            return model.half()

    def device_to_id(self, device_str=None):
        device_str = self.torch_device if device_str is None else device_str
        if device_str.lower() == "cpu":
            return -1
        elif device_str.lower() == "cuda":
            return 0
        elif device_str.lower().startswith("cuda:"):
            _, device_id = device_str.split(":")[1]
            device_id = int(device_id)
            return device_id
        else:
            warnings.warn("Could not determine device ID - defaulting to -1")
            return -1
