try:
    import librosa as librosa

    LIBROSA = True
except (ImportError, OSError):
    LIBROSA = False

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from ...torch_base import TorchBase


# duplicated from ktrain.utils for now
def batchify(X, size):
    """
    ```
    Splits X into separate batch sizes specified by size.
    Args:
        X(list): elements
        size(int): batch size
    Returns:
        list of evenly sized batches with the last batch having the remaining elements
    ```
    """
    return [X[x : x + size] for x in range(0, len(X), size)]


class Transcriber(TorchBase):
    """
    Transcriber: speech to text pipeline
    """

    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        """
        ```
        basic wrapper speech transcription

        Args:
          model_name(str): Helsinki-NLP model
          device(str): device to use (e.g., 'cuda', 'cpu')
        ```
        """
        super().__init__(device=device, quantize=False)
        # if not SOUNDFILE:
        # raise ImportError("SoundFile library not installed or libsndfile not found: pip install soundfile")
        if not LIBROSA:
            raise ImportError(
                "librosa library must be installed: pip install librosa. Conda users may also have to install ffmpeg: conda install -c conda-forge ffmpeg"
            )

        # load model and processor
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.torch_device)
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)

    def transcribe(self, afiles, batch_size=32):
        """
        ```
        Transcribes audio files to text.
        This method supports files as a string or a list. If the input is a string,
        the return type is string. If text is a list, the return type is a list.
        Args:
            afiles (str|list):  file path to audio file or a list of them
            batch_size (int): batch size
        Returns:
            list of transcribed text
        ```
        """

        # preprocess audio
        values = [afiles] if not isinstance(afiles, list) else afiles

        # parse audio files
        # speech = [sf.read(f) for f in values]
        speech = [librosa.load(f, sr=16000) for f in values]

        # get unique list of sampling rates (since we're resampling, should be {16000})
        unique = set(s[1] for s in speech)

        results = {}
        for sampling in unique:
            # get inputs for current sampling rate
            inputs = [(x, s[0]) for x, s in enumerate(speech) if s[1] == sampling]

            # transcribe in batches
            outputs = []
            for chunk in batchify([s for _, s in inputs], batch_size):
                outputs.extend(self._transcribe(chunk, sampling))

            # Store output value
            for y, (x, _) in enumerate(inputs):
                results[x] = outputs[y].capitalize()

        # Return results in same order as input
        results = [results[x] for x in sorted(results)]
        return results[0] if isinstance(afiles, str) else results

    def _transcribe(self, speech, sampling):
        """
        Transcribes audio to text.
        Args:
            speech: list of audio
            sampling: sampling rate
        Returns:
            list of transcribed text
        """
        import torch

        with torch.no_grad():
            # Convert samples to features
            inputs = self.processor(
                speech, sampling_rate=sampling, padding=True, return_tensors="pt"
            ).input_values
            # Place inputs on tensor device
            inputs = inputs.to(self.torch_device)
            # Retrieve logits
            logits = self.model(inputs).logits
            # Decode argmax
            ids = torch.argmax(logits, dim=-1)
            return self.processor.batch_decode(ids)
