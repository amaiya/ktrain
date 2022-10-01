from .. import utils as U
from ..imports import *
from . import textutils as TU


try:
    import textract

    TEXTRACT_INSTALLED = True
except ImportError:
    TEXTRACT_INSTALLED = False

JAVA_INSTALLED = U.checkjava()


class TextExtractor:
    """
    ```
    Text Extractor: a wrapper to textract package
    ```
    """

    def __init__(self, use_tika=True):
        if use_tika:
            try:
                from tika import parser
            except ImportError as e:
                raise ValueError(
                    "If use_tika=True, then TextExtractor requires tika: pip install tika"
                )
            except PermissionError as e:
                raise PermissionError(
                    f"There may already be a /tmp/tika.log file from another user - please delete it or change permissions: {e}"
                )
        if not use_tika and not TEXTRACT_INSTALLED:
            raise ValueError(
                "If use_tika=False, then TextExtractor requires textract: pip install textract"
            )
        self.use_tika = use_tika

    def extract(
        self, filename=None, text=None, return_format="document", lang=None, verbose=1
    ):
        """
        ```
        Extracts text from document given file path to document.
        filename(str): path to file,  Mutually-exclusive with text.
        text(str): string to tokenize.  Mutually-exclusive with filename.
                   The extract method can also simply accept a string and return lists of sentences or paragraphs.
        return_format(str): One of {'document', 'paragraphs', 'sentences'}
                          'document': returns text of document
                          'paragraphs': returns a list of paragraphs from document
                          'sentences': returns a list of sentences from document
        lang(str): language code. If None, lang will be detected from extracted text
        verbose(bool): verbosity
        ```
        """
        if filename is None and text is None:
            raise ValueError(
                "Either the filename parameter or the text parameter must be supplied"
            )
        if filename is not None and text is not None:
            raise ValueError("The filename and text parameters are mutually-exclusive.")
        if return_format not in ["document", "paragraphs", "sentences"]:
            raise ValueError(
                'return_format must be one of {"document", "paragraphs", "sentences"}'
            )
        if filename is not None:
            mtype = TU.get_mimetype(filename)
            try:
                if mtype and mtype.split("/")[0] == "text":
                    with open(filename, "r") as f:
                        text = f.read()
                        text = str.encode(text)
                else:
                    text = self._extract(filename)
            except Exception as e:
                if verbose:
                    print("ERROR on %s:\n%s" % (filename, e))
        try:
            text = text.decode(errors="ignore")
        except:
            pass
        if return_format == "sentences":
            return TU.sent_tokenize(text, lang=lang)
        elif return_format == "paragraphs":
            return TU.paragraph_tokenize(text, join_sentences=True, lang=lang)
        else:
            return text

    def _extract(self, filename):
        if self.use_tika:
            from tika import parser

            if JAVA_INSTALLED:
                parsed = parser.from_file(filename)
                text = parsed["content"]
            else:
                raise Exception("Please install Java for TIKA text extraction")
        else:
            text = textract.process(filename)
        return text.strip()
