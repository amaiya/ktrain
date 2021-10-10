from ..imports import *
from . import textutils as TU

class TextExtractor:
    """
    ```
    Text Extractor: a wrapper to textract package   
    ```
    """
    def __init__(self): pass

    def extract(self, filename, return_type='document', lang=None):
        """
        ```
        Extracts text from supplied filename
        filename(str): path to file
        return_type(str): One of {'document', 'paragraphs', 'sentences'}
                          'document': returns text of document
                          'paragraphs': returns a list of paragraphs from document
                          'sentences': returns a list of sentences from document
        lang(str): language code. If None, lang will be detected from extracted text
        ```
        """
        try:
            import textract
        except ImportError:
            raise Exception('TextExtractor requires textract: pip install textract')
        mtype = TU.get_mimetype(filename)
        try:
            if mtype and mtype.split('/')[0] == 'text':
                with open(filename, 'r') as f:
                    text = f.read()
                    text = str.encode(text)
            else:
                text = textract.process(filename)
        except Exception as e:
            if verbose:
                print('ERROR on %s:\n%s' % (filename, e))
        if return_type == 'sentences':
            return TU.sent_tokenize(text, lang=lang)
        elif return_type == 'paragraphs':
            return TU.paragraph_tokenize(text, join_sentences=True, lang=lang)
        else:
            return text

