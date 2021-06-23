Module ktrain.text.summarization.core
=====================================

Classes
-------

`TransformerSummarizer(model_name='facebook/bart-large-cnn', device=None)`
:   interface to Transformer-based text summarization
    
    interface to BART-based text summarization using transformers library
    
    Args:
      model_name(str): name of BART model for summarization
      device(str): device to use (e.g., 'cuda', 'cpu')

    ### Methods

    `summarize(self, doc)`
    :   summarize document text
        Args:
          doc(str): text of document
        Returns:
          str: summary text