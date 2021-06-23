Module ktrain.text.ner.anago.tagger
===================================
Model API.

Classes
-------

`Tagger(model, preprocessor, tokenizer=<method 'split' of 'str' objects>)`
:   A model API that tags input sentence.
    
    Attributes:
        model: Model.
        preprocessor: Transformer. Preprocessing data for feature extraction.
        tokenizer: Tokenize input sentence. Default tokenizer is `str.split`.

    ### Methods

    `analyze(self, text)`
    :   Analyze text and return pretty format.
        
        Args:
            text: string, the input text.
        
        Returns:
            res: dict.
        
        Examples:
            >>> text = 'President Obama is speaking at the White House.'
            >>> model.analyze(text)
            {
                "words": [
                    "President",
                    "Obama",
                    "is",
                    "speaking",
                    "at",
                    "the",
                    "White",
                    "House."
                ],
                "entities": [
                    {
                        "beginOffset": 1,
                        "endOffset": 2,
                        "score": 1,
                        "text": "Obama",
                        "type": "PER"
                    },
                    {
                        "beginOffset": 6,
                        "endOffset": 8,
                        "score": 1,
                        "text": "White House.",
                        "type": "ORG"
                    }
                ]
            }

    `predict(self, text)`
    :   Predict using the model.
        
        Args:
            text: string, the input text.
        
        Returns:
            tags: list, shape = (num_words,)
            Returns predicted values.

    `predict_proba(self, text)`
    :   Probability estimates.
        
        The returned estimates for all classes are ordered by the
        label of classes.
        
        Args:
            text : string, the input text.
        
        Returns:
            y : array-like, shape = [num_words, num_classes]
            Returns the probability of the word for each class in the model,