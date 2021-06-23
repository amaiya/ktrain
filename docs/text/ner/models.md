Module ktrain.text.ner.models
=============================

Functions
---------

    
`print_sequence_taggers()`
:   

    
`sequence_tagger(name, preproc, wv_path_or_url=None, bert_model='bert-base-multilingual-cased', bert_layers_to_use=[-2], word_embedding_dim=100, char_embedding_dim=25, word_lstm_size=100, char_lstm_size=25, fc_dim=100, dropout=0.5, verbose=1)`
:   Build and return a sequence tagger (i.e., named entity recognizer).
    
    Args:
        name (string): one of:
                      - 'bilstm-crf' for Bidirectional LSTM-CRF model
                      - 'bilstm' for Bidirectional LSTM (no CRF layer)
        preproc(NERPreprocessor):  an instance of NERPreprocessor
        wv_path_or_url(str): either a URL or file path toa fasttext word vector file (.vec or .vec.zip or .vec.gz)
                             Example valid values for wv_path_or_url:
    
                               Randomly-initialized word embeeddings:
                                 set wv_path_or_url=None
                               English pretrained word vectors:
                                 https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
                               Chinese pretrained word vectors:
                                 https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz
                               Russian pretrained word vectors:
                                 https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.vec.gz
                               Dutch pretrained word vectors:
                                 https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.nl.300.vec.gz
    
    
                             See these two Web pages for a full list of URLs to word vector files for 
                             different languages:
                                1.  https://fasttext.cc/docs/en/english-vectors.html (for English)
                                2.  https://fasttext.cc/docs/en/crawl-vectors.html (for non-English langages)
    
                            Default:None (randomly-initialized word embeddings are used)
    
        bert_model_name(str):  the name of the BERT model.  default: 'bert-base-multilingual-cased'
                               This parameter is only used if bilstm-bert is selected for name parameter.
                               The value of this parameter is a name of BERT model from here:
                                        https://huggingface.co/transformers/pretrained_models.html
                               or a community-uploaded BERT model from here:
                                        https://huggingface.co/models
                               Example values:
                                 bert-base-multilingual-cased:  Multilingual BERT (157 languages) - this is the default
                                 bert-base-cased:  English BERT
                                 bert-base-chinese: Chinese BERT
                                 distilbert-base-german-cased: German DistilBert
                                 albert-base-v2: English ALBERT model
                                 monologg/biobert_v1.1_pubmed: community uploaded BioBERT (pretrained on PubMed)
    
        bert_layers_to_use(list): indices of hidden layers to use.  default:[-2] # second-to-last layer
                                  To use the concatenation of last 4 layers: use [-1, -2, -3, -4]
        word_embedding_dim (int): word embedding dimensions.
        char_embedding_dim (int): character embedding dimensions.
        word_lstm_size (int): character LSTM feature extractor output dimensions.
        char_lstm_size (int): word tagger LSTM output dimensions.
        fc_dim (int): output fully-connected layer size.
        dropout (float): dropout rate.
    
        verbose (boolean): verbosity of output
    Return:
        model (Model): A Keras Model instance