from collections import Counter
import warnings

from ... import imports as I
from .. import textutils as TU
try:
    import textblob
    TEXTBLOB_INSTALLED = True
except ImportError:
    TEXTBLOB_INSTALLED = False

SUPPORTED_LANGS = {'en' : 'english',
                   'ar' : 'arabic',
                   'az' : 'azerbaijani',
                   'da' : 'danish',
                   'nl' : 'dutch',
                   'fi' : 'finnish',
                   'fr' : 'french',
                   'de' : 'german',
                   'el' : 'greek',
                   'hu' : 'hungarian',
                   'id' : 'indonesian',
                   'it' : 'italian',
                   'kk' : 'kazakh',
                   'ne' : 'nepali',
                   'no' : 'norwegian',
                   'pt' : 'portuguese',
                   'ro' : 'romanian',
                   'ru' : 'russian',
                   'sl' : 'slovene',
                   'es' : 'spanish',
                   'sv' : 'swedish',
                   'tg' : 'tajik',
                   'tr' : 'turkish',
                   'zh' : 'chinese'}

class KeywordExtractor:
    """
    Keyphrase Extraction
    """

    def __init__(self, lang='en', custom_stopwords = ['et al', 'et', 'al', "n't", 'did', 'does']):
        """
        ```
        Keyphrase Extraction

        Args:
          lang(str):  2-character language code: 
          custom_stopwords(list): list of custom stopwords to ignore
        ```
        """
        # error checks
        if not TEXTBLOB_INSTALLED:
            raise Exception('The textblob package is required for keyphrase extraction: pip install textblob; python -m textblob.download_corpora')
        if lang not in SUPPORTED_LANGS:
            raise ValueError(f'lang="{lang}" is not supported. Supported 2-character ISO 639-1 language codes are: {SUPPORTED_LANGS}')
        self.lang = lang

        # build blacklist
        from nltk.corpus import stopwords as nltk_stopwords
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        if lang == 'en':
            stopwords = list(ENGLISH_STOP_WORDS)+custom_stopwords
        elif lang == 'zh':
            stopwords = TU.chinese_stopwords()+custom_stopwords
        elif lang in SUPPORTED_LANGS:
            stopwords = nltk_stopwords.words(SUPPORTED_LANGS[lang])
        else:
            stopwords = []
        blacklist = stopwords+custom_stopwords
        self.blacklist = blacklist


    def extract_keywords(self, text, ngram_range=(1,3), top_n=10, n_candidates=50, omit_scores=False,
                         candidate_generator='ngrams', constrain_unigram_case=True, maxlen=64):
        """
        simple keyword extraction
        
        This is a simplified TextBlob implementation of the KERA algorithm from:
          https://arxiv.org/pdf/1308.2359.pdf
        Args:
          text (str): the text as unicode string
          ngram_range(tuple): the ngram range.  Example: (1,3) considers unigrams, bigrams, and trigrams as candidates
          top_n(int): number of keyphrases to return
          n_candidates(int): number of candidates considered, when ranking
          omit_scores(bool):  If True, no scores are returned.
          candidate_generator(str):  Either 'noun_phrases' or 'ngrams'.
                                     The default 'ngrams' method will be faster.
          contrain_unigram_case(bool): Only applies if candidate_generator=='ngrams'.
                                       If True, only unigrams in uppercase are returned (e.g., LDA, SVM, NASA).
                                       True is recommended.
          maxlen(int): maximum number of characters in keyphrase. Default:64

          Returns:
            list
                                        
        """
        if candidate_generator not in ['noun_phrases', 'ngrams']: 
            raise ValueError('candidate_generator must be one of {"noun_phrases", "ngrams"}')
        if self.lang == 'zh':
            text = ' '.join(I.jieba.cut(text, HMM=False))
        if candidate_generator == 'noun_phrases' and self.lang != 'en':
            warnings.warn(f'lang={self.lang} but candidate_generator="noun_phrases" is not supported. '+\
                          'Falling back to candidate_generator="ngrams"')
            candidate_generator = 'ngrams'

        blob = textblob.TextBlob(text)
        candidates = []
        min_n, max_n = ngram_range
        ngram_lens = list(range(min_n, max_n+1))

        # generate ngrams or noun phrases
        ngrams = {}
        if candidate_generator == 'ngrams':
            for n in ngram_lens:
                ngrams[n] = blob.ngrams(n=n)
        else:
            noun_phrases = blob.noun_phrases
            for np in noun_phrases:
                words = np.split()
                n = len(words)
                if n not in ngram_lens: continue
                lst = ngrams.get(n, [])
                lst.append(words)
                ngrams[n] = lst
                

        # generate candidates
        for n in range(min_n, max_n+1):
            if n == 1:
                grams = [k[0].lower() for k in ngrams.get(n, []) if not any(w.lower() in self.blacklist for w in k) \
                         and (not constrain_unigram_case or\
                             (candidate_generator=='ngrams' and constrain_unigram_case and k[0].isupper()))]
            else:
                grams = [" ".join(k).lower() for k in ngrams.get(n, []) if not any(w.lower() in self.blacklist for w in k) \
                          and len(set(k)) != 1 and len(k[0])>1 and len(k[1])>1]
            candidates.extend([kw for kw in grams if kw[0].isalpha() and len([w for w in kw if not w.isspace() and w not in ['-', '.']]) >= 3 and\
                               (kw[-1].isalpha() or kw[-1].isdigit()) and '@' not in kw])
        cnt = Counter(candidates)
        tups = cnt.most_common(n_candidates)

        # normalize and return
        keywords = [tup[0] for tup in tups if len(tup[0]) <= maxlen]
        scores =   [tup[1] for tup in tups if len(tup[0]) <= maxlen]
        scores = [float(i)/sum(scores) for i in scores]
        result = list(zip(keywords, scores))
        return [r[0] for r in result[:top_n]] if omit_scores else result[:top_n]
