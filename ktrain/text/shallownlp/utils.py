#
# The ShallowNLP is kept self-contained for now.
# Thus, some or all of the functions here are copied from
# ktrain.text.textutils

from .imports import *


def extract_filenames(corpus_path, follow_links=False):
    if os.listdir(corpus_path) == []:
        raise ValueError("%s: path is empty" % corpus_path)
    for root, _, fnames in os.walk(corpus_path, followlinks=follow_links):
        for filename in fnames:
            try:
                yield os.path.join(root, filename)
            except Exception:
                continue


def detect_lang(texts, sample_size=32):
    """
    detect language
    """
    if not LANGDETECT:
        raise ValueError('langdetect is missing - install with pip3 install langdetect')

    if isinstance(texts, str): texts = [texts]
    if not isinstance(texts, (list, np.ndarray)):
        raise ValueError('texts must be a list or NumPy array of strings')
    lst = []
    for doc in texts[:sample_size]:
        try:
            lst.append(langdetect.detect(doc))
        except:
            continue
    if len(lst) == 0:
        raise Exception('could not detect language in random sample of %s docs.'  % (sample_size))
    return max(set(lst), key=lst.count)


def is_chinese(lang):
    """
    include additional languages due to mistakes on short texts by langdetect
    """
    return lang is not None and lang.startswith('zh-') or lang in ['ja', 'ko']



def split_chinese(texts):
    if not JIEBA:
        raise ValueError('jieba is missing - install with pip3 install jieba')
    if isinstance(texts, str): texts=[texts]

    split_texts = []
    for doc in texts:
        seg_list = jieba.cut(doc, cut_all=False)
        seg_list = list(seg_list)
        split_texts.append(seg_list)
    return [" ".join(tokens) for tokens in split_texts]


def decode_by_line(texts, encoding='utf-8', verbose=1):
    """
    Decode text line by line and skip over errors.
    """
    if isinstance(texts, str): texts = [texts]
    new_texts = []
    skips=0
    num_lines = 0
    for doc in texts:
        text = ""
        for line in doc.splitlines():
            num_lines +=1
            try:
                line = line.decode(encoding)
            except:
                skips +=1
                continue
            text += line
        new_texts.append(text)
    pct = round((skips*1./num_lines) * 100, 1)
    if verbose:
        print('skipped %s lines (%s%%) due to character decoding errors' % (skips, pct))
        if pct > 10:
            print('If this is too many, try a different encoding')
    return new_texts


def detect_encoding(texts, sample_size=32):
    if not CHARDET:
        raise ValueError('cchardet is missing - install with pip3 install cchardet')
    if isinstance(texts, str): texts = [texts]
    lst = [chardet.detect(doc)['encoding'] for doc in texts[:sample_size]]
    encoding = max(set(lst), key=lst.count)
    encoding = 'utf-8' if encoding.lower() in ['ascii', 'utf8', 'utf-8'] else encoding
    return encoding


def read_text(filename):
    with open(filename, 'rb') as f:
        text = f.read()
    encoding = detect_encoding([text])
    try:
        decoded_text = text.decode(encoding) 
    except:
        U.vprint('Decoding with %s failed 1st attempt - using %s with skips' % (encoding,
                                                                                encoding),
                                                                                verbose=verbose)
        decoded_text = decode_by_line(text, encoding=encoding)
    return decoded_text.strip()


def sent_tokenize(text):
    """
    segment text into sentences
    """
    lang = detect_lang(text)
    sents = []
    if is_chinese(lang):
        for sent in re.findall(u'[^!?。\.\!\?]+[!?。\.\!\?]?', text, flags=re.U):
            sents.append(sent)
    else:
        for paragraph in segmenter.process(text):
            for sentence in paragraph:
                sents.append(" ".join([t.value for t in sentence]))
    return sents


