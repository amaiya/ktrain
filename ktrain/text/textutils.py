from ..imports import *
from subprocess import Popen, PIPE, DEVNULL


DEFAULT_TOKEN_PATTERN = (r"\b[a-zA-Z][a-zA-Z0-9]*(?:[_/&-][a-zA-Z0-9]+)+\b|"
                         r"\b\d*[a-zA-Z][a-zA-Z0-9][a-zA-Z0-9]+\b")



def extract_copy(corpus_path, output_path):
    """
    Crawl <corpus_path>, extract or read plain text from application/pdf
    and text/plain files and then copy them to output_path.
    Args:
        corpus_path(str):  root folder containing documents
        output_path(str):  root folder of output directory
    Returns:
        list: list of skipped filenames
    """
    skipped = set()
    num_skipped = 0
    corpus_path = os.path.normpath(corpus_path)
    output_path = os.path.normpath(output_path)
    for idx, filename in enumerate(extract_filenames(corpus_path)):
        if idx %1000 == 0: print('processed %s doc(s)' % (idx+1))
        mtype = get_mimetype(filename)
        if mtype == 'application/pdf':
            text = pdftotext(filename)
            text = text.strip()
        elif mtype and mtype.split('/')[0] == 'text':
            with open(filename, 'r') as f:
                text = f.read()
                text = str.encode(text)
        else:
            num_skipped += 1
            if not mtype:
                mtype =  os.path.splitext(filename)[1]
                if not mtype: mtype == 'unknown'
            skipped.add(mtype)
            continue
        if not text: 
            num_skipped += 1
            continue
        fpath, fname = os.path.split(filename)
        if mtype == 'application/pdf': fname = fname+'.txt'
        relfpath = fpath.replace(corpus_path, '')
        relfpath = relfpath[1:] if relfpath and relfpath[0] == os.sep else relfpath
        opath = os.path.join(output_path, relfpath)
        if not os.path.exists(opath):
            os.makedirs(opath)
        ofilename = os.path.join(opath, fname)
        with open(ofilename, 'wb') as f:
            f.write(text)
    print('processed %s docs' % (idx+1))
    print('done.')
    print('skipped %s docs' % (num_skipped))
    if skipped: print('%s' %(skipped))


def get_mimetype(filepath):
    return mimetypes.guess_type(filepath)[0]

def is_txt(filepath):
    return mimetypes.guess_type(filepath)[0] == 'text/plain'

def is_pdf(filepath):
    return mimetypes.guess_type(filepath)[0] == 'application/pdf'



def pdftotext(filename):
    """
    Use pdftotext program to convert PDF to text string.
    :param filename: of PDF file
    :return: text from file, or empty string if failure
    """
    output = Popen(['pdftotext', '-q', filename, '-'],
                   stdout=PIPE).communicate()[0]
    # None may indicate damage, but convert for consistency
    return '' if output is None else output



def requires_ocr(filename):
    """
    Uses pdffonts program to determine if the PDF requires OCR, i.e., it
    doesn't contain any fonts.
    :param filename: of PDF file
    :return: True if requires OCR, False if not
    """
    output = Popen(['pdffonts', filename], stdout=PIPE,
                   stderr=DEVNULL).communicate()[0]
    return len(output.split('\n')) < 4


def extract_filenames(corpus_path, follow_links=False):
    if os.listdir(corpus_path) == []:
        raise ValueError("%s: path is empty" % corpus_path)
    walk = os.walk
    for root, dirs, filenames in walk(corpus_path, followlinks=follow_links):
        for filename in filenames:
            try:
                yield os.path.join(root, filename)
            except:
                continue


def strip_control_characters(data):
    if data:
        # unicode invalid characters
        re_xml_illegal = (
            '([\u0000-\u0008\u000b-\u000c\u000e-\u001f\ufffe-\uffff])|'
            '([%s-%s][^%s-%s])|([^%s-%s][%s-%s])|([%s-%s]$)|(^[%s-%s])'
            % (chr(0xd800), chr(0xdbff), chr(0xdc00), chr(0xdfff), chr(0xd800),
               chr(0xdbff), chr(0xdc00), chr(0xdfff), chr(0xd800), chr(0xdbff),
               chr(0xdc00), chr(0xdfff))
        )
        data = re.sub(re_xml_illegal, "", data)
        # ascii control characters
        #data = re.sub(r"[\x01-\x1F\x7F]", "", data)
        # See:  http://w3.org/International/questions/qa-forms-utf-8.html
        # Printable utf-8 does not include any of these chars below x7F
        data = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", data)
    return data



def to_ascii(data):
    """Transform accentuated unicode symbols into ascii or nothing

    Warning: this solution is only suited for languages that have a direct
    transliteration to ASCII symbols.

    A better solution would be to use transliteration based on a precomputed
    unidecode map to be used by translate as explained here:

        http://stackoverflow.com/questions/2854230/

    """
    import unicodedata
    if isinstance(data, bytes):
        data = data.decode()
    nkfd_form = unicodedata.normalize('NFKD', data)
    only_ascii = nkfd_form.encode('ASCII', 'ignore')

    # Return a string
    return only_ascii.decode('ascii')



def load_text_files(corpus_path, truncate_len=None, 
                    clean=True, return_fnames=False):
    """
    load text files
    """
    
    texts = []
    filenames = []
    mb = master_bar(range(1))
    for i in mb:
        for filename in progress_bar(list(extract_filenames(corpus_path)), parent=mb):
            with open(filename, 'r') as f:
                text = f.read()
            if clean:
                text = strip_control_characters(text)
                text = to_ascii(text)
            if truncate_len is not None:
                text = " ".join(text.split()[:truncate_len])
            texts.append(text)
            filenames.append(filename)
        mb.write('done.')
    if return_fnames:
        return (texts, filenames)
    else:
        return texts


def filter_by_id(lst, ids=[]):
    """
    filter list by supplied IDs
    """
    return [x for i,x in enumerate(lst) if i in ids]
