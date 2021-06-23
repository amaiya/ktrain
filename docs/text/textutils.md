Module ktrain.text.textutils
============================

Functions
---------

    
`decode_by_line(texts, encoding='utf-8', verbose=1)`
:   Decode text line by line and skip over errors.

    
`detect_encoding(texts, sample_size=32)`
:   

    
`detect_lang(texts, sample_size=32)`
:   detect language

    
`extract_copy(corpus_path, output_path, verbose=0)`
:   Crawl <corpus_path>, extract plain text from documents
    and then copy them to output_path.
    Requires textract package
    Args:
        corpus_path(str):  root folder containing documents
        output_path(str):  root folder of output directory
        verbose(bool):  Default:0.  Set to 1 (or True) to see error details on why each skipped document was skipped.
    Returns:
        list: list of skipped filenames

    
`extract_filenames(corpus_path, follow_links=False)`
:   

    
`extract_noun_phrases(text)`
:   extracts noun phrases

    
`filter_by_id(lst, ids=[])`
:   filter list by supplied IDs

    
`get_mimetype(filepath)`
:   

    
`is_chinese(lang, strict=True)`
:   Args:
      lang(str): language code (e.g., en)
      strict(bool):  If False, include additional languages due to mistakes on short texts by langdetect

    
`is_nospace_lang(lang)`
:   

    
`is_pdf(filepath)`
:   

    
`is_txt(filepath, strict=False)`
:   

    
`load_text_files(corpus_path, truncate_len=None, clean=True, return_fnames=False)`
:   load text files

    
`paragraph_tokenize(text, join_sentences=False, lang=None)`
:   segment text into sentences

    
`pdftotext(filename)`
:   Use pdftotext program to convert PDF to text string.
    :param filename: of PDF file
    :return: text from file, or empty string if failure

    
`read_text(filename)`
:   

    
`requires_ocr(filename)`
:   Uses pdffonts program to determine if the PDF requires OCR, i.e., it
    doesn't contain any fonts.
    :param filename: of PDF file
    :return: True if requires OCR, False if not

    
`sent_tokenize(text, lang=None)`
:   segment text into sentences

    
`split_chinese(texts)`
:   

    
`strip_control_characters(data)`
:   

    
`to_ascii(data)`
:   Transform accentuated unicode symbols into ascii or nothing
    
    Warning: this solution is only suited for languages that have a direct
    transliteration to ASCII symbols.
    
    A better solution would be to use transliteration based on a precomputed
    unidecode map to be used by translate as explained here:
    
        http://stackoverflow.com/questions/2854230/

    
`tokenize(s, join_tokens=False, join_char=' ')`
: