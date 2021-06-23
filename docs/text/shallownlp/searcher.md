Module ktrain.text.shallownlp.searcher
======================================

Functions
---------

    
`build_ngrams(s, n=2)`
:   

    
`find_arabic(s)`
:   

    
`find_chinese(s)`
:   

    
`find_cyrillic(s)`
:   

    
`find_cyrillic2(s)`
:   

    
`find_russian(s)`
:   

    
`find_times(s)`
:   

    
`printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', printEnd='\r')`
:   Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "", "
    ") (Str)

    
`search(query, doc, case_sensitive=False, keys=[], progress=False)`
:   

Classes
-------

`Searcher(queries, lang=None)`
:   Search for keywords in text documents
    
    Args:
      queries(list of str): list of chinese text queries
      lang(str): language of queries.  default:None --> auto-detected

    ### Methods

    `search(self, docs, case_sensitive=False, keys=[], min_matches=1, progress=True)`
    :   executes self.queries on supplied list of documents
        Args:
          docs(list of str): list of chinese texts
          case_sensitive(bool):  If True, case sensitive search
          keys(list): list keys for supplied docs (e.g., file paths).
                      default: key is index in range(len(docs))
          min_matches(int): results must have at least these many word matches
          progress(bool): whether or not to show progress bar
        Returns:
          list of tuples of results of the form:
            (key, query, no. of matches)
          For Chinese, no. of matches will be number of unique Jieba-extracted character sequences that match