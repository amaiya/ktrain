from . import utils as U
from .imports import *


def search(query, doc, case_sensitive=False, keys=[], progress=False):
    searcher = Searcher(query)
    return searcher.search(
        doc, case_sensitive=case_sensitive, keys=keys, progress=progress
    )


class Searcher:
    """
    Search for keywords in text documents
    """

    def __init__(self, queries, lang=None):
        """
        ```
        Args:
          queries(list of str): list of chinese text queries
          lang(str): language of queries.  default:None --> auto-detected
        ```
        """
        self.queries = queries
        if isinstance(self.queries, str):
            self.queries = [self.queries]
        self.lang = lang
        if self.lang is None:
            self.lang = U.detect_lang(queries)
        # print("lang:%s" %(self.lang))

    def search(self, docs, case_sensitive=False, keys=[], min_matches=1, progress=True):
        """
        ```
        executes self.queries on supplied list of documents
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
        ```
        """
        if isinstance(docs, str):
            docs = [docs]
        if keys and len(keys) != len(docs):
            raise ValueError("lengths of keys and docs must be the same")
        results = []
        l = len(docs)
        for idx, text in enumerate(docs):
            for q in self.queries:
                if U.is_chinese(self.lang):
                    r = self._search_chinese(
                        q, [text], min_matches=min_matches, parse=1, progress=False
                    )
                elif self.lang == "ar":
                    r = self._search(
                        q,
                        [text],
                        case_sensitive=case_sensitive,
                        min_matches=min_matches,
                        progress=False,
                        substrings_on=True,
                    )
                else:
                    r = self._search(
                        q,
                        [text],
                        case_sensitive=case_sensitive,
                        min_matches=min_matches,
                        progress=False,
                        substrings_on=False,
                    )
                if not r:
                    continue
                r = r[0]
                k = idx
                if keys:
                    k = keys[idx]
                num_matches = len(set(r[2])) if U.is_chinese(self.lang) else len(r[2])
                results.append((k, q, num_matches))
            if progress:
                printProgressBar(
                    idx + 1, l, prefix="progress: ", suffix="complete", length=50
                )
        return results

    def _search(
        self,
        query,
        docs,
        case_sensitive=False,
        substrings_on=False,
        min_matches=1,
        progress=True,
    ):
        """
        ```
        search documents for query string.
        Args:
            query(str or list):  the word or phrase to search (or list of them)
                                 if list is provided, each element is combined using OR
            docs (list of str): list of text documents
            case_sensitive(bool):  If True, case sensitive search
            substrings_on(bool): whether to use "\b" in regex. default:True
                                 If True, will find substrings
        returns:
            list or tuple:  Returns list of results if len(docs) > 1.  Otherwise, returns tuple of results
        ```
        """
        if not isinstance(query, (list, tuple, str)):
            raise ValueError("query must be str or list of str")
        if isinstance(query, str):
            query = [query]
        if not isinstance(docs, (list, np.ndarray)):
            raise ValueError("docs must be list of str")

        flag = 0
        if not case_sensitive:
            flag = re.I
        qlist = []
        for q in query:
            qlist.append("\s+".join(q.split()))
        original_query = query
        query = "|".join(qlist)
        bound = r"\b"
        if substrings_on:
            bound = ""
        pattern_str = r"%s(?:%s)%s" % (bound, query, bound)
        pattern = re.compile(pattern_str, flag)

        results = []
        l = len(docs)
        for idx, text in enumerate(docs):
            matches = pattern.findall(text)
            if matches and len(matches) >= min_matches:
                results.append((idx, text, matches))
            if progress:
                printProgressBar(
                    idx + 1, l, prefix="progress: ", suffix="complete", length=50
                )
        return results

    def _search_chinese(
        self, query, docs, substrings_on=True, parse=1, min_matches=1, progress=False
    ):
        """
        convenience method to search chinese text
        """
        original_query = query
        if not isinstance(query, str):
            raise ValueError("query must be str")
        if parse > 0:
            q = U.split_chinese(query)[0]
            num_words = len(q.split())
            query = build_ngrams(q, n=parse)
            query = ["".join(q) for q in query]
        return self._search(query, docs, substrings_on=substrings_on, progress=progress)


# ------------------------------------------------------------------------------
# Non-English Language-Handling
# ------------------------------------------------------------------------------
def find_chinese(s):
    return re.findall(r"[\u4e00-\u9fff]+", s)


def find_arabic(s):
    return re.findall(r"[\u0600-\u06FF]+", s)


def find_cyrillic(s):
    return re.findall(r"[\u0400-\u04FF]+", s)


def find_cyrillic2(s):
    return re.findall(r"[а-яА-Я]+", s)


def find_russian(s):
    return find_cyrillic(s)


def find_times(s):
    return re.findall(r"\d{2}:\d{2}(?:am|pm)", s, re.I)


def build_ngrams(s, n=2):
    lst = s.split()
    ngrams = []
    for i in range(len(lst) - (n - 1)):
        ngram = []
        for j in range(n):
            ngram.append(lst[i + j])
        ngram = tuple(ngram)
        ngrams.append(ngram)
    return ngrams


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=50,
    fill="█",
    printEnd="\r",
):
    """
    ```
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    ```
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
