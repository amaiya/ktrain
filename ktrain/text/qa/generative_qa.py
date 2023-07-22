import sys
import os
import pickle
from typing import Optional
from pathlib import Path

try:
    from paperqa import Docs

    PAPERQA_INSTALLED = True
except ImportError:
    PAPERQA_INSTALLED = False

DOCS = "docs_obj.pkl"


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if "google.colab" in sys.modules:
            return True
        elif shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


class GenerativeQA:
    """
    Question-answering using OpenAI or open-source GPT or GPT-like generative LLM models
    """

    def __init__(self, llm=None):
        """
        ```
        GenerativeQA constructor

        Args:
          llm(str):  The LLM to use.  If None, gpt-3.5-turbo is used.
        ```
        """
        if not PAPERQA_INSTALLED:
            raise Exception(
                "GenerativeQA in ktrain requires the paper-qa package by Andrew White: pip install paper-qa==2.1.1"
            )
        self.docs = Docs(llm)
        if is_notebook():
            import nest_asyncio

            nest_asyncio.apply()

    def load(self, path: str):
        """
        ```
        load previously-saved document vector database from folder specified by path

        Args:
          path(str): folder path
        ```
        """
        with open(os.path.join(path, DOCS), "rb") as f:
            self.docs = pickle.load(f)

    def save(self, path: str):
        """
        ```
        Save current document vector database to folder represented by path
        Save the current vector database to disk

        Args:
          path(str): folder path
        ```
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.docs.index_path = Path(path)
        with open(os.path.join(path, DOCS), "wb") as f:
            pickle.dump(self.docs, f)

    def clear_index(self):
        """
        This will delete the entire index.
        """
        if input("are you sure you want to delete the vector index? (y/n)") != "y":
            print("ok - aborting")
            return
        index_path = self.docs.index_path.as_posix()
        self.docs.clear()
        self.save(index_path)

    def add_doc(
        self,
        path: Optional[str] = None,
        text: Optional[str] = None,
        citation: Optional[str] = None,
        key: Optional[str] = None,
        disable_check: bool = True,
        chunk_chars: Optional[int] = 3000,
    ):
        """
        ```
        Add documents to the data store

        Args:
          path(str): Path to the document.  Mutually-exclusive with text parameter.
          text(str): text of document. Mutually-exclusive with path parameter.
          citation(str):  The citation for document that will appear in references below answer.
                          If omitted, the LLM will be used to infer the correct citation from the document text.
          key(str): The key for the document that will appear within the body of the answer when referenced.
                    If omitted, the LLM will be used to infer the correct citaiton from the document text.
          disable_check(bool): A check of the text of the document.
          chunk_chars(int): This is how many characters documents are split into.

        Returns:
          None
        ```
        """
        if (path is not None and text is not None) or (path is None and text is None):
            raise ValueError(
                "The path and text parameters are mutually-exclusive and exactly one must be supplied."
            )
        if (
            path is not None
            and not path.lower().endswith(".pdf")
            and not path.lower().endswith(".txt")
        ):
            raise ValueError(
                "Currently, the path parameter only accepts files that end with either a .pdf or .txt extension."
            )

        if text is not None:
            import os
            import tempfile

            fd, fpath = tempfile.mkstemp()
            os.rename(fpath, fpath + ".txt")
            fpath = fpath + ".txt"
            try:
                with os.fdopen(fd, "w") as tmp:
                    # do stuff with temp file
                    tmp.write(text)
                key, citation = self.default_key_and_citation(
                    fpath, key=key, citation=citation
                )
                self.add_doc(
                    fpath,
                    citation=citation,
                    key=key,
                    disable_check=disable_check,
                    chunk_chars=chunk_chars,
                )
            finally:
                pass
            return
        key, citation = self.default_key_and_citation(path, key=key, citation=citation)
        self.docs.add(
            path=path,
            citation=citation,
            key=key,
            disable_check=disable_check,
            chunk_chars=chunk_chars,
        )
        return

    def query(
        self,
        query: str,
        k: int = 10,
        max_sources: int = 5,
        length_prompt: str = "about 100 words",
        marginal_relevance: bool = True,
        answer=None,
        key_filter: Optional[bool] = None,
        show_token_usage=False,
        # get_callbacks: Callable[[str], AsyncCallbackHandler] = lambda x: [],
    ):
        """
        ```
        Query for cited answers
        ```
        """
        try:
            result = self.docs.query(
                query=query,
                k=k,
                max_sources=max_sources,
                length_prompt=length_prompt,
                marginal_relevance=marginal_relevance,
                answer=answer,
                key_filter=key_filter,
            )
            if not show_token_usage:
                result.formatted_answer = result.formatted_answer.split("Tokens Used")[
                    0
                ]
            return result
        except RuntimeError:
            raise Exception(
                "There was a RuntimeError - try addding  the following to the top of your notebook:\nimport nest_asyncio\nnest_asyncio.apply()"
            )

    def default_key_and_citation(
        self, path: str, key: Optional[str] = None, citation: Optional[str] = None
    ):
        """
        ```
        Get default key and citation
        ```
        """
        if path.endswith(".pdf"):
            return (key, citation)
        default_key = self.compute_key(path)
        if key is None:
            key = default_key
        if citation is None:
            citation = f"Document {default_key}"
        return (key, citation)

    def compute_key(self, path: str):
        """
        ```
        compute MD5 hash
        ```
        """
        from paperqa.utils import md5sum

        return f"md5:{md5sum(path)}"
