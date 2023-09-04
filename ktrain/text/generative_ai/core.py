# As of v0.38.x, generative_ai is powered by our onprem.LLM package
try:
    from onprem import LLM
except ImportError:
    raise ImportError(
        "The generative_ai module requires the onprem package.  Please install it with: pip install onprem"
    )
import langchain

langchain.llm_cache = None
