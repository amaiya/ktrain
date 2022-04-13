def causal_inference_model(
    df,
    method="t-learner",
    metalearner_type=None,
    treatment_col="treatment",
    outcome_col="outcome",
    text_col=None,
    ignore_cols=[],
    include_cols=[],
    treatment_effect_col="treatment_effect",
    learner=None,
    effect_learner=None,
    min_df=0.05,
    max_df=0.5,
    ngram_range=(1, 1),
    stop_words="english",
    verbose=1,
):
    """
    ```
    Infers causality from the data contained in `df` using a metalearner.
    This function is a wrapper to the CausalNLP.CausalInferenceModel class.
    For more details on methods and capabilities of the returned `CausalInferenceModel` object,
    see the [CausalNLP documentation](https://amaiya.github.io/causalnlp/causalinference.html).

    Usage:
    >>> cm = causal_inference_model(df,
                                    treatment_col='Is_Male?',
                                    outcome_col='Post_Shared?', text_col='Post_Text',
                                    ignore_cols=['id', 'email'])
        cm.fit()

    **Parameters:**
    * **df** : pandas.DataFrame containing dataset
    * **method** : metalearner model to use. One of {'t-learner', 's-learner', 'x-learner', 'r-learner'} (Default: 't-learner')
    * **metalearner_type** : Alias of **method** parameter for backwards compatibility.  If not None, overrides method.
    * **treatment_col** : treatment variable; column should contain binary values: 1 for treated, 0 for untreated.
    * **outcome_col** : outcome variable; column should contain the categorical or numeric outcome values
    * **text_col** : (optional) text column containing the strings (e.g., articles, reviews, emails).
    * **ignore_cols** : columns to ignore in the analysis
    * **include_cols** : columns to include as covariates (e.g., possible confounders)
    * **treatment_effect_col** : name of column to hold causal effect estimations.  Does not need to exist.  Created by CausalNLP.
    * **learner** : an instance of a custom learner.  If None, a default LightGBM will be used.
        # Example
         learner = LGBMClassifier(num_leaves=1000)
    * **effect_learner**: used for x-learner/r-learner and must be regression model
    * **min_df** : min_df parameter used for text processing using sklearn
    * **max_df** : max_df parameter used for text procesing using sklearn
    * **ngram_range**: ngrams used for text vectorization. default: (1,1)
    * **stop_words** : stop words used for text processing (from sklearn)
    * **verbose** : If 1, print informational messages.  If 0, suppress.

    **Returns:**
    `CausalNLP.CausalInferenceModel` object
    ```
    """
    try:
        import causalnlp
    except ImportError:
        raise Exception("CausalNLP must be installed: pip install causalnlp")
    from causalnlp import CausalInferenceModel

    return CausalInferenceModel(
        df,
        method=method,
        metalearner_type=metalearner_type,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        text_col=text_col,
        ignore_cols=ignore_cols,
        include_cols=include_cols,
        treatment_effect_col=treatment_effect_col,
        learner=learner,
        effect_learner=effect_learner,
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        stop_words=stop_words,
        verbose=verbose,
    )
