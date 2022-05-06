"""Metrics to assess performance on sequence labeling task given prediction
Functions named as ``*_score`` return a scalar value to maximize: the higher
the better
Reference: seqeval==0.0.19
"""

from __future__ import absolute_import, division, print_function

import warnings
from collections import defaultdict

import numpy as np


def get_entities(seq, suffix=False):
    """Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """

    def _validate_chunk(chunk, suffix):
        if chunk in ["O", "B", "I", "E", "S"]:
            return

        if suffix:
            if not (
                chunk.endswith("-B")
                or chunk.endswith("-I")
                or chunk.endswith("-E")
                or chunk.endswith("-S")
            ):
                warnings.warn("{} seems not to be NE tag.".format(chunk))

        else:
            if not (
                chunk.startswith("B-")
                or chunk.startswith("I-")
                or chunk.startswith("E-")
                or chunk.startswith("S-")
            ):
                warnings.warn("{} seems not to be NE tag.".format(chunk))

    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ["O"]]

    prev_tag = "O"
    prev_type = ""
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ["O"]):
        _validate_chunk(chunk, suffix)

        if suffix:
            tag = chunk[-1]
            type_ = chunk[:-1].rsplit("-", maxsplit=1)[0] or "_"
        else:
            tag = chunk[0]
            type_ = chunk[1:].split("-", maxsplit=1)[-1] or "_"

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i - 1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == "E":
        chunk_end = True
    if prev_tag == "S":
        chunk_end = True

    if prev_tag == "B" and tag == "B":
        chunk_end = True
    if prev_tag == "B" and tag == "S":
        chunk_end = True
    if prev_tag == "B" and tag == "O":
        chunk_end = True
    if prev_tag == "I" and tag == "B":
        chunk_end = True
    if prev_tag == "I" and tag == "S":
        chunk_end = True
    if prev_tag == "I" and tag == "O":
        chunk_end = True

    if prev_tag != "O" and prev_tag != "." and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == "B":
        chunk_start = True
    if tag == "S":
        chunk_start = True

    if prev_tag == "E" and tag == "E":
        chunk_start = True
    if prev_tag == "E" and tag == "I":
        chunk_start = True
    if prev_tag == "S" and tag == "E":
        chunk_start = True
    if prev_tag == "S" and tag == "I":
        chunk_start = True
    if prev_tag == "O" and tag == "E":
        chunk_start = True
    if prev_tag == "O" and tag == "I":
        chunk_start = True

    if tag != "O" and tag != "." and prev_type != type_:
        chunk_start = True

    return chunk_start


def f1_score(y_true, y_pred, average="micro", suffix=False):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def precision_score(y_true, y_pred, average="micro", suffix=False):
    """Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import precision_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, average="micro", suffix=False):
    """Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import recall_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> recall_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


def performance_measure(y_true, y_pred):
    """
    Compute the performance metrics: TP, FP, FN, TN

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        performance_dict : dict

    Example:
        >>> from seqeval.metrics import performance_measure
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'O', 'B-ORG'], ['B-PER', 'I-PER', 'O', 'B-PER']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'O'], ['B-PER', 'I-PER', 'O', 'B-MISC']]
        >>> performance_measure(y_true, y_pred)
        {'TP': 3, 'FP': 3, 'FN': 1, 'TN': 4}
    """
    performance_dict = dict()
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]
    performance_dict["TP"] = sum(
        y_t == y_p for y_t, y_p in zip(y_true, y_pred) if ((y_t != "O") or (y_p != "O"))
    )
    performance_dict["FP"] = sum(
        ((y_t != y_p) and (y_p != "O")) for y_t, y_p in zip(y_true, y_pred)
    )
    performance_dict["FN"] = sum(
        ((y_t != "O") and (y_p == "O")) for y_t, y_p in zip(y_true, y_pred)
    )
    performance_dict["TN"] = sum(
        (y_t == y_p == "O") for y_t, y_p in zip(y_true, y_pred)
    )

    return performance_dict


def classification_report(y_true, y_pred, digits=2, suffix=False, output_dict=False):
    """Build a text report showing the main classification metrics.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a classifier.
        digits : int. Number of digits for formatting output floating point values.
        output_dict : bool(default=False). If True, return output as dict else str.

    Returns:
        report : string/dict. Summary of the precision, recall, F1 score for each class.

    Examples:
        >>> from seqeval.metrics import classification_report
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> print(classification_report(y_true, y_pred))
                     precision    recall  f1-score   support
        <BLANKLINE>
               MISC       0.00      0.00      0.00         1
                PER       1.00      1.00      1.00         1
        <BLANKLINE>
          micro avg       0.50      0.50      0.50         2
          macro avg       0.50      0.50      0.50         2
       weighted avg       0.50      0.50      0.50         2
        <BLANKLINE>
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    name_width = 0
    d1 = defaultdict(set)
    d2 = defaultdict(set)
    for e in true_entities:
        d1[e[0]].add((e[1], e[2]))
        name_width = max(name_width, len(e[0]))
    for e in pred_entities:
        d2[e[0]].add((e[1], e[2]))

    avg_types = ["micro avg", "macro avg", "weighted avg"]

    if output_dict:
        report_dict = dict()
    else:
        avg_width = max([len(x) for x in avg_types])
        width = max(name_width, avg_width, digits)
        headers = ["precision", "recall", "f1-score", "support"]
        head_fmt = "{:>{width}s} " + " {:>9}" * len(headers)
        report = head_fmt.format("", *headers, width=width)
        report += "\n\n"

        row_fmt = "{:>{width}s} " + " {:>9.{digits}f}" * 3 + " {:>9}\n"

    ps, rs, f1s, s = [], [], [], []
    for type_name in sorted(d1.keys()):
        true_entities = d1[type_name]
        pred_entities = d2[type_name]
        nb_correct = len(true_entities & pred_entities)
        nb_pred = len(pred_entities)
        nb_true = len(true_entities)

        p = nb_correct / nb_pred if nb_pred > 0 else 0
        r = nb_correct / nb_true if nb_true > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0

        if output_dict:
            report_dict[type_name] = {
                "precision": p,
                "recall": r,
                "f1-score": f1,
                "support": nb_true,
            }
        else:
            report += row_fmt.format(
                *[type_name, p, r, f1, nb_true], width=width, digits=digits
            )

        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)

    if not output_dict:
        report += "\n"

    # compute averages
    nb_true = np.sum(s)

    for avg_type in avg_types:
        if avg_type == "micro avg":
            # micro average
            p = precision_score(y_true, y_pred, suffix=suffix)
            r = recall_score(y_true, y_pred, suffix=suffix)
            f1 = f1_score(y_true, y_pred, suffix=suffix)
        elif avg_type == "macro avg":
            # macro average
            p = np.average(ps)
            r = np.average(rs)
            f1 = np.average(f1s)
        elif avg_type == "weighted avg":
            # weighted average
            p = np.average(ps, weights=s)
            r = np.average(rs, weights=s)
            f1 = np.average(f1s, weights=s)
        else:
            assert False, "unexpected average: {}".format(avg_type)

        if output_dict:
            report_dict[avg_type] = {
                "precision": p,
                "recall": r,
                "f1-score": f1,
                "support": nb_true,
            }
        else:
            report += row_fmt.format(
                *[avg_type, p, r, f1, nb_true], width=width, digits=digits
            )

    if output_dict:
        return report_dict
    else:
        return report
