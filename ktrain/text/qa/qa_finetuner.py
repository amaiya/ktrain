import warnings

import pandas as pd
import tensorflow as tf


def convert_to_dataset(list_of_dicts):
    try:
        from datasets import Dataset, load_dataset
    except ImportError:
        raise ImportError(
            "The datasets package is required for fine-tuning QA models: pip install datasets"
        )

    new_list = []
    for d in list_of_dicts:
        if "question" not in d or "context" not in d or "answers" not in d:
            raise ValueError(
                'All dictionaries in list must have the following keys: "question", "context", "answers"'
            )
        q = d["question"].strip()
        c = d["context"].strip()
        a = d["answers"].strip() if isinstance(d["answers"], str) else d["answers"]
        if a is None:
            answers = {"text": [], "answer_start": []}
        elif isinstance(a, dict):
            ans_start = a["answer_start"]
            ans_text = a["text"]
            answers = {"text": ans_text, "answer_start": ans_start}
        elif isinstance(a, str):
            ans_start = c.index(a)
            if ans_start < 0:
                raise ValueError("Could not locate answer in context: %s" % (d))
            ans_start = [ans_start]
            answers = {"text": [a], "answer_start": ans_start}
        else:
            raise ValueError(
                'Value for "answers" key must be a string, dictionary, or None'
            )
        new_d = {"question": q, "context": c, "answers": answers}
        new_list.append(new_d)
    dataset = Dataset.from_pandas(pd.DataFrame(new_list))

    # uncomment to test with squad sample
    # datasets = load_dataset('squad')
    # td = datasets['train']
    # td = Dataset.from_pandas(td.to_pandas().head(100))
    # return td
    return dataset


def convert_dataset_for_tensorflow(
    dataset,
    batch_size,
    dataset_mode="variable_batch",
    shuffle=True,
    drop_remainder=True,
):
    """Converts a Hugging Face dataset to a Tensorflow Dataset. The dataset_mode controls whether we pad all batches
    to the maximum sequence length, or whether we only pad to the maximum length within that batch. The former
    is most useful when training on TPU, as a new graph compilation is required for each sequence length.
    """
    try:
        from datasets import Dataset, load_dataset
    except ImportError:
        raise ImportError(
            "The datasets package is required for fine-tuning QA models: pip install datasets"
        )

    def densify_ragged_batch(features, label=None):
        features = {
            feature: (
                ragged_tensor.to_tensor(shape=batch_shape[feature])
                if feature in tensor_keys
                else ragged_tensor
            )
            for feature, ragged_tensor in features.items()
        }
        if label is None:
            return features
        else:
            return features, label

    tensor_keys = ["attention_mask", "input_ids"]
    label_keys = ["start_positions", "end_positions"]
    if dataset_mode == "variable_batch":
        batch_shape = {key: None for key in tensor_keys}
        data = {key: tf.ragged.constant(dataset[key]) for key in tensor_keys}
    elif dataset_mode == "constant_batch":
        data = {key: tf.ragged.constant(dataset[key]) for key in tensor_keys}
        batch_shape = {
            key: tf.concat(([batch_size], ragged_tensor.bounding_shape()[1:]), axis=0)
            for key, ragged_tensor in data.items()
        }
    else:
        raise ValueError("Unknown dataset mode!")

    if all([key in dataset.features for key in label_keys]):
        for key in label_keys:
            data[key] = tf.convert_to_tensor(dataset[key])
        dummy_labels = tf.zeros_like(dataset[key])
        tf_dataset = tf.data.Dataset.from_tensor_slices((data, dummy_labels))
    else:
        tf_dataset = tf.data.Dataset.from_tensor_slices(data)
    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(dataset))
    tf_dataset = tf_dataset.batch(
        batch_size=batch_size, drop_remainder=drop_remainder
    ).map(densify_ragged_batch)
    return tf_dataset


class QAFineTuner:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def finetune(
        self, data, epochs=3, learning_rate=2e-5, batch_size=8, max_seq_length=512
    ):
        """
        ```
        Finetune a QA model.

        Args:
          data(list): list of dictionaries of the form:
                      [{'question': 'What is ktrain?'
                       'context': 'ktrain is a low-code library for augmented machine learning.'
                       'answer': 'ktrain'}]
          epochs(int): number of epochs.  Default:3
          learning_rate(float): learning rate.  Default: 2e-5
        Returns:
          fine-tuned model (i.e., QAFineTuner.model)
        ```
        """
        if self.model.name_or_path.startswith("bert-large"):
            warnings.warn(
                "You are fine-tuning a bert-large-* model, which requires lots of memory.  "
                + "If you get a memory error, switch to distilbert-base-cased-distilled-squad."
            )
        if batch_size > len(data):
            batch_size = len(data)

        model = self.model
        tokenizer = self.tokenizer
        question_column_name = "question"
        context_column_name = "context"
        answer_column_name = "answers"

        # Padding side determines if we do (question|context) or (context|question).
        pad_on_right = tokenizer.padding_side == "right"

        # Training preprocessing
        def prepare_train_features(examples):
            # Some of the questions have lots of whitespace on the left, which is not useful and will make the
            # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
            # left whitespace
            examples[question_column_name] = [
                q.lstrip() for q in examples[question_column_name]
            ]

            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=128,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                # padding="max_length" if data_args.pad_to_max_length else False,
                padding="max_length",
            )

            # Since one example might give us several features if it has a long context, we need a map from a feature to
            # its corresponding example. This key gives us just that.
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # The offset mappings will give us a map from token to character position in the original context. This will
            # help us compute the start_positions and end_positions.
            offset_mapping = tokenized_examples.pop("offset_mapping")

            # Let's label those examples!
            tokenized_examples["start_positions"] = []
            tokenized_examples["end_positions"] = []

            for i, offsets in enumerate(offset_mapping):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]
                cls_index = input_ids.index(tokenizer.cls_token_id)

                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)

                # One example can give several spans, this is the index of the example containing this span of text.
                sample_index = sample_mapping[i]
                answers = examples[answer_column_name][sample_index]
                # If no answers are given, set the cls_index as answer.
                if len(answers["answer_start"]) == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])

                    # Start token index of the current span in the text.
                    token_start_index = 0
                    while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                        token_start_index += 1

                    # End token index of the current span in the text.
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                        token_end_index -= 1

                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if not (
                        offsets[token_start_index][0] <= start_char
                        and offsets[token_end_index][1] >= end_char
                    ):
                        tokenized_examples["start_positions"].append(cls_index)
                        tokenized_examples["end_positions"].append(cls_index)
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while (
                            token_start_index < len(offsets)
                            and offsets[token_start_index][0] <= start_char
                        ):
                            token_start_index += 1
                        tokenized_examples["start_positions"].append(
                            token_start_index - 1
                        )
                        while offsets[token_end_index][1] >= end_char:
                            token_end_index -= 1
                        tokenized_examples["end_positions"].append(token_end_index + 1)

            return tokenized_examples

        dataset = convert_to_dataset(data)
        print(dataset)
        print(dataset[0])

        train_dataset = dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=1,
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        def dummy_loss(y_true, y_pred):
            return tf.reduce_mean(y_pred)

        losses = {"loss": dummy_loss}
        model.compile(optimizer=optimizer, loss=losses)

        training_dataset = convert_dataset_for_tensorflow(
            train_dataset,
            batch_size=batch_size,
            dataset_mode="variable_batch",
            drop_remainder=True,
            shuffle=True,
        )
        model.fit(training_dataset, epochs=int(epochs))
