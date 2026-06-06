#!/usr/bin/env python
# coding=utf-8
# Berbasis run_classification.py (HuggingFace, Apache-2.0).
# Tambahan: auxiliary incongruity head. lambda_aux=0 => identik dengan baseline.
""" Finetuning text classification + auxiliary incongruity head."""

import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn
import datasets
import evaluate
import numpy as np
from datasets import Value, load_dataset, load_from_disk, concatenate_datasets
from sklearn.utils.class_weight import compute_class_weight

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

try:
    from transformers.utils import send_example_telemetry
except ImportError:
    def send_example_telemetry(*args, **kwargs):
        return None


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)


# ===========================================================================
# Model dua-head: head utama TAK DIUBAH (tetap AutoModelForSequenceClassification),
# aux head incongruity menempel di token CLS encoder. Saat eval aux tidak dihitung,
# sehingga jalur lambda_aux=0 identik bit-per-bit dengan baseline run_classification.
# ===========================================================================
class ModelWithAux(nn.Module):
    def __init__(self, base_model, hidden_size, num_aux=2, dropout_p=0.1):
        super().__init__()
        self.base = base_model
        self.config = base_model.config           # supaya Trainer/utility bisa baca config
        self.aux_dropout = nn.Dropout(dropout_p)
        self.aux_head = nn.Linear(hidden_size, num_aux)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                labels=None, clash=None, **kwargs):
        # labels & clash sengaja diserap di sini; loss dihitung di Trainer.
        out = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=self.training,    # hanya saat training (eval = persis baseline)
        )
        result = {"logits": out.logits}
        if self.training:
            result["cls"] = out.hidden_states[-1][:, 0]   # token CLS lapisan terakhir
        return result


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."})
    dataset_config_name: Optional[str] = field(default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."})
    do_regression: bool = field(default=None, metadata={"help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."})
    text_column_names: Optional[str] = field(default=None, metadata={"help": "The name of the text column in the input dataset or a CSV/JSON file."})
    text_column_delimiter: Optional[str] = field(default=" ", metadata={"help": "THe delimiter to use to join text columns into a single sentence."})
    train_split_name: Optional[str] = field(default=None, metadata={"help": 'The name of the train split.'})
    validation_split_name: Optional[str] = field(default=None, metadata={"help": 'The name of the validation split.'})
    test_split_name: Optional[str] = field(default=None, metadata={"help": 'The name of the test split.'})
    remove_splits: Optional[str] = field(default=None, metadata={"help": "The splits to remove from the dataset."})
    remove_columns: Optional[str] = field(default=None, metadata={"help": "The columns to remove from the dataset."})
    label_column_name: Optional[str] = field(default=None, metadata={"help": "The name of the label column."})
    max_seq_length: int = field(default=128, metadata={"help": "The maximum total input sequence length after tokenization."})
    overwrite_cache: bool = field(default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."})
    pad_to_max_length: bool = field(default=True, metadata={"help": "Whether to pad all samples to `max_seq_length`."})
    shuffle_train_dataset: bool = field(default=False, metadata={"help": "Whether to shuffle the train dataset or not."})
    shuffle_seed: int = field(default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "Truncate the number of training examples to this value if set."})
    max_eval_samples: Optional[int] = field(default=None, metadata={"help": "Truncate the number of evaluation examples to this value if set."})
    max_predict_samples: Optional[int] = field(default=None, metadata={"help": "Truncate the number of prediction examples to this value if set."})
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the training data."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the validation data."})
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    do_augment: bool = field(default=False, metadata={"help": "Whether to augment with iSarcasm dataset."})
    do_weighted_loss: bool = field(default=False, metadata={"help": "Whether to use weighted cross-entropy loss."})
    weight_multiplier: float = field(default=1.0, metadata={"help": "Weighted loss multiplier factor."})

    # --- tambahan auxiliary incongruity ---
    lambda_aux: float = field(default=0.0, metadata={"help": "Bobot auxiliary incongruity loss. 0 = baseline tanpa aux."})
    incongruity_source_column: str = field(default="content", metadata={"help": "Kolom teks MENTAH untuk hitung label clash (BUKAN teks ber-marker)."})
    inset_pos_path: Optional[str] = field(default=None, metadata={"help": "Path positive.tsv InSet (wajib jika lambda_aux>0)."})
    inset_neg_path: Optional[str] = field(default=None, metadata={"help": "Path negative.tsv InSet (wajib jika lambda_aux>0)."})

    def __post_init__(self):
        if self.dataset_name is None:
            if self.train_file is None or self.validation_file is None:
                raise ValueError(" training/validation file or a dataset name.")
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert validation_extension == train_extension, "`validation_file` should have the same extension as `train_file`."


@dataclass
class ModelArguments:
    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Where do you want to store the pretrained models"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Whether to use one of the fast tokenizer or not."})
    model_revision: str = field(default="main", metadata={"help": "The specific model version to use."})
    token: str = field(default=None, metadata={"help": "The token to use as HTTP bearer authorization for remote files."})
    use_auth_token: bool = field(default=None, metadata={"help": "Deprecated. Use `token`."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Whether to allow custom models defined on the Hub."})
    ignore_mismatched_sizes: bool = field(default=False, metadata={"help": "Enable to load a pretrained model whose head dimensions are different."})


def get_label_list(raw_dataset, split="train") -> List[str]:
    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    label_list = [str(label) for label in label_list]
    return label_list


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated. Please use `token` instead.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only `token`.")
        model_args.token = model_args.use_auth_token

    send_example_telemetry("run_classification", model_args, data_args)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    set_seed(training_args.seed)

    if data_args.dataset_name is not None:
        if os.path.isdir(data_args.dataset_name):
            raw_datasets = load_from_disk(data_args.dataset_name)
        else:
            raw_datasets = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name,
                cache_dir=model_args.cache_dir, token=model_args.token,
            )
        logger.info(f"Dataset loaded: {raw_datasets}")
        logger.info(raw_datasets)
    else:
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
        if training_args.do_predict:
            if data_args.test_file is not None:
                train_extension = data_args.train_file.split(".")[-1]
                test_extension = data_args.test_file.split(".")[-1]
                assert test_extension == train_extension, "`test_file` should have the same extension as `train_file`."
                data_files["test"] = data_args.test_file
            else:
                raise ValueError("Need either a dataset name or a test file for `do_predict`.")
        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")
        if data_args.train_file.endswith(".csv"):
            raw_datasets = load_dataset("csv", data_files=data_files, cache_dir=model_args.cache_dir, token=model_args.token)
        else:
            raw_datasets = load_dataset("json", data_files=data_files, cache_dir=model_args.cache_dir, token=model_args.token)

    if data_args.do_augment:
        isarcasm = load_dataset("w11wo/isarcasm_id")
        isarcasm = isarcasm.filter(lambda x: x["label"] == 1)
        if data_args.text_column_names is not None:
            text_column_name = data_args.text_column_names.split(",")[0]
            isarcasm = isarcasm.rename_column("text", text_column_name)
        raw_datasets = raw_datasets.remove_columns(
            set(raw_datasets["train"].features.keys()) - set([text_column_name, "label"])
        )
        raw_datasets["train"] = concatenate_datasets([raw_datasets["train"], isarcasm["train"]])

    if data_args.remove_splits is not None:
        for split in data_args.remove_splits.split(","):
            logger.info(f"removing split {split}")
            raw_datasets.pop(split)

    if data_args.train_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["train"] = raw_datasets[data_args.train_split_name]
        raw_datasets.pop(data_args.train_split_name)

    if data_args.validation_split_name is not None:
        logger.info(f"using {data_args.validation_split_name} as validation set")
        raw_datasets["validation"] = raw_datasets[data_args.validation_split_name]
        raw_datasets.pop(data_args.validation_split_name)

    if data_args.test_split_name is not None:
        logger.info(f"using {data_args.test_split_name} as test set")
        raw_datasets["test"] = raw_datasets[data_args.test_split_name]
        raw_datasets.pop(data_args.test_split_name)

    if data_args.remove_columns is not None:
        for split in raw_datasets.keys():
            for column in data_args.remove_columns.split(","):
                logger.info(f"removing column {column} from split {split}")
                raw_datasets[split].remove_columns(column)

    if data_args.label_column_name is not None and data_args.label_column_name != "label":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

    is_regression = (
        raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
        if data_args.do_regression is None else data_args.do_regression
    )

    is_multi_label = False
    if is_regression:
        label_list = None
        num_labels = 1
        for split in raw_datasets.keys():
            if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                logger.warning(f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}")
                features = raw_datasets[split].features
                features.update({"label": Value("float32")})
                try:
                    raw_datasets[split] = raw_datasets[split].cast(features)
                except TypeError as error:
                    logger.error(f"Unable to cast {split} set to float32, please check the labels.")
                    raise error
    else:
        if raw_datasets["train"].features["label"].dtype == "list":
            is_multi_label = True
            logger.info("Label type is list, doing multi-label classification")
        label_list = get_label_list(raw_datasets, split="train")
        for split in ["validation", "test"]:
            if split in raw_datasets:
                val_or_test_labels = get_label_list(raw_datasets, split=split)
                diff = set(val_or_test_labels).difference(set(label_list))
                if len(diff) > 0:
                    logger.warning(f"Labels {diff} in {split} set but not in training set, adding them to the label list")
                    label_list += list(diff)
        for label in label_list:
            if label == -1:
                logger.warning("Label -1 found in label list, removing it.")
                label_list.remove(label)
        label_list.sort()
        num_labels = len(label_list)
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="text-classification",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if is_regression:
        config.problem_type = "regression"
        logger.info("setting problem type to regression")
    elif is_multi_label:
        config.problem_type = "multi_label_classification"
        logger.info("setting problem type to multi label classification")
    else:
        config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )

    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        padding = False

    if training_args.do_train and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}
        if model.config.label2id != label_to_id:
            logger.warning("The label2id key in the model config.json is not equal to the label2id key of this run.")
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}
    elif not is_regression:
        logger.info("using label infos in the model config")
        logger.info("label2id: {}".format(model.config.label2id))
        label_to_id = model.config.label2id
    else:
        label_to_id = None

    # Bungkus dengan aux head. Head utama (model.base) tetap apa adanya, jadi
    # lambda_aux=0 identik dengan baseline.
    model = ModelWithAux(model, hidden_size=config.hidden_size,
                         dropout_p=getattr(config, "hidden_dropout_prob", 0.1))

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Muat InSet hanya jika aux dipakai (lambda_aux>0). Jadi run baseline lambda=0
    # tidak butuh InSet sama sekali.
    inset_obj, inset_maxlen = None, 1
    if data_args.lambda_aux > 0:
        if not data_args.inset_pos_path or not data_args.inset_neg_path:
            raise ValueError("--inset_pos_path dan --inset_neg_path wajib jika --lambda_aux > 0")
        from incongruity import load_inset
        inset_obj, inset_maxlen = load_inset(data_args.inset_pos_path, data_args.inset_neg_path)
        logger.info(f"InSet: {len(inset_obj)} entri, frasa terpanjang {inset_maxlen} kata")

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label_to_id)
        for label in labels:
            ids[label_to_id[label]] = 1.0
        return ids

    def preprocess_function(examples):
        if data_args.text_column_names is not None:
            text_column_names = data_args.text_column_names.split(",")
            examples["sentence"] = examples[text_column_names[0]]
            for column in text_column_names[1:]:
                for i in range(len(examples[column])):
                    examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
        result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            if is_multi_label:
                result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            else:
                result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
        # label clash dihitung dari kolom teks MENTAH (bukan teks ber-marker).
        if data_args.lambda_aux > 0 and data_args.incongruity_source_column in examples:
            from sarcasm_preprocess import to_lexicon_tokens
            from incongruity import incongruity_label
            result["clash"] = [
                incongruity_label(to_lexicon_tokens(str(t)), inset_obj, inset_maxlen)[0]
                for t in examples[data_args.incongruity_source_column]
            ]
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        # sanity: clash rate harus ~50-62%, bukan 0% atau 100%.
        if data_args.lambda_aux > 0 and "clash" in train_dataset.column_names:
            logger.info(f"Clash rate (train): {float(np.mean(train_dataset['clash'])):.1%}")

    if training_args.do_eval:
        if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
            if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
            else:
                logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                eval_dataset = raw_datasets["test"]
        else:
            eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")
    precision = evaluate.load("precision")
    recall = evaluate.load("recall")

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=p.label_ids)["accuracy"],
            "f1": f1.compute(predictions=preds, references=p.label_ids)["f1"],
            "precision": precision.compute(predictions=preds, references=p.label_ids)["precision"],
            "recall": recall.compute(predictions=preds, references=p.label_ids)["recall"],
        }

    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.01)

    classes = np.array(sorted([int(l) for l in label_list]))
    weights = (
        compute_class_weight(class_weight="balanced", classes=classes, y=train_dataset["label"])
        * data_args.weight_multiplier
    )

    class AuxWeightedTrainer(Trainer):
        # `num_items_in_batch` ditambahkan di transformers >= 4.46 -> tampung via **kwargs.
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            clash = inputs.pop("clash", None)
            outputs = model(**inputs)
            logits = outputs["logits"]
            loss_fct = nn.CrossEntropyLoss(
                weight=torch.tensor(weights, device=logits.device, dtype=torch.float)
                if data_args.do_weighted_loss
                else None
            )
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            if data_args.lambda_aux > 0 and clash is not None and "cls" in outputs:
                aux_fct = nn.CrossEntropyLoss()
                aux_logits = self.model.aux_head(self.model.aux_dropout(outputs["cls"]))
                loss = loss + data_args.lambda_aux * aux_fct(aux_logits, clash.view(-1))
            return (loss, outputs) if return_outputs else loss

    trainer = AuxWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[early_stopping],
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.save_model()
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=predict_dataset)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        if "label" in predict_dataset.features:
            predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        if is_regression:
            predictions = np.squeeze(predictions)
        elif is_multi_label:
            predictions = np.array([np.where(p > 0, 1, 0) for p in predictions])
        else:
            predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(training_args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                logger.info("***** Predict results *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if is_regression:
                        writer.write(f"{index}\t{item:3.3f}\n")
                    elif is_multi_label:
                        item = [label_list[i] for i in range(len(item)) if item[i] == 1]
                        writer.write(f"{index}\t{item}\n")
                    else:
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")
        logger.info("Predict results saved at {}".format(output_predict_file))
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        try:
            trainer.create_model_card(**kwargs)
        except Exception:
            logger.warning("create_model_card dilewati (model dibungkus aux, bukan PreTrainedModel).")


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()