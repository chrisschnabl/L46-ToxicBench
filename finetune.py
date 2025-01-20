import numpy as np
import evaluate
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, toxicity = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=toxicity)

id2label = {0: "NON_TOXIC", 1: "TOXIC"}
label2id = {"NON_TOXIC": 0, "TOXIC": 1}
required_columns = ["input_ids", "attention_mask", "label"] 

def remove_unused_features(dataset):
    return dataset.remove_columns([col for col in dataset.column_names if col not in required_columns])

def get_trainer(base_model: str, output_model: str, tokenizer, train, test, batch_sizes=32) -> Trainer:
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model, num_labels=2, id2label=id2label, label2id=label2id
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=output_model,
        learning_rate=2e-6,
        per_device_train_batch_size=batch_sizes,
        per_device_eval_batch_size=batch_sizes,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=True,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )