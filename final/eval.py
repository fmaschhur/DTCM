import torch
from transformers import BertForSequenceClassification

from data import get_test_dl
from utils import get_cli_args, save_predictions

DEFAULT_EVAL_DATA_PATH = "test1.csv"
EVAL_RESULT_PATH = "eval.csv"

BASE_BERT_MODEL = "deepset/gbert-base"
FINETUNED_BERT_MODEL_PATH = "deepset-gbert-base-finetuned"


def make_predictions(test_dl, model):
    predictions = []

    # select GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # switch to eval mode
    model.eval()

    # iterate batches
    for batch in test_dl:
        # move data to GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        prediction = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        prediction = prediction.logits.detach().numpy()

        # store
        for idx in range(prediction.shape[0]):
            predictions.append({'sent_id': int(batch['id'][idx]), 'mos': float(prediction[idx])})

    return predictions


if __name__ == "__main__":
    cli_args = get_cli_args(DEFAULT_EVAL_DATA_PATH)
    eval_data_file_path = cli_args.input

    print(f"Loading evaluation data from '{eval_data_file_path}' ...")
    test_dl = get_test_dl(eval_data_file_path, BASE_BERT_MODEL)
    print(f"Evaluation data loaded!")

    print(f"Loading model ...")
    model = BertForSequenceClassification.from_pretrained(FINETUNED_BERT_MODEL_PATH)
    print(f"Model loaded!")

    print(f"Evaluating model...")
    predictions = make_predictions(test_dl, model)
    save_predictions(predictions, EVAL_RESULT_PATH)
    print(f"Evaluation results saved in '{EVAL_RESULT_PATH}'.")

