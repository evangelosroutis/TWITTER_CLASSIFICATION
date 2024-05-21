import torch
import torch.nn as nn
import json
import argparse
from model import TransformerClassifier
from utilities.tokenizers import CharacterTokenizer
from hparam_tuning_eval import filter_params
import yaml


def load_hyperparameters(file_path):
    """
    Load hyperparameters from a JSON file.
    Args:
        file_path (str): The path to the JSON file containing the hyperparameters.
    Returns:
        dict: A dictionary containing the hyperparameters.
    """
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters


def write_predictions(predictions, output_file):
    """
    Write predictions to a file.
    Args:
        predictions (list): List of predicted labels.
        output_file (str): File path where predictions will be saved.
    """
    # Mapping numeric predictions to text labels 
    label_map = {0: 'positive', 1: 'negative', 2: 'neutral'}

    with open(output_file, 'w') as file:
        for pred in predictions:
            file.write(f"{label_map[pred]}\n")


def predict(model, tokenizer, input_lines, batch_size=16):
    """
    Predict sentiment for batches of lines using the loaded model.
    Args:
        model (nn.Module): Trained model for prediction.
        tokenizer (CharacterTokenizer): Tokenizer for processing input text.
        input_lines (list): List of input text lines for prediction.
        batch_size (int): The number of lines to process in each batch.
    Returns:
        list: Predicted sentiments for each line.
    """
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(input_lines), batch_size):
            batch = input_lines[i:i+batch_size]
            tokenized = [tokenizer.encode(line) for line in batch]
            # Create a padded tensor for the batch
            padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(tokens) for tokens in tokenized],
                                                     batch_first=True, padding_value=tokenizer._token_to_id[tokenizer.pad_token])
            # Create a mask where padding tokens are True (masked) and all others are False (unmasked)
            src_key_padding_mask = (padded == tokenizer._token_to_id[tokenizer.pad_token])
            
            output = model(padded, src_key_padding_mask)
            preds = torch.argmax(output, dim=1).tolist()  # Get predicted class indices for the whole batch
            predictions.extend(preds)
    return predictions




def main(input_file, output_file, config_file,batch_size):
    """
    Load the model, make predictions on input data in batches, and write predictions to an output file.
    """
    with open(config_file, 'r') as file:
        config_file = yaml.safe_load(file)


    # load trained tokenizer
    tokenizer=CharacterTokenizer()
    tokenizer.load(config_file['trained_tokenizer_path'])
    #tokenizer.load('output/eval_result/trained_tokenizer.json')
    vocab_size=tokenizer.vocab_size

    # Load hyperparameters and initialize the model
    best_params = load_hyperparameters(config_file['best_params_path'])
    model_params = filter_params(TransformerClassifier, best_params) #get tunable parameters 
    model = TransformerClassifier(vocab_size=vocab_size, num_classes=3,**model_params)
    model.load_state_dict(torch.load(config_file['eval_result_path']))
    

    # Read input data
    with open(input_file, 'r') as file:
        input_lines = file.read().splitlines()

    # Make predictions
    predictions = predict(model, tokenizer, input_lines, batch_size) 

    # Write predictions to file
    write_predictions(predictions, output_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict Sentiment from Tweets')
    parser.add_argument('--input', type=str, required=True, help='Input file path with tweets')
    parser.add_argument('--output', type=str, required=True, help='Output file path for predictions')
    parser.add_argument('--config', type=str, default='config.yaml', help='Configuration file path for model settings')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of lines to process in each batch')


    args = parser.parse_args()

    main(args.input, args.output, args.config, args.batch_size)
