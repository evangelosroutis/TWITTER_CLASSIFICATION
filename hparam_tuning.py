import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from itertools import product
from utilities.preprocessing import remove_urls
from utilities.dataloading import create_dataloader
from train import train_epoch,train_model,evaluate
from utilities.tokenizers import CharacterTokenizer
from model import TransformerClassifier
import yaml
import argparse
import json
import inspect
from sklearn.model_selection import train_test_split
from utilities.save_load import read_labels_from_folder,read_list_from_folder



def filter_params(func, params):
    """
    Filters parameters to match the signature of the provided function.
    
    Args:
    func (function): The function whose parameters are used as a filter.
    params (dict): Dictionary with potential parameters that might match
                   the function's signature.

    Returns:
    dict: A dictionary containing only those key-value pairs from `params` that
          are valid arguments for the function `func`.
    """
    sig = inspect.signature(func)
    return {k: v for k, v in params.items() if k in sig.parameters}
 

def save_best_params(best_params, file_path):
    """
    Saves the best hyperparameters to a JSON file and prints a confirmation message.

    Args:
    best_params (dict): Dictionary containing the best hyperparameter values.
    file_path (str): The path to the file where the hyperparameters will be saved.
    """
    with open(file_path, 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters have been successfully saved to {file_path}")




def main(config_file,output_path):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)



    X_train = read_list_from_folder('./data/split', 'X_train')
    y_train = read_labels_from_folder('./data/split', 'y_train')
    X_val = read_list_from_folder('./data/split', 'X_val')
    y_val = read_labels_from_folder('./data/split', 'y_val')


    body=set(''.join(X_train))
    vocab_size=len(body)+2##### CHANGE hardcoded 2 to num_special_tokens    
    tokenizer=CharacterTokenizer()
    tokenizer.fit(body)
    num_classes=3

    hyperparameters = config['hyperparameters']
    # Convert dictionary to a list of all combinations
    param_combinations = [dict(zip(hyperparameters.keys(), values)) for values in product(*hyperparameters.values())]

    best_params = None
    best_val_loss = float('inf')

    # Iterate over all combinations of hyperparameters
    for params in param_combinations:
        print(f'Training model with parameters {params}')
        # get tunable arguments for each function/class
        dataloader_params = filter_params(DataLoader, params)
        model_params = filter_params(TransformerClassifier, params)
        optimizer_params = filter_params(optim.Adam, params)
        scheduler_params = filter_params(optim.lr_scheduler.StepLR, params)
        train_model_params=filter_params(train_model,params)

        # Prepare DataLoaders

        train_dataloader = create_dataloader(X_train, y_train, tokenizer=tokenizer, **dataloader_params)
        val_dataloader = create_dataloader(X_val, y_val, tokenizer=tokenizer, shuffle=False, drop_last=False,  **dataloader_params)

        # Initialize model with current params
        model = TransformerClassifier(vocab_size=vocab_size, num_classes=num_classes,**model_params)

        # Set up optimizer and scheduler with current lr, gamma, step_size
        optimizer = optim.Adam(model.parameters(),**optimizer_params)
        scheduler = StepLR(optimizer, **scheduler_params)

        criterion = nn.CrossEntropyLoss()

        # Train the model
        train_loss,val_loss= train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, save_path='output/hparam_result/best_model.pt',**train_model_params)
        # Update best parameters if current model is better
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params

        print(f"Best Validation Loss: {best_val_loss}")
        print(f"Best Hyperparameters: {best_params}")
        save_best_params(best_params, output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter Tuning Script')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration YAML file')
    parser.add_argument('--output', type=str, default='output/hparam_result/best_params.json', help='Path to save best parameters')
    args = parser.parse_args()

    main(args.config,args.output)