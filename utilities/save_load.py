from pathlib import Path
import os
import torch

def save_list_to_folder(data, folder_name, prefix):
    os.makedirs(folder_name, exist_ok=True)
    file_path = Path(folder_name) / f"{prefix}.txt"
    with open(file_path, 'w', encoding='utf-8') as file:
        for item in data:
            file.write(f"{item}\n")

def read_list_from_folder(folder_name, prefix):
    file_path = Path(folder_name) / f"{prefix}.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [line.strip() for line in file]
    return data

def read_labels_from_folder(folder_name, prefix):
    file_path = Path(folder_name) / f"{prefix}.txt"
    with open(file_path, 'r', encoding='utf-8') as file:
        labels = [int(line.strip()) for line in file]
    return labels


