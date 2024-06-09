import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def train_epoch(model, train_dataloader, criterion, optimizer):
    model.train()  # Set the model to training mode
    total_loss = 0

    for batch in train_dataloader:
        src, mask, labels=batch['input_ids'],batch['attention_masks'].bool(),torch.tensor(batch['labels'])
        optimizer.zero_grad()
        outputs = model(src, src_key_padding_mask=~mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_dataloader)

def evaluate(model, val_dataloader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0

    with torch.no_grad():
        for batch in val_dataloader:
            src, mask, labels=batch['input_ids'],batch['attention_masks'].bool(),torch.tensor(batch['labels'])
            outputs = model(src, src_key_padding_mask=~mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(val_dataloader)

def train_model(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs, patience,  save_path='best_model.pt',return_best_loss=False):
    """
    Args:
    return_best_loss (bool): flag to indicate whether to return the pair of training loss and validation loss corresponding to the best validation loss during training
    """
    best_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_dataloader, criterion, optimizer)
        val_loss = evaluate(model, val_dataloader, criterion)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)  # Save the best model state
            epochs_no_improve = 0
            record_pair=(train_loss,best_loss)
            print(f'Validation loss decreased to {val_loss:.4f}, saving model to {save_path}')
        else:
            epochs_no_improve += 1
            print(f'Validation loss did not improve. Count: {epochs_no_improve}/{patience}')

        if epochs_no_improve >= patience:
            print('Early stopping triggered.')
            break
        
        scheduler.step()

    if return_best_loss:
        return record_pair

    return train_loss,val_loss
 