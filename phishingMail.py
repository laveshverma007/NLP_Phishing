import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import requests
import zipfile
import io
import os

"""
Phishing Email Detection using DistilBERT

This implementation uses the following key components:
1. DistilBERT: It refers to a model that is an approximation of the BERT model. It reduces the size of the BERT model by 40% while maintaining 97% of its performance.
2. Transfer Learning: We fine-tune a pre-trained model for our specific task
3. Binary Classification: Outputs whether an email is phishing (1) or legitimate (0)
"""

def download_dataset(url, sample_size=1000):
    """
    Downloads and samples the dataset to reduce training time.
    Using random sampling to maintain distribution of classes.
    
    Args:
        url: Dataset URL
        sample_size: Number of examples to use (default: 1000)
    """
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        zip_ref.extractall(".")
    df = pd.read_csv("Phishing_Email.csv")
    return df.sample(n=min(sample_size, len(df)), random_state=42)

class EmailDataset(Dataset):
    """
    Custom PyTorch Dataset for email classification.
    
    Key features:
    1. Tokenization: Converts text to token IDs using DistilBERT tokenizer
    2. Padding: Ensures all sequences have same length
    3. Attention Masks: Identifies real tokens vs padding tokens
    
    Args:
        texts: List of email texts
        labels: Binary labels (0: legitimate, 1: phishing)
        tokenizer: DistilBERT tokenizer
        max_len: Maximum sequence length (default: 128)
    """
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenization with special tokens ([CLS], [SEP]) and padding
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_model(model, train_loader, val_loader, device, epochs=2):
    """
    Trains the model using the following techniques:
    1. AdamW Optimizer: Better weight decay regularization than Adam
    2. Learning Rate: 3e-5 (standard for fine-tuning transformers)
    3. Early Stopping: Reduced epochs for faster training
    4. Batch Size: 16 for better gradient estimates
    
    Training process:
    1. Forward pass: Compute predictions and loss
    2. Backward pass: Calculate gradients
    3. Optimization: Update model weights
    4. Validation: Evaluate on held-out data
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Move batch to device (CPU/GPU)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Calculate loss and backpropagate
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        # Quick validation after each epoch
        model.eval()
        val_loss = 0
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                predictions.extend(outputs.logits.argmax(dim=1).cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())
        
        print(f'Epoch {epoch + 1}:')
        print('Training Loss:', train_loss / len(train_loader))
        print('Validation Loss:', val_loss / len(val_loader))
        print('\nClassification Report:')
        print(classification_report(actual_labels, predictions))

def main():
    model_path = './phishing_model'
    
    # Check if model already exists
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("Model loaded successfully!")
        return

    # Download and prepare data (reduced sample size)
    print("Downloading and preparing dataset...")
    df = download_dataset("https://storage.googleapis.com/kaggle-data-sets/3487818/6090437/compressed/Phishing_Email.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20241023%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20241023T165911Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=605a8501eb0975d437514d9bc6abce89b850095d0fc57dd20a6e373cd0fbb3187ee6c87a5e446869fe9f2c66a92eaddf6cbbcb346fef3516f46f8f11b58cefd3ca28e377ab10c316512ca0a050e15934a40b31daf11b083467d4bc4bc711fc86e453fd95e7c982f2aca012cdced04a93a57ba9a6ea7f3af2001dad8178b6000c124c53d6edc558ebb006daeb804bd6cb7ff42997800e9a2ebdae04bbf257534f1a18d777ee845861b55b716db14564e71119c3f1e4ac6d8f327983ff4cc474e614d76db92ebe414d3d2e2b7bdd0dbe5e410fa976111f5590f929cc5e9f997c501fc521f1b7d2cba57f1e42cd235671bf4f739df80abfe294548142daf8df70b3", sample_size=1000)
    
    # Convert labels to numeric
    df['Label'] = (df['Email Type'] == 'Phishing Email').astype(int)
    
    # Split data into train and validation sets (80/20 split)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Email Text'].values, 
        df['Label'].values,
        test_size=0.2,
        random_state=42
    )

    # Initialize smaller, faster model (DistilBERT instead of BERT)
    print("Initializing DistilBERT model...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )

    # Create datasets with smaller max_len
    train_dataset = EmailDataset(train_texts, train_labels, tokenizer)
    val_dataset = EmailDataset(val_texts, val_labels, tokenizer)

    # Create dataloaders with larger batch size for faster training
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Set device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Train model
    print("Training model...")
    train_model(model, train_loader, val_loader, device)

    # Save model and tokenizer for later use
    print("Saving model...")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print("Model saved successfully!")

if __name__ == "__main__":
    main()
