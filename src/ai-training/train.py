import argparse
import sys
import os
import pandas as pd
import pickle
import random
from tqdm import tqdm

import transformers
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from transformers import TrainingArguments, Trainer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

import torch
from torch.utils.data import DataLoader, Dataset


def get_pytorch_device():
    """
    Checks for CUDA, then MPS, and falls back to CPU if neither is available.
    Returns a torch.device object.
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS device (Apple Silicon GPU)")
    else:
        device = torch.device('cpu')
        print("Using CPU device")
    return device

class AutoCorrectionDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inputs, target = self.data[idx]
        model_inputs = self.tokenizer(inputs, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        labels = self.tokenizer(text_target=target, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        
        model_inputs['labels'] = labels['input_ids']

        return {type: data[0] for type, data in model_inputs.items()}


def main():
    parser = argparse.ArgumentParser(description="Train an encoder-decoder model given a parallel corpus")
    parser.add_argument("-training_file", required=True, help="Name of the training file, the parallel corpus")
    parser.add_argument("-input_field", required=False, help="Name of the input field in the training/test files", default='input')
    parser.add_argument("-output_field", required=False, help="Name of the output field in the training/test files", default='output')
    parser.add_argument("-model", required=False, help="Type of the model to be trained: bart | mbart | t5 | mt5", default='bart')
    parser.add_argument("-lr", required=False, help="Learning rate", default=2e-5)
    parser.add_argument("-epochs", required=False, help="Training epochs", default=20)
    parser.add_argument("-model_file", required=True, help="Name of the output file to save the trained model")
    args = parser.parse_args()

    training_file = args.training_file
    print(f"The provided training file is: {training_file}")
    input_field = args.input_field
    output_field = args.output_field
    print(f"The provided input and output fields: {input_field},{output_field}")
    

    model_name = args.model
    model_lr = float(args.lr)
    model_epochs = int(args.epochs)
    print(f"Model name: {model_name} LR: {model_lr} epochs: {model_epochs}")

    model_file = args.model_file
    print(f"The provided model output file is: {model_file}")

    print('Loading data')
    annotated_data = pd.read_csv( training_file )

    total_samples = len(annotated_data)
    validation_proportion = 0.2
    validation_samples = int(total_samples * validation_proportion)
    
    annotated_data = annotated_data.sample(frac=1)

    training_data = annotated_data.iloc[:-validation_samples]
    validation_data = annotated_data.iloc[-validation_samples:]

    print(f'Len training set {len(training_data)}')
    print(f'Len validation set {len(validation_data)})')


    wrong, correct = training_data[input_field], training_data[output_field]
    train_data = [tup for tup in zip(wrong, correct)]


    wrong, correct = validation_data[input_field], validation_data[output_field]
    validation_data = [tup for tup in zip(wrong, correct)]

    print('Loading tokenizer and model')
    match model_name:
        case 'bart-base':
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
            model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')   
        case 'bart-large':
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

        case 'mbart-large':
            tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50', src_lang="pt_XX", tgt_lang="pt_XX")
            model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50')

        case 't5-small':
            tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
            model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
        case 't5-base':
            tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-base')
            model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-base')
        case 't5-large':
            tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-large')
            model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-large')

        case 'mt5-small':
            tokenizer = MT5Tokenizer.from_pretrained('google/mt5-small')
            model = MT5ForConditionalGeneration.from_pretrained('google/mt5-small')
        case 'mt5-base':
            tokenizer = MT5Tokenizer.from_pretrained('google/mt5-base')
            model = MT5ForConditionalGeneration.from_pretrained('google/mt5-base')
        case 'mt5-large':
            tokenizer = MT5Tokenizer.from_pretrained('google/mt5-large')
            model = MT5ForConditionalGeneration.from_pretrained('google/mt5-large')

        case _:
            print(f"Invalid model name {model_name}")
            quit()

    device = get_pytorch_device()
    model.to(device)

    print('Tokenizing and organizing data')

    # Create datasets and dataloaders
    train_dataset = AutoCorrectionDataset(train_data, tokenizer, max_length=128)
    val_dataset = AutoCorrectionDataset(validation_data, tokenizer, max_length=128) 

    print('Training model')
    batch_size=8

    training_args = Seq2SeqTrainingArguments(
        output_dir="results",
        eval_strategy="epoch",
        learning_rate=model_lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=model_epochs,
        fp16=False,
        save_strategy='epoch',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    print('Saving model')
    trainer.save_model( model_file )

if __name__ == "__main__":
    main()