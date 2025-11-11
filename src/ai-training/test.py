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
    parser = argparse.ArgumentParser(description="Train a BART model given a parallel corpus")
    parser.add_argument("-input_field", required=False, help="Name of the input field in the training/test files", default='wrong_text')
    parser.add_argument("-output_field", required=False, help="Name of the output field in the training/test files", default='correct_text')
    parser.add_argument("-model", required=False, help="Type of the model to be evaluated: bart | mbart | t5 | mt5", default='bart')
    parser.add_argument("-model_file", required=True, help="Name of the output file to save the trained model")
    parser.add_argument("-test_file", required=True, help="Name of the test file")
    parser.add_argument("-eval_output", required=True, help="Name of the output XLSX file to save the evaluation results")
    args = parser.parse_args()

    input_field = args.input_field
    output_field = args.output_field
    print(f"The provided input and output fields: {input_field},{output_field}")
    
    model_name = args.model
    print(f"Model name: {model_name}")

    model_file = args.model_file
    print(f"The provided model output file is: {model_file}")

    test_file = args.test_file
    print(f"The provided test file is: {test_file}")
    eval_file = args.eval_output
    print(f"The provided XLSX output file is: {eval_file}")

    print('Loading tokenizer and model')

    match model_name:
        case 'bart-base' | 'bart-large':
            tokenizer = BartTokenizer.from_pretrained(model_file)
            model = BartForConditionalGeneration.from_pretrained(model_file)

        case 'mbart-large':
            tokenizer = MBart50Tokenizer.from_pretrained(model_file, src_lang="pt_XX", tgt_lang="pt_XX")
            model = MBartForConditionalGeneration.from_pretrained(model_file)

        case 't5-small' | 't5-base' | 't5-large':
            tokenizer = T5Tokenizer.from_pretrained(model_file)
            model = T5ForConditionalGeneration.from_pretrained(model_file)

        case 'mt5-small' | 'mt5-base' | 'mt5-large':
            tokenizer = MT5Tokenizer.from_pretrained(model_file)
            model = MT5ForConditionalGeneration.from_pretrained(model_file)

        case _:
            print(f"Invalid model name {model_name}")
            quit()


    print('Model saved')
    # Set up GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model.to(device)

    print('Loading test data')
    input_data = pd.read_csv( test_file )

    print(f'Len test set {len(input_data)}')

    wrong = input_data[input_field]
    correct = input_data[output_field]
    test_data = [tup for tup in zip(wrong, correct)]

    # Create datasets and dataloaders
    test_dataset = AutoCorrectionDataset(test_data, tokenizer, max_length=128) 
    test_loader = DataLoader(test_dataset, batch_size=16)

    print("Test the model")
    total_samples = 0
    total_correct = 0

    output_dict = { "inputs": [], "outputs": [], "references": []}

    for test_sample in tqdm(test_data):
        input_text, expected_correct = test_sample
        
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        outputs = model.generate(input_ids.to(device), num_beams=10, num_return_sequences=10,  return_dict_in_generate=True, output_scores=True)

        suggestions = []
        for output_id, sequence_score in zip(outputs.sequences, outputs.sequences_scores):
            suggestions.append( (tokenizer.decode(output_id, skip_special_tokens=True), float(sequence_score.cpu().detach().numpy()) )[0] )

        output_dict["inputs"].append( input_text )
        output_dict["references"].append( expected_correct )
        output_dict["outputs"].append( suggestions )


    df = pd.DataFrame.from_dict(output_dict)
    df.to_excel(eval_file)

if __name__ == "__main__":
    main()