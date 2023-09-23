from transformers import GPT2Tokenizer

# Instantiate the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
from nltk.corpus import gutenberg
from collections import Counter
import nltk

class LanguageModelDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_sequence_length, batch_size, sample_size):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.sample_size = sample_size

    def DataTokenization(self, text, max_sequence_length):
        value = self.tokenizer.encode(text, add_special_tokens=False)
        if len(value) > max_sequence_length:
            value = value[:max_sequence_length]
        else:
            value = value + [self.tokenizer.pad_token_id] * (max_sequence_length - len(value))
        return value

    def TrainDataPreprocess(self, max_sequence_length, sample_size, device):
        # Extract the columns 'system_prompt', 'question', and 'response'
        system_prompts = self.dataset['train']['system_prompt']
        questions = self.dataset['train']['question']
        responses = self.dataset['train']['response']

        # Preprocess and pad the sequences
        preprocessed_data = []
        for i in range(sample_size):
            submax_sequence_length = max_sequence_length // 2
            padded_system_prompt = self.DataTokenization(system_prompts[i], submax_sequence_length)
            padded_question = self.DataTokenization(questions[i], submax_sequence_length)
            padded_response = self.DataTokenization(responses[i], max_sequence_length)
            preprocessed_data.append({
                'system_prompt': padded_system_prompt,
                'question': padded_question,
                'response': padded_response
            })

        # Convert the preprocessed data to tensors
        system_prompt_tensors = torch.tensor([item['system_prompt'] for item in preprocessed_data], dtype=torch.long)
        question_tensors = torch.tensor([item['question'] for item in preprocessed_data], dtype=torch.long)
        response_tensors = torch.tensor([item['response'] for item in preprocessed_data], dtype=torch.long)

        # Concatenate system_prompts and questions along the appropriate dimension
        input_ids = torch.cat((system_prompt_tensors, question_tensors), dim=1)
        response_tensors = response_tensors.to(dtype=torch.long, device=device)

        # Create DataLoader with the preprocessed tensors
        data = torch.utils.data.TensorDataset(input_ids, response_tensors)
        return data

    def TestDataPreprocess(self, max_sequence_length, sample_size, device):
        # Extract the columns 'system_prompt', 'question', and 'response'
        system_prompts = self.dataset['test']['system_prompt']
        questions = self.dataset['test']['question']
        responses = self.dataset['test']['response']

        # Preprocess and pad the sequences
        preprocessed_data = []
        for i in range(sample_size):
            submax_sequence_length = max_sequence_length // 2
            padded_system_prompt = self.DataTokenization(system_prompts[i], submax_sequence_length)
            padded_question = self.DataTokenization(questions[i], submax_sequence_length)
            padded_response = self.DataTokenization(responses[i], max_sequence_length)
            preprocessed_data.append({
                'system_prompt': padded_system_prompt,
                'question': padded_question,
                'response': padded_response
            })

        # Convert the preprocessed data to tensors
        system_prompt_tensors = torch.tensor([item['system_prompt'] for item in preprocessed_data], dtype=torch.long)
        question_tensors = torch.tensor([item['question'] for item in preprocessed_data], dtype=torch.long)
        response_tensors = torch.tensor([item['response'] for item in preprocessed_data], dtype=torch.long)

        # Concatenate system_prompts and questions along the appropriate dimension
        input_ids = torch.cat((system_prompt_tensors, question_tensors), dim=1)
        response_tensors = response_tensors.to(dtype=torch.long, device=device)

        # Create DataLoader with the preprocessed tensors
        data = torch.utils.data.TensorDataset(input_ids, response_tensors)
        return data

    def DataDivision(self, data, test_size=0.2, random_state=42):
        # Split the data into training and validation sets
        train_data, val_data = train_test_split(data, test_size=test_size, random_state=random_state)

        # Create DataLoader for training data
        train_dataloader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        # Create DataLoader for validation data
        validation_dataloader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)
        return train_dataloader, validation_dataloader

from datasets import load_dataset

dataset = load_dataset("shirsh10mall/LLM_Instruct_Learning_Project_Preprocessed_Tokenized_Open_Orca_Dataset_Flan_T5")