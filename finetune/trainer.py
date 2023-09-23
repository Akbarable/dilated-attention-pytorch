import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import gc
from torch.autograd import profiler

class LongNetTrainer:
    def __init__(self, model, train_dataloader, validation_dataloader, device,learning_rate,num_training_steps,num_warmup_steps):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.device = device

        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.learning_rate = learning_rate
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, self.num_warmup_steps, self.num_training_steps)

        self.batch_size = None  # Set your batch size
        self.sequence_length = None  # Set your sequence length
        self.vocab_size = None  # Set your vocabulary size

    def calculate_loss(self, logits, targets):
        # Calculate the Cross-Entropy loss between predicted logits and target indices
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss

    def train_epoch(self, epoch):
        self.model.train()

        for batch in self.train_dataloader:
            batch = [item.to(self.device) for item in batch]
            input_ids, response_tensor = batch

            with profiler.profile(record_shapes=True, use_cuda=True) as prof:
                self.optimizer.zero_grad()
                logits = self.model(input_ids)
                responses = response_tensor.to(dtype=torch.long, device=self.device)
                loss = self.calculate_loss(logits, responses)
                print(loss.tolist())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

    def validate_epoch(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.validation_dataloader:
                batch = [item.to(self.device) for item in batch]
                input_ids, response_tensor = batch
                logits = self.model(input_ids)
                responses = response_tensor.to(dtype=torch.long, device=self.device)
                loss = self.calculate_loss(logits, responses)

            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.train_epoch(epoch)
            gc.collect()
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.empty_cache()
            self.validate_epoch()
            # Move the model's parameters and buffers to the desired device
            self.model.to(device)
            filepath = 'trained_model.pth'
            torch.save(self.model.state_dict(), filepath)


device = torch.device('cuda')
model = LongNetLM(num_tokens=tokenizer.vocab_size).to(device)
model = nn.DataParallel(model)

num_training_steps = 10
num_warmup_steps = 2
learning_rate = 6e-4  # Define an appropriate learning rate value

# Create an instance
trainer = LongNetTrainer(model, train_dataloader, validation_dataloader, device, learning_rate, num_training_steps, num_warmup_steps)

# Train the model
trainer.train(num_epochs)
