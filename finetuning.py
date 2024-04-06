import os
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from datasets import load_dataset
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

torch.cuda.empty_cache()
torch.manual_seed(42)


class VQADataset(torch.utils.data.Dataset):
    """VQA (v2) dataset."""
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get text
        question = self.dataset[idx]['question']
        answer = self.dataset[idx]['answer']
        context = self.dataset[idx]['context']
        question_id = self.dataset[idx]['question_id']
        text = question
        
        # Get images
        images = []
        for image_name in self.dataset[idx]['image']:
            image_path = f"Data/train_images/{image_name}"
            image = Image.open(image_path).convert("RGB")
            images.append(image)
        
        # Process images and text
        encodings = self.processor(images, text, padding="max_length", truncation=True, return_tensors="pt")
        
        # Process labels
        labels = self.processor.tokenizer.encode(answer, max_length=8, pad_to_max_length=True, return_tensors='pt')
        
        # Assign labels to each image
        encodings["labels"] = labels.squeeze().repeat(len(images), dim=0)
        
        # Add additional fields
        encodings["context"] = [context] * len(images)
        encodings["question_id"] = [question_id] * len(images)
        
        return encodings

training_dataset = load_dataset("json", data_files="Data/vqa_context_train.jsonl", split="train[:90%]")
valid_dataset = load_dataset("json", data_files="Data/vqa_context_train.jsonl", split="train[90%:]")
print("Training sets: {} - Validating set: {}".format(len(training_dataset), len(valid_dataset)))

train_dataset = VQADataset(dataset=training_dataset,
                          processor=processor)
valid_dataset = VQADataset(dataset=valid_dataset,
                          processor=processor)

batch_size = 12
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False)

num_epochs = 10
patience = 10
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    for idx, batch in zip(tqdm(range(len(train_dataloader)), desc='Training batch: ...'), train_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        # attention_mask=attention_masked,
                        labels=labels)
            
        loss = outputs.loss
        epoch_loss += loss.item()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    
    model.eval()
    eval_loss = 0
    for idx, batch in zip(tqdm(range(len(valid_dataloader)), desc='Validating batch: ...'), valid_dataloader):
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        attention_mask=attention_masked,
                        labels=labels)
        
        loss = outputs.loss
        eval_loss += loss.item()

    tracking_information.append((epoch_loss/len(train_dataloader), eval_loss/len(valid_dataloader), optimizer.param_groups[0]["lr"]))
    print("Epoch: {} - Training loss: {} - Eval Loss: {} - LR: {}".format(epoch+1, epoch_loss/len(train_dataloader), eval_loss/len(valid_dataloader), optimizer.param_groups[0]["lr"]))
    scheduler.step()
    if eval_loss < min_eval_loss:
        model.save_pretrained("Model/blip2-saved-model", from_pt=True) 
        print("Saved model to Model/blip2-saved-model")
        min_eval_loss = eval_loss
        early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        if early_stopping_hook > patience:
            break
    
pickle.dump(tracking_information, open("tracking_information.pkl", "wb"))
print("The finetuning process has done!")
