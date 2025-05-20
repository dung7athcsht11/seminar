import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd

# 1. Định nghĩa lớp Dataset với Instruction Tuning
class SentimentDataset(Dataset):
    def __init__(self, contexts, targets, tokenizer, max_length):
        self.contexts = contexts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = {"negative": 0, "neutral": 1, "positive": 2}

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        target = self.targets[idx]
        encoding = self.tokenizer(
            context,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        label = self.label_map[target]
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# 2. Hàm huấn luyện
def train_model(model, train_loader, device, epochs=8):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
    return model

# 3. Hàm đánh giá
def evaluate_model(model, eval_loader, device):
    model.eval()
    predictions, true_labels = [], []
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
            predictions.extend([label_map[p] for p in preds])
            true_labels.extend([label_map[l.item()] for l in labels])
    f1 = f1_score(true_labels, predictions, average="weighted")
    return f1, predictions

# 4. Chuẩn bị dữ liệu FPB với Instruction Tuning
dataset = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
df = dataset["train"].to_pandas()
df = df.rename(columns={"sentence": "input", "label": "output"})
df["instruction"] = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
df["output"] = df["output"].map({0: "negative", 1: "neutral", 2: "positive"})
df["context"] = df.apply(lambda x: f"Instruction: {x['instruction']}\nInput: {x['input']}\nAnswer: ", axis=1)

# Chia dữ liệu
train_df = df.sample(frac=0.8, random_state=42)
eval_df = df.drop(train_df.index)

# 5. Khởi tạo mô hình DistilBERT
model_name = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\n=== Đánh giá mô hình {model_name} ===")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Chuẩn bị dataset
train_dataset = SentimentDataset(
    train_df["context"].tolist(),
    train_df["output"].tolist(),
    tokenizer,
    max_length=128
)
eval_dataset = SentimentDataset(
    eval_df["context"].tolist(),
    eval_df["output"].tolist(),
    tokenizer,
    max_length=128
)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=2)

# Đánh giá trước khi tinh chỉnh (pre-tuning)
print("Đánh giá trước khi tinh chỉnh...") 8
pre_tune_f1, pre_tune_predictions = evaluate_model(model, eval_loader, device)
print(f"{model_name} F1-Score (Pre-Tuning): {pre_tune_f1:.3f}")

# Áp dụng LoRA
lora_config = LoraConfig(
    task_type="SEQ_CLS",
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]
)
model = get_peft_model(model, lora_config)
model.to(device)

# Huấn luyện
model = train_model(model, train_loader, device, epochs=8)

# Đánh giá sau khi tinh chỉnh (post-tuning)
print("Đánh giá sau khi tinh chỉnh...")
post_tune_f1, post_tune_predictions = evaluate_model(model, eval_loader, device)
print(f"{model_name} F1-Score (Post-Tuning): {post_tune_f1:.3f}")

# Lưu mô hình
model.save_pretrained(f"./fingpt_{model_name.lower()}_sentiment")
tokenizer.save_pretrained(f"./fingpt_{model_name.lower()}_sentiment")

# So sánh kết quả
print("\n=== So sánh kết quả ===")
print(f"{model_name} F1-Score (Pre-Tuning): {pre_tune_f1:.3f}")
print(f"{model_name} F1-Score (Post-Tuning): {post_tune_f1:.3f}")
print(f"Cải thiện: {(post_tune_f1 - pre_tune_f1):.3f}")

# Lưu kết quả
eval_df["DistilBERT_pred_pre_tune"] = pre_tune_predictions
eval_df["DistilBERT_pred_post_tune"] = post_tune_predictions
eval_df.to_csv("fingpt_comparison_distilbert.csv")
print("Kết quả đã được lưu vào fingpt_comparison_distilbert.csv")