import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

df = pd.read_csv('data.csv', encoding='utf-8')  # 替换为你的数据文件名
train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

# 2. 定义Dataset
class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['text'].tolist()
        self.labels = df['label'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 3. 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('./bert-base-chinese', num_labels=2)

# 4. 构建Dataset和DataLoader
train_dataset = TextDataset(train_df, tokenizer)
val_dataset = TextDataset(val_df, tokenizer)

# 5. 训练参数
training_args = TrainingArguments(
    output_dir='./bert_cls_output',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=50,
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

# 6. 定义Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 7. 开始训练
trainer.train()

# 8. 保存模型
model.save_pretrained('./bert_cls_model')
tokenizer.save_pretrained('./bert_cls_model')
