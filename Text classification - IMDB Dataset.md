#Explaination:
### **1️⃣ Loading the IMDB Dataset**
```python
from datasets import load_dataset
imdb = load_dataset("imdb")
```
- Loads the **IMDB dataset**, which contains **50,000** labeled movie reviews (positive/negative).
- It is structured like a **dictionary** with `train`, `test`, and `unsupervised` splits.

```python
imdb['train'] = imdb['train'].shuffle(seed=1).select(range(2000))
```
- The training set is **shuffled** and reduced to **2,000 samples** for faster fine-tuning.

```python
imdb_train_validation = imdb['train'].train_test_split(train_size=0.8)
imdb_train_validation['validation'] = imdb_train_validation.pop('test')
imdb.update(imdb_train_validation)
```
- The training set is further **split into training (80%) and validation (20%)**.

```python
imdb['test'] = imdb['test'].shuffle(seed=1).select(range(400))
imdb.pop('unsupervised')
```
- The test set is **reduced to 400 samples**.
- The **'unsupervised'** part (unlabeled data) is **removed**.

---

### **2️⃣ Exploratory Data Analysis**
```python
df = imdb['train'][:]
df['text'] = df.text.str.replace('<br />', '')  # Remove HTML tags
df.label.value_counts()
```
- Converts dataset into a **Pandas DataFrame**.
- Removes `<br />` tags.
- **Counts the number of positive and negative reviews**.

```python
df["Words per review"] = df["text"].str.split().apply(len)
df.boxplot("Words per review", by="label", grid=False, showfliers=False, color="black")
```
- Computes the **word count per review** and plots a **box plot**.

---

### **3️⃣ Tokenizing Text with BERT**
```python
from transformers import AutoTokenizer
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```
- Loads **BERT's tokenizer**.
- `"bert-base-cased"` means:
  - **Base model** (not large).
  - **Cased** (distinguishes uppercase and lowercase).

```python
def tokenize_function(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

imdb_encoded = imdb.map(tokenize_function, batched=True, batch_size=None)
```
- **Tokenizes** each review while ensuring:
  - **Padding** (all sequences have the same length).
  - **Truncation** (trims long reviews).

---

### **4️⃣ Creating a Tiny Subset for Quick Testing**
```python
from datasets import DatasetDict
tiny_imdb = DatasetDict()
tiny_imdb['train'] = imdb['train'].shuffle(seed=1).select(range(50))
tiny_imdb['validation'] = imdb['validation'].shuffle(seed=1).select(range(10))
tiny_imdb['test'] = imdb['test'].shuffle(seed=1).select(range(10))

tiny_imdb_encoded = tiny_imdb.map(tokenize_function, batched=True, batch_size=None)
```
- A **mini dataset** is created with **only 50 training, 10 validation, and 10 test samples**.
- **Tokenized** again.

---

### **5️⃣ Initializing BERT for Sentiment Classification**
```python
from transformers import AutoModelForSequenceClassification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 2
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).to(device)
```
- Loads **BERT model** with a **classification head** (`num_labels=2` for positive/negative).
- Moves the model to **GPU (if available)**.

---

### **6️⃣ Training BERT on the Tiny IMDB Dataset**
```python
from transformers import Trainer, TrainingArguments

batch_size = 8
logging_steps = len(tiny_imdb_encoded["train"]) // batch_size
training_args = TrainingArguments(
    output_dir=f"{checkpoint}-finetuned-tiny-imdb",
    num_train_epochs=2,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_steps=logging_steps,
    optim='adamw_torch'
)
```
- **Training parameters**:
  - `2` **epochs** (full passes over the dataset).
  - **Batch size** of `8`.
  - **Learning rate** of `2e-5` (small for fine-tuning).
  - **AdamW optimizer**.

```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tiny_imdb_encoded["train"],
    eval_dataset=tiny_imdb_encoded["validation"],
    tokenizer=tokenizer
)
trainer.train()
```
- **Trainer** handles training and evaluation.
- **Trains on the tiny dataset**.

---

### **7️⃣ Evaluating on the Tiny IMDB Dataset**
```python
preds = trainer.predict(tiny_imdb_encoded['test'])
preds.predictions.argmax(axis=-1)
```
- **Predictions** are extracted and converted to class labels.

```python
from sklearn.metrics import accuracy_score
accuracy_score(preds.label_ids, preds.predictions.argmax(axis=-1))
```
- Computes **accuracy**.

---

### **8️⃣ Full Training on IMDB Dataset**
```python
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=get_accuracy,
    train_dataset=imdb_encoded["train"],
    eval_dataset=imdb_encoded["validation"],
    tokenizer=tokenizer
)
trainer.train()
trainer.evaluate()
```
- **Fine-tunes on the full IMDB dataset**.
- Evaluates the final performance.

```python
trainer.save_model()
```
- Saves the **fine-tuned model**.

---

### **9️⃣ Using the Model for Sentiment Classification**
```python
from transformers import pipeline
classifier = pipeline('text-classification', model=model_name)

classifier('This is not my idea of fun')
# Output: [{'label': 'NEGATIVE', 'score': 0.99}]

classifier('This was beyond incredible')
# Output: [{'label': 'POSITIVE', 'score': 0.99}]
```
- Loads the fine-tuned model into a **pipeline**.
- **Classifies** movie reviews.

---

### **🔑 Summary**
✔ **Loaded and preprocessed** the IMDB dataset.  
✔ **Tokenized** text using BERT.  
✔ **Fine-tuned BERT** on IMDB using Hugging Face’s Trainer.  
✔ **Evaluated model** performance.  
✔ **Used the model** for real-world sentiment classification.

This workflow **fine-tunes BERT for sentiment analysis**, turning it into a **movie review classifier**. 🚀
