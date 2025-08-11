# GPT-2 Fine-Tuning with Custom Dataset

This repository contains a Jupyter notebook for **fine-tuning the GPT-2 language model** using the wikitext dataset.
It leverages the Hugging Face 🤗 `transformers` library to adapt GPT-2 for domain-specific text generation.

## 📌 Features

* Tokenize text using the GPT-2 tokenizer.
* Fine-tune GPT-2 with Hugging Face `Trainer` API.
* Save and reload trained models for inference.
* Generate text from your fine-tuned model.

## 📂 Files

* **`TG-GPT2.ipynb`** — Main notebook for preprocessing, training, and text generation.
* **`requirements`** — Dependencies.
* **`README.md`** — Project documentation.

## 🛠 Installation

Clone the repository:

```bash
git clone https://github.com/8Whoknow3/Text-Generator-with-GPT2.git
cd Text-Generator-with-GPT2
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### 1. Open Notebook

If running locally:

```bash
jupyter notebook TG-GPT2.ipynb
```

If using Google Colab:

* Upload the notebook to Colab.
* Mount Google Drive if storing data there.

### 2. Configure Parameters

In the notebook:

* **Model**: Choose from `"gpt2"`, `"gpt2-medium"`, etc.
* **Training Hyperparameters**: batch size, learning rate, epochs.
* **Max sequence length**: Adjust for your dataset.

### 3. Run Fine-Tuning

The notebook will:

1. Load the tokenizer and model.
2. Tokenize your dataset.
3. Train the model.
4. Save the model to the specified directory.

### 4. Generate Text

After training, run the **Text Generation** section in the notebook:

```python
from transformers import pipeline

generator = pipeline("text-generation", model="path/to/saved/model")
print(generator("Your prompt here", max_length=100, num_return_sequences=1))
```

## 💾 Saving & Loading Models

**Saving:**

```python
model.save_pretrained("model/")
tokenizer.save_pretrained("model/")
```

**Loading:**

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
model = GPT2LMHeadModel.from_pretrained("model/")
tokenizer = GPT2Tokenizer.from_pretrained("model/")
```

## ⚡ Tips for Better Results

* Use more training data for better generalization.
* Adjust learning rate (`5e-5` to `1e-4` often works well).
* Experiment with `gpt2-medium` or `gpt2-large` if you have enough GPU memory.
* Use **Google Colab Pro** or a local GPU for faster training.

## 🙌 Acknowledgments

* [Hugging Face Transformers](https://github.com/huggingface/transformers)
* [PyTorch](https://pytorch.org/)
