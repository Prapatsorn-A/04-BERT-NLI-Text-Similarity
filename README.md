# Text Similarity and Natural Language Inference (NLI) with BERT and Sentence-BERT

This project focuses on training a BERT model from scratch, fine-tuning it for sentence embeddings using a Siamese network architecture (Sentence-BERT), and developing a web application to demonstrate Natural Language Inference (NLI) predictions.

## Project Overview

**1. Training BERT from Scratch:** Implementing and training a BERT model from scratch using a Masked Language Modeling (MLM) objective.

**2. Sentence-BERT:** Fine-tuning the trained BERT model with a Siamese network architecture to generate semantically meaningful sentence embeddings.

**3. Evaluation:** Evaluating the model on the SNLI/MNLI datasets for Natural Language Inference (NLI) tasks.

**4. Web Application:** Developing a Flask-based web application to demonstrate Natural Language Inference (NLI) predictions.

## Datasets

**1. Training BERT:**
- **Dataset:** BookCorpus
- **Source:** [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus)

**2. Fine-Tuning Sentence-BERT:**
- **Dataset:** SNLI and MNLI
- **Source:** [SNLI (Stanford Natural Language Inference)](https://huggingface.co/datasets/stanfordnlp/snli), [MNLI (Multi-Genre Natural Language Inference)](https://huggingface.co/datasets/nyu-mll/glue)

## Tasks

### Task 1: Training BERT from Scratch

**1. Objective:** Implement and train a BERT model from scratch using a Masked Language Modeling (MLM) objective.

**2. Dataset:** The [BookCorpus](https://huggingface.co/datasets/bookcorpus/bookcorpus) dataset was used for training. A subset of 100,000 sentences was used for this task.

**3. Implementation:**

- **Tokenization:** The text was preprocessed by converting it to lowercase and removing punctuation. A vocabulary (`word2id`) was created from the unique words in the dataset.

- **BERT Architecture:** The BERT model was implemented from scratch, including:
  - **Embedding Layer:** Combines token, positional, and segment embeddings.
  - **Multi-Head Attention:** Implements self-attention with multiple heads.
  - **Feed-Forward Network:** A two-layer feed-forward network with GELU activation.
  - **Layer Normalization:** Applied after each sub-layer.

- **Training:** 
  - The model was trained using the MLM objective, where 15% of the tokens were masked, and the model was tasked with predicting the masked tokens.
  - The Next Sentence Prediction (NSP) task was also implemented to predict whether two sentences are consecutive.
  - The model was trained for 1000 epochs using the Adam optimizer with a learning rate of 0.001.

**4. Output:** The trained model weights were saved as `bert_model.pth` for use in Task 2.

### Task 2: Sentence Embedding with Sentence-BERT

**1. Objective:** Fine-tune the trained BERT model using a Siamese network architecture to generate sentence embeddings.

**2. Dataset:** The [SNLI (Stanford Natural Language Inference)](https://huggingface.co/datasets/stanfordnlp/snli) and [MNLI (Multi-Genre Natural Language Inference)](https://huggingface.co/datasets/nyu-mll/glue) datasets were used for fine-tuning.

**3. Implementation:**

- **Data Preprocessing:**
  - The SNLI and MNLI datasets were loaded and combined.
  - Invalid labels (e.g., `-1`) were filtered out.
  - The datasets were tokenized using the `bert-base-uncased` tokenizer.

- **Model Architecture:**
  - A Siamese network structure was implemented, where two BERT models share weights.
  - The model was trained using a classification objective function:
 $$ 
 o = softmax(W^T ⋅ (u, v, |u-v|))
 $$
  where $u$ and $v$ are sentence embeddings, and $W$ is a learnable weight matrix.

- **Training:**
  - The model was trained for 5 epochs using the Adam optimizer with a learning rate of $5 \times 10^{-5}$.
  - A linear learning rate scheduler with warmup was used.

**4. Output:** 

The following files were saved for use in the web application:
- `best_model.pth`: Fine-tuned BERT model weights.
- `classifier_head.pth`: Classifier head weights.
- `tokenizer/`: Tokenizer files (`special_tokens_map.json`, `tokenizer_config.json`, `vocab.txt`).

### Task 3: Evaluation and Analysis

**1. Objective:** Evaluate the model's performance on the SNLI and MNLI datasets for the NLI task.

**2. Metrics:** 
- **Accuracy:** The model achieved an accuracy of **34.2%** on the validation set.
- **Cosine Similarity:** The average cosine similarity between sentence embeddings was **0.9989**.

**3. Challenges:**
- **Limited Computational Resources:**
  - Training BERT from scratch and fine-tuning Sentence-BERT require significant computational power.
  - Long training times due to the large number of parameters in the BERT model.

- **Overfitting on Smaller Datasets:**
  - The model may overfit when trained on smaller datasets like SNLI, leading to poor generalization on unseen data.
  - Limited diversity in smaller datasets can restrict the model's ability to learn robust representations.

- **Data Imbalance:**
  - The dataset may have imbalanced classes (e.g., more "entailment" examples than "contradiction"), which can bias the model's predictions.

- **Complexity of Sentence Structures:**
  - The model may struggle with complex sentence structures, such as long sentences, nested clauses, or ambiguous phrasing.
  - Handling negation, sarcasm, or idiomatic expressions can be challenging.

- **Hyperparameter Tuning:**
  - Finding the optimal hyperparameters (e.g., learning rate, batch size, number of layers) can be time-consuming and computationally expensive.
  - Poorly chosen hyperparameters can lead to suboptimal model performance.

**4. Improvements:**
- **Use Larger Datasets:**
  - Train on larger datasets like MNLI, which contains more diverse examples across multiple genres, improving generalization.
  - Combine multiple datasets to increase the variety of training data.

- **Data Augmentation:**
  - Apply data augmentation techniques such as back-translation (translating sentences to another language and back) to generate additional training examples.
  - Use synonym replacement or paraphrasing to create variations of existing sentences.

- **Regularization Techniques:**
  - Implement dropout and weight decay to prevent overfitting.
  - Use early stopping during training if the model stops getting better on the validation set.

- **Transfer Learning:**
  - Start with a pre-trained BERT model (e.g., `bert-base-uncased`) instead of training from scratch to save time and computational resources.
  - Fine-tune the pre-trained model on the specific task (e.g., NLI) to leverage its learned representations.

- **Advanced Architectures:**
  - Experiment with more advanced architectures like RoBERTa, ALBERT, or DistilBERT, which are optimized for better performance and efficiency.
  - Use ensemble methods to combine predictions from multiple models for improved accuracy.

### Task 4: Web Application Development

**1. Objective:** to demonstrate Natural Language Inference (NLI) predictions

**2. Implementation:**
-  A **Flask-based** web application was developed with two input boxes for premise and hypothesis.
- The custom-trained **Sentence-BERT model** was used to predict the NLI label (entailment, neutral, contradiction).

**3. Web Application Sample Screenshot**

![](images/web-sample.png)

## Installation

**1. Clone the Repository:** Clone the repository to your local machine.
```bash
git clone https://github.com/Prapatsorn-A/04-BERT-NLI-Text-Similarity.git
cd 04-BERT-NLI-Text-Similarity
```

**2. Install Dependencies:** Install the dependencies listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

**3. Run the Flask App:**
```bash
python app.py
```

**4. Access the Web Application:**
- Open your browser and go to `http://127.0.0.1:5000`.
- Enter a premise and hypothesis to get the NLI label.

## Acknowledgements

This notebook is based on the work of **Professor Chaklam Silpasuwanchai** and Teaching Assistant **Todsavad Tangtortan**, specifically the **'BERT-update.ipynb'** and **'S-BERT.ipynb'** notebooks. Their foundational structure and ideas played a key role in implementing the model. Grateful appreciation is extended to both Professor Chaklam Silpasuwanchai and Teaching Assistant Todsavad Tangtortan for providing these invaluable resources.

**Link to the notebooks:** 
- [BERT-update.ipynb](https://github.com/chaklam-silpasuwanchai/Python-fo-Natural-Language-Processing/blob/f19297bca9731337cab873e072c1d04f1588587d/Code/02%20-%20DL/04%20-%20Masked%20Language%20Model/BERT-update.ipynb#L4)
- [S-BERT.ipynb](https://github.com/chaklam-silpasuwanchai/Python-fo-Natural-Language-Processing/blob/main/Code/04%20-%20Huggingface/Appendix%20-%20Sentence%20Embedding/S-BERT.ipynb)