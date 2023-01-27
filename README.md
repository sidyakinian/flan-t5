# FLAN-T5 implementation in PyTorch

FLAN-T5 is a PyTorch implementation of the Fine-tuned Language Analysis Network (FLAN) using the T5 model architecture. FLAN is a multi-task learning model that can be fine-tuned for various natural language understanding tasks, such as sentiment analysis, named entity recognition, and text classification.

### Installation

To install FLAN-T5, clone the repository and install the required packages:

```
git clone https://github.com/[username]/FLAN-T5.git
cd FLAN-T5
pip install -r requirements.txt
```

### Loading weights

TBD

### Inference

To run the model, first tokenize the inputs with the default T5 tokenizer from Hugging Face, then call the model, and decode the outputs.
