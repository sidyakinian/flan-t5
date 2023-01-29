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

To load the pretrained weights (from the paper) into the model, please refer to `load_weights.py`.

### Inference

To run the model, first tokenize the inputs with the default T5 tokenizer from Hugging Face, then call the model, and decode the outputs.

```
tokenizer = T5Tokenizer.from_pretrained("t5-small")
input_sentence = "translate english to french: trees are very green today"
inputs = tokenizer(input_sentence, return_tensors="pt")
transformer_outputs = transformer(inputs.input_ids, debug=True, max_tokens=20)
decoded_outputs = tokenizer.decode(transformer_outputs[0], skip_special_tokens=True)
decoded_outputs
```
