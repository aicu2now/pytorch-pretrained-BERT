#!/usr/bin/env python
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

def preprocess_text(text):
      """
          Process the input text using BERT tokenizer.
              """
      # Initialize tokenizer
      tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the input text
      tokens = tokenizer.tokenize(text)

    # Convert tokens to input IDs
      input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Convert to PyTorch tensor
      input_tensor = torch.tensor([input_ids])

    return input_tensor

def extract_features(input_tensor):
      """
          Extract features from the input tensor using BERT model.
              """
      model = BertModel.from_pretrained('bert-base-uncased')
      model.eval()

    # Extract hidden states
      with torch.no_grad():
                hidden_states, _ = model(input_tensor)

      return hidden_states
