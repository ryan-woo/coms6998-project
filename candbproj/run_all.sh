#!/bin/bash

echo "Running gpt2_trained.py"
python gpt2_trained.py

echo "Running gpt2_varied_embeddings.py"
python gpt2_varied_embeddings.py

echo "Running gpt2_varied_attn_heads.py"
python gpt2_varied_attn_heads.py

echo "Running gpt2_random_init_variations.py"
python gpt2_random_init_variations.py

echo "Running char_tokenizer.py"
python char_tokenizer.py