# Mirza Ghalib Language Model

**Welcome to the Mirza Ghalib Language Model, an Artistry by Aashish Kant**

Embark on a mesmerizing journey through the exquisite world of Mirza Ghalib's poetry, reimagined and brought to life by the advanced Bigram Language Model crafted by Aashish Kant. Immerse yourself in the timeless elegance of Ghalib's verses, where love, loss, and existential contemplation intertwine in poetic harmony.

## 📜 Dataset

Dive into the poetic brilliance of Mirza Ghalib with our meticulously curated dataset sourced from the "mirza.txt" file. This comprehensive collection encapsulates a rich array of Ghalib's ghazals and poetry, providing the fertile ground for the model's creative expressions.

## 🚀 Model Architecture

Powered by PyTorch, our Bigram Language Model boasts a sophisticated architecture designed by Aashish Kant. With a dynamic Embedding layer, a GRU (Gated Recurrent Unit) layer for nuanced context understanding, and a judicious dropout layer for regularization, the model aims to capture the essence of Ghalib's poetic cadence by predicting subsequent characters in a sequence.

## 🎓 Training

Embark on the poetic journey orchestrated by Aashish Kant. Train the model effortlessly using the provided Python script, assimilating data from "mirza.txt," and witness its prowess through insightful evaluations on a validation set during training.

```bash
python train_language_model.py
```

## ✨ Model Generation

After training, witness the magic as the model gracefully composes new text in Ghalib's style. Use the provided code snippet to generate 500 characters of enchanting poetry and experiment with hyperparameters to unleash a diverse range of captivating text.

```python
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_indices = model.generate(context, max_new_tokens=500)[0].tolist()
generated_chars = ''.join([index_to_char[idx] for idx in generated_indices])
print(generated_chars)
```

## 🌟 Results

Experience the project's success, spearheaded by Aashish Kant, and track the model's evolving prowess through periodic evaluations. Delve into the nuances of optimization by adjusting key hyperparameters such as `block_size` and `batch_size` for optimal results.

## 🙏 Acknowledgments

This project is a heartfelt ode to the enduring literary legacy of Mirza Ghalib, masterfully brought to life by Aashish Kant. Special gratitude extends to the dataset source and the vibrant PyTorch community, instrumental in providing cutting-edge tools for unraveling the beauty of natural language processing.

## License
This project operates under the MIT License, offering freedom and flexibility. Refer to the [LICENSE.txt](LICENSE.txt) file for comprehensive details.

Elevate your understanding of Mirza Ghalib's poetic universe, where tradition meets innovation in a symphony of words. 🌌✨
