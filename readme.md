# English-to-French Transformer Translator

A custom English-to-French machine translation model built from scratch using a PyTorch Transformer architecture. The model has been exported to ONNX for highly optimized, lightweight local inference.
Trained using google colab on A100 gpu 

It achieves a BLEU score of **~26**, demonstrating strong structural understanding of cross-lingual semantics.

## Architecture & Hosting
* **Architecture:** Sequence-to-Sequence Transformer with Flash Attention and vocabulary sharing.
* **Format:** ONNX (Open Neural Network Exchange).
* **Weights:** Model weights (~330 MB) and the custom SentencePiece tokenizer are hosted externally on Hugging Face at [`abxy12344/en_fr_translator`](https://huggingface.co/abxy12344/en_fr_translator). 

*Note: You do not need to download the weights manually. The inference script will automatically fetch and cache them directly from Hugging Face on the first run.*

## Quickstart

**1. Clone the repository**
```bash
git clone [https://github.com/vyomk18576-droid/en_fr_translator.git]
cd en_fr_translator 
```

**2. Install the dependencies**
```bash
pip install -r requirements.txt
```

**3. Run inference**
```bash
python inference.py --text "Hello, how are you?"
```


## Roadmap
- [x] Export PyTorch Transformer to ONNX format.
- [x] Host model weights on Hugging Face and build local inference pipeline.
- [X] Upload the original Google Colab training source code and dataset preparation scripts.