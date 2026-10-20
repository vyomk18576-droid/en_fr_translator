import onnxruntime as ort
import numpy as np
import sentencepiece as spm
import argparse
from huggingface_hub import hf_hub_download

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def translate_onnx_beam_search(text, session, tokenizer, beam_width=3, max_len=128):
    source_ids = tokenizer.encode(text, out_type=int, add_bos=True, add_eos=True)
    src_input = np.array([source_ids], dtype=np.int64)
    
    beams = [(0.0, [tokenizer.bos_id()])]
    completed_beams = []

    for _ in range(max_len):
        new_beams = []
        for score, seq in beams:
            if seq[-1] == tokenizer.eos_id():
                completed_beams.append((score, seq))
                continue
                
            tgt_input = np.array([seq], dtype=np.int64)
            logits = session.run(["logits"], {"source_ids": src_input, "target_ids": tgt_input})[0]
            
            probs = softmax(logits[0, -1, :])
            log_probs = np.log(probs + 1e-10) 
            
            top_k_indices = np.argsort(log_probs)[-beam_width:]
            
            for token_idx in top_k_indices:
                new_score = score + log_probs[token_idx]
                new_seq = seq + [int(token_idx)]
                new_beams.append((new_score, new_seq))
                
        if not new_beams:
            break
            
        new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_width]

    final_beams = completed_beams + beams
    best_beam = max(final_beams, key=lambda x: x[0])
    best_seq = best_beam[1]

    if tokenizer.eos_id() in best_seq:
        best_seq = best_seq[:best_seq.index(tokenizer.eos_id())]
        
    return tokenizer.decode(best_seq[1:])

def main():
    parser = argparse.ArgumentParser(description="English to French ONNX Translator")
    parser.add_argument("--text", type=str, required=True, help="Text to translate")
    args = parser.parse_args()

    print("Loading model from Hugging Face (will download on first run)...")
    
    repo_id = "abxy12344/en_fr_translator"
    
    tokenizer_path = hf_hub_download(repo_id=repo_id, filename="shared_tokenizer.model")
    onnx_data_path = hf_hub_download(repo_id=repo_id, filename="transformer_translation.onnx.data")
    onnx_path = hf_hub_download(repo_id=repo_id, filename="transformer_translation.onnx")

    tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

    translation = translate_onnx_beam_search(args.text, session, tokenizer, beam_width=3)
    print(f"\nFrench: {translation}")

if __name__ == "__main__":
    main()