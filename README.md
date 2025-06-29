# Kiswahili EdgeVoice Challenge 2025() - Starter Notebooks

![image alt](https://github.com/Sartify/swahili-challenge-competition/blob/main/image.jpeg)

> **Tiny-Lugha: Efficient Kiswahili AI Speech Stack Challenge**
> 
> Build lightweight STT + LLM + TTS pipeline optimized for NVIDIA T4 GPU (≤16GB)

## Challenge Overview

This repository provides starter notebooks for the ITU AI/ML Kiswahili EdgeVoice Challenge. The goal is to create voice to text AI system that:

1. **Speech-to-Text (STT)**: Transcribe Kiswahili audio with low Word Error Rate (WER)
2. **Language Understanding**: Process text (translate from Swahili to English) through Pawa-Gemma-Swahili-2B

**All running on a single NVIDIA T4 GPU within Google Colab's free tier!**

```
## Quick Start

```

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/kiswahili-edgevoice-starter/blob/main/notebooks/01_stt_training.ipynb)

### 2. Run the Notebooks in Order

1. **STT Training** (`asr_starter.ipynb`)
2. **TTS Training** (`tts_starter_notebook.ipynb`) 


##  Dataset Information

### Speech-to-Text
- **Source**: [Mozilla Common Voice 17.0 (Swahili)](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
- **Size**: 100+ hours of labeled Kiswahili speech
- **License**: CC-0 1.0

### Language Model
- **Model**: [Pawa-Gemma-Swahili-2B](https://huggingface.co/sartifyllc/Pawa-Gemma-Swahili-2B)
- **Parameters**: 2B
- **License**: Apache 2.0

##  Model Architecture

### Speech-to-Text
- **Base Model**: OpenAI Whisper-small
- **Optimization**: Fine-tuned on Kiswahili data
- **Target WER**: less the better

### Integration
- **Memory Budget**: ≤16GB GPU memory
- **Real-Time Factor**: <1.0 for STT/TTS
- **End-to-end Latency**: <3 seconds

##  Evaluation Criteria

| Component | Metric | Weight | Target |
|-----------|---------|---------|---------|
| **STT** | Word Error Rate (WER) | 85% | <15% |
| **STT** | Real-Time Factor | 5% | <1.0 |
| **System** | Peak GPU Memory | 5% | ≤16GB |
| **Documentation** | Technical Report | 5% | Clear & reproducible |

##  Key Features

### Word Error Rate 
```python
import re
from typing import List, Tuple, Union
import numpy as np

def calculate_wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis strings.
    
    WER = (S + D + I) / N
    Where:
    - S = number of substitutions
    - D = number of deletions  
    - I = number of insertions
    - N = number of words in reference
    
    Args:
        reference (str): Ground truth text
        hypothesis (str): Predicted/transcribed text
        normalize (bool): Whether to normalize text (lowercase, remove punctuation)
    
    Returns:
        float: Word Error Rate (0.0 = perfect match, higher = more errors)
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Get edit distance and operations
    distance, operations = levenshtein_distance_with_operations(ref_words, hyp_words)
    
    # Count operations
    substitutions = operations.count('S')
    deletions = operations.count('D')
    insertions = operations.count('I')
    
    # Calculate WER
    n_ref_words = len(ref_words)
    if n_ref_words == 0:
        return 0.0 if len(hyp_words) == 0 else float('inf')
    
    wer = (substitutions + deletions + insertions) / n_ref_words
    return wer

def calculate_wer_detailed(reference: str, hypothesis: str, normalize: bool = True) -> dict:
    """
    Calculate detailed WER metrics with breakdown of error types.
    
    Returns:
        dict: Contains WER, error counts, and additional metrics
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Get edit distance and operations
    distance, operations = levenshtein_distance_with_operations(ref_words, hyp_words)
    
    # Count operations
    substitutions = operations.count('S')
    deletions = operations.count('D')
    insertions = operations.count('I')
    matches = operations.count('M')
    
    n_ref_words = len(ref_words)
    n_hyp_words = len(hyp_words)
    
    # Calculate metrics
    wer = (substitutions + deletions + insertions) / n_ref_words if n_ref_words > 0 else 0.0
    accuracy = matches / n_ref_words if n_ref_words > 0 else 0.0
    
    return {
        'wer': wer,
        'accuracy': accuracy,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'matches': matches,
        'reference_words': n_ref_words,
        'hypothesis_words': n_hyp_words,
        'edit_distance': distance
    }

def levenshtein_distance_with_operations(ref_words: List[str], hyp_words: List[str]) -> Tuple[int, List[str]]:
    """
    Calculate Levenshtein distance with operation tracking.
    
    Returns:
        Tuple[int, List[str]]: (edit_distance, operations_list)
        Operations: 'M' = match, 'S' = substitution, 'D' = deletion, 'I' = insertion
    """
    n, m = len(ref_words), len(hyp_words)
    
    # Create distance matrix
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]  # Match
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Deletion
                    dp[i][j-1],      # Insertion
                    dp[i-1][j-1]     # Substitution
                )
    
    # Backtrack to find operations
    operations = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            operations.append('M')  # Match
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            operations.append('S')  # Substitution
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            operations.append('D')  # Deletion
            i -= 1
        else:
            operations.append('I')  # Insertion
            j -= 1
    
    operations.reverse()
    return dp[n][m], operations

def normalize_text(text: str) -> str:
    """
    Normalize text for WER calculation.
    - Convert to lowercase
    - Remove punctuation
    - Normalize whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (keep only letters, numbers, spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_batch_wer(references: List[str], hypotheses: List[str], normalize: bool = True) -> float:
    """
    Calculate WER for a batch of reference-hypothesis pairs.
    
    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        normalize: Whether to normalize texts
    
    Returns:
        float: Average WER across all pairs
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    
    total_errors = 0
    total_words = 0
    
    for ref, hyp in zip(references, hypotheses):
        if normalize:
            ref = normalize_text(ref)
            hyp = normalize_text(hyp)
        
        ref_words = ref.split()
        hyp_words = hyp.split()
        
        distance, _ = levenshtein_distance_with_operations(ref_words, hyp_words)
        
        total_errors += distance
        total_words += len(ref_words)
    
    return 100*total_errors / total_words if total_words > 0 else 0.0

# Example usage and testing
if __name__ == "__main__":
    # Test examples
    reference = "Mheshimiwa samia suluhu hassan ni Rais wa Jamhuri ya Muungano wa Tanzania "
    hypothesis1 = "Mheshimiwa samia suluhu hassan ni Rais wa Jamhuri ya Muungano wa Tanzania "  # Perfect match
    hypothesis2 = "Mheshimiwa samia suluhu hassan ni Rai wa Jamhuri ya Muungano wa Tanzania "  # 1 substitution
    hypothesis3 = "Mheshimiwa samia suluhu hassan ni  wa Jamhuri ya Muungano wa Tanzania "  # 1 deletion
    hypothesis4 = "Mheshimiwa samia suluhu hassan ni Rais wa Jamhuri ya Muungano wa Tanzania Nzima"  # 1 insertion
    
    print("WER Examples:")
    print(f"Perfect match: {calculate_wer(reference, hypothesis1):.3f}")
    print(f"1 substitution: {calculate_wer(reference, hypothesis2):.3f}")
    print(f"1 deletion: {calculate_wer(reference, hypothesis3):.3f}")
    print(f"1 insertion: {calculate_wer(reference, hypothesis4):.3f}")
    
    print("\nDetailed WER for substitution example:")
    detailed = calculate_wer_detailed(reference, hypothesis3)
    for key, value in detailed.items():
        print(f"{key}: {value}")
    
    print("\nBatch WER example:")
    refs = [reference, reference]
    hyps = [hypothesis1, hypothesis2]
    batch_wer = calculate_batch_wer(refs, hyps)
    print(f"Batch WER: {batch_wer:.3f}")
```

### Running Pawa HF API
```python
!pip install --upgrade "transformers>=4.52.0" "accelerate" 
!pip install -U nemo_toolkit["asr"]
!pip install -U datasets


from datasets import load_dataset, Features, Value, Audio
import soundfile as sf
import re
from typing import List, Tuple, Union
import numpy as np
import nemo.collections.asr as nemo_asr
from transformers import AutoModelForCausalLM, AutoTokenizer
import nemo.collections.asr as nemo_asr
import torch

def calculate_wer(reference: str, hypothesis: str, normalize: bool = True) -> float:
    """
    Calculate Word Error Rate (WER) between reference and hypothesis strings.
    
    WER = (S + D + I) / N
    Where:
    - S = number of substitutions
    - D = number of deletions  
    - I = number of insertions
    - N = number of words in reference
    
    Args:
        reference (str): Ground truth text
        hypothesis (str): Predicted/transcribed text
        normalize (bool): Whether to normalize text (lowercase, remove punctuation)
    
    Returns:
        float: Word Error Rate (0.0 = perfect match, higher = more errors)
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Get edit distance and operations
    distance, operations = levenshtein_distance_with_operations(ref_words, hyp_words)
    
    # Count operations
    substitutions = operations.count('S')
    deletions = operations.count('D')
    insertions = operations.count('I')
    
    # Calculate WER
    n_ref_words = len(ref_words)
    if n_ref_words == 0:
        return 0.0 if len(hyp_words) == 0 else float('inf')
    
    wer = (substitutions + deletions + insertions) / n_ref_words
    return wer

def calculate_wer_detailed(reference: str, hypothesis: str, normalize: bool = True) -> dict:
    """
    Calculate detailed WER metrics with breakdown of error types.
    
    Returns:
        dict: Contains WER, error counts, and additional metrics
    """
    if normalize:
        reference = normalize_text(reference)
        hypothesis = normalize_text(hypothesis)
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Get edit distance and operations
    distance, operations = levenshtein_distance_with_operations(ref_words, hyp_words)
    
    # Count operations
    substitutions = operations.count('S')
    deletions = operations.count('D')
    insertions = operations.count('I')
    matches = operations.count('M')
    
    n_ref_words = len(ref_words)
    n_hyp_words = len(hyp_words)
    
    # Calculate metrics
    wer = (substitutions + deletions + insertions) / n_ref_words if n_ref_words > 0 else 0.0
    accuracy = matches / n_ref_words if n_ref_words > 0 else 0.0
    
    return {
        'wer': wer,
        'accuracy': accuracy,
        'substitutions': substitutions,
        'deletions': deletions,
        'insertions': insertions,
        'matches': matches,
        'reference_words': n_ref_words,
        'hypothesis_words': n_hyp_words,
        'edit_distance': distance
    }

def levenshtein_distance_with_operations(ref_words: List[str], hyp_words: List[str]) -> Tuple[int, List[str]]:
    """
    Calculate Levenshtein distance with operation tracking.
    
    Returns:
        Tuple[int, List[str]]: (edit_distance, operations_list)
        Operations: 'M' = match, 'S' = substitution, 'D' = deletion, 'I' = insertion
    """
    n, m = len(ref_words), len(hyp_words)
    
    # Create distance matrix
    dp = np.zeros((n + 1, m + 1), dtype=int)
    
    # Initialize first row and column
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    
    # Fill the matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]  # Match
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # Deletion
                    dp[i][j-1],      # Insertion
                    dp[i-1][j-1]     # Substitution
                )
    
    # Backtrack to find operations
    operations = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            operations.append('M')  # Match
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            operations.append('S')  # Substitution
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            operations.append('D')  # Deletion
            i -= 1
        else:
            operations.append('I')  # Insertion
            j -= 1
    
    operations.reverse()
    return dp[n][m], operations

def normalize_text(text: str) -> str:
    """
    Normalize text for WER calculation.
    - Convert to lowercase
    - Remove punctuation
    - Normalize whitespace
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation (keep only letters, numbers, spaces)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_batch_wer(references: List[str], hypotheses: List[str], normalize: bool = True) -> float:
    """
    Calculate WER for a batch of reference-hypothesis pairs.
    
    Args:
        references: List of reference texts
        hypotheses: List of hypothesis texts
        normalize: Whether to normalize texts
    
    Returns:
        float: Average WER across all pairs
    """
    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")
    
    total_errors = 0
    total_words = 0
    
    for ref, hyp in zip(references, hypotheses):
        if normalize:
            ref = normalize_text(ref)
            hyp = normalize_text(hyp)
        
        ref_words = ref.split()
        hyp_words = hyp.split()
        
        distance, _ = levenshtein_distance_with_operations(ref_words, hyp_words)
        
        total_errors += distance
        total_words += len(ref_words)
    
    return 100*total_errors / total_words if total_words > 0 else 0.0




# 1. Define a schema that matches the *actual* Arrow types in CV-17
cv_sw_features = Features({
    "client_id":      Value("string"),
    "path":           Value("string"),
    "sentence_id":    Value("string"),
    "sentence":       Value("string"),
    "sentence_domain":Value("string"),
    "up_votes":       Value("string"),      # <- string, not int64
    "down_votes":     Value("string"),      # <- string, not int64
    "age":            Value("string"),
    "gender":         Value("string"),
    "variant":        Value("string"),
    "locale":         Value("string"),
    "segment":        Value("string"),
    "accent":         Value("string"),
    # keep audio decoded so we get "array" + "sampling_rate"
    "audio":          Audio(sampling_rate=48_000, mono=True, decode=True),
})

# 2. Stream the train split with that schema
cv_17 = load_dataset(
    "mozilla-foundation/common_voice_17_0",
    "sw",
    split="train",
    streaming=True,
    features=cv_sw_features,   # <- custom schema solves CastError
)

# 3. Grab the first row
first_row = next(iter(cv_17))

# 4. Save the audio clip
audio_array = first_row["audio"]["array"]
sr          = first_row["audio"]["sampling_rate"]
sf.write("first_row_audio.wav", audio_array, sr)
print("Saved:", first_row["sentence"])
print("→ first_row_audio.wav  |  shape:", audio_array.shape, "  sr:", sr)



# ---- Step 1: Capture initial memory stats ----
gpu_stats = torch.cuda.get_device_properties(0)
max_memory = round(gpu_stats.total_memory / 1024**3, 3)
start_reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 3)

print(f"🖥️ GPU: {gpu_stats.name}")
print(f"📊 Max memory: {max_memory} GB")
print(f"🔹 Reserved before Inference: {start_reserved} GB")

asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

# Model configuration
model_name = "sartifyllc/pawa-min-alpha"
max_seq_length = 2048

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use float16 for efficiency
    device_map="auto",  # Automatically handle device placement
    trust_remote_code=True  # In case the model requires custom code
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Configure chat template (ChatML format)
def apply_chatml_template(messages):
    """Apply ChatML template to conversation messages"""
    formatted_text = ""
    for message in messages:
        role = message.get("role", message.get("from", ""))
        content = message.get("content", message.get("value", ""))
        
        # Map role names
        if role == "human" or role == "user":
            role = "user"
        elif role == "gpt" or role == "assistant":
            role = "assistant"
        elif role == "system":
            role = "system"
            
        formatted_text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    
    # Add assistant start token for generation
    formatted_text += "<|im_start|>assistant\n"
    return formatted_text

# Example usage function
def generate_response(messages, max_new_tokens=64):
    """Generate response for a conversation"""
    
    # Format the conversation with ChatML template
    if isinstance(messages, str):
        # If single string input, treat as user message
        messages = [{"role": "user", "content": messages}]
    
    formatted_input = apply_chatml_template(messages)
    
    # Tokenize input
    inputs = tokenizer(
        formatted_input,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
        padding=True
    )
    
    # Move to same device as model
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
    
    # Decode and return response
    response = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    return response


# Example usage:
if __name__ == "__main__":
    # Method 1: Manual approach
    output = asr_model.transcribe(['./first_row_audio.wav'])
    message = output[0].text

    refs = [first_row["sentence"], first_row["sentence"]]
    hyps = [message, message]
    batch_wer = calculate_batch_wer(refs, hyps)
    print(f"Batch WER: {batch_wer:.3f}")



    print(message)
    messages = [
        {"role": "user", "content": f"Translate to English from swahili the following world: '{message}'"}
    ]
    
    response = generate_response(messages, max_new_tokens=64)
    print("Generated response:", response)
    

# ---- Step 3: Capture final memory stats ----
end_reserved = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
delta_reserved = round(end_reserved - start_reserved, 3)
percent_used = round(end_reserved / max_memory * 100, 3)
percent_delta = round(delta_reserved / max_memory * 100, 3)

print(f"\n📈 Peak reserved memory after Inference: {end_reserved} GB")
print(f"📉 Additional memory used for Inference: {delta_reserved} GB")
print(f"💯 Total memory used (%): {percent_used} %")
print(f"🧠 Inference memory usage (%): {percent_delta} %")
```


### Real-Time Performance
```python
import time

def measure_rtf(audio_duration, inference_time):
    rtfx = audio_duration / inference_time 
    print(f"Real-Time Factor: {rtfx:.3f}")
    return rtfx
```

### End-to-End Pipeline
```python
# Complete voice assistant pipeline
def voice_assistant(audio_input):
    # STT: Audio → Text
    transcript = stt_model.transcribe(audio_input)
    
    # LLM: Text → Response
    response = llm_model.generate(transcript)

    return transcript, response
```

##  Submission Requirements

Your final submission should include:

1. **Code Repository**: All training and inference code
2. **Model Weights**: Fine-tuned checkpoints under 10GB total
3. **Technical Report**: Max of 4-page PDF with methodology and results
4. **Demo Video**: Working prototype demonstration
5. **Performance Metrics**: WER, RTFx, memory usage

## 🎯 Tips for Success

### Model Optimization
- Use **mixed precision (FP16)** to reduce memory usage
- Apply **gradient checkpointing** during training
- Consider **model pruning** for further compression
- Implement **dynamic batching** for efficient inference

### Performance Tuning
- Profile memory usage with `nvidia-smi`
- Measure inference latency on T4 GPU
- Optimize audio preprocessing pipelines
- Cache commonly used speaker embeddings

### Quality Improvements
- Augment training data with noise/speed variations
- Use **knowledge distillation** from larger models
- Fine-tune with **domain-specific Kiswahili text**
- Implement **beam search** for better STT accuracy

##  Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request with detailed description

##  License

This starter code is released under Apache 2.0 license. Please ensure compliance with dataset licenses:

- Mozilla Common Voice: CC-0 1.0
- TTS Dataset: CC-BY 4.0 (attribution required)
- Pawa-Gemma-Swahili-2B: Apache 2.0

##  Support

- **Competition Questions**: michael.mollel@sartify.com
- **Technical Issues**: Create GitHub issue
- **Documentation**: Check [competition guidelines](https://docs.google.com/document/d/1W-qOtOd4Q9AkAmCOv3SSoRGixhT-5l7UHNmIE35lRDI/edit?usp=sharing)

## 🏆 Leaderboard

Current baselines:
- **STT WER**: 25.3%
- **TTS MOS**: 3.1
- **Memory Usage**: 12.8GB
- **RTF**: 0.8x

**Can you beat these numbers?** 🚀

---

*Good luck building the future of Kiswahili voice technology!* 🎤🇹🇿
