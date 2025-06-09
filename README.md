# Kiswahili EdgeVoice Challenge 2025 - Starter Notebooks

> **Tiny-Lugha: Efficient Kiswahili AI Speech Stack Challenge**
> 
> Build lightweight STT + LLM + TTS pipeline optimized for NVIDIA T4 GPU (≤16GB)

## Challenge Overview

This repository provides starter notebooks for the ITU AI/ML Kiswahili EdgeVoice Challenge. The goal is to create an end-to-end voice AI system that:

1. **Speech-to-Text (STT)**: Transcribe Kiswahili audio with low Word Error Rate (WER)
2. **Language Understanding**: Process text through Pawa-Gemma-Swahili-2B
3. **Text-to-Speech (TTS)**: Generate natural Kiswahili speech with high Mean Opinion Score (MOS)

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

### Text-to-Speech  
- **Source**: [Mendeley TTS Dataset](https://data.mendeley.com/datasets/vbvj6j6pm9/1)
- **Size**: 7K utterance pairs
- **License**: CC-BY 4.0

### Language Model
- **Model**: [Pawa-Gemma-Swahili-2B](https://huggingface.co/sartifyllc/Pawa-Gemma-Swahili-2B)
- **Parameters**: 2B
- **License**: Apache 2.0

##  Model Architecture

### Speech-to-Text
- **Base Model**: OpenAI Whisper-small
- **Optimization**: Fine-tuned on Kiswahili data
- **Target WER**: <15%

### Text-to-Speech
- **Base Model**: Microsoft SpeechT5
- **Optimization**: Fine-tuned with Kiswahili speaker embeddings
- **Target MOS**: >3.5

### Integration
- **Memory Budget**: ≤16GB GPU memory
- **Real-Time Factor**: <1.0 for STT/TTS
- **End-to-end Latency**: <3 seconds

##  Evaluation Criteria

| Component | Metric | Weight | Target |
|-----------|---------|---------|---------|
| **STT** | Word Error Rate (WER) | 35% | <15% |
| **STT** | Real-Time Factor | 5% | <1.0 |
| **TTS** | Mean Opinion Score (MOS) | 35% | >3.5 |
| **TTS** | Mel-Cepstral Distortion | 5% | Lower is better |
| **System** | Peak GPU Memory | 10% | ≤16GB |
| **Documentation** | Technical Report | 10% | Clear & reproducible |

##  Key Features

### Resource Monitoring
```python
import torch

def check_gpu_memory():
    if torch.cuda.is_available():
        memory_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory: {memory_gb:.2f}GB / 16GB")
        return memory_gb
    return 0
```

### Real-Time Performance
```python
import time

def measure_rtf(audio_duration, inference_time):
    rtf = inference_time / audio_duration
    print(f"Real-Time Factor: {rtf:.3f}")
    return rtf
```

### End-to-End Pipeline
```python
# Complete voice assistant pipeline
def voice_assistant(audio_input):
    # STT: Audio → Text
    transcript = stt_model.transcribe(audio_input)
    
    # LLM: Text → Response
    response = llm_model.generate(transcript)
    
    # TTS: Text → Audio
    audio_output = tts_model.synthesize(response)
    
    return audio_output, transcript, response
```

##  Submission Requirements

Your final submission should include:

1. **Code Repository**: All training and inference code
2. **Model Weights**: Fine-tuned checkpoints under 16GB total
3. **Technical Report**: 4-page PDF with methodology and results
4. **Demo Video**: Working prototype demonstration
5. **Performance Metrics**: WER, MOS, RTF, memory usage

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