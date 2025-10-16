# AgriBot - Agricultural Domain Chatbot Using Transformer Models

## Project Overview

AgriBot is an AI-powered chatbot specialized in agriculture, designed to assist farmers and agricultural professionals with queries about crop management, pest control, soil management, and farming practices. The chatbot leverages the power of Transformer models, specifically the T5 (Text-to-Text Transfer Transformer) model, fine-tuned on the AgriQA dataset.

## Domain: Agriculture

Agriculture is a critical sector where farmers often need quick access to expert advice. AgriBot provides:

- Crop management guidance
- Pest and disease control recommendations
- Fertilizer application advice
- Soil management tips
- Irrigation best practices
- General agricultural problem-solving

## Dataset

**Dataset Name:** AgriQA  
**Source:** [Hugging Face - shchoi83/agriQA](https://huggingface.co/datasets/shchoi83/agriQA)

The dataset contains real agricultural questions and expert answers, structured as:

- **Questions:** Queries about farming challenges (e.g., "How to control aphid infestation in mustard crops?")
- **Answers:** Expert recommendations (e.g., "Spray rogor@2ml/lit at evening time")

### Dataset Statistics

- High-quality question-answer pairs
- Diverse agricultural topics
- Real-world farming scenarios

## Model Architecture

### Pre-trained Model: T5-small

- **Framework:** TensorFlow with Hugging Face Transformers
- **Approach:** Generative Question Answering
- **Tokenization:** T5 Tokenizer with max length of 128 tokens

### Why T5?

T5 is a text-to-text transformer that treats all NLP tasks as text generation, making it ideal for generative QA systems. It can understand context and generate coherent, relevant responses.

## Data Preprocessing

### Steps Performed:

1. **Text Cleaning**

   - Removal of extra whitespace
   - Text normalization
   - Handling of missing values

2. **Tokenization**

   - Using T5 tokenizer
   - Max input length: 128 tokens
   - Max target length: 128 tokens
   - Padding and truncation applied

3. **Data Splitting**
   - Training set: 85%
   - Validation set: 15%

## Hyperparameter Tuning

### Experiment 1 (Baseline)

- **Learning Rate:** 5e-5
- **Batch Size:** 16
- **Epochs:** 3
- **Optimizer:** AdamW with warmup

### Experiment 2 (Optimized)

- **Learning Rate:** 3e-5
- **Batch Size:** 8
- **Epochs:** 5
- **Optimizer:** AdamW with warmup

### Results

Experiment 2 showed improved performance with:

- Lower final validation loss
- Better convergence
- More coherent generated responses
- Performance improvement > 10% over baseline

## Performance Metrics

### Quantitative Evaluation

1. **BLEU Score:** Measures n-gram overlap between generated and reference answers
2. **ROUGE-1:** Unigram-based evaluation
3. **ROUGE-2:** Bigram-based evaluation
4. **ROUGE-L:** Longest common subsequence evaluation

### Evaluation Summary

| Metric  | Score |
| ------- | ----- |
| BLEU    | 17.09 |
| ROUGE-1 | 42.14 |
| ROUGE-2 | 25.36 |
| ROUGE-L | 41.36 |



### Qualitative Testing

- Tested with in-domain agricultural questions
- Tested with out-of-domain questions to assess domain specificity
- Analyzed response relevance and coherence

## Deployment

### Gradio Web Interface

The chatbot is deployed using Gradio, providing:

- **Intuitive UI:** Easy-to-use web interface
- **Real-time Responses:** Instant answers to user queries
- **Example Questions:** Pre-loaded examples for guidance
- **Shareable Link:** Accessible via public URL

### Features

- Natural language input
- Context-aware responses
- Domain-specific guidance
- User-friendly design

## Installation and Setup

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
CUDA (optional, for GPU support)
```

### Install Dependencies

```bash
pip install transformers datasets tensorflow pandas numpy scikit-learn gradio sacrebleu rouge-score nltk matplotlib seaborn
```

### Download NLTK Data

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

## Running the Chatbot

### Option 1: Run Jupyter Notebook

1. Open `notebook.ipynb` in Jupyter or VS Code
2. Execute all cells sequentially
3. The Gradio interface will launch automatically

### Option 2: Run Python Script

```bash
python chatbot_app.py
```

### Option 3: Load Pre-trained Model

```python
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

model = TFAutoModelForSeq2SeqLM.from_pretrained("./agribot_model_exp2")
tokenizer = AutoTokenizer.from_pretrained("./agribot_model_exp2")

# Generate answer
question = "How to control pests in wheat crops?"
inputs = tokenizer(question, return_tensors='tf', max_length=128, truncation=True)
outputs = model.generate(inputs['input_ids'], max_length=128, num_beams=4)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

## Project Structure

```
SproutBot/
│
├── notebook.ipynb              # Main Jupyter notebook with complete workflow
├── README.md                   # This file
├── agribot_model_exp1/         # Trained model - Experiment 1
├── agribot_model_exp2/         # Trained model - Experiment 2 (Best)
├── requirements.txt            # Python dependencies
└── demo_video.mp4             # Demo video (5-10 minutes)
```

## Example Conversations

### Example 1: Pest Control

**User:** "How do I control aphid infestation in mustard crops?"  
**AgriBot:** "Apply systemic insecticide like rogor at 2ml per liter during evening time. Ensure thorough coverage of plant foliage."

### Example 2: Fertilizer Application

**User:** "What is the best fertilizer for wheat crops?"  
**AgriBot:** "Apply NPK fertilizer with nitrogen at the time of sowing, followed by top dressing with urea during tillering stage."

### Example 3: Disease Management

**User:** "How to manage fungal disease in tomato plants?"  
**AgriBot:** "Spray copper-based fungicide or mancozeb at recommended doses. Ensure proper spacing and avoid overhead irrigation."

## Code Quality

### Best Practices Followed

- Clean, well-documented code
- Modular functions with clear purposes
- Meaningful variable and function names
- Comprehensive comments explaining logic
- Error handling and validation
- Type hints where applicable

## Evaluation Results

### Model Performance

- Training loss convergence achieved
- Validation metrics show generalization
- BLEU and ROUGE scores indicate quality
- Qualitative testing confirms domain relevance

### Strengths

- Generates domain-specific responses
- Understands agricultural terminology
- Provides actionable advice
- Fast inference time

### Limitations

- Limited to training data knowledge
- May require more diverse dataset for edge cases
- Response quality depends on question clarity

## Future Enhancements

1. **Dataset Expansion:** Include more agricultural scenarios and crops
2. **Multilingual Support:** Add support for regional languages
3. **RAG Integration:** Combine with retrieval systems for more accurate answers
4. **Larger Models:** Fine-tune T5-base or T5-large for better performance
5. **Mobile App:** Develop native mobile application
6. **Voice Interface:** Add speech-to-text and text-to-speech capabilities
7. **Image Support:** Allow users to upload crop images for diagnosis

## Technologies Used

- **Python 3.x**
- **TensorFlow 2.x**
- **Hugging Face Transformers**
- **Gradio**
- **Pandas & NumPy**
- **Matplotlib & Seaborn**
- **NLTK**
- **SacreBLEU & ROUGE**

## Contributors

- **Developer:** Alain Michael Muhirwa
- **Course:** Machine Learning Techniques I
- **Institution:** ALU
- **Date:** October 2025

## License

This project is for educational purposes as part of a university course assignment.

## Acknowledgments

- Hugging Face for the Transformers library and model hosting
- Dataset creators of AgriQA
- TensorFlow team for the deep learning framework
- Gradio for the web interface framework

## Contact

For questions or feedback about this project, please contact:
a.muhirwa@alustudent.com

---

**Note:** This chatbot is designed for educational and informational purposes. For critical agricultural decisions, always consult with professional agronomists or agricultural extension services.
