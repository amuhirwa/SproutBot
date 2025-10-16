"""
AgriBot - Agricultural Domain Chatbot
A Transformer-based chatbot specialized in agriculture using T5 model
"""

import gradio as gr
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import warnings

warnings.filterwarnings('ignore')

# Model configuration
MODEL_PATH = "./models/agribot_model_exp3"
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 128

# Load model and tokenizer
print("Loading AgriBot model...")
try:
    model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model is trained and saved in './agribot_model_exp2' directory")
    exit(1)


def generate_answer(question, max_length=MAX_OUTPUT_LENGTH):
    """
    Generate answer for a given agricultural question
    
    Args:
        question (str): User's agricultural question
        max_length (int): Maximum length of generated answer
    
    Returns:
        str: Generated answer
    """
    inputs = tokenizer(
        question,
        return_tensors='tf',
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding='max_length'
    )
    
    outputs = model.generate(
        inputs['input_ids'],
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
        no_repeat_ngram_size=2,
        temperature=0.7
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def agribot_chat(question):
    """
    Chatbot function that handles user queries and returns responses
    
    Args:
        question (str): User's question
    
    Returns:
        str: Bot's response
    """
    if not question or question.strip() == "":
        return "⚠️ Please enter a question about agriculture to get started."
    
    # Generate answer
    answer = generate_answer(question)
    
    # Domain awareness check
    agriculture_keywords = [
        'crop', 'plant', 'soil', 'pest', 'fertilizer', 'farm', 'seed',
        'harvest', 'irrigation', 'disease', 'insect', 'weed', 'agriculture',
        'farming', 'cultivation', 'pesticide', 'herbicide', 'fungicide'
    ]
    
    is_agriculture = any(keyword in question.lower() for keyword in agriculture_keywords)
    
    # If question seems out of domain and answer is too short
    if not is_agriculture and len(answer.split()) < 5:
        return ("🌱 I am AgriBot, your specialized agricultural assistant. "
                "I'm designed to help with:\n\n"
                "• Crop management and cultivation\n"
                "• Pest and disease control\n"
                "• Soil health and fertilizers\n"
                "• Irrigation practices\n"
                "• Harvesting techniques\n\n"
                "Please ask me a question related to agriculture or farming.")
    
    return f"🌾 {answer}"


def create_gradio_interface():
    """
    Create and configure the Gradio chatbot interface
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    # Modern custom CSS with enhanced styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        max-width: 1200px !important;
        margin: auto !important;
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        font-weight: 700 !important;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95) !important;
        font-size: 1.1rem !important;
        margin-top: 0.5rem !important;
    }
    
    .instruction-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    
    .instruction-card h3 {
        color: #2d3748;
        margin-bottom: 0.8rem;
        font-size: 1.3rem;
    }
    
    .instruction-list {
        list-style: none;
        padding-left: 0;
    }
    
    .instruction-list li {
        padding: 0.5rem 0;
        color: #4a5568;
        font-size: 1rem;
        line-height: 1.6;
    }
    
    .instruction-list li:before {
        content: "✓ ";
        color: #48bb78;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem;
        border: 2px solid #e2e8f0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        border-color: #667eea;
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    
    .feature-title {
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 0.3rem;
    }
    
    .feature-desc {
        color: #718096;
        font-size: 0.9rem;
    }
    
    .input-container, .output-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    .gr-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    .gr-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    .gr-box {
        border-radius: 10px !important;
        border: 2px solid #e2e8f0 !important;
    }
    
    .gr-box:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .examples-section {
        background: #f7fafc;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .examples-section h3 {
        color: #2d3748;
        margin-bottom: 1rem;
        font-size: 1.2rem;
    }
    
    footer {
        text-align: center;
        padding: 2rem 1rem;
        color: #718096;
        font-size: 0.9rem;
    }
    """
    
    # Create the Gradio Blocks interface for more control
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue")) as demo:
        # Header
        gr.HTML("""
            <div class="main-header">
                <h1>🌱 AgriBot - Your Agricultural Assistant</h1>
                <p>AI-Powered Expert Guidance for Modern Farming</p>
            </div>
        """)
        
        # Instructions Card
        gr.HTML("""
            <div class="instruction-card">
                <h3>📋 How to Use AgriBot</h3>
                <ul class="instruction-list">
                    <li><strong>Step 1:</strong> Type your agricultural question in the text box below</li>
                    <li><strong>Step 2:</strong> Click the "Get Answer" button or press Enter</li>
                    <li><strong>Step 3:</strong> Read the AI-generated response in the output section</li>
                    <li><strong>Tip:</strong> Try the example questions below for quick start!</li>
                </ul>
            </div>
        """)
        
        # Features Section
        gr.HTML("""
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🌾</div>
                    <div class="feature-title">Crop Management</div>
                    <div class="feature-desc">Expert advice on cultivation techniques and best practices</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🐛</div>
                    <div class="feature-title">Pest Control</div>
                    <div class="feature-desc">Identify and manage pests effectively</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">💧</div>
                    <div class="feature-title">Irrigation</div>
                    <div class="feature-desc">Water management and irrigation strategies</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🌱</div>
                    <div class="feature-title">Soil Health</div>
                    <div class="feature-desc">Soil fertility and fertilizer recommendations</div>
                </div>
            </div>
        """)
        
        # Main Chat Interface
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<div class='input-container'>")
                question_input = gr.Textbox(
                    label="💬 Ask Your Question",
                    placeholder="Example: How do I control aphids in my wheat crop?",
                    lines=4,
                    max_lines=8,
                    info="Enter your agricultural question here. Be specific for better results!"
                )
                
                with gr.Row():
                    submit_btn = gr.Button("Get Answer 🚀", variant="primary", size="lg")
                    clear_btn = gr.Button("Clear 🗑️", variant="secondary", size="lg")
                gr.HTML("</div>")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.HTML("<div class='output-container'>")
                answer_output = gr.Textbox(
                    label="🤖 AgriBot Response",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    show_copy_button=True,
                    info="AI-generated response will appear here"
                )
                gr.HTML("</div>")
        
        # Examples Section
        gr.HTML("<div class='examples-section'><h3>💡 Try These Example Questions</h3></div>")
        
        gr.Examples(
            examples=[
                ["How do I control aphid infestation in mustard crops?"],
                ["What is the best fertilizer for wheat cultivation?"],
                ["How to manage fungal disease in tomato plants?"],
                ["When should I apply nitrogen fertilizer to rice crops?"],
                ["What are the symptoms of iron deficiency in plants?"],
                ["How to improve soil fertility naturally?"],
                ["What is the best time to harvest corn?"],
                ["How to control whitefly in cotton crops?"],
                ["What are the best practices for organic farming?"],
                ["How to prevent root rot in vegetables?"]
            ],
            inputs=question_input,
            label="Click any example to try it out:"
        )
        
        # Additional Information
        gr.HTML("""
            <div class="instruction-card" style="margin-top: 2rem;">
                <h3>ℹ️ Important Information</h3>
                <ul class="instruction-list">
                    <li>AgriBot is powered by advanced AI trained on agricultural knowledge</li>
                    <li>For best results, ask specific questions about crops, pests, soil, or farming practices</li>
                    <li>The chatbot specializes in agricultural topics and may redirect non-agricultural queries</li>
                    <li>Responses are generated instantly using state-of-the-art language models</li>
                </ul>
            </div>
        """)
        
        # Footer
        gr.HTML("""
            <footer>
                <p>🌍 AgriBot - Empowering Farmers with AI Technology | Built with ❤️ for Agriculture</p>
                <p style="font-size: 0.8rem; margin-top: 0.5rem;">Disclaimer: Always consult with local agricultural experts for critical farming decisions.</p>
            </footer>
        """)
        
        # Event Handlers
        submit_btn.click(
            fn=agribot_chat,
            inputs=question_input,
            outputs=answer_output
        )
        
        question_input.submit(
            fn=agribot_chat,
            inputs=question_input,
            outputs=answer_output
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            inputs=None,
            outputs=[question_input, answer_output]
        )
    
    return demo


def main():
    """
    Main function to launch the AgriBot application
    """
    print("="*80)
    print("🌱 AgriBot - Agricultural Domain Chatbot")
    print("="*80)
    print("\n✨ Initializing Gradio interface...")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    print("\n🚀 Launching AgriBot...")
    print("📱 The chatbot interface will open in your default web browser.")
    print("🌐 A public URL will be generated for sharing (if share=True)")
    print("="*80)
    
    # Launch with share=True to create a public link
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()