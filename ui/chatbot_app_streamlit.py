"""
SproutBot - Agricultural Domain Chatbot (Streamlit Version)
A Transformer-based chatbot specialized in agriculture using T5 model
"""

import streamlit as st
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
import warnings

warnings.filterwarnings('ignore')

# Model configuration
MODEL_PATH = "./models/agribot_model_exp3"
MAX_INPUT_LENGTH = 128
MAX_OUTPUT_LENGTH = 128


@st.cache_resource
def load_model():
    """Load model and tokenizer (cached for performance)"""
    try:
        model = TFAutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()


def generate_answer(question, model, tokenizer, max_length=MAX_OUTPUT_LENGTH):
    """
    Generate answer for a given agricultural question
    
    Args:
        question (str): User's agricultural question
        model: Loaded model
        tokenizer: Loaded tokenizer
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


def agribot_chat(question, model, tokenizer):
    """
    Chatbot function that handles user queries and returns responses
    
    Args:
        question (str): User's question
        model: Loaded model
        tokenizer: Loaded tokenizer
    
    Returns:
        str: Bot's response
    """
    if not question or question.strip() == "":
        return "Please enter a question about agriculture to get started."
    
    # Generate answer
    answer = generate_answer(question, model, tokenizer)
    
    # Domain awareness check
    agriculture_keywords = [
        'crop', 'plant', 'soil', 'pest', 'fertilizer', 'farm', 'seed',
        'harvest', 'irrigation', 'disease', 'insect', 'weed', 'agriculture',
        'farming', 'cultivation', 'pesticide', 'herbicide', 'fungicide'
    ]
    
    is_agriculture = any(keyword in question.lower() for keyword in agriculture_keywords)
    
    # If question seems out of domain and answer is too short
    if not is_agriculture and len(answer.split()) < 5:
        return ("I am SproutBot, your specialized agricultural assistant. "
                "I'm designed to help with:\n\n"
                "• Crop management and cultivation\n"
                "• Pest and disease control\n"
                "• Soil health and fertilizers\n"
                "• Irrigation practices\n"
                "• Harvesting techniques\n\n"
                "Please ask me a question related to agriculture or farming.")
    
    return f"{answer}"


def main():
    """Main function to run the Streamlit app"""
    
    # Page configuration
    st.set_page_config(
        page_title="SproutBot - Agricultural Assistant",
        page_icon="🌱",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for professional design
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Color Palette */
        :root {
            --primary-green: #2D5016;
            --light-green: #5A7C3E;
            --accent-green: #7FA650;
            --brown: #6B4423;
            --light-brown: #8B6B47;
            --white: #FFFFFF;
            --off-white: #F8F9FA;
            --light-gray: #E8EBE8;
            --text-dark: #2C3E2C;
            --text-gray: #5A6B5A;
        }
        
        /* Main container */
        .main {
            background-color: var(--off-white);
            padding: 0 !important;
        }
        
        .block-container {
            padding: 0 !important;
            max-width: 100% !important;
        }
        
        /* Header */
        .chat-header {
            background: transparent;
            padding: 2rem 2rem 1rem 2rem;
            border-radius: 0;
            margin-bottom: 0;
            box-shadow: none;
            position: relative;
            overflow: visible;
        }
        
        .header-content {
            max-width: 100%;
            margin: 0 auto;
            position: relative;
            z-index: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 1rem;
        }
        
        .header-icon {
            font-size: 2.5rem;
            filter: none;
        }
        
        .header-text {
            flex: 0;
            text-align: center;
        }
        
        .chat-header h1 {
            color: var(--primary-green);
            font-size: 2.5rem;
            margin: 0;
            font-weight: 700;
            letter-spacing: -0.5px;
            text-shadow: none;
        }
        
        .chat-header p {
            color: var(--text-gray);
            font-size: 1rem;
            margin: 0.3rem 0 0 0;
            font-weight: 400;
        }
        
        /* Instructions banner */
        .instructions-banner {
            background-color: var(--white);
            color: var(--text-dark);
            padding: 1rem 2rem;
            border-radius: 0;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            font-weight: 500;
            border-left: none;
            border-bottom: 1px solid var(--light-gray);
            text-align: center;
        }
        
        .instructions-banner strong {
            color: var(--primary-green);
        }
        
        /* Examples section */
        .examples-container {
            background-color: var(--white);
            padding: 1.5rem 2rem;
            border-bottom: 1px solid var(--light-gray);
            margin-bottom: 0;
        }
        
        .examples-title {
            color: var(--primary-green);
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .examples-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 0.75rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .example-button {
            background-color: var(--off-white);
            color: var(--primary-green);
            border: 1.5px solid var(--light-gray);
            border-radius: 6px;
            padding: 0.75rem 1rem;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            text-align: left;
        }
        
        .example-button:hover {
            background-color: var(--primary-green);
            color: var(--white);
            border-color: var(--primary-green);
        }
        
        /* Chat container */
        .chat-container {
            background-color: var(--white);
            border-radius: 0;
            padding: 2rem;
            margin-bottom: 0;
            min-height: calc(100vh - 500px);
            max-height: calc(100vh - 500px);
            overflow-y: auto;
            box-shadow: none;
            border: none;
            border-bottom: 1px solid var(--light-gray);
        }
        
        /* User message */
        .user-message {
            display: flex;
            justify-content: flex-end;
            margin-bottom: 1.25rem;
        }
        
        .user-message-content {
            background-color: var(--primary-green);
            color: var(--white);
            padding: 1rem 1.25rem;
            border-radius: 8px 8px 2px 8px;
            max-width: 70%;
            box-shadow: 0 1px 2px rgba(45, 80, 22, 0.2);
        }
        
        .user-label {
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            opacity: 0.9;
            margin-bottom: 0.5rem;
        }
        
        .user-text {
            font-size: 0.95rem;
            line-height: 1.6;
            font-weight: 400;
        }
        
        /* Bot message */
        .bot-message {
            display: flex;
            justify-content: flex-start;
            margin-bottom: 1.25rem;
        }
        
        .bot-message-content {
            background-color: var(--off-white);
            color: var(--text-dark);
            padding: 1rem 1.25rem;
            border-radius: 8px 8px 8px 2px;
            max-width: 70%;
            box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
            border: 1px solid var(--light-gray);
        }
        
        .bot-label {
            font-weight: 600;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: var(--primary-green);
            margin-bottom: 0.5rem;
        }
        
        .bot-text {
            font-size: 0.95rem;
            line-height: 1.7;
            color: var(--text-dark);
            font-weight: 400;
        }
        
        /* Empty state */
        .empty-chat {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-gray);
        }
        
        .empty-chat-icon {
            width: 80px;
            height: 80px;
            margin: 0 auto 1.5rem;
            background-color: var(--light-gray);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5rem;
        }
        
        .empty-chat-text {
            font-size: 1.1rem;
            color: var(--text-gray);
            font-weight: 500;
        }
        
        .empty-chat-subtext {
            font-size: 0.9rem;
            color: var(--text-gray);
            margin-top: 0.5rem;
            opacity: 0.8;
        }
        
        /* Loading spinner */
        .loading-container {
            text-align: center;
            padding: 3rem 2rem;
        }
        
        .loading-spinner {
            width: 60px;
            height: 60px;
            margin: 0 auto 1.5rem;
            border: 4px solid var(--light-gray);
            border-top: 4px solid var(--primary-green);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-text {
            font-size: 1.1rem;
            color: var(--text-gray);
            font-weight: 500;
        }
        
        /* Fixed input area */
        .fixed-input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: var(--white);
            padding: 1.5rem 2rem;
            box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
            z-index: 1000;
            border-top: 1px solid var(--light-gray);
        }
        
        .input-wrapper {
            max-width: 1200px;
            margin: 0 auto;
            display: flex;
            gap: 0.75rem;
            align-items: center;
        }
        
        /* Examples section - removed duplicate */
        
        /* Button styling */
        .stButton > button {
            background-color: var(--primary-green);
            color: var(--white);
            border: none;
            border-radius: 6px;
            padding: 0.75rem 1.5rem;
            font-size: 0.95rem;
            font-weight: 600;
            transition: all 0.2s ease;
            box-shadow: 0 1px 3px rgba(45, 80, 22, 0.3);
            height: 100%;
        }
        
        .stButton > button:hover {
            background-color: var(--light-green);
            box-shadow: 0 2px 6px rgba(45, 80, 22, 0.4);
            transform: translateY(-1px);
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            border-radius: 6px;
            border: 1.5px solid var(--light-gray);
            padding: 0.75rem 1rem;
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: var(--primary-green);
            box-shadow: 0 0 0 3px rgba(45, 80, 22, 0.1);
        }
        
        /* Add padding to bottom of content to account for fixed input */
        .main .block-container {
            padding-bottom: 120px !important;
        }
        
        /* Scrollbar styling */
        .chat-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: var(--light-gray);
            border-radius: 10px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: var(--accent-green);
            border-radius: 10px;
        }
        
        .chat-container::-webkit-scrollbar-thumb:hover {
            background: var(--light-green);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Stats styling */
        [data-testid="stMetricValue"] {
            color: var(--primary-green);
            font-weight: 600;
        }
        
        [data-testid="stMetricLabel"] {
            color: var(--text-gray);
            font-weight: 500;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: var(--white);
            border-radius: 6px;
            border: 1px solid var(--light-gray);
            font-weight: 500;
            color: var(--text-dark);
        }
        
        .streamlit-expanderHeader:hover {
            background-color: var(--off-white);
        }
        
        /* Section styling */
        .info-section {
            background-color: var(--white);
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid var(--light-gray);
        }
        
        .section-title {
            color: var(--primary-green);
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        
        .section-content {
            color: var(--text-dark);
            line-height: 1.7;
            font-size: 0.9rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load model
    model, tokenizer = load_model()
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'question_count' not in st.session_state:
        st.session_state.question_count = 0
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    # Header
    st.markdown("""
        <div class="chat-header">
            <div class="header-content">
                <div class="header-icon">🌱</div>
                <div class="header-text">
                    <h1>SproutBot</h1>
                    <p>AI-Powered Agricultural Assistant</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Instructions Banner
    st.markdown("""
        <div class="instructions-banner">
            <strong>How to Use:</strong> Type your agricultural question in the input box at the bottom • 
            Click example questions below for quick start • Use "Clear" to start fresh
        </div>
    """, unsafe_allow_html=True)
    
    # Example Questions Section - Moved to top
    examples = [
        "How do I control aphid infestation in mustard crops?",
        "What is the best fertilizer for wheat cultivation?",
        "How to manage fungal disease in tomato plants?",
        "When should I apply nitrogen fertilizer to rice crops?",
        "What are the symptoms of iron deficiency in plants?",
        "How to improve soil fertility naturally?",
        "What is the best time to harvest corn?",
        "How to control whitefly in cotton crops?",
    ]
    
    st.markdown('<div class="examples-container">', unsafe_allow_html=True)
    st.markdown('<div class="examples-title">💡 Try These Questions</div>', unsafe_allow_html=True)
    
    # Display examples in a grid using columns
    cols = st.columns(4)
    for idx, example in enumerate(examples):
        with cols[idx % 4]:
            if st.button(f"📝 {example}", key=f"example_{idx}", use_container_width=True):
                st.session_state.is_processing = True
                st.session_state.temp_question = example
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process example question if flagged
    if st.session_state.is_processing and hasattr(st.session_state, 'temp_question'):
        response = agribot_chat(st.session_state.temp_question, model, tokenizer)
        st.session_state.chat_history.append({
            "question": st.session_state.temp_question,
            "answer": response
        })
        st.session_state.question_count += 1
        st.session_state.is_processing = False
        delattr(st.session_state, 'temp_question')
        st.rerun()
    
    # Chat Container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
    if st.session_state.is_processing:
        # Loading state
        st.markdown("""
            <div class="loading-container">
                <div class="loading-spinner"></div>
                <div class="loading-text">🤔 Processing your question...</div>
            </div>
        """, unsafe_allow_html=True)
    elif len(st.session_state.chat_history) == 0:
        # Empty state
        st.markdown("""
            <div class="empty-chat">
                <div class="empty-chat-icon">🌾</div>
                <div class="empty-chat-text">Welcome to SproutBot</div>
                <div class="empty-chat-subtext">Ask me anything about agriculture and farming</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Display chat messages
        for chat in st.session_state.chat_history:
            # User message
            st.markdown(f"""
                <div class="user-message">
                    <div class="user-message-content">
                        <div class="user-label">You</div>
                        <div class="user-text">{chat['question']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Bot message
            st.markdown(f"""
                <div class="bot-message">
                    <div class="bot-message-content">
                        <div class="bot-label">SproutBot</div>
                        <div class="bot-text">{chat['answer']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Fixed Input Area at bottom
    st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([8, 1, 1])
    
    with col1:
        user_question = st.text_input(
            "Type your question here:",
            placeholder="e.g., How do I control aphids in my wheat crop?",
            key="question_input",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", use_container_width=True, type="primary")
    
    with col3:
        clear_button = st.button("Clear", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle clear button
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.question_count = 0
        st.session_state.is_processing = False
        st.rerun()
    
    # Handle send button
    if send_button and user_question and user_question.strip():
        st.session_state.is_processing = True
        st.rerun()
    
    # Process the question if flagged
    if st.session_state.is_processing:
        response = agribot_chat(user_question, model, tokenizer)
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": response
        })
        st.session_state.question_count += 1
        st.session_state.is_processing = False
        st.rerun()
    
    # Stats Section
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Questions Asked", st.session_state.question_count)
    
    with col2:
        st.metric("Domain", "Agriculture")
    
    with col3:
        st.metric("Model", "T5 Transformer")
    
    # Add spacing for fixed input
    st.markdown("<br><br><br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()