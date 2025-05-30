import streamlit as st
import numpy as np
import pandas as pd
import re
import pickle
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import json

# Import the chatbot classes (assuming they're in the same file or imported)
# You can copy the classes from the previous code here or import them

# Data structures
@dataclass
class Intent:
    name: str
    confidence: float

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int

@dataclass
class ChatbotResponse:
    intent: Intent
    entities: List[Entity]
    response: str
    confidence: float

class CustomerSupportDataset:
    """Sample dataset for training intent classification"""
    
    @staticmethod
    def get_training_data():
        return [
            # Track Order Intent
            ("Where is my order?", "TrackOrder"),
            ("Can you track my package?", "TrackOrder"),
            ("What's the status of order #12345?", "TrackOrder"),
            ("I need to check on my delivery", "TrackOrder"),
            ("When will my order arrive?", "TrackOrder"),
            ("Track order 98765", "TrackOrder"),
            
            # Cancel Order Intent
            ("I want to cancel my order", "CancelOrder"),
            ("Can I cancel order #54321?", "CancelOrder"),
            ("Please cancel my recent purchase", "CancelOrder"),
            ("How do I cancel my order?", "CancelOrder"),
            ("Stop my order from shipping", "CancelOrder"),
            
            # Refund Request Intent
            ("I want my money back", "RefundRequest"),
            ("Can I get a refund?", "RefundRequest"),
            ("This product is defective, I need a refund", "RefundRequest"),
            ("How do I return this item?", "RefundRequest"),
            ("I'm not satisfied with my purchase", "RefundRequest"),
            ("Process refund for order #11111", "RefundRequest"),
            
            # General Help Intent
            ("I need help", "GeneralHelp"),
            ("Can you assist me?", "GeneralHelp"),
            ("I have a question", "GeneralHelp"),
            ("What can you help me with?", "GeneralHelp"),
            ("Hello", "GeneralHelp"),
            ("Hi there", "GeneralHelp"),
            
            # Account Issues Intent
            ("I can't login to my account", "AccountIssue"),
            ("Reset my password", "AccountIssue"),
            ("My account is locked", "AccountIssue"),
            ("Update my profile information", "AccountIssue"),
            ("Change my email address", "AccountIssue"),
        ]

class IntentClassifier:
    """ML-based intent detection using TF-IDF and Logistic Regression"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.classifier = LogisticRegression(random_state=42)
        self.is_trained = False
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess input text"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def train(self, training_data: List[Tuple[str, str]]):
        """Train the intent classification model"""
        texts, labels = zip(*training_data)
        
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42
        )
        
        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train classifier
        self.classifier.fit(X_train_vec, y_train)
        
        self.is_trained = True
    
    def predict(self, text: str) -> Intent:
        """Predict intent for input text"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        processed_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([processed_text])
        
        # Get prediction and probability
        prediction = self.classifier.predict(text_vec)[0]
        probabilities = self.classifier.predict_proba(text_vec)[0]
        confidence = max(probabilities)
        
        return Intent(name=prediction, confidence=confidence)

class NamedEntityRecognizer:
    """NER for extracting entities like order numbers, dates, products"""
    
    def __init__(self):
        # Custom patterns for order numbers
        self.order_pattern = re.compile(r'#?(\d{4,6})')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text"""
        entities = []
        
        # Extract order numbers
        for match in self.order_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(1),
                label="ORDER_ID",
                start=match.start(),
                end=match.end()
            ))
        
        # Extract emails
        for match in self.email_pattern.finditer(text):
            entities.append(Entity(
                text=match.group(),
                label="EMAIL",
                start=match.start(),
                end=match.end()
            ))
        
        return entities

class ResponseGenerator:
    """Template-based and rule-based response generation"""
    
    def __init__(self):
        self.templates = {
            "TrackOrder": [
                "I'll help you track your order. Let me look up order #{order_id} for you.",
                "Your order #{order_id} is currently {status}. Expected delivery: {delivery_date}.",
                "I can help you track your order. Could you please provide your order number?"
            ],
            "CancelOrder": [
                "I can help you cancel your order. Let me process the cancellation for order #{order_id}.",
                "To cancel your order, I'll need to verify some details. Is this for order #{order_id}?",
                "I understand you want to cancel your order. Let me check if it's still possible to cancel."
            ],
            "RefundRequest": [
                "I'm sorry to hear you're not satisfied. Let me process your refund request for order #{order_id}.",
                "I can help you with a refund. Could you please tell me more about the issue with your order?",
                "Refund requests typically take 3-5 business days to process. I'll start the process for you."
            ],
            "GeneralHelp": [
                "Hello! I'm here to help you with your orders, returns, and account questions. What can I assist you with today?",
                "Hi there! How can I help you today? I can help with tracking orders, cancellations, refunds, and more.",
                "Welcome! I'm your customer support assistant. What would you like help with?"
            ],
            "AccountIssue": [
                "I can help you with your account. Let me assist you with your account issue.",
                "For account security, I'll need to verify some information. What specific account issue are you experiencing?",
                "I'm here to help with your account problem. Let's get this resolved for you."
            ]
        }
        
        # Mock database for demonstration
        self.order_db = {
            "12345": {"status": "shipped", "delivery_date": "2024-02-15", "items": ["Laptop", "Mouse"], "total": "$1,299.99"},
            "54321": {"status": "processing", "delivery_date": "2024-02-18", "items": ["Headphones"], "total": "$199.99"},
            "98765": {"status": "delivered", "delivery_date": "2024-02-10", "items": ["Keyboard", "Monitor"], "total": "$899.99"},
            "11111": {"status": "cancelled", "delivery_date": "N/A", "items": ["Phone Case"], "total": "$29.99"},
        }
    
    def generate_response(self, intent: Intent, entities: List[Entity], user_input: str) -> str:
        """Generate response based on intent and entities"""
        intent_name = intent.name
        
        if intent_name not in self.templates:
            return "I'm sorry, I didn't understand that. Could you please rephrase your question?"
        
        # Get appropriate template
        templates = self.templates[intent_name]
        
        # Extract specific entities
        order_ids = [e.text for e in entities if e.label == "ORDER_ID"]
        
        # Select template based on available information
        if order_ids and intent_name == "TrackOrder":
            order_id = order_ids[0]
            if order_id in self.order_db:
                order_info = self.order_db[order_id]
                return templates[1].format(
                    order_id=order_id,
                    status=order_info["status"],
                    delivery_date=order_info["delivery_date"]
                )
            else:
                return f"I couldn't find order #{order_id} in our system. Please check the order number and try again."
        
        elif order_ids and intent_name in ["CancelOrder", "RefundRequest"]:
            return templates[0].format(order_id=order_ids[0])
        
        # Default response for intent
        if intent_name == "TrackOrder" and not order_ids:
            return templates[2]
        elif intent_name in ["CancelOrder", "RefundRequest"] and not order_ids:
            return templates[2] if len(templates) > 2 else templates[0]
        else:
            return templates[0]
    
    def get_order_details(self, order_id: str) -> Dict:
        """Get order details for display"""
        return self.order_db.get(order_id, None)

class CustomerSupportChatbot:
    """Main chatbot class integrating all components"""
    
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.ner = NamedEntityRecognizer()
        self.response_generator = ResponseGenerator()
        self.conversation_history = []
        self.is_trained = False
        
    def train(self):
        """Train the chatbot components"""
        if not self.is_trained:
            training_data = CustomerSupportDataset.get_training_data()
            self.intent_classifier.train(training_data)
            self.is_trained = True
    
    def process_message(self, user_input: str) -> ChatbotResponse:
        """Process user message and generate response"""
        if not self.is_trained:
            self.train()
            
        # Step 1: Intent Detection
        intent = self.intent_classifier.predict(user_input)
        
        # Step 2: Named Entity Recognition
        entities = self.ner.extract_entities(user_input)
        
        # Step 3: Response Generation
        response_text = self.response_generator.generate_response(intent, entities, user_input)
        
        # Step 4: Create response object
        response = ChatbotResponse(
            intent=intent,
            entities=entities,
            response=response_text,
            confidence=intent.confidence
        )
        
        # Store in conversation history
        self.conversation_history.append({
            "user_input": user_input,
            "response": response,
            "timestamp": datetime.now()
        })
        
        return response

# Streamlit App Configuration
st.set_page_config(
    page_title="🤖 Customer Support Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .intent-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background-color: #ff9800;
        color: white;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.25rem;
    }
    .entity-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background-color: #9c27b0;
        color: white;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 0.25rem;
    }
    .confidence-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #ff5722, #ffc107, #4caf50);
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = CustomerSupportChatbot()
    st.session_state.messages = []
    st.session_state.training_complete = False

# Main App Layout
st.markdown('<h1 class="main-header">🤖 Customer Support Chatbot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Chatbot Controls")
    
    # Training Section
    st.subheader("🧠 Model Training")
    if not st.session_state.training_complete:
        if st.button("🚀 Train Chatbot", type="primary"):
            with st.spinner("Training the chatbot... Please wait."):
                st.session_state.chatbot.train()
                st.session_state.training_complete = True
            st.success("✅ Training completed!")
            st.rerun()
    else:
        st.success("✅ Chatbot is trained and ready!")
    
    # Clear Chat
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chatbot.conversation_history = []
        st.rerun()
    
    # Sample Queries
    st.subheader("💡 Try These Examples")
    sample_queries = [
        "Track my order #12345",
        "Cancel order 54321",
        "I want a refund",
        "I can't login to my account",
        "Hello, I need help"
    ]
    
    for query in sample_queries:
        if st.button(query, key=f"sample_{query}"):
            st.session_state.messages.append({"role": "user", "content": query})
            st.rerun()

# Main Chat Interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("💬 Chat Interface")
    
    # Chat Container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>👤 You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message bot-message">
                    <strong>🤖 Bot:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show additional info if available
                if "metadata" in message:
                    metadata = message["metadata"]
                    
                    # Intent and Confidence
                    st.markdown(f"""
                    <div style="margin-left: 1rem; margin-top: 0.5rem;">
                        <span class="intent-badge">Intent: {metadata['intent']} ({metadata['confidence']:.1%})</span>
                    """, unsafe_allow_html=True)
                    
                    # Entities
                    if metadata['entities']:
                        for entity in metadata['entities']:
                            st.markdown(f'<span class="entity-badge">{entity["label"]}: {entity["text"]}</span>', unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat Input
    if st.session_state.training_complete:
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # Process with chatbot
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.chatbot.process_message(user_input)
            
            # Add bot response with metadata
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response.response,
                "metadata": {
                    "intent": response.intent.name,
                    "confidence": response.intent.confidence,
                    "entities": [{"text": e.text, "label": e.label} for e in response.entities]
                }
            })
            
            st.rerun()
    else:
        st.info("Please train the chatbot first using the sidebar controls.")

# Analytics Panel
with col2:
    st.subheader("📊 Analytics")
    
    if st.session_state.chatbot.conversation_history:
        # Intent Distribution
        intents = [conv["response"].intent.name for conv in st.session_state.chatbot.conversation_history]
        intent_counts = pd.Series(intents).value_counts()
        
        fig_pie = px.pie(
            values=intent_counts.values, 
            names=intent_counts.index,
            title="Intent Distribution"
        )
        fig_pie.update_layout(height=300)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Confidence Over Time
        confidences = [conv["response"].intent.confidence for conv in st.session_state.chatbot.conversation_history]
        timestamps = [conv["timestamp"] for conv in st.session_state.chatbot.conversation_history]
        
        fig_line = px.line(
            x=timestamps, 
            y=confidences,
            title="Confidence Over Time",
            labels={"x": "Time", "y": "Confidence"}
        )
        fig_line.update_layout(height=300)
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Statistics
        st.subheader("📈 Statistics")
        avg_confidence = np.mean(confidences)
        total_messages = len(st.session_state.chatbot.conversation_history)
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Total Messages", total_messages)
        with col_stat2:
            st.metric("Avg Confidence", f"{avg_confidence:.1%}")
    else:
        st.info("Start chatting to see analytics!")

# Order Details Section
if st.session_state.chatbot.conversation_history:
    # Check if any recent message has order entities
    recent_orders = []
    for conv in st.session_state.chatbot.conversation_history[-5:]:  # Last 5 conversations
        for entity in conv["response"].entities:
            if entity.label == "ORDER_ID":
                recent_orders.append(entity.text)
    
    if recent_orders:
        st.subheader("📦 Order Details")
        
        # Get unique order IDs
        unique_orders = list(set(recent_orders))
        
        for order_id in unique_orders:
            order_details = st.session_state.chatbot.response_generator.get_order_details(order_id)
            if order_details:
                with st.expander(f"Order #{order_id}"):
                    col_detail1, col_detail2 = st.columns(2)
                    
                    with col_detail1:
                        st.write(f"**Status:** {order_details['status']}")
                        st.write(f"**Total:** {order_details['total']}")
                    
                    with col_detail2:
                        st.write(f"**Delivery Date:** {order_details['delivery_date']}")
                        st.write(f"**Items:** {', '.join(order_details['items'])}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    🤖 Customer Support Chatbot powered by Machine Learning & NLP<br>
    Built with Streamlit, scikit-learn, and ❤️
</div>
""", unsafe_allow_html=True)