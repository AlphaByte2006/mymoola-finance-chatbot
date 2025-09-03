import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# --- Page config ---
st.set_page_config(page_title="MyMoola - Finance Buddy", layout="wide")

# --- Style ---
page_bg = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Baloo+2:wght@600&display=swap');
[data-testid="stAppViewContainer"] {
    background-color: #0e0e0e;
    color: #f5f5f5;
    font-family: 'Baloo 2', cursive;
}
/* Simplified sparkle container removed for brevity */
/* Add shimmer heading style */
h1 {
  text-align: center;
  font-family: 'Baloo 2', cursive;
  font-size: 3rem;
  color: #ffe066;
  text-shadow: 2px 2px #ff6b6b, 4px 4px #1e1e1e;
  position: relative;
  overflow: hidden;
}
h1::before {
  content: "";
  position: absolute;
  top: 0; left: -75%;
  width: 50%;
  height: 100%;
  background: linear-gradient(
      120deg,
      rgba(255,255,255,0) 0%,
      rgba(255,255,255,0.6) 50%,
      rgba(255,255,255,0) 100%
  );
  transform: skewX(-25deg);
  animation: shimmer 2.5s infinite;
}
@keyframes shimmer {
  0% { left: -75%; }
  100% { left: 125%; }
}
/* Chat bubbles simplified */
.chat-bubble-user {
  background: #ffe066; color: #111; padding: 14px; max-width: 70%;
  margin: 10px; border-radius: 18px; text-align: right; display: inline-block;
  font-family: 'Baloo 2', cursive; box-shadow: 3px 3px 0 #222;
}
.chat-bubble-bot {
  background: #7bed9f; color: #111; padding: 14px; max-width: 70%;
  margin: 10px; border-radius: 18px; text-align: left; display: inline-block;
  font-family: 'Baloo 2', cursive; box-shadow: 3px 3px 0 #222;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)
st.markdown("<h1>💰 MyMoola - Your Finance Buddy 🐷✨</h1>", unsafe_allow_html=True)

# --- Finance QA dataset (example subset; expand as needed) ---
finance_qa_dataset = [
    {"question": "What is the 50/30/20 budgeting rule?",
     "answer": "The 50/30/20 rule divides your income into 50% for needs, 30% for wants, and 20% for savings or debt payment."},
    {"question": "How can I start saving money?",
     "answer": "Track your income and expenses, set a realistic monthly savings goal, and automate transfers to savings accounts."},
    {"question": "How to build an emergency fund?",
     "answer": "Save 3-6 months of essential expenses in a separate account for unexpected situations."},
    {"question": "How to improve credit score?",
     "answer": "Pay bills on time, keep credit utilization low, and avoid multiple credit inquiries."},
    {"question": "What are good beginner investments?",
     "answer": "Low-cost index funds, ETFs, and retirement accounts are great for beginners."},
    # Add more QAs here up to 500+ for better coverage
]

# --- Initialize embedding model and FAISS index ---
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
questions = [item['question'] for item in finance_qa_dataset]
embeddings = embed_model.encode(questions, convert_to_numpy=True)
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# --- Load Hugging Face model ---
@st.cache_resource
def load_hf_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tokenizer, model

tokenizer, model = load_hf_model()

# --- Sidebar inputs ---
st.sidebar.header("Your Finance Profile")
monthly_income = st.sidebar.number_input("Monthly Income (₹)", min_value=0.0, step=1000.0, format="%.2f")
monthly_expenses = st.sidebar.number_input("Monthly Expenses (₹)", min_value=0.0, step=1000.0, format="%.2f")
employment_status = st.sidebar.selectbox("Employment Status", ["Working", "Student", "Unemployed", "Retired"])

# --- FAQ buttons in sidebar ---
faq_questions = [item['question'] for item in finance_qa_dataset[:5]]  # Use first 5 or more

st.sidebar.subheader("FAQs")
for q in faq_questions:
    if st.sidebar.button(q):
        st.session_state.input_text = q
        submit_trigger = st.experimental_rerun()  # To force immediate processing after button press.

# --- Helper functions ---

def retrieve_answer(query, top_k=1):
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = faiss_index.search(query_emb, top_k)
    return finance_qa_dataset[indices[0][0]]['answer']

def get_prompt_with_context(user_input):
    # Retrieve best dataset answer
    retrieved_answer = retrieve_answer(user_input)
    prompt = f"""
You are a helpful, witty, and accurate financial advisor chatbot named Moola. A user profile: Monthly Income ₹{monthly_income:.2f}, Expenses ₹{monthly_expenses:.2f}, Employment status: {employment_status}.

Use the following background info extracted from finance FAQs or dataset to answer the user's question:

Context: {retrieved_answer}

User: {user_input}
Answer:"""
    return prompt

def generate_bot_response(user_input):
    prompt = get_prompt_with_context(user_input)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=180,
        temperature=0.3,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def get_budget_summary(income, expenses):
    savings_goal = income * 0.20
    wants_budget = income * 0.30
    needs_budget = income * 0.50
    actual_savings = income - expenses
    savings_status = "on track" if actual_savings >= savings_goal else "below goal"
    return (
        f"Based on your income of ₹{income:.2f} and expenses of ₹{expenses:.2f}:\n"
        f"- Needs budget (50%): ₹{needs_budget:.2f}\n"
        f"- Wants budget (30%): ₹{wants_budget:.2f}\n"
        f"- Savings goal (20%): ₹{savings_goal:.2f}\n"
        f"You are currently saving ₹{actual_savings:.2f}, which is {savings_status}."
    )

# --- Chat state handler ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "feedback" not in st.session_state:
    st.session_state.feedback = []

def submit_message():
    user_text = st.session_state.input_text.strip()
    if user_text:
        st.session_state.messages.append(("You", user_text))

        # Budget summary trigger
        if "budget summary" in user_text.lower() or "how am i doing" in user_text.lower():
            summary = get_budget_summary(monthly_income, monthly_expenses)
            st.session_state.messages.append(("Moola", summary))
        else:
            response = generate_bot_response(user_text)
            st.session_state.messages.append(("Moola", response))
            st.session_state.feedback.append(None)  # placeholder for feedback

    st.session_state.input_text = ""

# --- Feedback buttons ---
def feedback_buttons(idx):
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("👍", key=f"up_{idx}"):
            st.session_state.feedback[idx] = "up"
    with col2:
        if st.button("👎", key=f"down_{idx}"):
            st.session_state.feedback[idx] = "down"

# --- Chat input UI ---
st.text_input("Ask Moola:", key="input_text", on_change=submit_message)

# --- Display conversation ---
for i, (role, text) in enumerate(st.session_state.messages):
    if role == "You":
        st.markdown(f"<div class='chat-bubble-user'><b>{role}:</b> {text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bubble-bot'><b>{role}:</b> {text}</div>", unsafe_allow_html=True)
        feedback_buttons(i - (len(st.session_state.messages) // 2))  # adjust index for feedback list


