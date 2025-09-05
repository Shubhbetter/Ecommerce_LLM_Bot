import streamlit as st
from chatbot import create_vectordb, get_response

st.set_page_config(page_title="E-commerce LLM Bot", page_icon="🛍️")

st.title("🛍️ Retail & E-commerce Chatbot")
st.write("Ask me anything about products, shipping, returns, or support.")

# Initialize VectorDB (only first run)
if "db_ready" not in st.session_state:
    create_vectordb()
    st.session_state.db_ready = True

query = st.text_input("💬 Your Question:")

if query:
    response = get_response(query)
    st.markdown(f"**🤖 Bot:** {response}")
