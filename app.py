import streamlit as st

st.title("🛍️ Retail & E-commerce LLM Bot")

query = st.text_input("Ask me anything about our store:")

if query:
    response = get_response(query)  # from your chatbot script
    st.write("🤖 Bot:", response)
