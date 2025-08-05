import streamlit as st

# Ultra-minimal test app
st.title("🧪 Minimal Test App")
st.write("Hello, Streamlit!")
st.success("If you see this, deployment works!")

# Simple interaction
name = st.text_input("Enter your name:")
if name:
    st.write(f"Hello, {name}!")

st.write("✅ Basic Streamlit deployment successful!")
