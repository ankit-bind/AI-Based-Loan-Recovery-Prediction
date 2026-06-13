# app.py — Streamlit Cloud entry point
# Redirects to Home.py which is the actual landing page

import streamlit as st

st.set_page_config(page_title="Home", layout="wide")

# Redirect to Home.py so sidebar shows "Home" instead of "app"
try:
    st.switch_page("Home.py")
except Exception:
    # Fallback: load Home.py content directly
    with open("Home.py", "r", encoding="utf-8") as f:
        code = f.read()
    exec(code, {"__name__": "__main__"})
