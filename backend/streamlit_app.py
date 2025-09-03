# streamlit_app.py
# Minimal Streamlit client that talks to the FastAPI backend.

import requests
import streamlit as st

# API = st.secrets.get("API_URL", "http://localhost:8000")  # set in .streamlit/secrets.toml optionally

API = "http://127.0.0.1:8000"

st.set_page_config(page_title="SwipeSense", layout="wide")
st.title("SwipeSense — Streamlit UI")

# User id
user_id = st.text_input("User ID", value="default")

# State
if "current" not in st.session_state:
    st.session_state.current = None

def fetch_next():
    r = requests.post(f"{API}/next", json={"user_id": user_id}).json()
    if r.get("status") == "end":
        st.warning("No more candidates. You may need to relax filters or restart the backend.")
        st.session_state.current = None
        return None
    st.session_state.current = r
    return r

col1, col2 = st.columns([3, 1])

with col1:
    if st.session_state.current:
        idx = st.session_state.current["idx"]
        img_resp = requests.get(f"{API}/image/{idx}")
        st.image(img_resp.content, caption=f"idx={idx} | sim={st.session_state.current.get('sim_score')}", use_column_width=True)
    else:
        st.info("Press **Start / Next** to begin.")

with col2:
    st.subheader("Controls")
    if st.button("Start / Next ▶"):
        fetch_next()

    if st.session_state.current:
        idx = st.session_state.current["idx"]
        if st.button("🔥 Like"):
            requests.post(f"{API}/feedback", json={"user_id": user_id, "idx": idx, "action": "like", "dwell": None})
            fetch_next()
        if st.button("➡️ Skip"):
            requests.post(f"{API}/feedback", json={"user_id": user_id, "idx": idx, "action": "swipe", "dwell": 0.0})
            fetch_next()
        if st.button("Get user profile"):
            r = requests.get(f"{API}/user_profile/{user_id}").json()
            st.write(r)

st.divider()
st.caption("Backend API must be running. Start with:  `uvicorn app:app --reload`  and run this with:  `streamlit run streamlit_app.py`")
