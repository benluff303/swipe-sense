import streamlit as st
import requests
import pandas as pd

#SwipeSense frontend

#This front queries the SwipeSense [SBERT model API](https://........)

images = [
    "https://picsum.photos/400/300?random=1",
    "https://picsum.photos/400/300?random=2",
    "https://picsum.photos/400/300?random=3",
    "https://picsum.photos/400/300?random=4",
    "https://picsum.photos/400/300?random=5",
    "https://picsum.photos/400/300?random=6",
    "https://picsum.photos/400/300?random=7",
    "https://picsum.photos/400/300?random=8",
    "https://picsum.photos/400/300?random=9",
    "https://picsum.photos/400/300?random=10",
    "https://picsum.photos/400/300?random=11",
    "https://picsum.photos/400/300?random=12",
    "https://picsum.photos/400/300?random=13",
    "https://picsum.photos/400/300?random=14",
    "https://picsum.photos/400/300?random=15",
    "https://picsum.photos/400/300?random=16",
    "https://picsum.photos/400/300?random=17",
    "https://picsum.photos/400/300?random=18",
    "https://picsum.photos/400/300?random=19",
    "https://picsum.photos/400/300?random=20"
    ]

MAX_IMAGES = 20

# Initialising session state
if "index" not in st.session_state:
    st.session_state.index = 0
if "responses" not in st.session_state:
    st.session_state.responses = []
if "num_days" not in st.session_state:
    st.session_state.num_days = None

# Title of page
st.title("SwipeSense")

# Loop that displays images one at a time for dislike/like being clicked up to 20
if st.session_state.index < min(MAX_IMAGES, len(images)):
    img_url = images[st.session_state.index]
    st.image(img_url, caption=f"{st.session_state.index+1}/20")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Dislike"):
            st.session_state.responses.append({"image": img_url, "response": "dislike"})
            st.session_state.index += 1
            st.rerun()
    with col2:
        if st.button("Like"):
            st.session_state.responses.append({"image": img_url, "response": "like"})
            st.session_state.index += 1
            st.rerun()

# Once 20 complete use inputs how long their trip will be
elif st.session_state.num_days is None:
    st.write("Before we show your destinations...")
    days = st.slider("How many days will your trip be?", min_value=1, max_value=14, value=7)
    if st.button("Confirm trip length"):
        st.session_state.num_days = days
        st.rerun()

else:
    st.success("✅ Thank you for your input!")
    st.write("Your journey awaits...")


    if st.session_state.responses:
        SwipeSense_url = "https://api.com/endpoint"

        user = {"responses": st.session_state.responses,
                "num_days": st.session_state.num_days
        }

        try:
            response = requests.post(SwipeSense_url, json=user)
            response.raise_for_status()

            # API call succeeds -->
            data = response.json()

            # Convert to DataFrame
            df = pd.DataFrame(data)

            # Show locations
            if "location" in df.columns:
                df = df[["location"]]

                st.header("Top 5 destinations for you:")
                for i, row in df.iterrows():
                    st.write(f"{i+1}. **{row['location']}**")

            else:
                st.warning("No destinations returned. Try again later.")

        # Something needed here for itiniraries

        except Exception as e:
            st.error(f"Something went wrong while fetching predictions: {e}")


    if st.button("Restart"):
        st.session_state.index = 0
        st.session_state.responses = []
        st.rerun()
