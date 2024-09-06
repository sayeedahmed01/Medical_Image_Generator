# streamlit_app.py
import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# Set the API URL to ngrok URL
API_URL = "https://bdf6-34-19-97-136.ngrok-free.app"  # Replace with running ngrok URL

@st.cache_data
def generate_image(color, rash_type, body_part):
    try:
        url = f"{API_URL}/generate_image"
        payload = {"color": color, "rash_type": rash_type, "body_part": body_part}

        response = requests.post(url, json=payload)
        response.raise_for_status()

        img_data = base64.b64decode(response.json()["image"])
        return Image.open(BytesIO(img_data))
    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with the API: {str(e)}")
    except ValueError as e:
        st.error(f"Error decoding the image: {str(e)}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
    return None

st.title("Skin Condition Image Generator")

# Input fields with unique keys
color = st.text_input("Skin Color", "fair", key="skin_color_input")
rash_type = st.text_input("Rash Type", "eczema", key="rash_type_input")
body_part = st.text_input("Body Part", "arm", key="body_part_input")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        img = generate_image(color, rash_type, body_part)
        if img:
            st.image(img, caption="Generated Image", use_column_width=True)
        else:
            st.error("Failed to generate image. See error message above for details.")