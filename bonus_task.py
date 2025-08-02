# --- Step 1: Install necessary libraries ---
# This command installs the required packages for the UI, model, and tunneling.
%pip install streamlit pyngrok ultralytics requests -q

# --- Step 2: Define the complete Streamlit app code ---
# This block contains the entire, corrected user interface and logic for the AetherVision AI application.
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import requests

# --- 1. Page & State Configuration ---
st.set_page_config(
    page_title="BoinkVision AI | ISS Asset Tracker",
    page_icon="🛰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.40 # Default confidence

# --- 2. Custom CSS Styling ---
def load_custom_css():
    st.markdown('''
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

        /* --- General Styles --- */
        .stApp {
            background-color: #0D1117;
            color: #F0F2F6;
        }
        body, .stMarkdown, .stButton>button, .stSlider>div>div>div>div {
            font-family: 'Inter', sans-serif;
        }
        /* --- Main Header --- */
        .header {
            font-size: 2.8rem;
            font-weight: 700;
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 2rem;
            color: #FFFFFF;
            border-bottom: 1px solid #30363d;
        }
        /* --- Sidebar Styles --- */
        [data-testid="stSidebar"] {
            background-color: #0D1117;
            border-right: 1px solid #30363d;
        }
        [data-testid="stSidebar"] h1 {
            color: #00AEEF; /* Highlight color for sidebar title */
        }
        /* --- Container Styles --- */
        .control-module, .results-container {
            background-color: #161B22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 4px 12px 0 rgba(0, 0, 0, 0.15);
        }
        .control-module {
            max-width: 800px;
            margin: 1rem auto;
        }
        /* --- Widget Styles --- */
        .stButton>button {
            background-image: linear-gradient(to right, #00AEEF, #008fbf);
            color: #FFFFFF;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            width: 100%;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            box-shadow: 0 0 15px #00AEEF;
            transform: translateY(-2px);
        }
        [data-testid="stFileUploader"] {
            border: 2px dashed #30363d;
            border-radius: 8px;
            padding: 1rem;
        }
    </style>
    ''', unsafe_allow_html=True)

# --- 3. Helper Functions & Model Loading ---

# UPDATED function to handle Google Drive's large file warning
def download_file_from_url(url, save_path):
    if os.path.exists(save_path):
        return True

    try:
        with st.spinner(f"Downloading model file... (this may take a moment)"):
            session = requests.Session()
            response = session.get(url, stream=True)

            # Get the confirmation token
            token = None
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    token = value

            # If a token was found, make a second request with the token
            if token:
                params = {'confirm': token}
                response = session.get(url, params=params, stream=True)

            # Now save the file
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
        st.success("Model downloaded successfully!")
        return True
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading model: {e}")
        return False


@st.cache_resource
def load_yolo_model(path):
    'Loads the YOLO model from a local file path.'
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- 4. UI Rendering ---
load_custom_css()

# --- Sidebar ---
with st.sidebar:
    st.title("🛰 BoinkVision AI")




    st.markdown("---")
    st.header("System Information")
    st.info(
        "BoinkVision is an AI-powered asset tracker for the ISS, trained on synthetic data "
        "to identify critical equipment with high accuracy."
    )
    st.metric(label="Current Model mAP", value="92.5%")

# --- Main Application Body ---


# --- Model Loading Logic ---
MODEL_URL = "https://drive.google.com/uc?export=download&id=161zFNNnU8Ct6lf2_8qG9Qe_rSPzJwx8D"

MODEL_PATH = "model.pt"

model_downloaded = download_file_from_url(MODEL_URL, MODEL_PATH)

model = None
if model_downloaded and os.path.exists(MODEL_PATH):
    model = load_yolo_model(MODEL_PATH)
else:
    st.error("Model file could not be loaded. Please check the URL or file path.")

# --- Primary UI Flow ---
# View 1: Uploader
if not st.session_state.analysis_done:

    st.subheader("Ready for Analysis")
    st.write("Upload an image from an ISS camera feed to begin asset detection.")

    uploaded_file = st.file_uploader(
        "Select an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if st.button("Run AI Analysis", disabled=(model is None)):
        if uploaded_file is None:
            st.warning("Please upload an image first.", icon="⚠")
        else:
            st.session_state.uploaded_image = uploaded_file
            st.session_state.analysis_done = True
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# View 2: Analysis Results
else:

    st.header("🔬 Analysis Results")

    image = Image.open(st.session_state.uploaded_image)
    with st.spinner('Analyzing visual telemetry... Please wait.'):
        results = model.predict(image, conf=st.session_state.confidence)
        annotated_image = results[0].plot()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True, caption="Uploaded telemetry scan.")
    with col2:
        st.subheader("AI Detected Assets")
        st.image(annotated_image, use_container_width=True, caption="Assets detected by the YOLOv8 model.")

    st.markdown("---")

    st.subheader("Detection Summary")
    names = model.names
    detected_counts = {}
    for r in results:
        for c in r.boxes.cls:
            name = names[int(c)]
            detected_counts[name] = detected_counts.get(name, 0) + 1

    if not detected_counts:
        st.success("✅ No critical assets were detected with the current confidence setting.")
    else:
        summary_cols = st.columns(len(detected_counts))
        for i, (name, count) in enumerate(detected_counts.items()):
            with summary_cols[i]:
                st.metric(label=name.title(), value=count)

    if st.button("Scan Another Image"):
        st.session_state.analysis_done = False
        st.session_state.uploaded_image = None
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# --- Step 3: Write the app code to a file ---
with open("app.py", "w") as f:
    f.write(app_code)

print("✅ 'AetherVision AI' app code has been successfully written to app.py.")

# --- Step 4: Launch the app with a public URL using ngrok ---
from pyngrok import ngrok
import os

# --- Add your ngrok authtoken here ---
# Get your token from https://dashboard.ngrok.com/get-started/your-authtoken
NGROK_AUTH_TOKEN = "30k3GBmcxKhrzVE9NLC4oTK6CdP_5PE6RBhQqfq5z2jy5mGDV"  # <--- PASTE YOUR NGROK TOKEN HERE

# Kill any existing ngrok tunnels and set the auth token
ngrok.kill()
try:
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
except Exception as e:
    print(f"Error setting ngrok auth token: {e}. Please ensure it is correct.")

# Run the Streamlit app in the background using nohup
if os.path.exists("app.py"):
    print("🚀 Launching Streamlit app...")
    !nohup streamlit run app.py &
    try:
        # Create ngrok tunnel to the Streamlit port (8501)
        public_url = ngrok.connect(8501)
        print("🎉 Your app is live! Click this link to open it:")
        print(public_url)
    except Exception as e:
        print(f"Could not connect to ngrok. Error: {e}")
else:
    print("❌ Error: app.py not found. Please run the script again to create the file.")
