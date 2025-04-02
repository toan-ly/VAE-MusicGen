import streamlit as st
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from utils import *
from model.model import CVAE

device = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_DIR = "samples"

st.set_page_config(
    page_title="Audio Reconstruction",
    page_icon="🎵",
    layout="centered"
)

st.markdown("""
    <style>
        .main-title {
            text-align: left;
            font-size: 40px;
            font-weight: bold;
            color: #FF6F61;
            margin-bottom: 20px;
        }
        .stButton > button {
            width: 21%;
            border-radius: 12px;
            background: linear-gradient(to right, #4A90E2, #FF6F61);
            color: white;
            font-size: 18px;
            font-weight: bold;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background: linear-gradient(to right, #FF6F61, #4A90E2);
        }
        .audio-container {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<p class='main-title'>🎶 New Genres Audio Reconstruction</p>", unsafe_allow_html=True)

@st.cache_data 
def load_model():
    with st.spinner('🔄 Loading model...'):
        model = CVAE(64, 128, 256, 130, len(uni_genres_list)).to(device)
        model.load_state_dict(torch.load('model/model_checkpoint123.pth', map_location=torch.device('cpu')))
        model.eval()
    return model

def gen_audio(model, audio_source, genres_list, fixed_length_seconds=3):
    with st.spinner('🔄 Processing audio...'):
        audio_data, sr = load_and_resample_audio(audio_source)
        segment_length_frame = int(fixed_length_seconds * sr)
        n_segments = len(audio_data) // segment_length_frame
        
        split_audio_text_placeholder = st.empty()
        split_audio_text_placeholder.text("Splitting audio... ✂")
        progress_bar_placeholder = st.empty()
        progress_bar = progress_bar_placeholder.progress(0)
        
        audios = []
        for i in range(n_segments):
            start = i * segment_length_frame
            end = (i + 1) * segment_length_frame
            segment = audio_data[start:end]
            mel_spec = audio_to_melspec(segment, sr, to_db=True)
            mel_spec_norm = normalize_melspec(mel_spec)
            mel_spec = torch.tensor(mel_spec, dtype=torch.float32)
            mel_spec_norm = torch.tensor(mel_spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            audios.append((mel_spec_norm, mel_spec))
            progress_bar.progress(int((i + 1) / n_segments * 100))
        
        progress_bar_placeholder.empty()
        split_audio_text_placeholder.empty()
        
        audios_input = torch.cat([audio[0] for audio in audios], dim=0)
        
        genres_input = onehot_encode(tokenize(genres_list), len(uni_genres_list))
        genres_input = torch.tensor(genres_input, dtype=torch.long).unsqueeze(0).unsqueeze(0)
        genres_input = genres_input.repeat(audios_input.shape[0], 1, 1)
        
        with st.spinner('🔄 Reconstructing audio...'):
            recons, _, _ = model(audios_input, genres_input)
        
        recon_audio_text_placeholder = st.empty()
        recon_audio_text_placeholder.text("Reconstructing audio... 🎵")
        progress_bar_placeholder = st.empty()
        progress_bar = progress_bar_placeholder.progress(0)
        
        recon_audios = []
        for i in range(len(recons)):
            spec_denorm = denormalize_melspec(recons[i].detach().numpy().squeeze(), audios[i][1])
            audio_reconstructed = melspec_to_audio(spec_denorm)
            recon_audios.append(audio_reconstructed)
            progress_bar.progress(int((i + 1) / len(recons) * 100))
        recon_audios = np.concatenate(recon_audios)
        
        progress_bar_placeholder.empty()
        recon_audio_text_placeholder.empty()
        
        return recon_audios

def plot_waveform(audio, sr, title="Waveform"):
    fig, ax = plt.subplots(figsize=(8, 2))
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig

def main():
    model = load_model()
    uploaded_audio = st.file_uploader("Upload an audio file (First 15s will be processed)", type=['wav', 'mp3'])
    
    sample_audio = st.selectbox(
        "Or choose a sample audio:",
        options=["None"] + [file for file in os.listdir(SAMPLE_DIR) if file.endswith(('.wav', '.mp3'))],
        index=0,
    )
    
    if uploaded_audio or sample_audio != "None":
        audio_path = uploaded_audio if uploaded_audio else os.path.join(SAMPLE_DIR, sample_audio)
        audio_data, sr = load_and_resample_audio(audio_path)

        st.audio(audio_path, format='audio/wav')
        # st.pyplot(plot_waveform(audio_data, 22050, title="Original Audio"))
            
        genres_list = st.multiselect('Select genres:', uni_genres_list)
        

        if st.button('🚀 Process Audio'):
            result = gen_audio(model, audio_path, genres_list)
            st.subheader("Reconstructed Audio:")
            
            st.audio(result, format='audio/wav', sample_rate=22050)
            st.pyplot(plot_waveform(result, 22050, title="Reconstructed Audio"))

if __name__ == '__main__':
    main()