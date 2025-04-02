import librosa
import numpy as np
from sklearn.preprocessing import MinMaxScaler

uni_genres_list = ['House', 'Soundtrack', 'Composed Music', 'Drone', 'Instrumental', 'Ambient Electronic', 'Blues', 'Easy Listening', 'Classical', 'Jazz', 'Christmas', 'Electronic', 'Ambient', 'Lo-fi Instrumental', 'Lounge', 'Contemporary Classical', 'Indie-Rock', 'Dance', 'New Age', 'Halloween', 'Lo-fi Electronic', '20th Century Classical', 'Piano', 'Chill-out', 'Pop']
genres2idx = {genre: idx for idx, genre in enumerate(uni_genres_list)}
idx2genres = {idx: genre for genre, idx in genres2idx.items()}

def tokenize(genres):
    return [genres2idx[genre] for genre in genres if genre in genres2idx]

def detokenize_tolist(tokens):
    return [idx2genres[token] for token in tokens if token in idx2genres]

def onehot_encode(tokens, max_genres):
    onehot = np.zeros(max_genres)
    onehot[tokens] = 1
    return onehot

def onehot_decode(onehot):
    return [idx for idx, val in enumerate(onehot) if val == 1]

def load_and_resample_audio(file_path, target_sr=22050, max_duration=15):
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    if len(audio) > target_sr * max_duration:
        audio = audio[:target_sr * max_duration]
    return audio, target_sr

def audio_to_melspec(audio, sr, n_mels=256, n_fft=2048, hop_length=512, to_db=False):
    spec = librosa.feature.melspectrogram(y=audio,
                                          sr=sr,
                                          n_fft=n_fft,
                                          hop_length=hop_length,
                                          win_length=None,
                                          window='hann',
                                          center=True,
                                          pad_mode='reflect',
                                          power=2.0,
                                          n_mels=n_mels)
    
    if to_db:
        spec = librosa.power_to_db(spec, ref=np.max)
    
    return spec

def normalize_melspec(melspec, norm_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=norm_range)
    melspec = melspec.T
    melspec_normalized = scaler.fit_transform(melspec)
    return melspec_normalized.T
 
def denormalize_melspec(melspec_normalized, original_melspec, norm_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=norm_range)
    melspec = original_melspec.T
    scaler.fit(melspec)
    melspec_denormalized = scaler.inverse_transform(melspec_normalized.T)
    return melspec_denormalized.T

def melspec_to_audio(melspec, sr=22050, n_fft=2048, hop_length=512, n_iter=64):
    if np.any(melspec < 0):
        melspec = librosa.db_to_power(melspec)
    
    audio_reconstructed = librosa.feature.inverse.mel_to_audio(melspec,
                                                              sr=sr,
                                                              n_fft=n_fft,
                                                              hop_length=hop_length,
                                                              win_length=None,
                                                              window='hann',
                                                              center=True,
                                                              pad_mode='reflect',
                                                              power=2.0,  # Ensure the correct inverse transformation
                                                              n_iter=n_iter)
    return audio_reconstructed