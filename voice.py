import librosa
import numpy as np

def get_voice_score(audio_path):
    try:
        # Load audio (sr=None preserves native sampling rate)
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return 0.0

    # Ensure the audio isn't completely silent/empty
    if len(y) == 0:
        return 0.0

    # 1. CORRECT PITCH EXTRACTION
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    
    # Extract the actual dominant pitch for each time frame
    dominant_pitches = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        p = pitches[index, t]
        if p > 0 and 50 < p < 500:  # Restrict to normal human speech range (50Hz - 500Hz)
            dominant_pitches.append(p)

    # Calculate average pitch and pitch variation (standard deviation)
    if len(dominant_pitches) > 0:
        avg_pitch = np.mean(dominant_pitches)
        pitch_std = np.std(dominant_pitches)  # Measures variation
    else:
        avg_pitch = 0
        pitch_std = 0

    # 2. ENERGY (VOLUME) EXTRACTION
    rms_features = librosa.feature.rms(y=y)
    energy = np.mean(rms_features)

    # 3. SMART HEURISTIC SCORING
    score = 0.0

    # Pitch Scoring: Award points if there is normal human voice variation
    # (Prevents rewarding a flat robotic monotone or pure static noise)
    if 10 < pitch_std < 50:
        score += 0.5
    elif 50 <= pitch_std < 100:
        score += 0.3

    # Energy Scoring: Ensure there's actual sound, but it isn't deafening clipping noise
    if 0.005 < energy < 0.3:
        score += 0.5

    return min(max(score, 0.0), 1.0)
