import librosa
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
import tempfile
from PIL import Image
import gradio as gr
from gradio_imageslider import ImageSlider

def generate_mel_spectrogram(audio_path, sr=22050, n_mels=128, fmin=0, fmax=7000):
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Generate Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmin=fmin, fmax=fmax)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    return S_dB, y, sr

def detect_zero_db(spectrogram,threshold,tol):
    # Create a binary mask where the spectrogram values are close to 0 dB
    # +0 dB threshold
    mask = np.isclose(spectrogram, threshold, atol=tol)  # Use a tolerance to include values close to 0 dB
    
    return mask

def plot_spectrogram(spectrogram, file_path):
    # Plot the Mel Spectrogram and save it to a file
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    librosa.display.specshow(spectrogram, sr=22050, x_axis='time', y_axis='mel', fmin=0, fmax=7000)
    plt.savefig(file_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_edge_spectrogram(edges, file_path):
    # Plot the Edge Detected Spectrogram and save it to a file
    plt.figure(figsize=(6, 6))
    plt.axis('off')
    plt.imshow(edges, cmap='gray', aspect='auto', origin='lower')
    plt.savefig(file_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_frequency(times, frequencies, label, color, file_path):
    plt.figure(figsize=(12, 6))
    plt.plot(times, frequencies, label=label, color=color, linewidth=2)
    plt.title(f'{label} Frequency')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    
    # Save to file
    plt.savefig(file_path, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()

def process_audio( threshold, audio_file,tol):
    mel_spectrogram, y, sr = generate_mel_spectrogram(audio_file)
    edges = detect_zero_db(mel_spectrogram,threshold,tol)

    # Create temporary files to save the generated images
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as mel_file, \
        tempfile.NamedTemporaryFile(suffix=".png", delete=False) as edge_file, \
        tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f0_file, \
        tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1_file, \
        tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2_file:
        
        mel_spectrogram_img = mel_file.name
        edge_spectrogram_img = edge_file.name
        f0_img = f0_file.name
        f1_img = f1_file.name
        f2_img = f2_file.name

        # Save the Mel spectrogram and edge-detected spectrogram to the temporary files
        plot_spectrogram(mel_spectrogram, mel_spectrogram_img)
        plot_edge_spectrogram(edges, edge_spectrogram_img)
        
        # Extract and save individual frequency plots
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        times = librosa.times_like(f0, sr=sr)
        
        plot_frequency(times, f0, 'F0', 'cyan', f0_img)
        
        # Formant frequency (F1 and F2) detection using LPC
        lpc_order = 5 # LPC order for formant estimation
        formants = np.empty((times.shape[0], 2))  # F1 and F2
        formants[:] = np.nan  # Initialize with NaN for unvoiced frames
        
        for i in range(len(times)):
            if voiced_flag[i] and i * sr < len(y):
                frame = y[int(i * sr):int(i * sr + sr)]  # 1 frame
                if len(frame) == 0:
                    continue

                # Apply LPC
                A = librosa.lpc(frame, order = lpc_order)
                rts = np.roots(A)
                rts = rts[np.imag(rts) >= 0]
                angz = np.arctan2(np.imag(rts), np.real(rts))
                frqs = angz * (sr / (2 * np.pi))
                frqs = np.sort(frqs)
                
                if len(frqs) >= 2:
                    formants[i, 0] = frqs[0]  # F1
                    formants[i, 1] = frqs[1]  # F2
        
        plot_frequency(times, formants[:, 0], 'F1', 'magenta', f1_img)
        plot_frequency(times, formants[:, 1], 'F2', 'yellow', f2_img)
    
    return [mel_spectrogram_img, edge_spectrogram_img], f0_img, f1_img, f2_img

with gr.Blocks() as demo:
    with gr.Group():
        threshold_slider =gr.Slider(-100,0,value=-2,info="Choose between -100 and 0", label = "db Level")
        tol_slider =gr.Slider(0,45,value=30,info="Choose between 0 and 25", label = "Tolerance")
        audio_input = gr.Audio(label="Upload an audio file in WAV format", type="filepath")
        submit_button = gr.Button("Submit")
        img_slider = ImageSlider(label="Before and After Edge Detection", type="filepath", slider_color="pink")
        f0_plot = gr.Image(label="F0 Frequency Plot", type="filepath")
        f1_plot = gr.Image(label="F1 Frequency Plot", type="filepath")
        f2_plot = gr.Image(label="F2 Frequency Plot", type="filepath")
        
        
        submit_button.click(process_audio, inputs=[ threshold_slider, audio_input,tol_slider], outputs=[img_slider, f0_plot, f1_plot, f2_plot])

if __name__ == "__main__":
    demo.launch()