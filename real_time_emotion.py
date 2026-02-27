import torch
import sounddevice as sd
import numpy as np
import cv2
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
from deepface import DeepFace
import emoji
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification


MODEL_NAME = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
speech_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME)
speech_processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)


SPEECH_EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# Open camera
cap = cv2.VideoCapture(0)

# Emotion trend tracking
emotion_trends = defaultdict(list)

# GUI Setup
root = tk.Tk()
root.title("Multi-Modal Emotion Recognition")
root.geometry("1200x700")
root.configure(bg="#2C3E50")

# Video Frame
video_frame = Frame(root, bg="#34495E", width=640, height=480)
video_frame.pack(side="left", padx=10, pady=10)
video_label = Label(video_frame)
video_label.pack()

# Info Frame
info_frame = Frame(root, bg="#2C3E50")
info_frame.pack(side="right", fill="both", padx=10, pady=10)

title_label = Label(info_frame, text="Emotion Detection (Face & Speech)", font=("Helvetica", 20, "bold"), fg="white",
                    bg="#2C3E50")
title_label.pack(pady=10)

emoji_label = Label(info_frame, text="", font=("Helvetica", 50), bg="#2C3E50")
emoji_label.pack(pady=10)

face_emotion_label = Label(info_frame, text="Face Emotion: None (0.00%)", font=("Helvetica", 14), fg="white",
                           bg="#2C3E50")
face_emotion_label.pack(anchor="nw", pady=5)

speech_emotion_label = Label(info_frame, text="Speech Emotion: None", font=("Helvetica", 14), fg="white", bg="#2C3E50")
speech_emotion_label.pack(anchor="nw", pady=5)

emotion_percentages_label = Label(info_frame, text="Emotion Percentages: None", font=("Helvetica", 12), fg="white",
                                  bg="#2C3E50", justify="left")
emotion_percentages_label.pack(anchor="nw", pady=10)

# Emotion Trends Graph
fig, ax = plt.subplots(figsize=(6, 3))
ax.set_title("Emotion Trends Over Time")
ax.set_xlabel("Time")
ax.set_ylabel("Emotion Confidence (%)")
ax.set_ylim(0, 100)
lines = {}
colors = ["r", "g", "b", "y", "m", "c", "k"]

emotion_keys = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
for idx, emotion in enumerate(emotion_keys):
    lines[emotion], = ax.plot([], [], label=emotion, color=colors[idx % len(colors)])

ax.legend(loc="upper right")
canvas = FigureCanvasTkAgg(fig, master=info_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(pady=10)


# Map emotions to emojis
def get_emoji(emotion):
    emoji_map = {
        "angry": emoji.emojize(":angry_face:"),
        "disgust": emoji.emojize(":nauseated_face:"),
        "fear": emoji.emojize(":fearful_face:"),
        "happy": emoji.emojize(":grinning_face_with_big_eyes:"),
        "sad": emoji.emojize(":crying_face:"),
        "surprise": emoji.emojize(":face_with_open_mouth:"),
        "neutral": emoji.emojize(":neutral_face:")
    }
    return emoji_map.get(emotion, emoji.emojize(":face_with_monocle:"))


# Update graph function
def update_graph():
    x_data = list(range(len(next(iter(emotion_trends.values()), []))))
    for emotion, y_data in emotion_trends.items():
        lines[emotion].set_data(x_data, y_data)
    ax.set_xlim(0, max(1, len(x_data)))
    canvas.draw()


# Face Emotion Detection with DeepFace
def update_frame():
    ret, frame = cap.read()
    if ret:
        frame_resized = cv2.resize(frame, (640, 480))

        # Use DeepFace for face emotion detection
        try:
            result = DeepFace.analyze(frame_resized, actions=['emotion'], enforce_detection=False)
            dominant_emotion = result[0]['dominant_emotion']
            dominant_accuracy = result[0]['emotion'][dominant_emotion] * 100

            face_emotion_label.config(text=f"Face Emotion: {dominant_emotion} ({dominant_accuracy:.2f}%)")
            emoji_label.config(text=get_emoji(dominant_emotion))

            emotion_percentages = "\n".join(
                [f"{emotion}: {score * 100:.2f}%" for emotion, score in result[0]['emotion'].items()])
            emotion_percentages_label.config(text="Emotion Percentages:\n" + emotion_percentages)

            for emotion, score in result[0]['emotion'].items():
                emotion_trends[emotion].append(score * 100)
                if len(emotion_trends[emotion]) > 50:
                    emotion_trends[emotion].pop(0)

        except Exception as e:
            print(f"Error in face emotion detection: {e}")

        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

        update_graph()

    video_label.after(10, update_frame)


# Speech Emotion Recognition
def record_and_predict_speech():
    duration = 3  # Record for 3 seconds
    sr = 16000  # Sample rate
    speech_emotion_label.config(text="Recording... 🎤")
    root.update()

    # Record audio
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype=np.float32)
    sd.wait()  # Wait until recording is finished

    # Preprocess audio
    inputs = speech_processor(audio.squeeze(), sampling_rate=sr, return_tensors="pt", padding=True)

    # Predict emotion
    with torch.no_grad():
        logits = speech_model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()

    # Update GUI with predicted emotion
    speech_emotion = SPEECH_EMOTIONS[predicted_class]
    speech_emotion_label.config(text=f"Speech Emotion: {speech_emotion}")


# Buttons
record_button = Button(info_frame, text="🎤 Record Speech", font=("Helvetica", 14), bg="#2980B9", fg="white",
                       command=record_and_predict_speech)
record_button.pack(pady=10)

exit_button = Button(info_frame, text="Exit", command=root.quit, font=("Helvetica", 14), bg="#E74C3C", fg="white")
exit_button.pack(pady=10)

update_frame()
root.mainloop()