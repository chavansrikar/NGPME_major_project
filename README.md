# Next-Gen Prosthetic Mobility Enhancer (NGPME)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://[YOUR-HUGGING-FACE-SPACE-LINK-HERE])

**An AI-powered blueprint for accessible, video-based gait analysis designed for prosthetic users.**

This is the final year diploma project for my AI/ML program. It's an end-to-end web application that uses deep learning and computer vision to analyze a user's gait from a simple video, providing a fall risk score, pattern analysis, and real-time visual feedback.

---

## 🚀 Live Demo

You can try the live demo of this blueprint, hosted on Hugging Face Spaces:

**[➡️ Try the NGPME Live Demo Here](https://huggingface.co/spaces/srikarrr/NGPME_major_project)**

---

## My Motivation

This project starts with a childhood memory.

Some of my earliest memories are of my father and his "leg." I must have been around 7 years old, and I remember seeing 4 or 5 different ones. Back then, I didn't know the difference between a "plastic leg" and a "prosthetic leg." I just knew my father walked differently, but I was too young to understand the daily challenges he faced.

Now, as a diploma student in AI/ML, I *do* understand. And I decided to use my final year project to build a **blueprint** for a tool I've always wished existed—a tool to make gait analysis accessible, affordable, and understandable for users like him.

## 🌟 Key Features

* **Secure User Authentication:** Local-first user signup and login system using `bcrypt` hashing and an `SQLite` database.
* **Video Gait Analysis:** Users can upload a simple video of themselves walking (e.g., from a smartphone).
* **18-Point Biomechanical Analysis:** The system uses `MediaPipe` to extract 33 body landmarks and calculates 18 key metrics, including:
    * Cadence (steps/min)
    * Knee Asymmetry (L/R difference)
    * Pelvic Tilt Range (stability)
    * Stride Variability (consistency)
    * **Min. Toe Clearance** (a critical trip-risk metric)
* **AI-Powered Insights:** A multi-model AI pipeline:
    * **Deep Learning (LSTM):** Classifies the overall gait pattern (e.g., "Asymmetric," "Shuffling").
    * **Anomaly Detection (Isolation Forest):** Acts as a safety net to flag highly irregular or outlier movements.
* **AI Fall Risk Score:** Generates a single, quantifiable score (0-100) to help users track their risk and progress.
* **Annotated Video Output:** Creates a new video file with a "Head-Up Display" (HUD) showing a skeleton, live metrics, and real-time alerts for detected issues.
* **Interactive Charts:** Displays `Plotly` graphs for knee dynamics and pelvic stability.
* **Session History:** Saves all past sessions for a user, allowing them to track their progress over time.

## 📸 Screenshots

| Login / Signup | Main Dashboard | Analysis & Annotated Video |
|  |![Uploading Screenshot 2025-10-29 161218.png…]()


| <img width="1807" height="919" alt="Screenshot 2025-10-29 162406" src="https://github.com/user-attachments/assets/042a38a5-7721-4ad1-b485-745803affa25" />
![Uploading Screenshot 2025-10-29 162421.png…]()
![Uploading Screenshot 2025-10-29 162448.png…]()
![Uploading Screenshot 2025-10-29 162711.png…]()
![Uploading Screenshot 2025-10-29 162725.png…]()
![Uploading Screenshot 2025-10-29 162750.png…]()
![Uploading Screenshot 2025-10-29 162932.png…]()
![Uploading Screenshot 2025-10-29 163149.png…]()
![Uploading Screenshot 2025-10-29 163214.png…]()
![Uploading Screenshot 2025-10-29 163223.png…]()|
| 

[Image of Login Screen]
 |  |  |
| **Performance Charts** | **Session History** | **Real-Time HUD** |
|  |  | [![Uploading Screenshot 2025-10-29 162421.png…]()
]|

## 🛠️ Technology Stack

* **Backend:** Python
* **Web Framework / UI:** Gradio
* **Computer Vision:** OpenCV, MediaPipe (Pose)
* **Deep Learning:** TensorFlow (Keras)
* **Machine Learning:** Scikit-learn (Isolation Forest, StandardScaler)
* **Data Handling:** Pandas, NumPy
* **Database:** SQLite
* **Security:** `bcrypt` for password hashing
* **Video Processing:** `ffmpeg-python`

## 🤖 The AI Pipeline (How It Works)

1.  **Upload:** The user logs in and uploads a video file via the `Gradio` interface.
2.  **CV Engine:** The `gait_analysis_processor.py` file takes over. `OpenCV` reads the video frame by frame, and `MediaPipe Pose` extracts 33 (x,y,z) body landmarks from each frame.
3.  **Feature Extraction:** The system calculates the 18+ biomechanical metrics (angles, distances, velocities) from these landmarks.
4.  **AI Analysis:**
    * The time-series data (e.g., knee angle per frame) is fed into the pre-trained **`LSTM`** model for pattern classification.
    * The summary metrics (e.g., total asymmetry, stride variability) are fed into the **`Isolation Forest`** to check for anomalies.
5.  **Risk Scoring:** The outputs from both models are combined with the metric violations (like low toe clearance) to generate a final, unified **Fall Risk Score**.
6.  **Report Generation:** The system renders the annotated video, generates the `Plotly` charts, and saves the full report to the user's `SQLite` history.

## 🚧 Project Status & Future Work

This project is a **functional blueprint** and a **work in progress**.

### The "Dummy Model"
The current AI models (`dl_gait_model.h5`, `anomaly_detector.joblib`) are **proof-of-concept models** trained on *synthetically generated data*. This was a crucial step to build and prove that the *entire end-to-end architecture works*—from video upload, to multi-model AI processing, to the final complex report.

### Future Work
The clear next step is to take this blueprint and develop it into a clinically-validated tool.
* **Collect Real Data:** Collaborate with a clinical partner or physical therapist to collect an anonymized, labeled dataset of real prosthetic user videos.
* **Re-Train Models:** Use this real-world data to train a new, highly accurate `LSTM` and `Isolation Forest`.
* **Scale Calibration:** Implement a feature to calibrate video measurements from "pixels" to real-world units (e.g., by having the user place a known-size object in the frame).
* **Expand Model:** Train the AI to detect more specific, named gait abnormalities (e.g., "Vaulting," "Hip Hiking," "Circumduction").

## ⚙️ Setup and Installation

To run this project locally:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR-USERNAME]/[YOUR-REPO-NAME].git
    cd [YOUR-REPO-NAME]
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    (It's recommended to have a `requirements.txt` file. If not, install the core packages.)
    ```bash
    pip install -r requirements.txt
    
    # Or install manually:
    # pip install tensorflow gradio opencv-python-headless mediapipe scikit-learn pandas bcrypt ffmpeg-python
    ```

4.  **Run the application:**
    The first time you run `app.py`, it will automatically call the `initialize_models()` function. This will train and save the synthetic ("dummy") models if they don't already exist.
    ```bash
    python app.py
    ```

5.  **Open the app:**
    Open your browser and go to the local URL provided (usually `http://127.0.0.1:7860`).

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 🙏 Acknowledgments

* This project was inspired by my father, whose resilience as a prosthetic user motivated me to find a way to apply my AI/ML skills to a real-world problem.
* Thanks to the teams at Google for **MediaPipe** and TensorFlow, the **Gradio** team for making ML web apps so accessible, and the entire open-source community.
* I also utilized AI-powered tools like **Gemini** during development to brainstorm solutions, debug complex code, and accelerate my learning process.
