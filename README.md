# 👁️ Pupil Heart Disease Detection using Deep Learning & Live AI Camera

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/indukurunaveenreddy/Pupil-Heart-Disease-Detection)
[![CI](https://github.com/indukurunaveenreddy/Pupil-Heart-Disease-Detection/actions/workflows/test.yml/badge.svg)](https://github.com/indukurunaveenreddy/Pupil-Heart-Disease-Detection/actions)

An AI-powered web application that identifies heart disease risk through non-invasive pupil image analysis using **MobileNetV3** transfer learning. Features both traditional image upload analysis and **real-time live camera screening** with automated reticle targeting.

> **💡 Run directly in your browser on GitHub**: Click the **[Open in GitHub Codespaces](https://codespaces.new/indukurunaveenreddy/Pupil-Heart-Disease-Detection)** button above to launch and test this application in the cloud with zero local setup!

---

## 🌟 Key Features

- **High-Accuracy Deep Learning**: Powered by a fine-tuned **MobileNetV3** architecture yielding **98.05% test accuracy**.
- **Instant Inference (~15ms)**: Pre-trained model loaded into memory for rapid diagnosis.
- **📷 Dual Camera Capabilities**:
  - **Live Camera Snapshot (`/upload1`)**: In-browser webcam capture with circular eye alignment reticle.
  - **Continuous Live Video Stream (`/live`)**: Real-time video analysis (~6.6 FPS) displaying continuous risk level, confidence progress gauge, and class probabilities.
- **📁 File Upload**: Drag-and-drop support for high-resolution retinal/pupil photographs.
- **📊 Diagnostic Visualization**: Auto-generated prediction confidence charts (Seaborn/Matplotlib) showing probability distributions.
- **🔐 User & Admin Management**: Built-in SQLite database with user registration, login authentication, and admin view.

---

## 🏗️ Project Architecture

```
├── app.py                      # Flask backend & prediction routes
├── model.py                    # Model training & evaluation pipeline
├── alg/
│   ├── MobileNet_v3.keras     # Trained MobileNetV3 deep learning model
│   └── MobileNet.png          # Training accuracy visualization
├── templates/
│   ├── index.html              # Landing page
│   ├── user.html               # User login
│   ├── ureg.html               # User registration
│   ├── userhome.html           # User dashboard
│   ├── upload.html             # Upload & webcam snapshot interface
│   ├── live.html               # Continuous real-time live camera stream
│   ├── template.html           # Diagnosis report & chart view
│   ├── admin.html              # Admin login
│   └── adminhome.html          # Admin dashboard
├── static/                     # CSS, JS, and UI assets
├── sample_images/              # Test pupil images for demonstration
│   ├── heart_risk/
│   └── no_heart_risk/
├── images/                     # Runtime upload directory (ephemeral)
├── Procfile                    # Gunicorn start command for Render/Heroku
├── render.yaml                 # Render.com deployment configuration
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 🚀 Quick Start & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/indukurunaveenreddy/Pupil-Heart-Disease-Detection.git
cd Pupil-Heart-Disease-Detection
```

### 2. Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python3 app.py
```

Open your browser and navigate to:
**[http://127.0.0.1:5001](http://127.0.0.1:5001)**

### 5. Production Server (Gunicorn)
```bash
gunicorn app:app --bind 0.0.0.0:5001 --workers 1 --timeout 120
```

---

## 📸 How to Use

1. **Register & Login**: Create an account or sign in through the user portal.
2. **Choose Mode**:
   - **Upload Image**: Drag and drop any image from `sample_images/` into the upload tab.
   - **Live Camera**: Switch to the Camera tab to snap a live photo of your eye.
   - **Live Stream**: Click **"Live Stream"** in the top navigation to start continuous real-time pupil scanning.
3. **View Report**: Review the diagnostic result (**Heart Risk Detected** vs **No Heart Risk**) along with the confidence percentage and bar chart.

---

## ☁️ Deploy on Render.com

This project is pre-configured for one-click deployment on [Render](https://render.com).

### Step-by-step Render Deployment

1. **Push to GitHub** (see Git commands below)
2. Go to [https://dashboard.render.com](https://dashboard.render.com) → **New** → **Web Service**
3. Connect your GitHub account and select the `Pupil-Heart-Disease-Detection` repository
4. Render will auto-detect `render.yaml` and pre-fill settings:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120`
   - **Runtime**: Python 3.11
   - **Plan**: Free
5. Click **Create Web Service** — Render will build and deploy automatically

### ⚠️ Important: SQLite on Render (Ephemeral Filesystem)

Render's free tier uses an **ephemeral filesystem** — the `users.db` SQLite database is **recreated on every deploy** (registered users will be lost). This is expected behavior on free tier.

**Options to persist data:**
- **Render Disk** (paid): Add a persistent disk mounted at `/data` and set `DB_PATH=/data/users.db`
- **Migrate to PostgreSQL**: Use Render's free PostgreSQL addon and switch to `psycopg2`
- **For demo/testing**: SQLite is fine — the app auto-creates the DB on startup

---

## 🔧 Git Commands — Push to GitHub

```bash
# Check current status
git status

# Stage all changes
git add .

# Commit
git commit -m "deploy: fix .gitignore, update requirements for production"

# Push to GitHub
git push origin main
```

---

## 🛠️ Tech Stack

- **Deep Learning**: TensorFlow 2.x, Keras 3, MobileNetV3
- **Computer Vision**: OpenCV (`cv2`), PIL
- **Backend**: Python 3, Flask, SQLite3
- **Production Server**: Gunicorn
- **Frontend**: HTML5, Vanilla JavaScript, CSS3, Bootstrap 5, FontAwesome
- **Data & Plotting**: NumPy, Pandas, Matplotlib, Seaborn
- **Deployment**: Render.com, GitHub Actions CI/CD

---

## 📄 License
This project is licensed under the MIT License.
