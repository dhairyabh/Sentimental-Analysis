# MoodSense AI - Emotion Detection Platform

MoodSense AI is a premium sentiment analysis web application that combines a local Machine Learning model with Groq's Llama-3 AI for deep emotional insights.

## 🚀 Deployment on Render

This project is pre-configured for **Render**. Follow these steps for a one-click deployment:

1.  **Push to GitHub**: Ensure your latest code is on GitHub.
2.  **Create a New Blueprint**:
    *   Go to [Render Dashboard](https://dashboard.render.com/).
    *   Click **New +** > **Blueprint**.
    *   Connect your GitHub repository.
    *   Render will automatically detect `render.yaml` and configure the service.
3.  **Set Environment Variables**:
    *   In the Render Dashboard, go to your Web Service > **Environment**.
    *   Add `GROQ_API_KEY` with your API key from [Groq Cloud](https://console.groq.com/).

## 🛠 Local Development

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/dhairyabh/Sentimental-Analysis.git
    cd Sentimental-Analysis
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables**:
    Create a `.env` file in the root:
    ```env
    GROQ_API_KEY=your_groq_api_key_here
    ```

4.  **Run the app**:
    ```bash
    python app.py
    ```
    The app will be available at `http://localhost:5000`.

## 📱 Mobile Optimized
The interface is fully responsive and optimized for mobile devices, ensuring a smooth experience across all screen sizes.

## 🧠 Model Information
- **Type**: Logistic Regression (v8)
- **Features**: TF-IDF (100K features) with negation handling and contrastive weighting.
- **Accuracy**: Optimized for nuanced emotional detection.
