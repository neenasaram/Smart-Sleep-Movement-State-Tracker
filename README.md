# Smart-Sleep-Movement-State-Tracker
### Project Overview
SleepSense is a machine learning-powered project that tracks and classifies sleep states—onset, wakeup, and none—using accelerometer data. It utilizes Random Forest, PCA, and ENMO (Euclidean Norm Minus One) calculations to analyze and predict sleep phases based on movement.
### Key Features
+ Real-time prediction of sleep states using wearable sensor data.
+ Random Forest Classifier for accurate and interpretable results.
+ Dimensionality Reduction using PCA for improved model performance.
+ ENMO calculation to quantify physical activity level from accelerometer signals.
+ Deployed with Streamlit for easy interaction and visualization.
### Technologies Used
+ Python
+ Pandas, NumPy, Scikit-learn
+ Matplotlib / Seaborn (optional for EDA)
+ Streamlit (for UI)
+ Jupyter Notebook
+ CSV Accelerometer Data

### Web App
🔗 Live Demo:https://huggingface.co/spaces/saramneena/Smart_Sleep_Movement_and_State_Tracker
### How It Works
+ Preprocessing: Data is cleaned, normalized, and missing values handled.
+ ENMO Calculation: Used to detect intensity of movement.
+ Feature Engineering: Important statistical features extracted.
+ Dimensionality Reduction: PCA is applied to reduce noise.
+ Classification: Random Forest Classifier is trained on labeled data.
+ Deployment: A user-friendly interface built with Streamlit allows users to upload data and view predictions interactively.
