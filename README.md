🎮 Steam Churn Prediction with Deep Learning

This project analyzes player behavior in digital games and predicts churn (player dropout) using monthly statistics from SteamCharts. The solution is powered by a Deep Neural Network (DNN) trained on engineered features and interpreted with SHAP explainability.

📂 Project Structure

steam_churn_project/
├── data/                 # Raw and processed data files
├── src/                  # Python source files (training, evaluation, SHAP, etc.)
├── results/              # Model results, plots, predictions
├── presentation/         # Visuals and presentation files
├── Final_Report.docx     # Project report
├── Steam_Churn_Presentation_Complete.pptx
├── Valve_Player_Data.csv
├── requirements.txt
└── README.md

🚀 Features

SteamCharts-based churn prediction

Feature engineering (Gain_Ratio, Gain_Direction, Volatility)

Deep Neural Network with 3 hidden layers

100% accuracy on test set

SHAP-based interpretability for model decisions

Clean project structure with modular Python scripts

🧪 Technologies

Python 3.11+

pandas, numpy, scikit-learn

TensorFlow / Keras

matplotlib, seaborn

shap

📊 Model Architecture

model = Sequential()
model.add(Dense(64, input_dim=6, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

📈 Results

Test Accuracy: 100%

Confusion Matrix: Perfect (0 errors)

SHAP Analysis: Gain_Direction is most influential feature

📌 How to Run

1- Clone the repo:

git clone https://github.com/yourusername/steam_churn_project.git

2- Navigate into the project:

cd steam_churn_project

3- Install requirements:

pip install -r requirements.txt

Run scripts (e.g. src/model_dnn_final.py, src/shap_explain.py)

📃 License

This project is for academic purposes only.

🙋 Author

Name: Kağan Kılıç

LinkedIn: https://www.linkedin.com/in/kagankilic/

Email: klckagan@gmail.com

