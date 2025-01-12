# StreamlitClassifer

This repository contains two Streamlit applications showcasing classification tasks using machine learning. Each app demonstrates a different approach to interactive data analysis and model building. The foundation of this work is based on the Coursera project ["Build a Machine Learning Web App with Streamlit and Python"](https://www.coursera.org/projects/machine-learning-streamlit-python) by instructor Snehan Kekre. The code has been updated to reflect recent changes in Streamlit and to include enhanced documentation and explanations of the machine learning algorithms.

---
<image scr= "/images/MushroomSafetyDashboard.pdf"/>

## Mushroom Classification Dashboard

### Overview
The **Mushroom Classification Dashboard** is a Streamlit-based web application designed to classify mushrooms as either edible or poisonous based on their characteristics. The app uses three machine learning algorithms for binary classification:
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest Classifier

Users can select a classifier, tune hyperparameters, and view evaluation metrics such as:
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

### Features
- **Dataset Preprocessing**: Encodes categorical features using Label Encoding.
- **Interactive Model Training**: Allows users to tune hyperparameters and train models in real-time.
- **Visualization**: Displays key evaluation metrics to assess model performance.
- **Documentation**: Offers built-in explanations for each machine learning algorithm.

### File Path
- Code: `mushroom_classification_app.py`

### How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/data-portfolio.git
    ```
2. Navigate to the directory:
    ```bash
    cd data-portfolio
    ```
3. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Streamlit app:
    ```bash
    streamlit run mushroom_classification_app.py
    ```
5. Access the application in your browser at `http://localhost:8501`.

---

## Dynamic Classifier with CSV Upload

### Overview
The **Dynamic Classifier with CSV Upload** is a Streamlit app that enables users to upload a custom CSV file and perform classification tasks. This app supports both binary and multiclass classification scenarios, providing flexibility for a variety of datasets. This app has been enhanced with several features, including the ability to select specific fields for classification, update scores dynamically around accuracy, and download the processed CSV for further analysis. The app provides flexibility to train models with user-selected classifiers and visualize model performance.

### Features
- **Dynamic CSV Upload**: Users can upload any dataset, choose the target variable for classification, and select fields for processing.
- **Classifier Selection**: Supports multiple algorithms, including SVM, Logistic Regression, and Random Forest.
- **Hyperparameter Tuning**: Offers sliders and input fields for fine-tuning model parameters.
- **Evaluation Metrics**: Visualizes confusion matrix, ROC curve, and precision-recall curve. These metrics are updated dynamically based on the selected classifier.
- **Download Processed Data**: Allows users to download the processed CSV file after classification for further use, a key enhancement that improves interactivity and usability.

### File Path
- Code: `dynamic_csv_classifier.py`

### How to Run
1. Follow the same setup instructions as the **Mushroom Classification Dashboard**.
2. Run the following command:
    ```bash
    streamlit run dynamic_csv_classifier.py
    ```
3. Access the application in your browser at `http://localhost:8501`.

---

## Requirements
Ensure you have Python 3.7 or later installed. Install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Libraries Used
- `streamlit`: For creating interactive web apps
- `pandas`: For data manipulation and analysis
- `numpy`: For numerical computations
- `sklearn`: For machine learning algorithms and evaluation metrics
- `matplotlib`: For visualizing evaluation metrics

---

## Contact
For any questions or feedback, please contact:
- **Name**: John Warner
- **Email**: [your-email@example.com](mailto:your-email@example.com)
- **GitHub**: [your-username](https://github.com/your-username)

