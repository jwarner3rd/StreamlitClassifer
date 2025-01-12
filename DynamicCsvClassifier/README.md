## Dynamic Classifier with CSV Upload

### Overview
The **Dynamic Classifier with CSV Upload** is a Streamlit app that enables users to upload a custom CSV file and perform classification tasks. This app supports both binary and multiclass classification scenarios, providing flexibility for a variety of datasets. This app has been enhanced with several features, including the ability to select specific fields for classification, update scores dynamically around accuracy, and download the processed CSV for further analysis. The app provides flexibility to train models with user-selected classifiers and visualize model performance.

### Features
- **Dynamic CSV Upload**: Users can upload any dataset, choose the target variable for classification, and select fields for processing.
- **Classifier Selection**: Supports multiple algorithms, including SVM, Logistic Regression, and Random Forest.
- **Hyperparameter Tuning**: Offers sliders and input fields for fine-tuning model parameters.
- **Evaluation Metrics**: Visualizes confusion matrix, ROC curve, and precision-recall curve. These metrics are updated dynamically based on the selected classifier.
- **Download Processed Data**: Allows users to download the processed CSV file after classification for further use, a key enhancement that improves interactivity and usability.
