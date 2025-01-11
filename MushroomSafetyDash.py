import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    precision_score,
    recall_score,
    ConfusionMatrixDisplay,
)

# Main function: Initializes the Streamlit app and handles user interactions.
def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are your mushrooms edible or poisonous?")
    st.sidebar.markdown("Are your mushrooms edible or poisonous?")

    # Function to load and preprocess the dataset, caching results for efficiency.
    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('/Users/johnwarner/Documents/data projects/Mushroom Safety dashboard/mushrooms.csv')
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    # Function to split the dataset into training and testing sets.
    @st.cache_data(persist=True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        return x_train, x_test, y_train, y_test

    # Function to plot evaluation metrics like Confusion Matrix, ROC Curve, and Precision-Recall Curve.
    def plot_metrics(metrics_list, model, x_test, y_test, y_pred, class_names):
        import matplotlib.pyplot as plt

        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(cmap="viridis", ax=ax)
            st.pyplot(fig)

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            if hasattr(model, "decision_function"):
                scores = model.decision_function(x_test)
            else:
                scores = model.predict_proba(x_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, scores)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label="ROC Curve")
            ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="best")
            st.pyplot(fig)

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            if hasattr(model, "decision_function"):
                scores = model.decision_function(x_test)
            else:
                scores = model.predict_proba(x_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, scores)
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label="Precision-Recall Curve")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.legend(loc="best")
            st.pyplot(fig)

    # Load and split the dataset.
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['edible', 'poisonous']

    # Sidebar for selecting the classifier.
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    # Option to display classifier documentation.
    show_docs = st.sidebar.checkbox("Show Classifier Documentation")

    if show_docs:
        st.subheader(f"Documentation for {classifier}")
        if classifier == "Support Vector Machine (SVM)":
            st.markdown("""
            **Support Vector Machine (SVM)**:
            - SVM is a supervised learning algorithm used for classification and regression.
            - It works by finding the hyperplane that best separates the classes in the feature space.
            - Hyperparameters:
              - **C**: Regularization parameter. Smaller values specify stronger regularization.
              - **Kernel**: The type of kernel to use (e.g., linear, RBF).
              - **Gamma**: Kernel coefficient for RBF and polynomial kernels.
            """)
        elif classifier == "Logistic Regression":
            st.markdown("""
            **Logistic Regression**:
            - A statistical method for binary classification that predicts the probability of an outcome.
            - It assumes a linear relationship between the input features and the log-odds of the outcome.
            - Hyperparameters:
              - **C**: Regularization strength. Smaller values mean stronger regularization.
              - **Max Iter**: Maximum number of iterations for the solver.
            """)
        elif classifier == "Random Forest":
            st.markdown("""
            **Random Forest**:
            - An ensemble learning method that builds multiple decision trees and merges their results.
            - Useful for both classification and regression tasks.
            - Hyperparameters:
              - **n_estimators**: Number of trees in the forest.
              - **Max Depth**: Maximum depth of each tree.
              - **Bootstrap**: Whether to sample with replacement when building trees.
            """)

    # Hyperparameters and results for Support Vector Machine.
    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, average="binary"), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, average="binary"), 2))
            plot_metrics(metrics, model, x_test, y_test, y_pred, class_names)

    # Hyperparameters and results for Logistic Regression.
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, average="binary"), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, average="binary"), 2))
            plot_metrics(metrics, model, x_test, y_test, y_pred, class_names)

    # Hyperparameters and results for Random Forest.
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap') == 'True'

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", round(accuracy, 2))
            st.write("Precision: ", round(precision_score(y_test, y_pred, average="binary"), 2))
            st.write("Recall: ", round(recall_score(y_test, y_pred, average="binary"), 2))
            plot_metrics(metrics, model, x_test, y_test, y_pred, class_names)

    # Option to display raw dataset.
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Mushroom Data set (Classification)")
        st.write(df)

# Entry point for the application.
if __name__ == "__main__":
    main()
