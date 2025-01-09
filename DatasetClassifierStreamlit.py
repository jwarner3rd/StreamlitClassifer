import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Function to preprocess data
def preprocess_data(df, target_column):
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        if col != target_column:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

    X = df.drop(columns=[target_column])
    y = df[target_column]

    if y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)
    else:
        target_encoder = None

    return X, y, label_encoders, target_encoder

# Convert DataFrame to CSV for download
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Metrics plotting function
def plot_metrics(metrics_list, model, x_test, y_test, y_pred, class_names):
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
        st.subheader("Precision-Recall Curve")
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

# Generate and display a grid-like table for the classification report
def display_classification_report(y_test, y_pred, target_names):
    report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    st.write("Classification Report (Table View):")
    st.dataframe(report_df)

# Main application logic
def main():
    # Load data via file uploader
    st.title("Dynamic Dataset Classifier App")
    st.subheader("Upload your dataset to test classification models, optimize hyperparameters, and visualize model performance through key metrics like accuracy, confusion matrix, and ROC curves.")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Uploaded Successfully!")
        st.write(df)

        # Sidebar configuration
        st.sidebar.subheader("Dataset Configuration")
        target_column = st.sidebar.selectbox("Select the Target Column", options=df.columns)

        # Preprocess data
        try:
            X, y, label_encoders, target_encoder = preprocess_data(df, target_column)
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
            class_names = target_encoder.classes_ if target_encoder else list(sorted(set(df[target_column])))

            # Debugging outputs
            st.write("Unique values in y_test:", set(y_test))
            st.write("Unique values in y_pred (after model prediction):", set(y_train))
            st.write("Class names:", class_names)
        except Exception as e:
            st.error(f"Error in preprocessing: {e}")
            st.stop()

        st.sidebar.subheader("Metrics to Plot")
        metrics = st.sidebar.multiselect(
            "Select metrics to plot:",
            options=["Confusion Matrix", "ROC Curve", "Precision-Recall Curve"]
        )

        # Sidebar model selection
        st.sidebar.subheader("Model Configuration")
        classifier_name = st.sidebar.selectbox(
            "Classifier",
            ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest")
        )

        # Checkbox to show documentation
        show_docs = st.sidebar.checkbox("Show Classifier Documentation")

        if show_docs:
            st.subheader(f"Documentation for {classifier_name}")
            if classifier_name == "Support Vector Machine (SVM)":
                st.markdown("""
                **Support Vector Machine (SVM)**:
                - SVM is a supervised learning algorithm used for classification and regression.
                - It works by finding the hyperplane that best separates the classes in the feature space.
                - Hyperparameters:
                  - **C**: Regularization parameter. Smaller values specify stronger regularization.
                  - **Kernel**: The type of kernel to use (e.g., linear, RBF).
                  - **Gamma**: Kernel coefficient for RBF and polynomial kernels.
                """)
            elif classifier_name == "Logistic Regression":
                st.markdown("""
                **Logistic Regression**:
                - A statistical method for binary classification that predicts the probability of an outcome.
                - It assumes a linear relationship between the input features and the log-odds of the outcome.
                - Hyperparameters:
                  - **C**: Regularization strength. Smaller values mean stronger regularization.
                  - **Max Iter**: Maximum number of iterations for the solver.
                """)
            elif classifier_name == "Random Forest":
                st.markdown("""
                **Random Forest**:
                - An ensemble learning method that builds multiple decision trees and merges their results.
                - Useful for both classification and regression tasks.
                - Hyperparameters:
                  - **n_estimators**: Number of trees in the forest.
                  - **Max Depth**: Maximum depth of each tree.
                  - **Bootstrap**: Whether to sample with replacement when building trees.
                """)

        # Model hyperparameter configuration
        def configure_classifier(clf_name):
            if clf_name == "Support Vector Machine (SVM)":
                C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
                kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"], key='kernel_SVM')
                gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"], key='gamma_SVM')
                return SVC(C=C, kernel=kernel, gamma=gamma, probability=True)

            elif clf_name == "Logistic Regression":
                C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
                max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter_LR')
                return LogisticRegression(C=C, max_iter=max_iter)

            elif clf_name == "Random Forest":
                n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators_RF')
                max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth_RF')
                bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ("True", "False"), key='bootstrap_RF') == 'True'
                return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)

        clf = configure_classifier(classifier_name)

        # Train the model
        clf.fit(x_train, y_train)

        # Metrics
        y_pred = clf.predict(x_test)
        acc = accuracy_score(y_test, y_pred)

        st.subheader(f"{classifier_name} Results")
        st.write(f"Accuracy: {acc:.2f}")

        # Classification Report table info
        display_classification_report(y_test, y_pred, class_names)

        # Plot
        plot_metrics(metrics, clf, x_test, y_test, y_pred, class_names)

if __name__ == "__main__":
    main()
