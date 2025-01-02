import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, silhouette_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import joblib

# lets cache this so we don't reload data every time
@st.cache_data
def load_data(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Diabetes":
        data = datasets.load_diabetes()
    elif dataset_name == "Digits":
        data = datasets.load_digits()
    elif dataset_name == "Wine":
        data = datasets.load_wine()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        st.error("Invalid dataset name.")
        return None, None

    if hasattr(data, 'data') and hasattr(data, 'target'):
        df = pd.DataFrame(data.data, columns=data.feature_names)
        if dataset_name in ["Iris", "Digits", "Wine", "Breast Cancer"]:
            encoder = LabelEncoder()
            df['target'] = encoder.fit_transform(data.target)
        else:
            df['target'] = pd.Series(data.target)
        return df, df['target']
    else:
        return data, None

# make the data look nice and clean
def preprocess_data(data, preprocessing_method):
    if preprocessing_method == "Standard Scaling":
        scaler = StandardScaler()
    elif preprocessing_method == "Min-Max Scaling":
        scaler = MinMaxScaler()
    elif preprocessing_method == "Robust Scaling":
        scaler = RobustScaler()
    else:
        return data
    
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, columns=data.columns.astype(str))  # Ensure column names are strings

# dimension reduction magic
def apply_pca(data, n_components):
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)
    columns = [f"PC{i+1}" for i in range(n_components)]
    return pd.DataFrame(pca_result, columns=columns), pca.explained_variance_ratio_

# grab the right model based on what we want to do
def get_model(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression
    elif model_name == "Linear Regression":
        return LinearRegression
    elif model_name == "SVM":
        return SVC
    elif model_name == "Random Forest Classifier":
        return RandomForestClassifier
    elif model_name == "Random Forest Regressor":
        return RandomForestRegressor
    elif model_name == "K-Means":
        return KMeans
    elif model_name == "KNN":
        return KNeighborsClassifier
    elif model_name == "XGBoost Classifier":
        return XGBClassifier
    elif model_name == "Naive Bayes":
        return GaussianNB
    elif model_name == "Decision Tree":
        return DecisionTreeClassifier
    else:
        st.error("Invalid model name.")
        return None

# let users play with model settings
def get_model_params(model_name):
    params = dict()
    if model_name == "Logistic Regression":
        params["C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        params["max_iter"] = st.sidebar.slider("max_iter", 100, 2000, 500)  # Increased default and range
        params["solver"] = st.sidebar.selectbox(
            "Solver", ["lbfgs", "liblinear", "sag", "saga"], index=0
        )  # Added solver selection
    elif model_name == "Linear Regression":
        params["fit_intercept"] = st.sidebar.checkbox("fit_intercept", True)
    elif model_name == "SVM":
        params["C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0)
        params["gamma"] = st.sidebar.slider("gamma", 0.01, 10.0, 0.1)
        params["probability"] = st.sidebar.checkbox("Enable probability estimates", value=True)  # Added
    elif model_name == "Random Forest Classifier" or model_name == "Random Forest Regressor":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 10, 500, 100)
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 32, 10)
    elif model_name == "K-Means":
        params["n_clusters"] = st.sidebar.slider("n_clusters", 2, 20, 5)
    elif model_name == "KNN":
        params["n_neighbors"] = st.sidebar.slider("n_neighbors", 1, 20, 5)
    elif model_name == "XGBoost Classifier":
        params["n_estimators"] = st.sidebar.slider("n_estimators", 50, 500, 100)
        params["learning_rate"] = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.1)
    elif model_name == "Decision Tree":
        params["max_depth"] = st.sidebar.slider("max_depth", 1, 32, 5)
        params["min_samples_split"] = st.sidebar.slider("min_samples_split", 2, 20, 2)
    return params

# do the actual training work
def train_model(model_class, params, data, target):
    if target is None:  # For clustering, target is None
        model = model_class(**params)
        model.fit(data)
        return model
    
    X = data.drop('target', axis=1)
    y = target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = model_class(**params)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

# check how well our model is doing
def evaluate_model(model, X_test, y_test, model_name):
    if y_test is not None:
      y_pred = model.predict(X_test)

      metrics = {}
      if model_name in ["Logistic Regression", "SVM", "Random Forest Classifier", "KNN", "Naive Bayes", "Decision Tree"]:
          metrics["accuracy"] = accuracy_score(y_test, y_pred)
          with st.expander("Classification Report"):
            st.text(classification_report(y_test, y_pred))
      elif model_name in ["Linear Regression", "Random Forest Regressor"]:
          metrics["mse"] = mean_squared_error(y_test, y_pred)
          metrics["r2"] = r2_score(y_test, y_pred)
      return metrics
    else:
      metrics = {}
      if model_name == "K-Means":
        metrics["silhouette_score"] = silhouette_score(X_test, model.labels_)
      return metrics

# make some pretty charts
def generate_plots(model, X_train, X_test, y_train, y_test, model_name, metrics, data):
    if y_test is not None:
        if model_name in ["Logistic Regression", "SVM", "Random Forest Classifier", "KNN", "Naive Bayes", "Decision Tree"]:
            # Confusion Matrix
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            with st.expander("Confusion Matrix"):
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", ax=ax)
                plt.xlabel("Predicted")
                plt.ylabel("True")
                st.pyplot(fig)

            # Scatter plot of data points (first two features)
            if X_test.shape[1] >= 2:
                with st.expander("Scatter Plot (First Two Features)"):
                    fig, ax = plt.subplots()
                    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_pred, cmap="viridis", alpha=0.7, label="Predicted")
                    plt.scatter(X_test.iloc[:, 0], X_test.iloc[:, 1], c=y_test, cmap="plasma", marker="x", alpha=0.7, label="Actual")
                    plt.xlabel(X_test.columns[0])
                    plt.ylabel(X_test.columns[1])
                    plt.legend()
                    st.pyplot(fig)

            # Modified ROC Curve for multiclass
            if hasattr(model, "predict_proba"):
                with st.expander("ROC Curve"):
                    fig, ax = plt.subplots()
                    
                    # Check if binary or multiclass
                    n_classes = len(np.unique(y_test))
                    if n_classes == 2:
                        # Binary classification
                        y_pred_proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                    else:
                        # Multiclass classification
                        y_pred_proba = model.predict_proba(X_test)
                        
                        # Compute ROC curve and ROC area for each class
                        fpr = dict()
                        tpr = dict()
                        roc_auc = dict()
                        for i in range(n_classes):
                            fpr[i], tpr[i], _ = roc_curve(
                                (y_test == i).astype(int), 
                                y_pred_proba[:, i]
                            )
                            roc_auc[i] = auc(fpr[i], tpr[i])
                            plt.plot(fpr[i], tpr[i], label=f'ROC curve class {i} (AUC = {roc_auc[i]:.2f})')
                    
                    plt.plot([0, 1], [0, 1], 'k--')
                    plt.xlim([0.0, 1.0])
                    plt.ylim([0.0, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (ROC)')
                    plt.legend(loc="lower right")
                    st.pyplot(fig)
            else:
                st.warning("ROC Curve is not available for this model.")

        elif model_name in ["Linear Regression", "Random Forest Regressor"]:
            # Scatter plot of actual vs predicted values
            y_pred = model.predict(X_test)

            with st.expander("Actual vs Predicted"):
                fig, ax = plt.subplots()
                plt.scatter(y_test, y_pred)
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
                st.pyplot(fig)

            # Residual Plot
            with st.expander("Residual Plot"):
                fig, ax = plt.subplots()
                residuals = y_test - y_pred
                plt.scatter(y_pred, residuals)
                plt.xlabel("Predicted")
                plt.ylabel("Residuals")
                plt.axhline(y=0, color='k', linestyle='--')
                st.pyplot(fig)

    else:
        if model_name == "K-Means":
            # Scatter plot of clusters (first two features)
            if data.shape[1] >= 2:
                with st.expander("Scatter Plot (First Two Features)"):
                    fig, ax = plt.subplots()
                    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=model.labels_, cmap='viridis')
                    plt.xlabel(data.columns[0])
                    plt.ylabel(data.columns[1])
                    st.pyplot(fig)

    if model_name in ["Random Forest Classifier", "Random Forest Regressor", "XGBoost Classifier"] and X_test is not None:
        with st.expander("Feature Importances"):
            fig, ax = plt.subplots()
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)
            plt.barh(range(len(sorted_idx)), importances[sorted_idx])
            plt.yticks(range(len(sorted_idx)), X_test.columns[sorted_idx])
            plt.xlabel("Importance")
            plt.title("Feature Importances")
            st.pyplot(fig)

# main UI stuff below
st.title("Chun's Interactive Machine Learning Model Explorer")

# quick lookup for what each dataset is good for
DATASET_TYPES = {
    "Iris": "classification",
    "Digits": "classification",
    "Wine": "classification",
    "Breast Cancer": "classification",
    "Diabetes": "regression"
}

# what kind of problems each model can solve
MODEL_TYPES = {
    "Logistic Regression": "classification",
    "SVM": "classification",
    "Random Forest Classifier": "classification",
    "KNN": "classification",
    "Naive Bayes": "classification",
    "Decision Tree": "classification",
    "XGBoost Classifier": "classification",
    "Linear Regression": "regression",
    "Random Forest Regressor": "regression",
    "K-Means": "clustering"
}

# sidebar stuff for all our controls
with st.sidebar:
    st.header("Settings")
    
    # Dataset Selection
    dataset_name = st.selectbox("Select Dataset", list(DATASET_TYPES.keys()))
    dataset_type = DATASET_TYPES[dataset_name]
    
    # Preprocessing
    st.subheader("Preprocessing")
    preprocessing_method = st.selectbox("Select Preprocessing", 
        ["None", "Standard Scaling", "Min-Max Scaling", "Robust Scaling"])
    
    # Dimensionality Reduction
    use_pca = st.checkbox("Apply PCA")
    if use_pca:
        n_components = st.slider("Number of Components", 2, 10, 2)
    
    # Model Selection
    st.subheader("Model Selection")
    
    # Filter models based on dataset type
    if dataset_type == "classification":
        available_models = [
            "Logistic Regression", "SVM",
            "Random Forest Classifier", "KNN",
            "Naive Bayes", "Decision Tree",
            "XGBoost Classifier"
        ]
    elif dataset_type == "regression":
        available_models = [
            "Linear Regression", "Random Forest Regressor"
        ]
    else:
        available_models = ["K-Means"]
    
    model_name = st.selectbox("Select Model", available_models)

# Load and preprocess data
data, target = load_data(dataset_name)
if data is not None:
    # Preprocessing
    if preprocessing_method != "None":
        if model_name != "K-Means":
            data = preprocess_data(data.drop('target', axis=1), preprocessing_method)
            data['target'] = target
        else:
            data = preprocess_data(data, preprocessing_method)
    
    # PCA
    if use_pca:
        if model_name != "K-Means":
            X_pca, explained_var = apply_pca(data.drop('target', axis=1), n_components)
            st.write("Explained variance ratio:", explained_var)
            data = pd.concat([X_pca, pd.Series(target, name='target')], axis=1)
        else:
            X_pca, explained_var = apply_pca(data, n_components)
            st.write("Explained variance ratio:", explained_var)
            data = X_pca
    
    # Model training and evaluation
    model_class = get_model(model_name)
    params = get_model_params(model_name)
    
    # Add cross-validation
    if st.sidebar.checkbox("Perform Cross-Validation"):
        n_folds = st.sidebar.slider("Number of folds", 2, 10, 5)
        if model_class is not None:
            model = model_class(**params)
            if dataset_type == "classification":
                cv_scores = cross_val_score(model, data.drop('target', axis=1), target, cv=n_folds)
                mean_score = cv_scores.mean()
                std_score = cv_scores.std() * 2
                metric = "Accuracy"
                st.write(f"Cross-validation scores: {cv_scores}")
                st.write(f"Mean CV {metric}: {mean_score:.3f} (+/- {std_score:.3f})")
            elif dataset_type == "regression":
                cv_scores = cross_val_score(model, data.drop('target', axis=1), target, cv=n_folds, scoring='neg_mean_squared_error')
                cv_scores = -cv_scores  # Convert to positive MSE
                mean_score = cv_scores.mean()
                std_score = cv_scores.std() * 2
                metric = "MSE"
                st.write(f"Cross-validation scores: {cv_scores}")
                st.write(f"Mean CV {metric}: {mean_score:.3f} (+/- {std_score:.3f})")
            elif dataset_type == "clustering":
                from sklearn.model_selection import KFold
                cv_scores = []
                kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                for train_idx, test_idx in kf.split(data):
                    model.fit(data.iloc[train_idx])
                    labels = model.predict(data.iloc[test_idx])
                    silhouette = silhouette_score(data.iloc[test_idx], labels)
                    cv_scores.append(silhouette)
                st.write(f"Cross-validation silhouette scores: {cv_scores}")
                st.write(f"Mean CV Silhouette Score: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores) * 2:.3f})")
    
    # Train final model
    if model_class is not None:
        if model_name != "K-Means":
            if dataset_type == "classification":
                model, X_train, X_test, y_train, y_test = train_model(model_class, params, data, target)
            elif dataset_type == "regression":
                model, X_train, X_test, y_train, y_test = train_model(model_class, params, data, target)
        else:
            model = train_model(model_class, params, data, target)
        
        st.write("Model trained successfully!")
        
        if model_name != "K-Means":
            metrics = evaluate_model(model, X_test, y_test, model_name)
            st.write("Performance Metrics:", metrics)
            generate_plots(model, X_train, X_test, y_train, y_test, model_name, metrics, data)
        else:
            metrics = evaluate_model(model, data, None, model_name)
            st.write("Performance Metrics:", metrics)
            generate_plots(model, None, data, None, None, model_name, metrics, data)
        
        # Add model export option
        if st.button("Export Model"):
            joblib.dump(model, f"{model_name}_{dataset_name}.joblib")
            st.success("Model exported successfully!")

# Add tabs for different visualizations
if 'model' in locals():
    tab1, tab2, tab3 = st.tabs(["Model Performance", "Feature Analysis", "Predictions"])
    
    with tab1:
        st.subheader("Model Performance")
        # Display relevant metrics
        if model_name == "K-Means":
            st.write("Silhouette Score:", metrics.get("silhouette_score", "N/A"))
        elif model_name in ["Linear Regression", "Random Forest Regressor"]:
            st.write("MSE:", metrics.get("mse", "N/A"))
            st.write("R²:", metrics.get("r2", "N/A"))
        else:
            st.write("Accuracy:", metrics.get("accuracy", "N/A"))
    
    with tab2:
        st.subheader("Feature Analysis")
        # Display feature importances if available
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)
            fig, ax = plt.subplots()
            plt.barh(range(len(sorted_idx)), importances[sorted_idx])
            plt.yticks(range(len(sorted_idx)), X_test.columns[sorted_idx])
            plt.xlabel("Importance")
            plt.title("Feature Importances")
            st.pyplot(fig)
        else:
            st.write("No feature importances for this model.")
    
    with tab3:
        if model_name != "K-Means":
            st.subheader("Make Predictions")
            # Add interactive prediction interface
            if st.checkbox("Enter custom values for prediction"):
                user_input = {}
                for feature in data.drop('target', axis=1).columns:
                    user_input[feature] = st.number_input(f"Enter {feature}", value=0.0)
                
                if st.button("Predict"):
                    user_df = pd.DataFrame([user_input])
                    prediction = model.predict(user_df)
                    st.write("Prediction:", prediction[0])
        else:
            st.warning("Prediction is not applicable for K-Means clustering.")
