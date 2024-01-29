**Overview**
This project focuses on predicting whether a customer is likely to default on a payment, with a specific emphasis on addressing the needs of credit card companies. The primary objective is to enhance the accuracy of identifying customers who might appear non-defaulting but are at risk.

**Goals**
1.	Default Prediction: Accurately predict the likelihood of a customer defaulting on their payment.
2.	**Enhanced Recall for Potential Risks:** Emphasize recall, particularly for customers who may not initially appear likely to default but are potential risks. The primary focus is on improving the recall score for the class representing customers who do not default, ensuring that potential risks are identified even if they don't exhibit obvious signs of default.

**Methodology**
_Data Preprocessing_
1.	Ordinal Encoding:
•	Applied ordinal encoding to the CAT_GAMBLING column to represent categorical variables in a meaningful order.
2.	Feature Removal:
•	Excluded the customer_id column, considering it irrelevant for predictive modeling.
3.	Outlier Removal:
•	Identified and removed outliers using the Interquartile Range (IQR) method to enhance the robustness of the dataset.
4.	Data Scaling:
•	Utilized the Standard Scaler to standardize numerical features, ensuring that all variables contribute equally to the model.
5.	Upsampling with SMOTE:
•	Addressed class imbalance by employing Synthetic Minority Over-sampling Technique (SMOTE), ensuring a balanced representation of target classes.
6.	Dimensionality Reduction with PCA:
•	Applied Principal Component Analysis (PCA) to reduce the dimensionality of the dataset while retaining essential information.

_Algorithm Exploration_
Implemented various classification algorithms to understand their performance on the given data:
•	K Neighbors Classifier (KNC): Explored the KNeighborsClassifier for classification based on the majority vote of its k-nearest neighbors.
•	Random Forest: Investigated the ensemble learning method using multiple decision trees to enhance predictive accuracy and control overfitting.
•	Support Vector Machine (SVM): Explored the powerful algorithm for both classification and regression tasks, finding the hyperplane that best separates classes.
•	Bagging: Investigated bagging, a parallel ensemble method, to improve the stability and accuracy of the models.
•	Voting Classifier: Explored a voting classifier to combine predictions from multiple algorithms to identify potential synergies.
•	Logistic Regression: Explored logistic regression for binary classification, providing insights into the probability of default.
This exploratory approach allows us to identify the most suitable algorithm for the given dataset and prediction task. The final model selection is based on the comprehensive evaluation of each algorithm's performance.

_Model Evaluation_
1.	Confusion Matrix and Classification Report:
•	Utilized confusion matrices and the classification_report function to generate a comprehensive report, including precision, recall, F1 score, and support for each class. This approach provides a detailed assessment of the models on both training and test datasets.
2.	Cross-Validation Score:
•	Utilized cross-validation to estimate the performance of the models on different subsets of the data, ensuring generalizability and reducing overfitting.
3.	Hyperparameter Tuning with Grid Search CV:
•	Conducted hyperparameter tuning using Grid Search Cross-Validation to systematically explore a range of hyperparameter combinations and identify the optimal set for each classification algorithm.

**Project App - Tailored for Credit Card Companies**
I have developed a user-friendly application using Streamlit that simplifies the process of obtaining default predictions based on customer data. The app takes input in the form of a CSV file and provides predictions in a downloadable CSV file, catering specifically to the needs of credit card companies.
Features
•	CSV Input: Upload a CSV file containing customer data for seamless predictions.
•	Prediction Output: Download the results as a CSV file for further analysis.
•	User-friendly Interface: Intuitive design for a smooth user experience.

**Contributor**
 Dhruv Desai
