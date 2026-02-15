## 📌 Loan RePayment Predicton (Project Summary)

 ### 📈 Objectives of this project
* Build an ML model to predict loan repayment status (not.fully.paid).
* Support credit-risk management by identifying high-risk borrowers.
* Apply supervised learning (binary classification) on historical loan data.
* Benchmark multiple ensemble models for accuracy and reliability.

### 🛠 Tools & Platform

1. Python (Jupyter Notebook).
2. Libraries: → pandas, numpy, matplotlib, seaborn, scikit-learn.
3. Techniques:→  Label Encoding, Log Transformation, Standard Scaling.
4. Validation: → StratifiedKFold, GridSearchCV (recall-weighted scoring).
5. 
🔍 Findings

→ Dataset highly imbalanced (85% fully paid vs 15% defaults).  
→ Random Forest (600 trees) achieved ~85% accuracy.  
→ Fully-paid loans predicted with recall = 1.0.  
→ Defaults (minority class) had near-zero recall, mostly misclassified.

🔹 Achievements

* Designed an end-to-end ML pipeline from preprocessing to prediction.
* Benchmarked Decision Tree, Bagging, AdaBoost, Gradient Boosting, and Random Forest.
* Optimized models with cross-validation and hyperparameter tuning.
* Identified the core limitation (class imbalance causing poor minority detection).

📊 Outcomes

→ Produced repayment predictions for unseen loan data.  
→ Delivered insights showing overall accuracy is misleading due to imbalance.  
→ Highlighted operational risk of false negatives (missed defaulters).  
→ Recommended SMOTE/ADASYN, threshold tuning, and cost-sensitive learning for improvement.

🗝 Key Points

Dataset: 9,578 loans, 14 features, no missing values.  
Train-test split: 70/30 (6,704 vs 2,874).  
Target variable: not.fully.paid (binary classification).
Best model: Random Forest with ~85% accuracy but poor minority recall.
