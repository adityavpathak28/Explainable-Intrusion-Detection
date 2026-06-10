# Explainable Intrusion Detection System using XGBoost and SHAP

This project presents an Explainable Intrusion Detection System (IDS) developed using the CICIDS2017 dataset. The framework combines XGBoost-based feature selection, multiclass attack classification, and SHAP explainability to provide both high detection performance and transparent decision-making.

The dataset was preprocessed by removing duplicates, handling missing values, and filtering minority classes. XGBoost feature importance was used to reduce the original 78 network traffic features to the 20 most informative attributes. A multiclass XGBoost classifier was then trained to detect various cyberattacks, including DDoS, DoS, PortScan, Botnet, FTP-Patator, SSH-Patator, and Web Attacks.

The proposed model achieved:

* Accuracy: 99.65%
* Precision: 99.63%
* Recall: 99.65%
* F1-Score: 99.62%

To improve interpretability, SHAP (SHapley Additive exPlanations) was integrated into the framework, enabling analysis of feature contributions and model decision behavior. The project demonstrates how Explainable AI (XAI) can enhance trust and transparency in cybersecurity applications while maintaining high intrusion detection performance.

## Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* XGBoost
* SHAP
* Matplotlib
* CICIDS2017 Dataset

## Key Features

* Large-scale network traffic preprocessing
* XGBoost-based feature selection
* Multiclass intrusion detection
* SHAP explainability analysis
* Confusion matrix and performance evaluation
* Research-oriented implementation suitable for cybersecurity studies
