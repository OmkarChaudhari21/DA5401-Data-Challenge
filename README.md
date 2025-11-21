# DA5401-2025 Data Challenge  
**Metric Learning for Conversational AI Evaluation**  
**Course:** DA5401 – Data Challenge (IIT Madras)  
**Student:** Omkar Ashok Chaudhari (NA22B059)

---

## Overview  
This project was completed as part of the DA5401 end-semester data challenge.  
The task focused on predicting a fitness score (0–10) for a given prompt–response pair with respect to a specific evaluation metric. Each metric was provided as a text embedding, and each training example contained the system prompt, user prompt, model response, and the corresponding ground-truth score.

The challenge represents a practical metric-learning problem where the model must understand how well a response aligns with the intent of an evaluation metric.

---

## What Was Done  
- Combined all text fields into a single representation and transformed them using TF-IDF followed by SVD for dense embeddings.  
- Used the provided metric-name embeddings to capture the meaning of each evaluation metric.  
- Trained two models:
  - A gradient-boosting regression model.  
  - An ordinal classification setup with 10 logistic models for thresholds 1–10.  
- Blended the outputs of both models (0.6 ordinal, 0.4 regression) to get the final predictions.  
- Generated the final submission file that achieved **3.742 RMSE** on Kaggle.

---

## Conclusion  
The project showed that a simple, lightweight pipeline can perform well even without large transformer models. Combining text features with metric embeddings helped the model connect the prompt–response pair to the metric definition. The blended approach gave stable and accurate predictions, leading to the best leaderboard score among all tried methods.

This work demonstrates a practical approach to automated evaluation of conversational AI agents using metric learning concepts.
