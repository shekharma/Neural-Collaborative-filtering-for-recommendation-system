
# Neural Collaborative Filtering for Recommendation Systems

## Overview
This project aims to build a recommendation system using neural collaborative filtering techniques. The system recommends a set of items to a user based on their past preferences.

## Objective
The primary objective is to leverage neural network architectures to perform collaborative filtering on implicit feedback data. This involves recommending items to users without explicit ratings, such as clicks, views, or purchases.
Certainly! Here's the revised methodology section for your README file:

## Methodology
## Neural Collaborative Filtering
![image](https://github.com/shekharma/Neural-Collaborative-filtering-for-recommendation-system/assets/122733304/bd97d5ce-b644-423c-b004-3eff159c9736)

### Fusion of GMF and MLP
- **Generalized Matrix Factorization (GMF)**: GMF is a collaborative filtering technique that learns latent factors for users and items through matrix factorization. Each user and item is represented as a vector of latent factors. The recommendation process involves computing the inner product between these vectors to predict preferences or ratings.
  
- **Multi-Layer Perceptron (MLP)**: MLP is a type of feedforward neural network with multiple layers of nodes (neurons) between the input and output layers. Each neuron calculates a weighted sum of its inputs, applies an activation function, and passes the result to the next layer. MLPs are effective at capturing complex non-linear relationships in the data.
  ![image](https://github.com/shekharma/Neural-Collaborative-filtering-for-recommendation-system/assets/122733304/0998c50e-4fa9-4a9e-8da9-8f1237d20912)

The fusion of GMF and MLP combines the strengths of both approaches:
- **GMF** focuses on capturing linear relationships between user and item latent factors through matrix factorization.
- **MLP** captures non-linear interactions and patterns in the data through its multi-layer architecture and activation functions.

By combining the outputs of GMF and MLP architectures, the recommendation system leverages both linear and non-linear features. This fusion approach enhances the model's ability to capture diverse aspects of user-item interactions, leading to more accurate and personalized recommendations based on users' past preferences.


## Usage
1. **Data Preparation**: Prepare the dataset for training and evaluation. This typically involves organizing user-item interactions, including positive and negative feedback.
   
2. **Model Training**: Train the neural collaborative filtering model using the prepared dataset. This includes:
   - Training the GMF and MLP components separately.
   - Combining the outputs of both architectures to generate recommendations.
   - Optimizing the model parameters using techniques like stochastic gradient descent.
   
3. **Evaluation**: Evaluate the performance of the trained model using the Hit Ratio (HR) metric. This indicates how often the recommended items are relevant to the user's preferences.

## Results
- The recommendation system achieved a Hit Ratio (HR) of 0.66 on the MovieLens dataset, indicating its effectiveness in recommending relevant items to users.
- Additional experiments and analyses may be conducted to further fine-tune the model and improve its performance on different datasets.

## Dependencies
- Python 3.11
- PyTorch 
- MovieLens dataset 

## References
- Xiangnan He, Lizi Liao, Hanwang Zhang, Liqiang Nie, Xia Hu & Tat-Seng Chua, Neural Collaborative Filtering, 2017, https://arxiv.org/abs/1708.05031

- Official NCF implementation [Keras with Theano]: https://github.com/hexiangnan/neural_collaborative_filtering

- Other nice NCF implementation [Pytorch]: https://github.com/LaceyChen17/neural-collaborative-filtering
---

This README file provides an overview of the project, including its objectives, methodology, usage instructions, results, dependencies, and references to relevant research papers. It serves as a guide for users and contributors to understand the purpose and implementation details of the neural collaborative filtering recommendation system.
