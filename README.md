# Ubiquant market predictions Time-Series Kaggle Competition
![cover](https://github.com/AmyRouillard/DSI-FCANS/blob/development/images/cover_image.png)

## 1. Introduction
This repository details the work done by Team C for the [Africa Data Science Intensive (DSI) program](http://dsi-program.com/) Module 2 task. The goal of the task was to compete in the time-series prediction competition by [Ubiquant](https://www.kaggle.com/c/ubiquant-market-prediction) on Kaggle. Time-series forecasting is a critical part of data science with many use cases such as epidimeology, inventory planning for businesses, stock trading etc. The different models and approaches used for the competition are detailed here.  

## 2. Competition Description
Competion Description taken from Kaggle:

"Regardless of your investment strategy, fluctuations are expected in the financial market. Despite this variance, professional investors try to estimate their overall returns. Risks and returns differ based on investment types and other factors, which impact stability and volatility. To attempt to predict returns, there are many computer-based algorithms and models for financial market trading. Yet, with new techniques and approaches, data science could improve quantitative researchers' ability to forecast an investment's return."

"In this competition, you’ll build a model that forecasts an investment's return rate. Train and test your algorithm on historical prices. Top entries will solve this real-world data science problem with as much accuracy as possible."

## 3. EDA
### Dataset Description
* **row_id** - A unique identifier for the row.
* **time_id** - The ID code for the time the data was gathered. The time IDs are in order, but the real time between the time IDs is not constant and will likely be shorter for the final private test set than in the training set.
* **investment_id** - The ID code for an investment. Not all investment have data in all time IDs.
* **target** - The target.
* **[f_0:f_299]** - Anonymized features generated from market data.

## 4. Models 

### Model 1
### Model 2
### Model 3

## 5. Approaches

With all the models we used for this project, we tried different fine-tuning approaches hoping for better model performance. In this section, we will discuss and show all the fine tunings and different activation functions we tried and show their performance and final scores.

### DNN with the Leaky Relu Activation Function

Leaky ReLU function is an improved version of the ReLU activation function. As for the ReLU activation function, the gradient is 0 for all the values of inputs that are less than zero, which would deactivate the neurons in that region and may cause a dying ReLU problem. Leaky ReLU addresses this problem. Instead of defining the ReLU activation function as 0 for negative values of inputs(x), we define it as an extremely small linear component of x.

Relu Function:

  f(x)=max(0,x)

Leaky Relu Function:

   f(x)=max(0.01*x , x)

*source:https://www.mygreatlearning.com/blog/relu-activation-function/*

For the hyperparameter tuning, we modified the Dropout Rate, Learning Rate, and Decay Steps as shown in the table below; 

|               |        |        |        | 
|---------------|--------|--------|--------|
| Dropout Rate  | 0.4    | 0.5    | 0.8    |  
| Learning Rate | 0.003  | 0.1    | 0.001  |   
| Decay Steps   | 9700   | 10000  | 10000  |   
| RMSE          | 0.9104 | 0.9123 | 0.9128 |   
| MSE           | 0.9151 | 0.9152 | 0.9143 |   
| Score         | 0.144  | 0.143  | 0.143  |   

### LightGBM

*source: https://neptune.ai/blog/lightgbm-parameters-guide*

|                             |             |            |            |
|-----------------------------|-------------|------------|------------|
| Objective                   |  Regression | Regression | Regression | 
| Metric                      |  MSE        | MSE        | MSE        |
| Boosting_type               |  gbdt       | gbdt       | gbdt       |
| lambda_l1                   | 2.3e-05     | 2.3e-05    | 2.3e-05    |   
| lambda_l2                   | 0.1         | 0.1        | 0.1        |
| num_leaves                  | 4           | 10         | 4          |
| Feature_fraction            | 0.5         | 0.6        | 0.5        | 
| Bagging_fraction            | 0.9         | 0.8        | 0.9        |
| Bagging_freq                | 7           | 6          | 7          |
| min_child_samples           | 20          | 20         | 20         |
| num_iterations              | 1000        | 1000       | 1000       |
| learning_rate               |             | 0.1        | 0.1        |
| max_depth                   |             | 10         | 10         |
| MSE                         | 0.8055      | 0.8055     | 0.8055     |    
| MSE(new features)           | 0.8052      | 0.8052     | 0.8975     |   
| RMSE                        | 0.8975      | 0.8975     | 0.8974     |  
| RMSE(new features)          | 0.8974      | 0.8974     | 0.8974     | 
| Pearson Corr.               | 0.1260      | 0.1260     | 0.1260     | 
| Pearson Corr.(new features) | 0.1272      | 0.1272     | 0.1272     |
| Score                       | 0.108       | 0.108      | 0.108      |

In the LightGBM notebook, we first used all 300 features before using the built-in function for plotting the feature importance which reduced the features to only 188. The metric for the important features is labeled 'new features' in the table above. The only difference the new features added was a drastic reduction in training time else, all other metrics remained the same.
Fine-tuning the parameters of the LightGBM did not improve the model in any way. As seen from the table above, all the metrics and scores remained the same.

### DNN with Swish Activation Function

Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks. It is unbounded above and bounded below & it is the non-monotonic attribute that actually creates the difference.
  
|               |        |        |        |        |        |        |
|---------------|--------|--------|--------|--------|--------|--------|
| Learning Rate | 0.001  | 0.001  | 0.0005 | 0.0025 | 0.0005 | 0.0005 |
| Epochs        | 30     | 50     | 50     | 50     | 30     | 20     |
| Pearson Corr. | 0.1220 | 0.1164 | 0.1140 | 0.1193 | 0.1100 |        |
| Score         | 0.15   | 0.149  | 0.146  | 0.142  | 0.147  | 0.144  |


### DNN with Mish Activation

*source: https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/*

The research we did showed Mish worked better than the Swish activation function but that was not the case when we applied it to our DNN model. The performance kept diminishing with every change we made to the hyperparameters.

|               |        |        |        |        |       |       |
|---------------|--------|--------|--------|--------|-------|-------|
| Dropout Rate  | 0.2    | 0.2    | 0.5    |   0.5  | 0.1   | 0.1   |
| Epochs        | 30     | 50     | 50     |  100   | 50    | 30    |
| Pearson Corr. |        |        |        |        |       |       |
| Score         | 0.143  | 0.146  | 0.143  |  0.143 | 0.146 | 0.143 |



## 6. Final Model Results

## 7. Conclusions


## 8. References
1. [Keras Documentation](https://keras.io/api/)
2. [Optuna](https://optuna.readthedocs.io/en/stable/)
## 9. Authors
* [Amy](https://github.com/AmyRouillard)
* [Nancy](https://github.com/NancyArmah)
* [Chris](https://github.com/chrisliti)
* [Sitwala](https://github.com/SitwalaM)

