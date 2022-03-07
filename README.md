# Ubiquant market predictions Time-Series Kaggle Competition
![cover](https://github.com/AmyRouillard/DSI-FCANS/blob/development/images/cover_image.png)

## 1. Introduction
This repository details the work done by Team C for the [Africa Data Science Intensive (DSI) program](http://dsi-program.com/) Module 2 task. The goal of the task was to compete in the time-series prediction competition by [Ubiquant](https://www.kaggle.com/c/ubiquant-market-prediction) on Kaggle. Time-series forecasting is a critical part of data science with many use cases such as epidimeology, inventory planning for businesses, stock trading etc. The different models and approaches used for the competition are detailed here and in our main notebook.  

### Code Files

| File  | Description |
|---|---|
| [Main Notebook](https://github.com/AmyRouillard/DSI-FCANS/blob/main/main-notebook-team-c.ipynb) | Main notebook with EDA and Discussions on Models |
| [Ensemble Inference](https://github.com/AmyRouillard/DSI-FCANS/blob/development/notebooks/ensemble-model-fscans.ipynb) | Notebook that loads weights and makes ensemble prediction |
| [Model 1 Training](https://github.com/AmyRouillard/DSI-FCANS/blob/main/notebooks/dnn-base-model-1.ipynb) | Model 1 DNN Model used in Ensemble Model |
| [Model 2 Training](https://github.com/AmyRouillard/DSI-FCANS/blob/main/notebooks/model-2-10fold-model-2.ipynb) | Model 2 DNN Model used in Ensemble Model |
| [Model 3 Training](https://github.com/AmyRouillard/DSI-FCANS/blob/main/notebooks/model-3-10fold.ipynb) | Model 3 DNN Model used in Ensemble Model |
| [Investment_ID Clustering](https://github.com/AmyRouillard/DSI-FCANS/blob/main/notebooks/investment-id-clustering.ipynb) | Kmeans Clustering of Investment IDs |
| [Model 1 DNN Optimization](https://github.com/AmyRouillard/DSI-FCANS/blob/main/scripts/base_optimize.py) | Study of relu activation Vs swish activation in model 1 |
| [Model 2 DNN Optimization](https://github.com/AmyRouillard/DSI-FCANS/blob/main/scripts/model2_optimize.py) | Study of dropout layers in model 2 |
| [Light GBM](https://github.com/AmyRouillard/DSI-FCANS/blob/main/notebooks/ubiquant-gbm2.ipynb) | Light GBM notebook for feature importance and model training |
| [EDA and Clustering](https://github.com/AmyRouillard/DSI-FCANS/blob/main/notebooks/lstm-0-eda-and-clustering.ipynb) | EDA and hierarchical clustering of investment IDs using Pearson correlations |
| [Data Preparation (LSTM)](https://github.com/AmyRouillard/DSI-FCANS/blob/main/notebooks/lstm-1-data-preperation.ipynb) | Data pre-proccessing for multi-variate time series model with LSTM |
| [Training (LSTM)](https://github.com/AmyRouillard/DSI-FCANS/blob/main/notebooks/lstm-2-training.ipynb) | Training of multi-variate time series model with LSTM |
| [Time Series Forcasting](https://github.com/AmyRouillard/DSI-FCANS/blob/main/notebooks/lstm-3-submission.ipynb) | Preditict target using multi-variate time series model with LSTM |

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



## 4. Approaches 

With all the models we used for this project, we tried different fine-tuning approaches hoping for better model performance. In this section, we will discuss and show all the fine tunings and different activation functions we tried and show their performance and final scores.

### DNN with the Leaky Relu Activation Function

Leaky ReLU function is an improved version of the ReLU activation function. As for the ReLU activation function, the gradient is 0 for all the values of inputs that are less than zero, which would deactivate the neurons in that region and may cause a dying ReLU problem. Leaky ReLU addresses this problem. Instead of defining the ReLU activation function as 0 for negative values of inputs(x), we define it as an extremely small linear component of x.

Relu Function:

  f(x)=max(0,x)

Leaky Relu Function:

   f(x)=max(0.01*x , x)

*source: https://www.mygreatlearning.com/blog/relu-activation-function/*

For the hyperparameter tuning, we modified the Dropout Rate, Learning Rate, and Decay Steps as shown in the table below to compare performance; 

|               |        |        |        | 
|---------------|--------|--------|--------|
| **Dropout Rate**  | 0.4    | 0.5    | 0.8    |  
| **Learning Rate** | 0.003  | 0.1    | 0.001  |   
| **Decay Steps**   | 9700   | 10000  | 10000  |   
|**RMSE**         | 0.9104 | 0.9123 | 0.9128 |   
|**MSE**           | 0.9151 | 0.9152 | 0.9143 |   
| **Score**         | 0.144  | 0.143  | 0.143  |   

### LightGBM

*source: https://neptune.ai/blog/lightgbm-parameters-guide*

In the first notebook, we ran the model wit a fixed learning rate and max_depth but adjusted these to the last two runs to compare results. It had no effect on the performance of the model
Also, all 300 features were used initially before using the built-in function for plotting the feature importance which reduced the features to only 188. The metric for the important features is labeled 'new features' in the table above. The only difference the new features added was a drastic reduction in training time else, all other metrics remained the same.
Fine-tuning the parameters of the LightGBM did not improve the model in any way. As seen from the table below, all the metrics and scores remained the same.

|                             |             |            |            |
|-----------------------------|-------------|------------|------------|
| **Objective**               |  Regression | Regression | Regression | 
| **Metric**                  |  MSE        | MSE        | MSE        |
| **Boosting_type**           |  gbdt       | gbdt       | gbdt       |
| **lambda_l1**               | 2.3e-05     | 2.3e-05    | 2.3e-05    |   
| **lambda_l2**               | 0.1         | 0.1        | 0.1        |
| **num_leaves**              | 4           | 10         | 4          |
| **Feature_fraction**        | 0.5         | 0.6        | 0.5        | 
| **Bagging_fraction**        | 0.9         | 0.8        | 0.9        |
| **Bagging_freq**            | 7           | 6          | 7          |
| **min_child_samples**       | 20          | 20         | 20         |
| **num_iterations**          | 1000        | 1000       | 1000       |
| **learning_rate**           |             | 0.1        | 0.1        |
| **max_depth**               |             | 10         | 10         |
| **MSE**                     | 0.8055      | 0.8055     | 0.8055     |    
| **MSE(new features)**       | 0.8052      | 0.8052     | 0.8975     |   
| **RMSE**                    | 0.8975      | 0.8975     | 0.8974     |  
| **RMSE(new features)**      | 0.8974      | 0.8974     | 0.8974     | 
| **Pearson Corr.**           | 0.1260      | 0.1260     | 0.1260     | 
| **Pearson Corr.(new features)** | 0.1272      | 0.1272     | 0.1272     |
| **Score**                       | 0.108       | 0.108      | 0.108      |


### DNN with Swish Activation Function

*source: [medium](https://medium.com/@neuralnets/swish-activation-function-by-google-53e1ea86f820#:~:text=Swish%20is%20a%20smooth%2C%20non,that%20actually%20creates%20the%20difference*) 

Swish is a smooth, non-monotonic function that consistently matches or outperforms ReLU on deep networks. 
|               |        |        |        |        |        |        |
|---------------|--------|--------|--------|--------|--------|--------|
| **Learning Rate** | 0.001  | 0.001  | 0.0005 | 0.0025 | 0.0005 | 0.0005 |
| **Epochs**        | 30     | 50     | 50     | 50     | 30     | 20     |
| **Pearson Corr.** | 0.1220 | 0.1164 | 0.1140 | 0.1193 | 0.1100 | 0.1194 |
| **Score**         | 0.15   | 0.149  | 0.146  | 0.142  | 0.147  | 0.144  |


### DNN with Mish Activation

*source: https://krutikabapat.github.io/Swish-Vs-Mish-Latest-Activation-Functions/*

The research we did showed Mish worked better than the Swish activation function when dropout rates between 0.2 to 0.75 is used but that was not the case when we applied it to our DNN model. Swish averagely produced better scores than Mish. 

|               |        |        |        |        |        |        |
|---------------|--------|--------|--------|--------|--------|--------|
|**Dropout Rate**  | 0.2    | 0.2    | 0.5    |   0.5  | 0.1    | 0.1    |
| **Epochs**        | 30     | 50     | 50     |  100   | 50     | 30     |
| **Pearson Corr.** | 0.1380 | 0.1280 | 0.1314 | 0.1338 | 0.1434 | 0.1319 |
| **Score**         | 0.143  | 0.146  | 0.143  | 0.143  | 0.146  | 0.143  |

### Optuna Study for Swish and Relu Activation on [Model 1 Base DNN](https://github.com/AmyRouillard/DSI-FCANS/blob/development/notebooks/dnn-base-model-1.ipynb)

An [optuna](https://optuna.org/) optimization study was carried out to evaluate the performance of the swish and activation functions using the script [here](https://github.com/AmyRouillard/DSI-FCANS/blob/development/scripts/base_optimize.py). The DNN was set to run 4 epochs for each trial run and 30 trials were carried out. The results for the MSE score obtained from the study are shown below,

![activation study](https://github.com/AmyRouillard/DSI-FCANS/blob/development/images/activation_study.png)

The swish activation function performs slightly better as earlier investigated. 

### Optuna Study for Dropout layers in [Model 2 DNN](https://github.com/AmyRouillard/DSI-FCANS/blob/development/scripts/model2_optimize.py)

A study was also carried out to investigate the effect of the two dropout layers in Model 2. The DNN was set to run 4 epochs for each trial run and 100 trials were carried out. The dropout variables "dropout_1" and "dropout_2" were optimized for a range of 0.1 to 0.9. The best value was obtained at {'dropout_1': 0.13889522793629328, 'dropout_2': 0.694167488259274}. The results for the MSE score obtained from the study are shown below,

![parameter importance](https://github.com/AmyRouillard/DSI-FCANS/blob/development/images/importance.png)

![parallel plot](https://github.com/AmyRouillard/DSI-FCANS/blob/development/images/parallel.png)

![optimization histor](https://github.com/AmyRouillard/DSI-FCANS/blob/development/images/optimization_history.png)


## 5. Conclusions

In conclusion we would like to mention a few ideas that due to time constraints we were unable to investigate further but that we believe could have the potential to improve our DNN ensemble model score. The first would be to find the optimal weighted average of the model predations. The second would be to perform more in-depth parameter tuning. For example, the optuna library could be used to optimize the number of layers. Finally, we would have liked to test the effectiveness of combining the results of the LGBM to reduce the number of features input to the DNN.

## 6. References
1. [Keras Documentation](https://keras.io/api/)
2. [Optuna](https://optuna.readthedocs.io/en/stable/)
3. [*Fast Data Loading and Low Mem with Parquet Files*](https://www.kaggle.com/robikscube/fast-data-loading-and-low-mem-with-parquet-files) by Rob Mulla
4. [*End to end simple and powerful DNN with LeakyReLU*](https://www.kaggle.com/pythonash/end-to-end-simple-and-powerful-dnn-with-leakyrelu) by pythonash
5. [*Using LightGBM for feature selection*](https://www.kaggle.com/melanie7744/using-lightgbm-for-feature-selection) by Melanie774
6. [*Ubiquant Market Prediction [ DNN ]*](https://www.kaggle.com/shamiaaftab/ubiquant-market-prediction-dnn) by Shamia Aftab
7. [*【Infer】DNN model ensemble*](https://www.kaggle.com/librauee/infer-dnn-model-ensemble) by 老肥
8. NVIDIA course: [Modeling Time Series Data with Recurrent Neural Networks in Keras](https://courses.nvidia.com/courses/course-v1:DLI+L-FX-24+V1/about)
9. [Keras: Multiple Inputs and Mixed Data](https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/)
## 7. Authors
* [Amy](https://github.com/AmyRouillard)
* [Nancy](https://github.com/NancyArmah)
* [Chris](https://github.com/chrisliti)
* [Sitwala](https://github.com/SitwalaM)

