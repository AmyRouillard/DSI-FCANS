# Ubiquant market predictions Time-Series Kaggle Competition
![cover](https://github.com/AmyRouillard/DSI-FCANS/blob/development/images/cover_image.png)

## 1. Introduction
add suff

## 2. Competition Description
Ubiquant market predictions

Regardless of your investment strategy, fluctuations are expected in the financial market. Despite this variance, professional investors try to estimate their overall returns. Risks and returns differ based on investment types and other factors, which impact stability and volatility. To attempt to predict returns, there are many computer-based algorithms and models for financial market trading. Yet, with new techniques and approaches, data science could improve quantitative researchers' ability to forecast an investment's return.

## 3. EDA/Feature Engineering 



## 4. Models 

### Model 1
### Model 2
### Model 3

## 5. Approaches

With all the models we used for this project, we tried different fine tuning approaches hoping for a better model performance. In this section, we will discuss and show all the fine tunings we did and show its performance and final scores.

### DNN with the Leaky Relu Activation Function
Leaky ReLU function is an improved version of the ReLU activation function. As for the ReLU activation function, the gradient is 0 for all the values of inputs that are less than zero, which would deactivate the neurons in that region and may cause dying ReLU problem. Leaky ReLU addresses this problem. Instead of defining the ReLU activation function as 0 for negative values of inputs(x), we define it as an extremely small linear component of x

Relu Function:

  f(x)=max(0,x)

Leaky Relu Function:

   f(x)=max(0.01*x , x)

*source:https://www.mygreatlearning.com/blog/relu-activation-function/*

For the hyper parameter tuning, we modified the Dropout Rate, Learning Rate and Decay Steps.


|--------------|---|---|---|---|
|Dropout Rate  |   |   |   |   |
|Learning Rate |   |   |   |   |
|Decay Steps   |   |   |   |   |
|RMSE          |   |   |   |   |
|MSE           |   |   |   |   |
|SCore         |   |   |   |   |



## 6. Final Model Results

## 7. Conclusions


## 8. References

## 9. Authors
* [Amy](https://github.com/AmyRouillard)
* [Nancy]
* [Chris]
* [Sitwala](https://github.com/SitwalaM)

