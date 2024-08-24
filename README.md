# Deep Learning-Based Modeling for Power Converters via Physics-Guided Hierarchical Res-LSTM Frameworks
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-11.3.1-green.svg?style=plastic)
![MATLAB 2021b](https://img.shields.io/badge/MATLAB-2024a-blue.svg?style=plastic)
![PLECS 4.5.6](https://img.shields.io/badge/PLECS-4.5.6-green.svg?style=plastic)
[![Build status](https://ci.appveyor.com/api/projects/status/8msiklxfbhlnsmxp/branch/master?svg=true)](https://ci.appveyor.com/project/TadasBaltrusaitis/openface/branch/master)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is a GitHub repository for the **`Physics-Guided Hierarchical Res-LSTM Framework (PhyRes-LSTM)`**.

## Introduction 

Power converters are extensively utilized in many applications such as consumer electronics, industrial electronics, electric vehicles (EVs), energy storage systems, and distributed generation systems to generate either a regulated dc or ac voltage source and manage the power flow by mainly controlling the switching actions of power semiconductor devices [[1]](https://ieeexplore.ieee.org/document/9351620). `In all of these applications, the ability to accurately describe the behavior of the converter is of critical importance, as it assists in component selection, controller design, and the preliminary evaluation of the entire DC system`.  Power converters exhibit an inherent nonlinear behavior due to the switching action of the power switches, thus making its modeling difficult due to the complexity of the required models.

![image](https://github.com/user-attachments/assets/85b4f94c-0c1b-49fe-9c5a-fb75fb48aa4e)

Fig. 1. Diagram of a hierarchical microgrid with power converters

Generally, physical modeling relies on a knowledge of the values of the dc–dc converter parameters, both the passive elements and the control circuit. In addition, an accurate model should be able to reproduce ripples in currents and voltages, transients caused by switching on and off operations, etc., which is difficult to generate with physical-based modeling methods [[2]](https://ieeexplore.ieee.org/document/9492829). Over the past few years, classical machine learning (ML) and deep learning (DL) algorithms have been used for the data-driven modeling of power converters with simple topologies. It is worth noting that this modeling approach falls under the task of time series forecasting, so models that are related to time series and their theoretical analysis are also applicable in this situation. However, conventional time-series data-driven approaches for power converter modeling are data-intensive, uninterpretable, and lack out-of-domain extrapolation capability. Recent physics-informed modeling methods combine physics into data-driven models using loss functions, but they inherently suffer from physical inconsistency, lower modeling accuracy, and require resource-intensive retraining for new case predictions. 

In this paper, we propose a **`physics-guided hierarchical network with the deep residual network (ResNet) and long short-term memory (LSTM) for the data-driven modeling of power converters`**, which can bridge the connection between knowledge-based model and data-driven model for enhancing converter modeling and industrial process modeling. The main contributions of this work can be summarized as follow.**

:triangular_flag_on_post:**News** 1)A general knowledge-guided framework based on DL is proposed, improving the performance of converter modeling and industrial process modeling.

We provide a neat code base to evaluate advanced deep time series models or develop your model, which covers five mainstream tasks: **long- and short-term forecasting, imputation, anomaly detection, and classification.**

:triangular_flag_on_post:**News** (2024.07) We wrote a comprehensive survey of [[Deep Time Series Models]](https://arxiv.org/abs/2407.13278) with a rigorous benchmark based on TSLib. In this paper, we summarized the design principles of current time series models supported by insightful experiments, hoping to be helpful to future research.

:triangular_flag_on_post:**News** (2024.04) Many thanks for the great work from [frecklebars](https://github.com/thuml/Time-Series-Library/pull/378). The famous sequential model [Mamba](https://arxiv.org/abs/2312.00752) has been included in our library. See [this file](https://github.com/thuml/Time-Series-Library/blob/main/models/Mamba.py), where you need to install `mamba_ssm` with pip at first.

:triangular_flag_on_post:**News** (2024.03) Given the inconsistent look-back length of various papers, we split the long-term forecasting in the leaderboard into two categories: Look-Back-96 and Look-Back-Searching. We recommend researchers read [TimeMixer](https://openreview.net/pdf?id=7oLshfEIC2), which includes both look-back length settings in experiments for scientific rigor.

:triangular_flag_on_post:**News** (2023.10) We add an implementation to [iTransformer](https://arxiv.org/abs/2310.06625), which is the state-of-the-art model for long-term forecasting. The official code and complete scripts of iTransformer can be found [here](https://github.com/thuml/iTransformer).

:triangular_flag_on_post:**News** (2023.09) We added a detailed [tutorial](https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb) for [TimesNet](https://openreview.net/pdf?id=ju_Uqw384Oq) and this library, which is quite friendly to beginners of deep time series analysis.

:triangular_flag_on_post:**News** (202
