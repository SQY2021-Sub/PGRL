# Deep Learning-Based Modeling for Power Converters via Physics-Guided Hierarchical Res-LSTM Frameworks
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 7.3.1](https://img.shields.io/badge/cudnn-11.3.1-green.svg?style=plastic)
![MATLAB 2021b](https://img.shields.io/badge/MATLAB-2024a-blue.svg?style=plastic)
![PLECS 4.5.6](https://img.shields.io/badge/PLECS-4.5.6-green.svg?style=plastic)
[![Build status](https://ci.appveyor.com/api/projects/status/8msiklxfbhlnsmxp/branch/master?svg=true)](https://ci.appveyor.com/project/TadasBaltrusaitis/openface/branch/master)

This is a GitHub repository for the **`Physics-Guided Hierarchical Res-LSTM Framework (PhyRes-LSTM)`**.

## Introduction 

Power converters are extensively utilized in many applications such as consumer electronics, industrial electronics, electric vehicles (EVs), energy storage systems, and distributed generation systems to generate either a regulated dc or ac voltage source and manage the power flow by mainly controlling the switching actions of power semiconductor devices (see Fig. 1)[[1]](https://ieeexplore.ieee.org/document/9351620). `In all of these applications, the ability to accurately describe the behavior of the converter is of critical importance, as it assists in component selection, controller design, and the preliminary evaluation of the entire DC system`.  Power converters exhibit an inherent nonlinear behavior due to the switching action of the power switches, thus making its modeling difficult due to the complexity of the required models.

![1](https://github.com/user-attachments/assets/48175392-9e72-47fa-a75d-5fe416eeffe5)

Fig. 1. Diagram of a hierarchical microgrid with power converters [[2]](https://ieeexplore.ieee.org/document/9525187)

Generally, physical modeling relies on a knowledge of the values of the dc–dc converter parameters, both the passive elements and the control circuit. In addition, an accurate model should be able to reproduce ripples in currents and voltages, transients caused by switching on and off operations, etc., which is difficult to generate with physical-based modeling methods [[3]](https://ieeexplore.ieee.org/document/9492829). Over the past few years, classical machine learning (ML) and deep learning (DL) algorithms have been used for the data-driven modeling of power converters with simple topologies. It is worth noting that this modeling approach falls under the task of time series forecasting, so models that are related to time series and their theoretical analysis are also applicable in this situation. However, conventional time-series data-driven approaches for power converter modeling are `data-intensive, uninterpretable, and lack out-of-domain extrapolation capability`. Recent physics-informed modeling methods combine physics into data-driven models using loss functions, but they inherently suffer from `physical inconsistency, lower modeling accuracy, and require resource-intensive retraining for new case predictions`. 

In this paper, we propose a **`physics-guided hierarchical network with the deep residual network (ResNet) and long short-term memory (LSTM) for the data-driven modeling of power converters`**, which can bridge the connection between knowledge-based model and data-driven model for enhancing converter modeling and industrial process modeling. The main contributions of this work can be summarized as follow.**

:triangular_flag_on_post:**(1)**  [A novel physics-guided framework based on DL is proposed](https://github.com/sub-p/PGRL), `improving the performance of converter modeling and industrial process modeling`.

:triangular_flag_on_post:**(2)** `Compared to the advanced existing physics-driven, data-driven, and physics-informed neural network (PINN)/ physics-guided machine learning (PGML) methods in the field of power electronics`, [our approach achieves superior physical consistency, data-light implementation and generalization ability](https://github.com/sub-p/PGRL).

:triangular_flag_on_post:**(3)** **`This paper envisions to democratize artificial intelligence for the modeling of power electronic converters and systems. Meanwhile, this exemplary application envisions to `**[provide a new perspective for tailoring existing machine learning tools for power electronics field](https://github.com/sub-p/PGRL).

[^back to top](#top)

## Requirements
- MATLAB == 2021b
- PLECS == 4.7.4
- Simulink
- Python
- PyTorch

## Problem Formulation
#### A. DAB Converters With Phase-Shift Modulation
Due to the advantages of galvanic isolation, high power densityand bidirectional energy flow, dual active bridge (DAB) dc–dc converter is gaining more and more popularity in industry applications. The topology of DAB dc–dc converter is depicted in Fig. 2. 

![image](https://github.com/user-attachments/assets/ff703b71-0dbc-4d9b-94dd-dfc2f000709b)

Fig. 2. Topology of dual-active-bridge DC–DC converter.

The most widely used control for DAB converters is the PSM from the invention of this topology. Consequently, single phase shift (SPS) control, extended phase shift (EPS) control, dual phase shift (DPS) control, and triple phase shift (TPS) control [[4]](https://ieeexplore.ieee.org/document/5776689) are proposed. 


[^back to top](#top)
## Methodology


[^back to top](#top)
## Main Results


[^back to top](#top)
## Baselines

We will continuously add PINN/PGML models applied in the field of power electronics to expand this repository.

- [x] Numerical methods [[7]](https://ieeexplore.ieee.org/document/8409299)
- [x] LSTM [[8]](https://ieeexplore.ieee.org/document/9492829)
- [x] ResNet [[9]](https://arxiv.fropet.com/abs/1603.08029)
- [x] ResNet-LSTM [[10]](https://ieeexplore.ieee.org/document/9798792)
- [x] PINN [[11]](https://ieeexplore.ieee.org/document/9779551)
- [ ] 

[^back to top](#top)


