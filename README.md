# Artificial Intelligence Capstone Project
This is a project which me and my friends worked on through our final semester topic of Artificial Intelligence, it investigates Electrical Energy Network Stress in New South Wales. 

It features two phases:
- A1: Data Ingestion
- A2: Machine Learning Symbolic Reasoning

This project focuses on the development of a hybrid artificial intelligence (AI) prediction and inference engine to predict potential electricity-network stress in New South Wales. Through A1, this pipeline sourced historical electricity-market and network data for NSW1 through Open Electricity API located at https://platform.openelectricity.org.au/. Responsibilities for the machine learning and symbolic reasoning portion of this project are divided into the four focus areas of electricity demand, network stress, renewable sufficiency, and electricity price spikes. Due to the ethical concerns with using AI to make decisions in critical infrastructure this tool is built purely to provide guidance, and in no case should be used to replace human judgement. For A1 I performed an exploration into the data for early analysis and for A2 I was responsible for the renewable energy forecasting and symbolic inference module. This module integrates with the rest of the system through utilising the shared ingestion pipeline to produce renewable forecasts and symbolic conclusions which contribute to the combined system’s assessment of electricity-network operating conditions. The modular separation of the machine learning and symbolic learning in each system, alongside explicit sufficiency rules, traceable explanations, and input validation allowed for essential governance and stakeholder requirements to be easily addressed. The resultant interpretable and consistent assessments allow for human oversight to be retained whilst still benefitting from the efficiency and depth provided by the hybrid system.

The following diagram displays an overview of the implemented pipeline.
<img width="1109" height="1163" alt="image" src="https://github.com/user-attachments/assets/13aa928a-ad62-49dd-a222-b5efcd8beeca" />

## Overview of my ML and subsystem
Random Forest Regressor and Histogram Gradient Boosting Regressor were both investigated as possible supervised regression algorithms to implement. The Histogram Gradient Boosting model performed well on the test dataset, achieving an MAE of 234.06 MW across 11,986 observations. This was a 58.22% improvement over the baseline, demonstrating the model was more effective than using the previous hour's generation. However, the higher RMSE of 338.17 MW suggests that some predictions had larger errors. The cross-validation and holdout results were also obtained using different testing periods, so their results cannot be directly compared. Histogram Gradient Boosting was selected as it performed better on Mean Absolute Error, this was implemented using 300 boosting iterations, a learning rate of 0.05 and 31 maximum leaf nodes. The trained model is stored in an NPZ archive, enabling subsequent inference without the need to retrain the whole model.

## Overview of my Symbolic Reasoning
Although the other symbolic reasoning components used Prolog, the renewable used miniKanren through the Python kanren package. This allowed for the symbolic reasoning to be operated directly within Python, independently of an external Prolog environment. Although architectural uniformity would have been improved through consistent use of Prolog, miniKanren allowed for further investigation into alternative tools, providing educational benefits whilst still allowing the required relational reasoning capabilities to be achieved.

## Full System
The inference pipeline was tested using observations retrieved on 9 November 2026. The prediction made by the renewable model was passed to miniKanren with the independently generated demand forecast. Despite renewable generation exceeding the project-defined sufficiency threshold, the network stress model identified critical conditions, and the demand model indicated high uncertainty. Consequently, the combined symbolic system prioritised urgent operator review rather than interpreting renewable sufficiency as evidence of overall network reliability.

The renewable model currently considers uncertainty in the demand forecast, but not in its own renewable generation prediction. Including this would allow for a more complete assessment of renewable sufficiency. The 50% threshold could also be investigated further to determine whether it is suitable under different network conditions. Another possible improvement would be to develop an interactive dashboard where users could view the predictions, classifications and explanations from each model in one place. This would make the results easier to access and understand, particularly for stakeholders who may not have experience working directly with machine learning systems.

## Execution 
This project is able to be run through the A2 folder, as guided by the readME, these instructions run both A1 and A2 combined, although A1 can additionally be run independently through the A1 folder.

A little overview of the whole pipeline will be uploaded below soon. I would love to extend the output to a visual format like a dashboard for readability. 





  
