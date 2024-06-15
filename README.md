This repository contains code and experiments for the paper:

# Fairness and Accuracy for Web-Based Content Analysis under Temporal Shifts and Delayed Labeling 

## Abstract:
Web-based content analysis tasks, such as labeling toxicity, misinformation, or spam often rely on machine learning models to achieve cost and scale efficiencies. As these models impact real human lives, ensuring accuracy and fairness of such models is critical. However, maintaining the performance of these models over time can be challenging due to the temporal shifts in the application context and the sub-populations represented. Furthermore, there is often a delay in obtaining human expert labels for the raw data, which hinders the timely adaptation and safe deployment of the models. To overcome these challenges, we propose a novel approach that anticipates future distributions of data, especially in settings where unlabeled data becomes available earlier than the labels to estimate the future distribution of labels per sub-population and adapt the model preemptively. We evaluate our approach using multiple temporally-shifting datasets and consider bias based on racial, political, and demographic identities. We find that the proposed approach yields promising performance with respect to both accuracy and fairness. Our paper contributes to the web science literature by proposing a novel method for enhancing the quality and equity of web-based content analysis using machine learning. 

## To cite:
@inproceedings{almuzaini2024accuracy,
  title={Accuracy and Fairness for Web-Based Content Analysis under Temporal Shifts and Delayed Labeling},
  author={Almuzaini, Abdulaziz A and Pennock, David M and Singh, Vivek K},
  booktitle={Proceedings of the 16th ACM Web Science Conference},
  pages={268--278},
  year={2024}
}

# Requirements

1. Access datasets [link](https://drive.google.com/file/d/1EAuFH9m2yuX-p38N-0NpJIYDvyvkDmOF/view?usp=sharing)
