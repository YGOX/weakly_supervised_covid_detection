This code was developed for the IEEE access paper "Weakly Supervised Deep Learning for COVID-19 Infection Detection and Classification From CT Images". linked here: https://ieeexplore.ieee.org/abstract/document/9127422 

Data used in the research is kept in IIAT OSMART server: the vesion is v7, containing covid, normal neumonia and healthy Lung CT images. 

Contrast_adjust.m- Correcting the contrast of Covid CT Images with the reference of a random healthy Lung CT image

generate_cross_validation.py- split subjects and return train and validation image lists for 5-fold cross-validation, saved in .npy files

model_cls2.py- build model for binary classification 

model_cls3.py- build model for three way classification

train_model_cls3.py- train three-way classification 

train_model_cls2.py train binary classification 

layers.py - custermorised layers 

plot_cams.py- plot multiscale class activation maps

plot_saliency.py- generate multi-scale saliency and joint saliency 

saliency folder- codes for integrated gradients: git clone https://github.com/PAIR-code/saliency.git

roc.py- plot roc curves 

test_metrics.py- classification metrics

