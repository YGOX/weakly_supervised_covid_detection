This code was developed for the IEEE access paper "Weakly Supervised Deep Learning for COVID-19 Infection Detection and Classification From CT Images". linked here: https://ieeexplore.ieee.org/abstract/document/9127422 

Data used in the research is kept in IIAT OSMART server: the vesion used is v3 and v7, containing covid, normal neumonia (CAP) and healthy Lung CT images.

Trained models and logs are kept here: https://drive.google.com/drive/folders/1R5PnZJt5e71Ala2mD37AEy0IM1lkCIDv?usp=sharing
For binary classification models i.e.(Covid/normal, Covid/CAP, normal/CAP), the models were named with the convention: multi_deep_classes involed_dataversion_learning hypaparameters. Multi_deep means mulri-scale deep learning. 
For three-way classification models i.e. Covid/normal/CAP, the models were named with the convention: multi_deep_3cls_foldid_learning hypaprameters. 

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

