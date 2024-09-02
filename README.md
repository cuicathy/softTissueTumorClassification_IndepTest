This is the code using machine learning model to classify benign and malignent tumor. Cross-validation is used for feature reduction and parameter tuning. And the trained model is tested on an independent testing set. 


Prerequisites: radiomics (version>3.0), SimpleITK, matplotlib, numpy, pandas, sklearn, xgboost, scipy

Tested successfully on Linux (Ubuntu 18.04) and Win10. Not sure for Mac.

extractRadiomicsFeatures.py: Extract features from images within the corresponding masks. You can run the createFilesPathTable.py to get the .csv table recoding the filepaths of images and masks.

tumorClassification_main.py: Main function to train and evaluate classifiers (to update).

model_cv.py: Contains functions will be used by tumorClassification_main.py.

Params_self_defined_rebuilt_clean_sumAvg_Normalized_binwidth_5.yaml: Parameters used for feature extraction. (You can got to the pyradiomics github for more parameter examples)

More information about radiomics feature extraction can be found here: https://github.com/AIM-Harvard/pyradiomics. https://pyradiomics.readthedocs.io/en/latest/

Please email can.cui.1@vanderbilt.edu if you have any questions.
