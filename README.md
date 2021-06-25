# IEEE-CIS Fraud Detection

Top-3% solution to the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) Kaggle competition on identifying fraudulent transactions.

![cover](https://i.postimg.cc/zBVTkP09/fraud-cover.jpg)


## Summary

This project works with a large data set containing the telescope data from different astronomical sources such as planets, supernovae and others. Using the time series of the objects brightness referred to as light curves and available object meta-data, we preprocess the data and build a classification pipeline with LightGBM and MLP models. Our solution represents a stacking ensemble of multiple base models for galactic and extragalactic objects. The reaches the top-2% of the corresponding Kaggle competition leaderboard.


## Project structure

The project has the following structure:
- `codes/`:  Jupyter notebooks with Python codes for data preparation, modeling and ensembling
- `data/`: input data (not included due to size constraints and can be downloaded [here](https://www.kaggle.com/c/ieee-fraud-detection/data))
- `output/`: figures produced when executing the modeling notebooks
- `oof_preds/`: out-of-fold predictions produced models within the cross-validation
- `submissions/`: test set predictions produced by the trained models
