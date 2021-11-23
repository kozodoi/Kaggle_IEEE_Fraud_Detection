# IEEE-CIS Fraud Detection

Top-3% solution to the [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) Kaggle competition on identifying fraudulent transactions.

![cover](https://i.postimg.cc/zBVTkP09/fraud-cover.jpg)


## Summary

Fraud detection is an important task for many businesses. Models that facilitate the accurate classification of potential fraud helps companies to reduce their fraud loss and increase revenue.

This project works with a large-scale e-commerce dataset to identify fraudulent transactions. The data contains a wide range of features from device type to product features. We perform extensive feature engineering and develop a stacking ensemble of multiple LightGBM models. The solution reaches the top-2% of the Kaggle competition leaderboard. More details are provided within the notebooks documentation.


## Project structure

The project has the following structure:
- `codes/`:  Jupyter notebooks and Python codes for data preparation, modeling and ensembling
- `data/`: input data (not included due to size constraints and can be downloaded [here](https://www.kaggle.com/c/ieee-fraud-detection/data))
- `output/`: figures produced when executing the modeling notebooks
- `oof_preds/`: out-of-fold predictions produced models within the cross-validation
- `submissions/`: test set predictions produced by the trained models
