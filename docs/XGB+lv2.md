# Adding XGBoost to ADS Injury Models

This work was primarily to add XGBoost onto previous models. See models.md for setup details. 
One major difference to highlight: XGBoost uses "scale_pos_weight" instead of having a balanced weight as the other models do.

## Results
| Model | Crash With | PR-AUC (95% range) | ROC-AUC | Macro-F1 |
|---|---|---|---|---|
| Baseline | either | 0.09 | 0.50 | 0.48 |
| Logistic regression | with | **0.18** (0.12–0.26) | 0.65 | 0.49 |
| Random forest | with | 0.16 (0.12–0.23) | 0.66 | 0.50 |
| XGBoost | with | 0.18 (0.13-0.26) | 0.661 | 0.554 |
| Logistic regression | without | 0.11 (0.09–0.15) | 0.57 | 0.44 |
| Random forest | without | 0.11 (0.09–0.15) | 0.59 | 0.48 |
| XGBoost | without | 0.10 (0.08-0.15) | 0.535 | 0.515 |

![With XGB](figures/pr_curves.png)

## Key Takeaways:
- XGBoost did not perform better than logistic regression, but does have a distinctly different PR curve.
- Some internal tests showed that XGB had must better results on the training data vs the validation data:
![train vs valid pr auc](figures/training_vs_validation_pr_auc.png)
- This suggests that XGB may be overfitting on the training data, so I tried playing around with some parameters such as reg_lambda and gamma, and adding more options for learning rate, max_depth, and min_child_weight, but none appeared to help as the model which was chosen to be the best actually had a worse F1 score than before (so I reverted this change because the runtime was extremely long with so many combinations of params)
- Still very limited by the size of our dataset, especially the number of injuries, even with the scaled weighting for XGBoost.
- Overall, XGBoost is at least doing similarly to logistic regression so far. 
- Note: best original params for XGB with Crash_With were: max_depth = 3, learning_rate = 0.03, 'min_child_weight' = 1.


# Level 2 work
- Kept all rows with NaN injury labels that were still valid on the other data cleaning conditons to expand the level 2 dataset.
- Built a classifier to predict dataset origination between level 2 and ADS.
    - See lv2_ads_classifier.py
    - Note that Report ID and Same Incident ID are excluded since it is likely to be able to distinguish between the encodings
    

## Level 2 Classifier Results
With "Crash With":
| Model | Validation Macro-F1 |
|---|---:|
| Majority baseline | 0.3820 |
| Logistic Regression | 0.9253 |
| Random Forest | **0.9339** |
| Extra Trees | 0.9200 |

Without "Crash With":
| Model | Validation Macro-F1 |
|---|---:|
| Majority baseline | 0.3820 |
| Logistic Regression | 0.9191 |
| Random Forest | **0.9260** |
| Extra Trees | 0.9117 |

So we selected Random Forest to be our model.

| Model | Crash With | PR-AUC (95% range) | ROC-AUC | Macro-F1 |
|---|---|---|---|---|
| Random Forest | with | 0.992 (0.990–0.995) | 0.987 | 0.936 |
| Random Forest | without | 0.991 (0.988–0.994) | 0.985 | 0.933 |

## Feature Importance Figure
![feature importance](figures/lv2_ads_rf_feature_importance.png)

## Key Takeaways:
- Able to distinguish between ADS and level 2 datasets
- I thought originally thought that the injury column could be an issue, since I just added back in a bunch of data containing unknown injury labels, so we completely removed it from the model. Meaning that our model can only learn on the actual crash conditions.
    - Also removed the report date information. 
- From the feature importance figure, the model is likely learning from the roadway type, the crash with partner, the type of pre-crash movement, and the pre-crash speed
    - I was concerned that the datasets are a bit sparse, especially for level 2, such that the model could just be learning that more unknowns = level 2, but it seems that we can still classify a difference. 
