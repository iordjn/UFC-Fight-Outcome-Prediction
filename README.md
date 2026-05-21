# UFC Fight Outcome Prediction

Machine learning pipeline for predicting UFC fight outcomes from pre-fight fighter history, rolling form metrics, and Elo-based strength ratings. The main workflow lives in [chavez-final-project.ipynb](chavez-final-project.ipynb) and is supported by the CSV files in this repo.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model](#model)
- [Feature Engineering](#feature-engineering)
- [Leakage Controls](#leakage-controls)
- [How To Predict Future Fights](#how-to-predict-future-fights)
- [Files](#files)

## Overview

The project estimates the probability that the red corner wins a fight before it happens. The model is trained only on information that would have been available before the bout, which is critical for keeping the evaluation honest.

## Dataset

The main input is [UFC.csv](UFC.csv), which contains fight-level records, fighter metadata, event data, and outcomes.

The CSV currently ends at **UFC Fight Night: Sterling vs. Zalal** on **April 25, 2026**.

Supporting files in the repo include [fight_details.csv](fight_details.csv), [fighter_details.csv](fighter_details.csv), and [event_details.csv](event_details.csv).

## Model

The notebook builds a chronological feature table and trains a time-aware classifier. The target is:

- `1` = red corner won
- `0` = blue corner won

The main model is XGBoost, with simpler baseline models used for comparison.

## Feature Engineering

The model is built around pre-fight snapshots, rolling averages, and matchup differences rather than raw career totals alone.

### Pre-Fight Snapshots

For each bout, the notebook stores fighter state before the fight result is applied. These snapshots include:

- total fights, wins, losses, and streaks
- total time fought and knockdowns
- title fight wins
- days since last fight
- age and date of birth
- striking, grappling, and defensive totals

These values are tracked separately for each side using `r_hist_...` and `b_hist_...` columns.

### Rolling Averages

Career totals can be misleading because older fighters naturally accumulate larger raw counts. To emphasize current form, the notebook adds 3-fight rolling averages built from the pre-fight history.

Examples include rolling values for:

- significant strikes landed and attempted
- ground strikes landed and attempted
- takedowns landed and attempted
- control time
- strikes absorbed
- derived accuracy metrics

These features help the model focus on recent performance instead of lifetime volume.

### Elo Features

Each fighter gets an Elo-style strength rating.

- Fighters start at `1000`.
- Elo updates after each fight using the result and opponent strength.
- The update rate is adaptive, so newer fighters can move faster and veteran fighters stabilize more slowly.
- Rating deviation is tracked so uncertainty is higher early in a fighter’s career.

The notebook also creates Elo trend features such as previous Elo, recent Elo change, and red-vs-blue Elo differences.

### Matchup Differences

Many final inputs are red-minus-blue differences so the model learns the matchup instead of corner order. Examples include:

- striking and grappling differences
- rolling average differences
- accuracy differences
- Elo difference
- age difference
- days-since-last-fight difference
- prospect flag difference

## Leakage Controls

Sports prediction is easy to contaminate with future information, so the notebook was adjusted to reduce leakage.

The main protections are:

- fights are processed strictly in chronological order
- fighter history is captured before the current result is applied
- training and test sets are split by date instead of random rows
- rolling features use only prior fights
- the prediction pipeline uses only the latest known pre-fight state

This is also why raw career stats are not used blindly. If you compute totals using the full dataset, you can accidentally include fights that happened after the bout you are trying to predict. Even when totals are historical, they can still act like a hidden time signal and bias the model toward career length instead of form.

## How To Predict Future Fights

To predict a fight that has not happened yet, change the fighter names in the fully trained model section:

```python
red_fighter = "Khamzat Chimaev"
blue_fighter = "Sean Strickland"
```

Then build the feature row and score it:

```python
upcoming = create_prediction_row(red_fighter, blue_fighter, pd.Timestamp.today(), df, X_cols)
X_up = upcoming[X_cols].fillna(0)
prob = final_model.predict_proba(X_up)[:, 1][0]
print(prob)
```

Interpretation:

- closer to `1.0` means the model favors the red corner
- closer to `0.0` means the model favors the blue corner

## Files

- [chavez-final-project.ipynb](chavez-final-project.ipynb): main notebook for cleaning, feature engineering, training, and prediction
- [UFC.csv](UFC.csv): main fight-level dataset
- [fight_details.csv](fight_details.csv): supporting fight metadata
- [fighter_details.csv](fighter_details.csv): fighter-level metadata
- [event_details.csv](event_details.csv): event-level metadata
- [UFC_PredictionV2.ipynb](UFC_PredictionV2.ipynb): earlier notebook version


## Notes

The pipeline is most reliable when every feature row is built from information that existed before the target fight date. If you extend the dataset, keep the chronological ordering and pre-fight snapshot logic intact so future results never leak into training.
- Python version - 3.13.9
