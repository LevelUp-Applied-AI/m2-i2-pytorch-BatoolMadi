# Housing Price Prediction — Integration 2 (PyTorch)

## Model Objective

The goal of this project is to train a neural network that predicts **housing prices in Jordanian Dinar (JOD)** based on property characteristics.

### Target Variable

* **price_jod** — the selling price of a property in Jordanian Dinars.

### Input Features (5)

The model uses the following property features as inputs:

1. **area_sqm** — property area in square meters
2. **bedrooms** — number of bedrooms
3. **floor** — floor number of the apartment
4. **age_years** — age of the property in years
5. **distance_to_center_km** — distance from city center in kilometers

---

## Model Architecture

Neural Network structure:

```
Linear(5 → 32) → ReLU → Linear(32 → 1)
```

The model performs **regression**, predicting a continuous price value.

---

## Training Configuration

* **Epochs:** 100
* **Optimizer:** Adam
* **Learning Rate:** 0.01
* **Loss Function:** Mean Squared Error (MSELoss)
* **Feature Scaling:** Standardization (mean = 0, std = 1)

Feature standardization was applied to ensure balanced gradient updates across features with different scales.

---

## Training Outcome

During training, the loss consistently decreased across epochs, indicating successful learning and convergence.

* The model converged smoothly without instability.
* Final loss value after 100 epochs: 1.94e9

The loss value appears large because housing prices are measured in Jordanian Dinars, and Mean Squared Error squares the prediction error, resulting in large numerical values.

---

## Behavioral Observation

The loss decreased rapidly during the early epochs and then gradually leveled off toward the end of training.
This suggests the model quickly learned general relationships between features and housing prices before fine-tuning smaller adjustments in later epochs.

---

## Output

After training, the model generates predictions saved in:

```
predictions.csv
```

The file contains:

* actual housing prices
* predicted housing prices
  for comparison and evaluation.

---
