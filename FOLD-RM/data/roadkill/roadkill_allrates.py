
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import sys
import random
random.seed(42)
sys.path.insert(0, "/home/pabmevi/CONFOLD")

import numpy as np
from foldrm import Classifier

def Imput_lessnoise_allsp_rates():
    attrs = ["Survey_interval_days","Road_length_km","Survey_period_days","Latitude","Longitude",
             "AdultBodyMass_g_median","Home_range_Km2","longevity_y","Ageofmaturity_d",
             "Diet_Invertebrates","Diet_Vertebrates.ectotherms","Diet_Scavenger","Diet_Seed","Diet_Plant","Activity_1Diurnal_2Nocturnal",
             "Litter_clutch_size","Litters_or_clutches_per_y","Diet_breadth","Artificial","Cropland","Grassland","Forest","Sparse.vegetation","Water.bodies",
             "Habitat_breadth","Population.density_IndKm2"]
    nums = ["Survey_interval_days","Road_length_km","Survey_period_days","Latitude","Longitude","AdultBodyMass_g_median",
            "Home_range_Km2","longevity_y","Ageofmaturity_d","Litter_clutch_size","Litters_or_clutches_per_y","Population.density_IndKm2"]
    model = Classifier(attrs=attrs, numeric=nums, label='risk_category')
    data = model.load_data('/home/pabmevi/CONFOLD/FOLD-RM/data/roadkill/Imput_lessnoise_allsp_rates_classified.csv')
    return model, data

model, data = Imput_lessnoise_allsp_rates()

# (80% train, 20% test) 
from utils import split_data
train_data, test_data = split_data(data, ratio=0.9, shuffle=True)

# Entrenar solo con el set de entrenamiento
model.fit(train_data, ratio=0.9)
model.confidence_fit(train_data, improvement_threshold=0.9)


print("\nLearned Answer Set Program rules:\n")
model.print_asp()

# ===========================
# Predicciones sobre test_data
# ===========================
Y_pred = model.predict(test_data)

print("\nEjemplo de predicciones (primeros 10):")
for i, (pred, obs) in enumerate(zip(Y_pred[:10], test_data[:10])):
    print(f"Obs {i+1}: pred = {pred}, entrada = {obs}")

# ===========================
# Evaluación del modelo
# ===========================
# Accuracy global (cuenta None como error)
all_pred_classes = [p[0] if p is not None else None for p in Y_pred]
all_true_classes = [row[-1] for row in test_data]
acc_global = sum([y1 == y2 for y1, y2 in zip(all_pred_classes, all_true_classes)]) / len(all_true_classes)
print("\nAccuracy global (incluyendo None como error):", acc_global)

# Extraer clases predichas y etiquetas reales, filtrando None
pred_classes = [p[0] for p in Y_pred if p is not None and p[0] is not None]
true_classes = [row[-1] for p, row in zip(Y_pred, test_data) if p is not None and p[0] is not None]

# Accuracy general
if pred_classes:
    acc = accuracy_score(true_classes, pred_classes)
    print("\nAccuracy general:", acc)
else:
    print("\nNo hay predicciones válidas para calcular accuracy.")

# Matriz de confusión
labels = ['low', 'medium', 'high']
if pred_classes:
    cm = confusion_matrix(true_classes, pred_classes, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nMatriz de confusión:")
    print(df_cm)
else:
    print("\nNo hay predicciones válidas para matriz de confusión.")




