
import os
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import sys
import random
random.seed(42)

import numpy as np
from foldrm import Classifier

def globalbirds_rates():
    attrs = ["road_length","survey_days","country","decimalLatitude","decimalLongitude",
             "IslandEndemic","Volancy","Mass","HWI","Habitat.x","Trophic.Level",
             "Trophic.Niche","Beak.Length.culmen","Beak.Length.nares",
             "Beak.Width","Beak.Depth","Tarsus.Length","Wing.Length","Kipps.Distance","Secondary1",
             "Tail.Length","LogRangeSize","Diet","Foraging","Migration","MatingSystem","NestPlacement",
             "Territoriality","IslandDwelling","LogClutchSize","LogNightLights","LogHumanPopulationDensity",
             "Extinct_full","Extinct_partial","Marine_full","Marine_partial","Migr_dir_full","Migr_dir_partial",
             "Migr_dir_local","Migr_disp_full","Migr_disp_partial","Migr_disp_local","Migr_altitudinal",
             "Irruptive","Nomad_full","Nomad_partial","Nomad_local","Resid_full","Resid_partial",
             "Unknown","Uncertain","Migratory_status","Migratory_status_2","Migratory_status_3"
]
    nums = ["road_length","survey_days","decimalLatitude","decimalLongitude",
             "Mass","HWI","Beak.Length.culmen","Beak.Length.nares",
             "Beak.Width","Beak.Depth","Tarsus.Length","Wing.Length","Kipps.Distance","Secondary1",
             "Tail.Length","LogRangeSize","LogClutchSize","LogNightLights","LogHumanPopulationDensity",
]
    model = Classifier(attrs=attrs, numeric=nums, label='rate_category')
    data = model.load_data('BirdsGlobalrk_traits.csv')
    return model, data

model, data = globalbirds_rates()

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
labels = ['Low', 'Medium', 'High']
if pred_classes:
    cm = confusion_matrix(true_classes, pred_classes, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print("\nMatriz de confusión:")
    print(df_cm)
else:
    print("\nNo hay predicciones válidas para matriz de confusión.")




