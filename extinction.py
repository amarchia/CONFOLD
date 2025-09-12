import sys
import random
random.seed(42)

import numpy as np
from foldrm import Classifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

def extinction():
    attrs = ["Mass","HWI","Habitat",
         "LAT","Beak.Length.culmen",
         "Beak.Length.nares","Beak.Width","Beak.Depth","Tarsus.Length",
         "Wing.Length","Kipps.Distance","Secondary1","Tail.Length"]
    nums = ["Mass","HWI","LAT","Beak.Length.culmen",
         "Beak.Length.nares","Beak.Width","Beak.Depth","Tarsus.Length",
         "Wing.Length","Kipps.Distance","Secondary1","Tail.Length"]

    model = Classifier(attrs=attrs, numeric=nums, label='Trophic.Level')
    data = model.load_data('BirdstraitsIUCN.csv')
    print('\n% dataset', np.shape(data))
    return model, data


model, data = extinction()

# Mostrar distribución de clases en todo el dataset
from collections import Counter
all_labels = [row[-1] for row in data]
print("\nDistribución de clases en todo el dataset:")
for label, count in Counter(all_labels).items():
     print(f"{label}: {count}")

from utils import split_data
train_data, test_data = split_data(data, ratio=0.9, shuffle=True)

# Training

model.fit(train_data, ratio=0.9)
model.confidence_fit(train_data, improvement_threshold=0.9)

print("\nLearned Answer Set Program rules:\n")
model.print_asp()

# Predicting over test_data

Y_pred = model.predict(test_data)

print("\nEjemplo de predicciones (primeros 10):")
for i, (pred, obs) in enumerate(zip(Y_pred[:10], test_data[:10])):
     print(f"Obs {i+1}: pred = {pred}, entrada = {obs}")

# Matriz de confusión

# Extraer clases predichas y etiquetas reales, filtrando None
pred_classes = [p[0] for p in Y_pred if p is not None and p[0] is not None]
true_classes = [row[-1] for p, row in zip(Y_pred, test_data) if p is not None and p[0] is not None]

# Obtener todas las categorías presentes en el test
all_labels = sorted(list(set(true_classes + pred_classes)))

if pred_classes:
     cm = confusion_matrix(true_classes, pred_classes, labels=all_labels)
     df_cm = pd.DataFrame(cm, index=all_labels, columns=all_labels)
     print("\nMatriz de confusión:")
     print(df_cm)
else:
     print("\nNo hay predicciones válidas para matriz de confusión.")








