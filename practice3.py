import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

Square_Footage_df = pd.read_csv('test_energy_data.csv')

numeric_col = Square_Footage_df.select_dtypes(include=[np.number])
input_col = numeric_col.columns.tolist()
print(input_col)
target_col='Building_Type'

Square_Footage_train, Square_Footage_temp, y_train, y_temp = train_test_split(Square_Footage_df[input_col], Square_Footage_df[target_col], train_size=0.7, random_state=42)
Square_Footage_val, Square_Footage_test, y_val, y_test = train_test_split(Square_Footage_temp, y_temp, train_size=0.3,random_state=42)

model = RandomForestClassifier()
model.fit(Square_Footage_train, y_train)

importances = model.feature_importances_

feature_names = Square_Footage_train.columns
feat_importances = pd.Series(importances, index=feature_names)

print(feat_importances.sort_values(ascending=False).head(10))

feat_importances.sort_values(ascending=False).head(10).plot(kind='bar')
plt.title("top 10 importance features")
plt.show()

y_val_pred = model.predict(Square_Footage_val)
print(classification_report(y_val, y_val_pred))



