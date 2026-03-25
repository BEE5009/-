"""Train RandomForest classifier on hand landmark data"""
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Create labels dictionary and numeric mapping
# If labels are '0'-'25', map to A-Z; else preserve custom labels
unique_labels = sorted(set(labels), key=lambda x: int(x) if str(x).isdigit() else str(x))
label_to_index = {lbl: idx for idx, lbl in enumerate(unique_labels)}
index_to_label = {idx: (chr(ord('A') + int(lbl)) if str(lbl).isdigit() else str(lbl)) for idx, lbl in enumerate(unique_labels)}
print(f"Labels: {index_to_label}")

numeric_labels = np.array([label_to_index[lbl] for lbl in labels])

x_train, x_test, y_train, y_test = train_test_split(
    data, numeric_labels, test_size=0.2, shuffle=True, stratify=numeric_labels
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'Accuracy: {score * 100:.2f}%')

# Save model and labels dictionary. labels_dict for predictor lookup.
labels_dict = index_to_label
pickle.dump({'model': model, 'labels_dict': labels_dict}, open('model.p', 'wb'))
print("Model saved!")