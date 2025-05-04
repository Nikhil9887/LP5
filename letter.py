import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input

# 1. Load the data
df = pd.read_csv('C:/Users/Admin/Downloads/letter-recognition.data', header=None)

# 2. Features and labels
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# 3. Encode labels (A–Z to 0–25)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

print(y_encoded)
print(y_categorical)

# 4. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_categorical, test_size=0.2, random_state=42
)

# 6. Build model
model = Sequential([
    Dense(64, activation='relu', input_shape=(16,)),
    Dense(64, activation='relu'),
    Dense(26, activation='softmax')
])

# 7. Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 8. Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# 9. Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy:.4f}")

# 10. Predict classes
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# 11. Classification report (includes precision, recall, F1)
target_names = label_encoder.classes_  # ['A', 'B', ..., 'Z']
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Convert predictions and true labels back to letters
predicted_letters = [letters[label] for label in y_pred]
actual_letters = [letters[label] for label in np.argmax(y_test, axis=1)]  # Convert y_test from one-hot to integer labels

# Print Predicted and Actual Labels side by side
print("Predicted Labels\tActual Labels")
for pred, actual in zip(predicted_letters, actual_letters):
    print(f"{pred}\t\t\t{actual}")
