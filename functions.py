
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def train_model(X, y, model_type):

    model = model_type

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    # 1. Confusion Matrix
    cm1 = confusion_matrix(y_test, y_pred)
    cm2 = confusion_matrix(y_train, y_pred_train)

    print(f'accuracy_score_test: {accuracy_score(y_test, y_pred)}')
    print(f'accuracy_score_train: {accuracy_score(y_train, y_pred_train)}\n')
    # 2. Relatório com Precision, Recall e F1-Score
    report1 = classification_report(y_test, y_pred, target_names=model.classes_, zero_division=1)
    print("Test Classification Report:")
    print(report1)

    report2 = classification_report(y_train, y_pred_train, target_names=model.classes_, zero_division=1)
    print("Train Classification Report:")
    print(report2)

    # 3. Exibição Alternativa da Confusion Matrix
    print("Confusion Matrix (Test):")
    disp1 = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=model.classes_)
    disp1.plot(cmap="Blues")
    plt.show()

    print("Confusion Matrix (Train):")
    disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=model.classes_)
    disp2.plot(cmap="Blues")
    plt.show()