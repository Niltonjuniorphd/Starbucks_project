
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import time



def train_model(X, y, model_type):

    model = model_type

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    cm1 = confusion_matrix(y_test, y_pred_test)
    cm2 = confusion_matrix(y_train, y_pred_train)

    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)

    print(f'accuracy_score_test: {test_accuracy}')
    print(f'accuracy_score_train: {train_accuracy}\n')
    report1 = classification_report(
        y_test, y_pred_test, target_names=model.classes_, zero_division=1)
    print("Test Classification Report:")
    print(report1)

    report2 = classification_report(
        y_train, y_pred_train, target_names=model.classes_, zero_division=1)
    print("Train Classification Report:")
    print(report2)

    print("Confusion Matrix (Test):")
    disp1 = ConfusionMatrixDisplay(
        confusion_matrix=cm1, display_labels=model.classes_, )
    disp1.plot(cmap="Blues")
    plt.show()

    print("Confusion Matrix (Train):")
    disp2 = ConfusionMatrixDisplay(
        confusion_matrix=cm2, display_labels=model.classes_)
    disp2.plot(cmap="Blues")
    plt.show()

    return train_accuracy, test_accuracy


def tunning_model_rf(
        X, y, 
        scoring, 
        param_grid = {
        'n_estimators': [50, 200],        
        'max_depth': [None, 10, 50],       
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 3],
        
    
    }):


    print(f'--- scoring with: {scoring} method\n---')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    


    # Instanciar o modelo
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Definir a grade de parâmetros


    # Configurar o GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,                  # Validação cruzada (5-fold)
        scoring=scoring,    # Métrica de avaliação
        verbose=2,             # Nível de detalhe das mensagens
        # Paralelização (usar todos os núcleos disponíveis)
        n_jobs=-1
    )

    # Ajustar o modelo aos dados de treino
    grid_search.fit(X_train, y_train)

    # Exibir os melhores parâmetros e a melhor pontuação
    print("Melhores parâmetros:", grid_search.best_params_)
    print("Melhor pontuação (treino):", grid_search.best_score_)

    # Avaliar o modelo com os melhores parâmetros nos dados de teste
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print("Acurácia nos dados de teste:", test_accuracy)

    y_pred_train = best_model.predict(X_train)

    # 1. Confusion Matrix
    cm1 = confusion_matrix(y_train, y_pred_train)
    cm2 = confusion_matrix(y_test, y_pred)

    # 2. Relatório com Precision, Recall e F1-Score
    report1 = classification_report(
        y_test, y_pred, target_names=best_model.classes_, zero_division=1)
    print("Test Classification Report:")
    print(report1)

    report2 = classification_report(
        y_train, y_pred_train, target_names=best_model.classes_, zero_division=1)
    print("Train Classification Report:")
    print(report2)

