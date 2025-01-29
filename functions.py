
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
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from scipy.stats import ks_2samp

def train_model(X, y, model_type, fig='no'):

    model = model_type

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y) #
    
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
    
    if fig == 'yes':
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
    else:
        pass

    return train_accuracy, test_accuracy


def tunning_model_rf(X, y, scoring, param_grid, cv):


    print(f'--- scoring with: {scoring} method\n---')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y)
    


    # Instanciar o modelo
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Definir a grade de parâmetros


    # Configurar o GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,                  # Validação cruzada (5-fold)
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

    print("Confusion Matrix (Test):")
    disp1 = ConfusionMatrixDisplay(
        confusion_matrix=cm1, display_labels=best_model.classes_, )
    disp1.plot(cmap="Blues")
    plt.show()
    print("Confusion Matrix (Train):")
    disp2 = ConfusionMatrixDisplay(
        confusion_matrix=cm2, display_labels=best_model.classes_)
    disp2.plot(cmap="Blues")
    plt.show()



def encode(X):
    encoder = OneHotEncoder(sparse_output=False)
    encoded_cols = encoder.fit_transform(X.select_dtypes(include=['object']))
    X_encoded = pd.DataFrame(
        encoded_cols,
        columns=encoder.get_feature_names_out(X.select_dtypes(include=['object']).columns),
        index=X.index
        )
    
    X_enc = pd.concat([X, X_encoded], axis=1).drop(columns=X.select_dtypes(include=['object']).columns)
 
    return X_enc 

def model_baseline(X, y):
    
    model1 = RandomForestClassifier(
        random_state=42,
        # max_depth=15,
        # n_estimators=100,
        # min_samples_leaf=2,
        # min_samples_split=10,
        # max_features=0.5,
        # min_impurity_decrease=0.000001,
        # n_jobs=-1,
        #class_weight='balanced'
        )

    model2 = DecisionTreeClassifier(
        random_state=42,
        #criterion='entropy',            # Função de impureza para divisão ('gini' ou 'entropy')
        #max_depth=100,                # Profundidade máxima da árvore (controle de overfitting)
        #min_samples_split=2,         # Número mínimo de amostras para dividir um nó
        #min_samples_leaf=2,          # Número mínimo de amostras em uma folha
        #max_features=5,           # Número máximo de features consideradas em cada divisão
        #splitter='random'            # Estratégia para escolher a divisão ('best' ou 'random')
        #class_weight='balanced'
        )

    model3 = GradientBoostingClassifier(
        random_state=42,
        # n_estimators=100,
        # learning_rate=0.05,
        # max_depth=5,
        # min_samples_split=2,
        # min_samples_leaf=1,
        # #max_features='sqrt',
        # subsample=1.0,
    )


    result = []
    for model in [model1, model2, model3]:
        print(f'--- training baseline model: {model.__class__.__name__} ---')
        y_pred_train, y_pred_test = train_model(X, y, model)
        result.append({'model': model.__class__.__name__, 'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test})
        
    model_baseline_metrics = pd.DataFrame(result).round(2)

    return model_baseline_metrics





def feature_selection(X, y, model=RandomForestClassifier(random_state=42), res=0.95):

    rf_model = model
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    rf_model.fit(X_train, y_train)

    feature_importances = model.feature_importances_
    # Criar um dataframe para visualizar as importâncias
    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": feature_importances
    })
    # Ordenar as features pela importância
    importance_df = importance_df.sort_values(by="Importance", ascending=False)
    print(importance_df)

    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # Calcular a importância acumulada
    importance_df["Cumulative"] = importance_df["Importance"].cumsum()

    # Selecionar features com importância acumulada até 95%
    selected_features = importance_df[importance_df["Cumulative"] <= res]["Feature"].tolist()
    print(f"\nSelected Features: {selected_features}")

    return selected_features

def ks_test(df1, df2, sig=0.05):
    # Generate random examples (replace with your actual data)

    # Kolmogorov-Smirnov test
    statistic, p_value = ks_2samp(df1, df2)

    # Display results
    print(f"KS Statistic: {statistic}")
    print(f"p-value: {p_value}")

    # Evaluation of the result
    alpha = sig
    if p_value < alpha:
        print("We reject the null hypothesis: the distributions are different.")
    else:
        print("We do not reject the null hypothesis: the distributions are the same.")
    
    return statistic, p_value