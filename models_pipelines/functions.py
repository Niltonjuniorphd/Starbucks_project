
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from scipy.stats import ks_2samp



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

def print_metrics(model, y_train, y_test, y_pred_train, y_pred_test):

    
    print('Train Metrics\n')
    print(classification_report(y_train, y_pred_train, target_names=(model.named_steps['classifier'].classes_).astype(str), zero_division=1))
    print('Test Metrics\n')
    print(classification_report(y_test, y_pred_test, target_names=(model.named_steps['classifier'].classes_).astype(str), zero_division=1))

    print('Confusion Matrix Train\n')
    cm2 = confusion_matrix(y_train, y_pred_train)
    disp1 = ConfusionMatrixDisplay(
                confusion_matrix=cm2, display_labels=model.named_steps['classifier'].classes_)
    disp1.plot(cmap="Blues")
    plt.show()

    print('Confusion Matrix Test\n')
    cm1 = confusion_matrix(y_test, y_pred_test)
    disp1 = ConfusionMatrixDisplay(
                confusion_matrix=cm1, display_labels=model.named_steps['classifier'].classes_, )
    disp1.plot(cmap="Blues")
    plt.show()



def print_metrics2(model, y, y_pred):

    print('Validation Metrics\n')
    print(classification_report(y, y_pred, target_names=(model.named_steps['classifier'].classes_).astype(str), zero_division=1))

    print('Confusion Matrix Train\n')
    cm2 = confusion_matrix(y, y_pred)
    disp1 = ConfusionMatrixDisplay(
                confusion_matrix=cm2, display_labels=model.named_steps['classifier'].classes_)
    disp1.plot(cmap="Blues")
    plt.show()



def feature_importance(model, percent=0.8, w=4, h=9 ):   

    feature_importances = model.named_steps['classifier'].feature_importances_
    importance_df = pd.DataFrame({
    "Feature": [x[5:] for x in model.named_steps['preprocessor'].get_feature_names_out()],
    "Importance": feature_importances}).sort_values(by="Importance", ascending=True).reset_index(drop=True)

    importance_df["Cumulative"] = importance_df["Importance"].cumsum()

    plt.figure(figsize=(w, h))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.show()

    print(importance_df['Feature'].tolist()[::-1])

    feature_selection = importance_df['Feature'].loc[(importance_df['Cumulative'] >=percent)][::-1].tolist()

    return feature_selection

