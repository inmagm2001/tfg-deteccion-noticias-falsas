import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



def plot_roc_auc(model, X_test, y_test, label='Modelo'):
    """
    Dibuja la curva ROC y muestra el AUC para un modelo binario.
    Compatible con modelos que usan .predict_proba() o .decision_function().
    
    Parámetros:
        model: pipeline o clasificador entrenado
        X_test: conjunto de prueba
        y_test: etiquetas (0/1 o 'Falsa'/'Real')
        label: nombre del modelo (opcional)
    """
    
    # Intentar obtener scores
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        raise AttributeError("El modelo no tiene ni predict_proba ni decision_function")

    # Calcular ROC y AUC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    auc = roc_auc_score(y_test, y_score)

    # Graficar
    plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("Tasa de falsos positivos (FPR)")
    plt.ylabel("Tasa de verdaderos positivos (TPR)")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid(True)


def plot_feature_importance(model, top_n=20):
    importances = model.named_steps['clf'].feature_importances_
    names = model.named_steps['prep'].get_feature_names_out()

    imp_df = pd.DataFrame({'feature': names, 'importance': importances})
    imp_df = imp_df.sort_values(by='importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(imp_df['feature'], imp_df['importance'])
    plt.title(f"Top {top_n} características más importantes")
    plt.xlabel("Importancia")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels=['Falsa', 'Real']):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Matriz de confusión")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.show()