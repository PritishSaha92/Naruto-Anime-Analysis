from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import accuracy_score


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(y_true=labels, y_pred=predictions)}


def get_class_weights(df):
    class_weights = compute_class_weight("balanced",
                                         classes=np.unique(df['label']),
                                         y=df['label'].tolist()
                                         )
    return class_weights