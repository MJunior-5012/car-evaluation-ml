import csv
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

from scratch_tree import ScratchTree

ROOT = Path(__file__).resolve().parent
FEATURES = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']


def load_data():
    with (ROOT / 'dataset' / 'car.data').open(newline='') as stream:
        rows = list(csv.reader(stream))
    if len(rows) != 1728 or any(len(row) != 7 or any(not value for value in row) for row in rows):
        raise ValueError('Expected 1728 complete rows with six features and a target')
    values = np.asarray(rows)
    return values[:, :6], values[:, 6]


def main():
    raw, targets = load_data()
    train_ids, test_ids = train_test_split(np.arange(len(targets)), test_size=0.2, stratify=targets, random_state=42)
    encoder = OrdinalEncoder()
    X_train = encoder.fit_transform(raw[train_ids])
    X_test = encoder.transform(raw[test_ids])
    labels = LabelEncoder().fit(targets[train_ids])
    y_train = labels.transform(targets[train_ids])
    y_test = labels.transform(targets[test_ids])
    models = {
        'majority_baseline': DummyClassifier(strategy='most_frequent'),
        'sklearn_depth_2': DecisionTreeClassifier(max_depth=2, criterion='gini', random_state=42),
        'sklearn_unrestricted': DecisionTreeClassifier(criterion='gini', random_state=42),
        'numpy_depth_2': ScratchTree(max_depth=2, min_size=1),
    }
    output = ROOT / 'study' / 'current-run'
    output.mkdir(parents=True, exist_ok=True)
    metrics = {}
    all_labels = np.arange(len(labels.classes_))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for (name, model), ax in zip(models.items(), axes.flat):
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        matrix = confusion_matrix(y_test, predicted, labels=all_labels)
        metrics[name] = {
            'train_accuracy': float(accuracy_score(y_train, model.predict(X_train))),
            'test_accuracy': float(accuracy_score(y_test, predicted)),
            'test_macro_f1': float(f1_score(y_test, predicted, average='macro', zero_division=0)),
            'confusion_matrix': matrix.tolist(),
            'classification_report': classification_report(y_test, predicted, labels=all_labels, target_names=labels.classes_.tolist(), output_dict=True, zero_division=0),
        }
        ax.imshow(matrix, cmap='Blues')
        ax.set(xticks=all_labels, yticks=all_labels, xticklabels=labels.classes_, yticklabels=labels.classes_, xlabel='Predicted', ylabel='Actual', title=name.replace('_', ' '))
        for i in all_labels:
            for j in all_labels:
                ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i,j] > matrix.max()/2 else 'black')
    fig.savefig(output / 'confusion-matrices.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    plot_tree(models['sklearn_depth_2'], feature_names=FEATURES, class_names=labels.classes_.tolist(), filled=True, rounded=True, ax=ax, fontsize=9)
    fig.savefig(output / 'shallow-tree.png', dpi=160)
    plt.close(fig)
    summary = {
        'seed': 42,
        'train_rows': len(train_ids),
        'test_rows': len(test_ids),
        'dataset_sha256': hashlib.sha256((ROOT / 'dataset' / 'car.data').read_bytes()).hexdigest(),
        'class_order': labels.classes_.tolist(),
        'class_counts': {value: int(np.sum(targets == value)) for value in labels.classes_},
        'feature_encodings': {feature: category.tolist() for feature, category in zip(FEATURES, encoder.categories_)},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scikit-learn': sklearn.__version__, 'matplotlib': matplotlib.__version__},
        'models': metrics,
    }
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    (output / 'split-indices.json').write_text(json.dumps({'train': train_ids.tolist(), 'test': test_ids.tolist()}, indent=2) + '\n')
    with (output / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'train_accuracy', 'test_accuracy', 'test_macro_f1'])
        for name, row in metrics.items():
            writer.writerow([name, row['train_accuracy'], row['test_accuracy'], row['test_macro_f1']])
            print(f"{name:24} train={row['train_accuracy']:.4f} test={row['test_accuracy']:.4f} macro-F1={row['test_macro_f1']:.4f}")


if __name__ == '__main__':
    main()
