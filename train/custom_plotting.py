import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve


def evaluate_model(test_labels
                   ,predictions
                   ,probs
                   ,train_labels
                   ,train_predictions
                   ,train_probs
                   ,save_png=False
                   ,png_name='roc.png'):
    
    # Plot formatting
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.size'] = 18

    baseline = {}
    
    baseline['recall'] = recall_score(test_labels, 
                                     [1 for _ in range(len(test_labels))])
    baseline['precision'] = precision_score(test_labels, 
                                      [1 for _ in range(len(test_labels))])
    baseline['roc'] = 0.5
    
    results = {}
    
    results['recall'] = recall_score(test_labels, predictions)
    results['precision'] = precision_score(test_labels, predictions)
    results['roc'] = roc_auc_score(test_labels, probs)
    
    train_results = {}
    train_results['recall'] = recall_score(train_labels, train_predictions)
    train_results['precision'] = precision_score(train_labels, train_predictions)
    train_results['roc'] = roc_auc_score(train_labels, train_probs)
    
    for metric in ['recall', 'precision', 'roc']:
        print(str(metric.capitalize())+ 'Baseline: ' + str(round(baseline[metric], 2)) + ' Test: '+ str(round(results[metric], 2)) + ' Train: ' +str(round(train_results[metric], 2)))
    
    # Calculate false positive rates and true positive rates
    base_fpr, base_tpr, _ = roc_curve(test_labels, [1 for _ in range(len(test_labels))])
    model_fpr, model_tpr, _ = roc_curve(test_labels, probs)

    plt.figure(figsize = (8, 8))
    plt.rcParams['font.size'] = 16
    
    # Plot both curves
    plt.plot(base_fpr, base_tpr, 'b', label = 'baseline')
    plt.plot(model_fpr, model_tpr, 'r', label = 'model')
    plt.legend();
    plt.xlabel('False Positive Rate'); 
    plt.ylabel('True Positive Rate'); 
    plt.title('ROC Curves');
    plt.show();
    
    if (save_png):
        plt.savefig(png_name)    
    

def plot_confusion_matrix(cm
                          ,classes
                          ,normalize=False
                          ,title='Confusion matrix'
                          ,cmap=plt.cm.Oranges
                          ,save_png=False
                          ,png_name='cm.png'):
    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)

    if (save_png):
        plt.savefig(png_name)

