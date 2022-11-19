#Confusion Matrix

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
#     plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

X_test=X_test.reshape('','','','')
y_pred_2 = model.predict(X_test)
y_enc = enc.fit_transform(y_test.reshape(-1, 1))
y_pred_2 = enc.inverse_transform(y_pred_2)

from sklearn.metrics import confusion_matrix
import itertools
cnf_matrix_2 = confusion_matrix(y_pred_2, enc.inverse_transform(y_test), labels=enc.classes_ )
#plt.figure(figsize=(20,20))
plot_confusion_matrix(cnf_matrix_2, classes=enc.classes_, title='Confusion matrix, using CNN')
plt.show()