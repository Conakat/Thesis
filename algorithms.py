from sklearn.model_selection import StratifiedKFold #type:ignore
from sklearn.neighbors import KNeighborsClassifier #type:ignore
from sklearn.model_selection import cross_val_score #type:ignore
from sklearn.model_selection import cross_val_predict #type:ignore
from sklearn.metrics import accuracy_score #type:ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score #type:ignore
from sklearn.model_selection import GridSearchCV #type:ignore
from sklearn.svm import SVC #type:ignore
import matplotlib.pyplot as plt #type:ignore
import numpy as np #type:ignore
from sklearn.preprocessing import StandardScaler #type:ignore
from sklearn.decomposition import PCA

def kNN_classification(x_train, x_test, y_train, y_test):

    num_train = len(x_train)
    flattened_trainset = x_train.reshape((num_train,-1))

    num_test = len(x_test)
    flattened_testset = x_test.reshape((num_test,-1))

    k_values = np.arange(1, 9)
    classes = ["all finger r", "all finger f", "all finger e", "thumb f", "thumb e", "index f", "index e", "middle f", "middle e", "ring f", "ring e", "pingy f", "pingy e"]
    accuracy_scores = np.empty(shape=(len(k_values),))

    for k in k_values:

        knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
        #scores = cross_val_score(knn, flattened_maps, labels, cv=5)

        knn.fit(flattened_trainset, y_train)
        
        # Calculate accuracy for this k value
        
        #accuracy = np.mean(scores)
        predicted_labels = knn.predict(flattened_testset)
        accuracy = np.mean(predicted_labels == y_test)
        #accuracy = accuracy_score(flattened_maps, labels)
        #Store the accuracy for this k value
        accuracy_scores[k-1] = accuracy
        '''
        plt.figure()
        sns.heatmap(cm_percentage1, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                    xticklabels=classes,
                    yticklabels=classes)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix for k={k}')
        plt.show()
        '''
    accuracy = np.array(accuracy)
    accuracy = np.mean(accuracy)
    print(f'Accuracy: {accuracy}')
    plt.figure(figsize=(6,6)) 
    plt.plot(k_values, accuracy_scores, marker='o')
    plt.grid()
    plt.title('NN accuracy for different number of nearest neighbors')
    plt.xlabel('k Value-Euclidean')
    plt.ylabel('Accuracy')
    plt.show()


def plot_hyperplane_and_decision_boundaries(clf, X, y):
    # Generate a mesh grid
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict class for each point in the mesh
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the contour and training points
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20, edgecolors='k')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title("SVM Decision Boundaries and Hyperplane")
    plt.show()


def svm_implementation(x_train, y_train, x_test, y_test):
    
    training_samples, axis, height, width  = x_train.shape
    testing_samples = x_test.shape[0]

    x_train = x_train.reshape(training_samples, height * width * axis)
    x_test = x_test.reshape(testing_samples, height * width * axis)  

    '''
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    '''

    param_grid = {'C': [0.1, 1, 10, 100],  
              'gamma': [0.1, 1, 10], 
              'kernel': ['linear']}  
    
    stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, cv=stratified_kfold)
    
    grid.fit(x_train, y_train)

    best_params = grid.best_params_
    best_score = grid.best_score_

    print(f"Best parameters found: {best_params}")
    print(f"Best cross-validated accuracy during grid search: {best_score}")
    
    best_model = grid.best_estimator_

    #clf = SVC(C=1000, gamma=10, kernel='linear')
    #clf = SVC(**best_params)
    #clf.fit(x_train, y_train)

    # Make predictions on the test set
    y_pred = best_model.predict(x_test)

    classes = ["all finger r", "all finger f", "all finger e", "thumb f", "thumb e", "index f", "index e", "middle f", "middle e", "ring f", "ring e", "pingy f", "pingy e"]
    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    #cm_percentage = cm / cm.sum(axis=1, keepdims=True) 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=classes)
    disp.plot(include_values=True, xticks_rotation='vertical', cmap='Blues')

    disp.ax_.set_xlabel('Predicted Labels')
    disp.ax_.set_ylabel('True Labels')
    plt.show()
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    precision = precision_score(y_test, y_pred, average='macro', zero_division='warn')
    print(f"Precision: {precision}")
    recall = recall_score(y_test, y_pred, average='macro', zero_division='warn')
    print(f"Recall: {recall}")

'''
    pca = PCA(n_components=2)
    x_test_pca = pca.fit_transform(x_test)
    plot_hyperplane_and_decision_boundaries(best_model, x_test_pca, y_test)
'''