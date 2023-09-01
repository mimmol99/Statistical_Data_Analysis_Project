import scipy.io
from scipy.io import loadmat
import numpy as np
import os
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns

#Caricamento dei dati
def load_data(file_name):
    #if mat not already in extension,add it
    appendmat = file_name.endswith(".mat")
    file = file_name.split(".")[0].split('/')[-1]
    mat_dict = loadmat(file_name,appendmat = appendmat)
    x = []
    y = []
    for raw in mat_dict[file]:
        if file == "train":
            x_raw = raw[:-1]
            y_raw = raw[-1:]
            x.append(x_raw)
            y.append(y_raw[0])
        else: 
            x.append(raw[:-1])
            y = None
    print('Features vectors', len(x))
    print('Features per array', len(x[0]))
    if y is not None:
        print('Labels', len(y))
        trues = 0
        falses = 0
        for i in y:
            if i == 1.0:
                trues = trues + 1
            elif i == -1.0:
                falses = falses + 1
        print('Trues', trues)
        print('Falses', falses)
    return x,y

#Riduzione della dimensionalità
def dimensionality_reduction(X,file_path,min_variance=0.95):
    features_standardized = standardization(X)
    n_components = len(features_standardized[0])
    pca_scaled = PCA(n_components)
    pca_scaled.fit(features_standardized)
    cumulative_variance_scaled = np.cumsum(pca_scaled.explained_variance_ratio_)
    plot_variance(pca_scaled.explained_variance_ratio_,cumulative_variance_scaled, np.arange(n_components), 'Principal components scaled', 'Variance ratio','variance on features standardized',file_path=file_path)
    # Find the number of components for at least min variance
    n_components_min_variance = np.argmax(cumulative_variance_scaled >= min_variance) + 1
    # Implementare il codice per la riduzione della dimensionalità
    pca_reducer = PCA(n_components_min_variance)
    features_reduced = pca_reducer.fit_transform(features_standardized)
    return features_reduced,pca_reducer
    
def standardization(features): 
    standard_scaler = preprocessing.StandardScaler()
    scaler_trained = standard_scaler.fit(features)
    features_standardized = scaler_trained.transform(features)
    return features_standardized

def plot_variance(variance, cum_variance, interval, x_label, y_label,title,file_path):  
    plt.bar(interval, variance, alpha = 0.5, align='center', label = 'Individual variance')
    plt.step(interval, cum_variance, where='mid', label = 'Cumulative variance')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='best')
    plt.savefig(file_path)
    plt.clf()

# Sigmoid function
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_gradient_logisticLoss(y, X, beta):
    y = np.array(y)
    X = np.array(X)
    #print('y',y.shape)
    #print('x',X.shape)
    #print('beta',beta.shape)
    #beta = np.expand_dims(beta, 0)
    return (-y * X.T) / (1 + np.exp(y * np.dot(X.T, beta)))

def logistic_class_SGD_logisticLoss(X, y, lr=0.01, n_epochs=10, file_path='gradient_vs_epoch.png'):
    X = np.insert(X, 0, 1, axis=1)
    m, n = X.shape
    weights = np.random.normal(size=n)
    epoch_avg_gradients = []  # Store average gradients for each epoch
    for epoch in tqdm(range(n_epochs)):
        epoch_gradients = []
        for i in range(m):
            random_idx = np.random.randint(m)
            xi = X[random_idx:random_idx+1]
            xi = np.reshape(xi, n)
            yi = y[random_idx:random_idx+1]
            #wi = weights[random_idx:random_idx+1]
            gradient = compute_gradient_logisticLoss(yi, xi, weights)
            weights -= lr * gradient
            epoch_gradients.append(np.linalg.norm(gradient))
        epoch_avg_gradient = np.mean(epoch_gradients)
        epoch_avg_gradients.append(epoch_avg_gradient)
    plot_gradients(epoch_avg_gradients, file_path)
    return weights

def plot_gradients(gradient_norms, file_path):
    plt.plot(range(len(gradient_norms)), gradient_norms, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Average Gradient Norm')
    plt.title('Gradient Norm vs Epoch')
    plt.grid(True)
    plt.savefig(file_path)
    plt.clf()

def predict(X, weights):
    X = np.insert(X, 0, 1, axis=1)
    probabilities = sigmoid(np.dot(X, weights))
    return [1.0 if p > 0.5 else -1.0 for p in probabilities]

def logistic_classifer(X, y, step_size=0.1, epochs=20,file_path='logistic_classifier.png'):
    weights = logistic_class_SGD_logisticLoss(X, y, step_size, epochs,file_path)
    predictions = predict(X, weights)
    return predictions, weights

def evaluate_step_sizes(X, y,step_sizes,epochs,file_path):
    pred_best = None
    beta_best = None
    highest_accuracy = 0
    best_step_size = None
    accuracies = []
    for step_size in step_sizes:
        pred, beta = logistic_classifer(X, y, step_size=step_size,epochs=epochs,file_path=os.path.join(images_path,'gradient_epochs_with_step_size_'+str(step_size)+'.png'))
        vett = [1 if pred[i]==y[i] else 0 for i in range(len(y))]
        accuracy = sum(vett)/len(vett)
        print('Accuracy for', step_size, accuracy)
        accuracies.append(round(accuracy,3))
        # Store the best model
        if accuracy >= highest_accuracy:
            highest_accuracy = accuracy
            beta_best = beta
            pred_best = pred
            best_step_size = step_size
    print('==============================================')
    print(f"Best Logistic Classifier Accuracy with step size {best_step_size}: {highest_accuracy}")
    plot_accuracy_step_sizes(accuracies,step_sizes,file_path)
    return pred_best, beta_best

def plot_accuracy_step_sizes(accuracies,step_sizes,file_path):
    # Plotting
    plt.plot(range(len(step_sizes)), accuracies, marker='o')  # Change here
    plt.xlabel('Step Size')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Step Size')
    plt.grid(True)
    plt.xticks(range(len(step_sizes)), [str(s) for s in step_sizes])
    # Annotate each point
    for i, (x, y) in enumerate(zip(step_sizes, accuracies)):
        plt.annotate(str(y), (i, y))  # Use index i as x-coordinate
    plt.savefig(file_path)  # Save the plot to a file
    plt.clf()  # Clear the plot

# Inference function for new X values
def logistic_inference(X_test, best_theta):
    predictions = []
    for i, x_value in enumerate(X_test):
        res = np.dot(x_value, best_theta)
        if res > 0:
            predictions.append(1.0)
        else:
            predictions.append(-1.0)
    print(f"Lenght of predictions {len(predictions)}")
    return predictions
    
# 6. Regola di decisione
def decision_rule(predictions):
    # Implementare la regola di decisione
    binaries = []
    for prediction in predictions:
        if prediction == -1.0:
            binaries.append(0)
        else:
            binaries.append(1)
    return binaries

# 7. Conversione in ASCII
def to_ascii(binary_bytes):
    binary_str = ''.join(map(str, binary_bytes))
    # Split the binary string into chunks of 8 bits and convert each chunk to ASCII
    ascii_chars = [chr(int(binary_str[i:i+8], 2)) for i in range(0, len(binary_str), 8)]
    return ascii_chars
    
def plot_correlation_matrix(x,filename):
    df = pd.DataFrame.from_dict(x)
    plt.figure("Correlation matrix")
    sns.heatmap(df.corr())
    plt.savefig(filename)
    plt.clf()

if __name__ == "__main__":
    
    images_path = os.path.join(os.getcwd(),'Images')
    if os.path.exists(images_path) is False:
        os.makedirs(images_path)

    #load data
    print('==============================================')
    print("Train data:")
    train_file_name = "Gruppo 3/train.mat"
    x_train,y_train = load_data(file_name=train_file_name)
    print('==============================================')
    print("Test data:")
    test_file_name = "Gruppo 3/test.mat"
    x_test,y_test = load_data(file_name=test_file_name)

    plot_correlation_matrix(x_train,filename=os.path.join(images_path,'correlation_matrix.png'))
    
    print('==============================================')
    #reduce dimensionality
    x_train_reduced,pca_reducer = dimensionality_reduction(x_train,file_path=os.path.join(images_path,'dimensionality_reduction.png'))
    print("Features arrays reduced:", len(x_train_reduced))
    print("Features per train array reduced:", len(x_train_reduced[0]))
    print('==============================================')
    #step sizes
    step_sizes = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
    #find best logistic classifier
    best_bias,best_beta = evaluate_step_sizes(x_train_reduced,y_train,step_sizes,epochs = 15,file_path=os.path.join(images_path,'accuracy on step size.png'))
    best_beta = best_beta[1:]
    #print(f"best beta:{best_beta}")
    x_test_reduced = pca_reducer.transform(x_test)
    print('==============================================')
    print("Lenght x test reduced:",len(x_test_reduced[0]))
    #predictions on test data
    predictions = logistic_inference(x_test_reduced,best_beta)
    #print(predictions)
    #convert predictions to binary
    binary_bytes = decision_rule(predictions)
    #convert binary to ascii
    characters = to_ascii(binary_bytes)
    print('==============================================')
    print(f"characters found :{characters}")

