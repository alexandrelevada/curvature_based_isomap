#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Curvature based Isometric Feature Mapping

@author: Alexandre L. M. Levada

"""

# Imports
import sys
import time
import warnings
import umap                 # precisa instalar com: pip install umap
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
import sklearn.utils.graph as sksp
import networkx as nx
from numpy import log
from numpy import trace
from numpy import dot
from numpy import sqrt
from numpy.linalg import det
from numpy.linalg import inv
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import RandomForestClassifier

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

# PCA implementation
def myPCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    
    return novos_dados

# ISOMAP implementation
def myIsomap(dados, k, d):
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='distance')
    # Computes geodesic distances
    A = knnGraph.toarray()
    G = nx.from_numpy_matrix(A)
    #D = sksp.graph_shortest_path(A, directed=False)
    D = nx.floyd_warshall_numpy(G)  
    n = D.shape[0]
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    
    return output

# K-ISOMAP implementation
def GeodesicIsomap(dados, k, d):
    
    n = dados.shape[0]
    m = dados.shape[1]

    # Componentes principais extraídos de cada patch (espaço tangente)
    matriz_pcs = np.zeros((n, m, m))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=k, mode='connectivity')
    A = knnGraph.toarray()
    
    # Computes the means and covariance matrices for each patch
    for i in range(n):       
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]
        if len(indices) == 0:   # Isolated points
            matriz_pcs[i, :, :] = np.eye(m)    # Autovetores nas colunas
        else:
            amostras = dados[indices]
            v, w = np.linalg.eig(np.cov(amostras.T))
            # Sort the eigenvalues
            ordem = v.argsort()
            # Select the d eigenvectors associated to the d largest eigenvalues
            maiores_autovetores = w[:, ordem[::-1]]     # Esse é o oficial!
            #maiores_autovetores = w[:, ordem[-1:]]      # Pega apenas os 2 primeiros (teste)
            # Projection matrix
            Wpca = maiores_autovetores  # Autovetores nas colunas
            #print(Wpca.shape)
            matriz_pcs[i, :, :] = Wpca
        
    # Defines the patch-based matrix (graph)
    B = A.copy()
    for i in range(n):
        for j in range(n):
            if B[i, j] > 0:
                delta = np.linalg.norm(matriz_pcs[i, :, :] - matriz_pcs[j, :, :], axis=0)
                B[i, j] = np.linalg.norm(delta)                
               
    # Computes geodesic distances in B
    #D = sksp.graph_shortest_path(B, directed=False)
    G = nx.from_numpy_matrix(B)
    D = nx.floyd_warshall_numpy(G)  
    # Computes centering matrix H
    H = np.eye(n, n) - (1/n)*np.ones((n, n))
    # Computes the inner products matrix B
    B = -0.5*H.dot(D**2).dot(H)
    # Remove infs or nans
    maximo = np.nanmax(B[B != np.inf])   
    B[np.isnan(B)] = 0
    B[np.isinf(B)] = maximo
    # Eigeendecomposition
    lambdas, alphas = sp.linalg.eigh(B)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    
    return output

'''
 Computes the Silhouette coefficient and the supervised classification
 accuracies for several classifiers: KNN, SVM, NB, DT, QDA, MPL, GPC and RFC
 dados: learned representation (output of a dimens. reduction - DR)
 target: ground-truth (data labels)
 method: string to identify the DR method (PCA, NP-PCAKL, KPCA, ISOMAP, LLE, LAP, ...)
'''
def Classification(dados, target, method):
    print()
    print('Supervised classification for %s features' %(method))
    print()
    
    lista = []

    # 50% for training and 40% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados.real.T, target, test_size=.5, random_state=42)

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train) 
    acc = neigh.score(X_test, y_test)
    lista.append(acc)
    #print('KNN accuracy: ', acc)

    # SMV
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    acc = svm.score(X_test, y_test)
    lista.append(acc)
    #print('SVM accuracy: ', acc)

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    acc = nb.score(X_test, y_test)
    lista.append(acc)
    #print('NB accuracy: ', acc)

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    acc = dt.score(X_test, y_test)
    lista.append(acc)
    #print('DT accuracy: ', acc)

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    acc = qda.score(X_test, y_test)
    lista.append(acc)
    #print('QDA accuracy: ', acc)

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    acc = mpl.score(X_test, y_test)
    lista.append(acc)
    #print('MPL accuracy: ', acc)

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    acc = gpc.score(X_test, y_test)
    lista.append(acc)
    #print('GPC accuracy: ', acc)

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    acc = rfc.score(X_test, y_test)
    lista.append(acc)
    #print('RFC accuracy: ', acc)

    # Computes the Silhoutte coefficient
    sc = metrics.silhouette_score(dados.real.T, target, metric='euclidean')
    print('Silhouette coefficient: ', sc)
    
    # Computes the average accuracy
    average = sum(lista)/len(lista)
    maximo = max(lista)

    print('Average accuracy: ', average)
    #print('Maximum accuracy: ', maximo)
    print()

    return [sc, average]


# Function for data plotting
def PlotaDados(dados, labels, metodo):
    
    nclass = len(np.unique(labels))

    if metodo == 'LDA':
        if nclass == 2:
            return -1

    # Convert labels to integers
    lista = []
    for x in labels:
        if x not in lista:  
            lista.append(x)     

    # Map labels to nunbers
    rotulos = []
    for x in labels:  
        for i in range(len(lista)):
            if x == lista[i]:  
                rotulos.append(i)

    # Numpy array
    rotulos = np.array(rotulos)

    if nclass > 11:
        cores = ['black', 'gray', 'silver', 'whitesmoke', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon']

    plt.figure(10)
    for i in range(nclass):
        indices = np.where(rotulos==i)[0]
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='*')
    
    nome_arquivo = metodo + '.png'
    plt.title(metodo+' clusters')

    plt.savefig(nome_arquivo)
    plt.close()



#%%%%%%%%%%%%%%%%%%%% Beginning of the script

#%%%%%%%%%%%%%%%%%%%%  Data loading

# OpenML datasets
X = skdata.load_iris()     
#X = skdata.fetch_openml(name='prnn_crabs', version=1) 
#X = skdata.fetch_openml(name='servo', version=1) 
#X = skdata.fetch_openml(name='tic-tac-toe', version=1)
#X = skdata.fetch_openml(name='visualizing_galaxy', version=2)
#X = skdata.fetch_openml(name='sleuth_ex1605', version=2)
#X = skdata.fetch_openml(name='mux6', version=1) 
#X = skdata.fetch_openml(name='car-evaluation', version=1)          
#X = skdata.fetch_openml(name='blogger', version=1)
#X = skdata.fetch_openml(name='sa-heart', version=1) 
#X = skdata.fetch_openml(name='pyrim', version=2)  
#X = skdata.fetch_openml(name='SPECTF', version=1) 
#X = skdata.fetch_openml(name='veteran', version=2) 
#X = skdata.fetch_openml(name='balance-scale', version=1)
#X = skdata.fetch_openml(name='parity5', version=1) 
#X = skdata.fetch_openml(name='kidney', version=2)      
#X = skdata.fetch_openml(name='hayes-roth', version=2)  
#X = skdata.fetch_openml(name='diabetes_numeric', version=2) 
#X = skdata.fetch_openml(name='parkinsons', version=1) 
#X = skdata.fetch_openml(name='rabe_131', version=2)
#X = skdata.fetch_openml(name='corral', version=1)
#X = skdata.fetch_openml(name='baskball', version=2)  
#X = skdata.fetch_openml(name='grub-damage', version=2)
#X = skdata.fetch_openml(name='bolts', version=2) 
#X = skdata.fetch_openml(name='haberman', version=1) 
#X = skdata.fetch_openml(name='TuningSVMs', version=1)
#X = skdata.fetch_openml(name='backache', version=1)  
#X = skdata.fetch_openml(name='prnn_synth', version=1) 
#X = skdata.fetch_openml(name='visualizing_environmental', version=2)  
#X = skdata.fetch_openml(name='mu284', version=2)
#X = skdata.fetch_openml(name='ar4', version=1) 
#X = skdata.fetch_openml(name='Engine1', version=1) 
#X = skdata.fetch_openml(name='prnn_viruses', version=1) 
#X = skdata.fetch_openml(name='vineyard', version=2) 
#X = skdata.fetch_openml(name='confidence', version=2) 
#X = skdata.fetch_openml(name='user-knowledge', version=1) 

dados = X['data']
target = X['target']

#%%%%%%%%%%%%%%%%%%%% Supervised classification for ISOMAP-KL features

n = dados.shape[0]
m = dados.shape[1]
c = len(np.unique(target))

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
input()

# Only for OpenML datasets
# Treat catregorical features
if not isinstance(dados, np.ndarray):
    cat_cols = dados.select_dtypes(['category']).columns
    dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
    # Convert to numpy
    dados = dados.to_numpy()
    target = target.to_numpy()

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

#%%%%%%%%%%% Simple PCA 
dados_pca = myPCA(dados, 2)

#%%%%%%%%%%% ISOMAP
model = Isomap(n_neighbors=20, n_components=2)
dados_isomap = model.fit_transform(dados)
dados_isomap = dados_isomap.T

#%%%%%%%%%%% LLE
model = LocallyLinearEmbedding(n_neighbors=20, n_components=2)
dados_LLE = model.fit_transform(dados)
dados_LLE = dados_LLE.T

#%%%%%%%%%%% Lap. Eig.
model = SpectralEmbedding(n_neighbors=20, n_components=2)
dados_Lap = model.fit_transform(dados)
dados_Lap = dados_Lap.T

#%%%%%%%%%%% LTSA
model = LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='ltsa')
dados_ltsa = model.fit_transform(dados)
dados_ltsa = dados_ltsa.T

#%%%%%%%%%%%%% t-SNE
model = TSNE(n_components=2, perplexity=30)
dados_tsne = model.fit_transform(dados)
dados_tsne = dados_tsne.T

#%%%%%%%%%%%% UMAP
model = umap.UMAP(n_components=2)
dados_umap = model.fit_transform(dados)
dados_umap = dados_umap.T

#%%%%%%%%%%% Supervised classification
L_pca = Classification(dados_pca, target, 'PCA')
L_iso = Classification(dados_isomap, target, 'ISOMAP')
L_lle = Classification(dados_LLE, target, 'LLE')
L_lap = Classification(dados_Lap, target, 'Lap. Eig.')
#L_ltsa = Classification(dados_ltsa, target, 'LTSA')
L_tsne = Classification(dados_tsne, target, 't-SNE')
L_umap = Classification(dados_umap, target, 'UMAP')


# Plota resultados
PlotaDados(dados_pca.T, target, 'PCA')
PlotaDados(dados_isomap.T, target, 'ISOMAP')
PlotaDados(dados_LLE.T, target, 'LLE')
PlotaDados(dados_Lap.T, target, 'LAP')
#PlotaDados(dados_ltsa.T, target, 'LTSA')
PlotaDados(dados_tsne.T, target, 't-SNE')
PlotaDados(dados_umap.T, target, 'UMAP')

#################### Geodesic ISOMAP features

inicio = 2
incremento = 1
fim = 51

lista_k = list(range(inicio, fim, incremento))

acuracias = []
silhouettes = []

for k in lista_k:

    print('K = ', k)
    # Geodesic ISOMAP
    dados_isokl = GeodesicIsomap(dados, k, 2)
    
    # 50% for training and 50% for testing
    X_train, X_test, y_train, y_test = train_test_split(dados_isokl.real, target, test_size=.5, random_state=42)
    acc = 0

    X_train_dt_rfc = X_train/X_train.max()
    X_test_dt_rfc = X_test/X_test.max()

    # KNN
    neigh = KNeighborsClassifier(n_neighbors=7)
    neigh.fit(X_train, y_train)
    acc += neigh.score(X_test, y_test)
    #print('KNN accuracy: ', neigh.score(X_test, y_test))

    # SMV
    svm = SVC(gamma='auto')
    svm.fit(X_train, y_train) 
    acc += svm.score(X_test, y_test)
    #print('SVM accuracy: ', svm.score(X_test, y_test))

    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    acc += nb.score(X_test, y_test)
    #print('NB accuracy: ', nb.score(X_test, y_test))

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    acc += dt.score(X_test, y_test)
    #print('DT accuracy: ', dt.score(X_test, y_test))

    # Quadratic Discriminant 
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    acc += qda.score(X_test, y_test)
    #print('QDA accuracy: ', qda.score(X_test, y_test))

    # MPL classifier
    mpl = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', max_iter=5000)
    mpl.fit(X_train, y_train)
    acc += mpl.score(X_test, y_test)
    #print('MPL accuracy: ', mpl.score(X_test, y_test))

    # Gaussian Process
    gpc = GaussianProcessClassifier()
    gpc.fit(X_train, y_train)
    acc += gpc.score(X_test, y_test)
    #print('GPC accuracy: ', gpc.score(X_test, y_test))

    # Random Forest Classifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    acc += rfc.score(X_test, y_test)
    #print('RFC accuracy: ', rfc.score(X_test, y_test))

    acuracia = acc/8
    sc = metrics.silhouette_score(dados_isokl.real, target, metric='euclidean')

    acuracias.append(acuracia)
    silhouettes.append(sc)

print('List of values for K: ', lista_k)
print('Supervised classification accuracies: ', acuracias)
acuracias = np.array(acuracias)
print('Max Acc: ', acuracias.max())
k_star = lista_k[acuracias.argmax()]
print('K* = ', k_star)
print()

plt.figure(1)
plt.plot(lista_k, acuracias)
plt.title('Mean accuracies for different values of K (neighborhood)')
plt.show()

print('Silhouette coefficients: ', silhouettes)
silhouettes = np.array(silhouettes)
print('Max SC: ', silhouettes.max())
print('SC* = ', lista_k[silhouettes.argmax()])
print()

plt.figure(2)
plt.plot(lista_k, silhouettes, color='red')
plt.title('Silhouette coefficients for different values of K (neighborhood)')
plt.show()

dados_isokl = GeodesicIsomap(dados, k_star, 2)
PlotaDados(dados_isokl, target, 'K-ISO')