# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 09:09:27 2020

@author: iad
"""
import numpy as np


def Normalization( x, height, width):
    '''
    Normalization of coordinates based on the slide 60 of the course
    Input
    -----
    x: the data to be normalized (directions at different columns and points at rows) (N ligne 3 colonne )
    height: the height of the input image
    width: the width of the input image
    ------
    Output
    Tr: the transformation matrix (translation plus scaling)
    x: the transformed data
    '''
    Tr = np.array([[2/height,0,-1],
                   [0,2/width,-1],
                   [0,0,1]])
    x = np.matmul(x,Tr.transpose())
    return Tr, x


def DLTcalib2(x, Hx):
    '''
    
    Camera calibration by DLT using known object points and their image points.
    Input
    -----
    x : size N,3. Homogenous coordinates of N 2D points
    Hx : size N,3. Homogenous coordinates of the same N 2D point x but after
    application of an unknown homographie H
    
    There must be at least 4 calibration points for the 2D DLT.
    Output
    ------
     H : array of 9 parameters of the calibration matrix. (the unknown homography)
     err: error of the DLT (mean residual of the DLT transformation in units of camera coordinates).
    '''
    # on ne travail que sur les points finis (notement pour calculer l'erreur à
    # la fin qui est difficilement interprétable pour les points infinis)
    eps = 10**(-6)
    finite_Hx = Hx[:,2] > eps * (np.abs(Hx[:,0]) + np.abs(Hx[:,1]))
    finite_x = x[:,2] > eps * (np.abs(x[:,0]) + np.abs(x[:,1]))
    finite = finite_Hx & finite_x
    if np.count_nonzero(finite) < 4:
        print('not enough finite point in DLTcalib2 input. A least 4 finite pairs required')
        return None
    # on impose que la troisieme coordonnées homogènes (x,y,z) des points finis
    # soit egale à 1
    Hx[finite_Hx,:] = Hx[finite_Hx,:] / Hx[finite_Hx,2].reshape(-1,1)
    x[finite_x,:] = x[finite_x,:] / x[finite_x,2].reshape(-1,1)
    # construction du systeme M à l'aide des paires de points finis
    ZerosArray = np.zeros((Hx[finite,:].shape))
    M = np.concatenate((np.concatenate((-x[finite,:],
                                        ZerosArray,
                                        Hx[finite,0].reshape(-1,1)*x[finite,:]),
                                       axis = 1),
                        np.concatenate((ZerosArray,
                                        -x[finite,:],
                                        Hx[finite,1].reshape(-1,1)*x[finite,:]),
                                       axis = 1)),
                       axis = 0)
    # résolution du systeme par SVD
    U, S, V = np.linalg.svd(M)
    # construction des sorties H, err_singVal et err_euclide2D
    bestSol = V[-1,:].reshape(1,-1)
    H = np.concatenate((bestSol[0:1,0:3],
                        bestSol[0:1,3:6],
                        bestSol[0:1,6:9]),
                       axis = 0)
    H = H/H[2,2] # on impose H[2,2] = 1
    err_singVal = S[-1] # cf la remarque en commentaire à la fin de cette fonction pour l'interpretation de err_singVal
    estimated_Hx = np.matmul(x,H.transpose())
    estimated_Hx[finite,:] = estimated_Hx[finite,:]/estimated_Hx[finite,2:3]
    err_euclide2D = np.linalg.norm(estimated_Hx[finite,0:2]-Hx[finite,0:2], axis = 1)
    err_euclide2D = err_euclide2D.sum()/err_euclide2D.size # distance euclidienne 2D moyenne entre les paires de points
    '''
    REMARQUE : Interpretation de : err_singVal = S[-1]
      On a choisi Hx et x fini et après application de H sont censé être egaux
    (en imposant leur troisième coeficient eqale à 1. On impose cette condition
    dans la suite de ce commentaire). 
      Si on appelle E le vecteur colonne de taille 2N * 1 contenant les erreurs
    entre les coefficients des points 2D de Hx et des points 2D de x après
    application de l'estimation de H, alors j'affirme que S[-1] est égale à la
    norme euclidienne de E dont chaque coefficient associé au ieme point a été
    pondéré par le produit scalaire xi * C avec C la troisième ligne de H estimé
    et xi le i-ieme point de x.
      De plus si l'estimation de H est réussi les xi * C valent
    approximativement 1 (car ici on a imposé que le troisième coef des points
    homogènes Hx vaux 1) donc S[-1] est "quasiment" la norme (euclidienne) des
    erreurs E entre les coefficients des points 2D esimé et vrai
    
    DEMONSTRATION DE LA REMARQUE :
      En effet en utilisant la formule slide 54 du cours, on peut voir, par
    orthonormailité des famille ui et vi, que :

    ||M*v9|| = ||u9s9|| = |s9| = s9 (s9 positif d'après l'enoncé du théorème de SVD)
    
    Ensuite si on reprend les slide 52 et 49 avec A, B et C les lignes de H
    estimé par SVD, (c'est à dire les coeficient de v9 regroupé 3 par 3) on a :
    
             [ -A(x1) + (x1)'C(x1)^t ]
             [ -B(x1) + (y1)'C(x1)^t ]
             [         .             ]
    M*v9 =   [         .             ]
             [         .             ]
             [ -A(xN) + (xN)'C(xN)^t ]
             [ -B(xN) + (yN)'C(xN)^t ]
    
    Puis en factorisant chaque ligne par (C(xi)^t) on obtient :
    
             [ (-A(x1)/(C(x1)^t)+(x1)') * (C(x1)^t) ]
             [ (-B(x1)/(C(x1)^t)+(y1)') * (C(x1)^t) ]
             [         .                            ]
    M*v9 =   [         .                            ]
             [         .                            ]
             [ (-A(xN)/(C(xN)^t)+(xN)') * (C(xN)^t) ]
             [ (-B(xN)/(C(xN)^t)+(yN)') * (C(xN)^t) ]

    On remarque alors que M*v9 est donc egale aux erreurs coefficients par
    coefficients entre les point 2D Hxi et H * xi pondéré par la 3 eme coordonnée
    homogène (C(xi)^t) de H * xi. autrement dit on a :

    s9 = norme euclidienne des eurreurs E pondéré par les (C(xi)^t)

    CQFD

    Et dans le cas ou l'estimation est correcte les (C(xi)^t) sont
    approximativement égaux à 1
    '''
    return H, err_singVal, err_euclide2D


def NormalizedDLTcalib2(x, Hx, height, width):
    Tr, normalized_x = Normalization(x ,height ,width)
    normalized_Hx = np.matmul(Hx,Tr.transpose())
    H, err_singVal = DLTcalib2(normalized_x, normalized_Hx)[0:2]
    invTr = np.linalg.inv(Tr)
    H = np.matmul(invTr, np.matmul(H, Tr))
    estimated_Hx = np.matmul(x,H.transpose())
    estimated_Hx[:,:] = estimated_Hx[:,:]/estimated_Hx[:,2:3]
    err_euclide2D = np.linalg.norm(estimated_Hx[:,0:2]-Hx[:,0:2], axis = 1)
    err_euclide2D = err_euclide2D.sum()/err_euclide2D.size # distance euclidienne 2D moyenne entre les paires de points
    return H, err_singVal, err_euclide2D


def run_ransac(x, Hx, threshold, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    """ RANSAC Coder for finding homography fit
    Input
    -----
    x : size N,3. Homogenous coordinates of N 2D points
    Hx : size N,3. Homogenous coordinates of the same N 2D point x but after
    application of an unknown homographie H
    threshold :  is the threshold to say if a data is inlier or not after
    reconstruction
    sample_size : is the size of the sample use to calculate the DLT
    goal_inliers : is the number of inlier on the dataset
    max_iterations : maximum number of iteration
    stop_at_goal : allow early stop
    random_seed : a seed to controle the process

    Output
    ------
     best_model: 3x3 array : the homographie H
     best_ic: best consensus value.
     """
    rg = np.random.default_rng(random_seed)
    H_list = []
    n_inlier = []
    n_iter = 0
    if sample_size < 4:
        print("sample_size should be an int equal to 4 or higher")
        return None
    Hx[:,:] = Hx[:,:]/Hx[:,2:3] # TODO si j'ai le temps : prendre en compte les points infinis (Hx[i,2:3] = 0)
    GoodEnough = False
    while ((not(GoodEnough & stop_at_goal)) & (n_iter < max_iterations)):
        # select randomly sample_size point pair
        randomIdx = rg.integers(x.shape[0],size = sample_size)
    
        # calculer le modèle à partir des points choisi
        H = DLTcalib2(x[randomIdx,:], Hx[randomIdx,:])[0]
    
        # determiner les inliers de ce modele
        isInlier = findHomographyInlier(H, x, Hx, threshold)
    
        if np.count_nonzero(isInlier) >= goal_inliers :
            GoodEnough = True
            # afine le modele a partir de tout les inliers
            H_list.append(DLTcalib2(x[isInlier,:], Hx[isInlier,:])[0]) 
            # inlier du nouveau modele 
            isInlier = findHomographyInlier(H_list[-1], x, Hx, threshold)
            n_inlier.append(np.count_nonzero(isInlier))
        n_iter = n_iter +1

    if len(n_inlier) == 0:
        return None,None,None
    
    best_idx = np.argmax(n_inlier)
    best_model = H_list[best_idx]
    best_ic = n_inlier[best_idx]
    
    # calcul les inliers à partir de la meilleure homographie
    isInlier = findHomographyInlier(best_model, x, Hx, threshold)
    return best_model, best_ic, isInlier

def findHomographyInlier(H, X, Y, threshold):
    # H 3x3 homography matrix
    # X Nx3 homogeneous 2D points
    # Y Nx3 homogeneous 2D points
    # threshold positive number 
    # test wether Y points are X points transformed by the homography H
    
    HX = np.matmul(X,H.transpose())
    HX[:,:] = HX[:,:]/HX[:,2:3]
    Y[:,:] = Y[:,:]/Y[:,2:3]
    err_euclide2D = np.linalg.norm(HX[:,0:2]-Y[:,0:2], axis = 1)
    isInlier = err_euclide2D < threshold    
    return isInlier

