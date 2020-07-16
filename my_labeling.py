__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np
import Kmeans as km
import KNN
from utils_data import read_dataset, visualize_k_means, visualize_retrieval, Plot3DCloud
import matplotlib.pyplot as plt
import cv2
import json
from more_itertools import locate

from scipy.spatial.distance import cdist
import time
#     print("--- Tiempo de ejecución: %s segundos ---" % (time.time() - start_time))


# ANALISI QUALITATIU


def retrieval_by_color(images, labels, question):  # question = color/s

    # lista = []

    return np.array(images)[list(locate(labels, lambda x: any(item in question for item in x)))]

    # for i, label in enumerate(labels):
      # if question in label:
        # lista.append(images[i])
    # visualize_retrieval(lista, len(lista))



def retrieval_by_shape(images, labels, question):
    lista = []

    for i, label in enumerate(labels):
        for j in question:
            if j == label:
                lista.append(images[i])
                break

    # lista = np.array(images)[list(locate(labels, lambda x: question == x))]
    # visualize_retrieval(lista, len(lista))


def retrieval_combined(images, color_labels, shape_labels, color_question, shape_question):

    lista = []

    combined_labels = list(zip(color_labels, shape_labels))
    for i, combined_labels in enumerate(combined_labels):
        bool_color = False
        bool_shape = False
        for color in color_question:
           if color in combined_labels[0]:
                bool_color = True
                break
        for shape in shape_question:
            if shape == combined_labels[1]:
                bool_shape = True
                break
        if bool_color and bool_shape:
            lista.append(images[i])

    # return np.array(images)[list(locate(list(zip(color_labels, shape_labels)), lambda x: color_question in x[0] and
                                        # shape_question == x[1]))]
    # visualize_retrieval(lista, len(lista))


# ANALISI QUANTITATIU


def kmean_statistics(element_kmeans, images, kmax=4):
    for i in range(2, kmax):
        wcd = element_kmeans.withinClassDistance()
        iteracions = element_kmeans.num_iter
        visualize_k_means(element_kmeans, 4800)


def get_shape_accuracy(knn_labels, shape_labels):

    accuracy = 100*sum(1 for x, y in zip(sorted(knn_labels), sorted(shape_labels)) if x == y) / len(knn_labels)
    print(accuracy)


def get_color_accuracy(kmeans_labels, color_labels):
    accuracy = 100 * sum(1 for x, y in zip(sorted(kmeans_labels), sorted(color_labels)) if x == y) / len(kmeans_labels)
    print(accuracy)


if __name__ == '__main__':
    start_time = time.time()

    #Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, \
    test_imgs, test_class_labels, test_color_labels = read_dataset(ROOT_FOLDER='./images/', gt_json='./images/gt.json')

    #List with all the existant classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))
    labels_complete = list(list(train_class_labels) + list(test_class_labels))

    # Qualitative functions
   # retrieval_by_color(test_imgs, test_color_labels, ['Pink', 'Black'])
    #retrieval_by_shape(test_imgs, test_class_labels, [Dresses', 'Flip Flops''])
    #retrieval_combined(test_imgs, test_color_labels, test_class_labels, ['Black', 'White'], ['Dresses', 'Flip Flops'])

    # Kmeans and KNN elements declaration
    element_kmeans = km.KMeans(test_imgs)
    element_kmeans.fit()
    kmeans_labels = km.get_colors(element_kmeans.centroids)
    get_color_accuracy(kmeans_labels, test_color_labels)
    kmean_statistics(element_kmeans, test_imgs, 5)

   # knn = KNN.KNN(train_imgs, test_class_labels)
   # knn_labels = knn.predict(test_imgs, 3)

    # Quantitative functions
    # get_shape_accuracy(knn_labels, test_class_labels)



# You can start coding your functions here









