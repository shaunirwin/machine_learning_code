import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import datasets, svm, metrics, manifold
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
# import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import os
import time


def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(digits.data.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    #         imagebox = offsetbox.AnnotationBbox(
    #             offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
    #             X[i])
    #         ax.add_artist(imagebox)
    # plt.xticks([]), plt.yticks([])
    # if title is not None:
    #     plt.title(title)


if __name__ == '__main__':
    folder = 'MNIST_data'
    training = pd.read_csv(os.path.join(folder, 'train.csv'), nrows=100000)
    target_names = [str(i) for i in range(10)]

    X_train = training.copy()
    y_train = X_train['label']
    X_train.drop('label', axis=1, inplace=True)

    X_train = X_train.values
    y_train = y_train.values

    # pca = PCA(n_components=3)
    # X_r = pca.fit(X_train).transform(X_train)
    # X_r_subset = X_r[::10, :]
    # y_train_subset = y_train[::10]
    # print(pca.explained_variance_ratio_)
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # colours = 'bgrcmykbgr'
    # for i, c, target_name in zip(range(10), colours, target_names):
    #     ax.scatter(X_r_subset[y_train_subset == i, 0],
    #                X_r_subset[y_train_subset == i, 1],
    #                X_r_subset[y_train_subset == i, 2],
    #                c=c, label=target_name)
    # plt.legend()
    # plt.title('PCA of MNIST dataset')
    # plt.show()

    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=2, learning_rate=100, n_iter=400, init='pca', random_state=0, verbose=3)
    t0 = time.time()
    # X_tsne = tsne.fit_transform(X_train[::10], y_train[::10])
    X_tsne = tsne.fit_transform(X_train[::10])

    plot_embedding(X_tsne,
                   y_train[::10],
                   "t-SNE embedding of the digits (time %.2f s)" %
                   (time.time() - t0))
    plt.show()

    print 'test'
