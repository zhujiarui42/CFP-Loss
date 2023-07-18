import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn import manifold


def draw(img: torch.Tensor, vis=True, save_path=None):
    """
    draw the image
    :param img: tensor with [H, W]
    :param vis: if visual?
    :param save_path: the path of saving image, if none it will not save
    :return:
    """
    _img = img.detach().cpu().numpy()
    plt.imshow(_img)
    plt.colorbar()
    if vis:
        plt.show()


def tsne(feature, k=3, vis=True, save_path=None):
    """
        draw the image
        :param img: tensor with [H, W]
        :param vis: if visual?
        :param save_path: the path of saving image, if none it will not save
        :return:
        """
    L = feature.size(0)
    classes = np.zeros([L])
    """
    classes[L // 6:L * 2 // 6] = 1
    classes[L * 2 // 6:L * 3 // 6] = 2
    classes[L * 3 // 6:L * 4 // 6] = 3
    classes[L * 4 // 6:L * 5 // 6] = 4
    classes[L * 5 // 6:] = 5
    """
    classes[L // 4:L * 2 // 4] = 1
    classes[L * 2 // 4: L * 3 // 4] = 2
    classes[L * 3 // 4:] = 0
    feature = feature.detach().cpu().numpy()
    feature_extract = manifold.TSNE(n_components=k, init='pca')
    feature_point = feature_extract.fit_transform(feature)
    plt.scatter(feature_point[:, 0], feature_point[:, 1], marker='o', c=classes)  # , cmap='coolwarm')

    if vis:
        plt.legend()
        plt.show()


def cam(feature_map, weight):
    """
    :param feature_map: [C, H, W]
    :param weight: [C]
    :return:
    """
    feature_map = (feature_map * weight).mean(dim=0)
    feature_map = feature_map.detach().cpu().numpy()
    feature_map = cv2.resize(feature_map, (256, 256))
    plt.imshow(feature_map)
    plt.show()


def draw_loss(path):
    df = pd.read_csv(path)
    df = df.fillna(0)
    df = np.array(df)
    df = df.sum(axis=1)
    plt.plot(np.arange(0, df.shape[0]), df)
    # plt.imsave("/output/df.jpg",df)
    plt.show()
    pass


# draw_loss(path='/home/servicing/projects/jiarui/CBCT_Jiarui/output/lossnet_plct_litterVGG_v5/iter_loss.csv')
# draw_loss(path='/home/servicing/projects/jiarui/CBCT_Jiarui/output/lossnet_plct_litterVGG_v5/log.csv')
