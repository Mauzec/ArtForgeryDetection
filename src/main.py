from BoVW.BoVW import BoVW
from CustomDescriptors.AkazeDescriptor.AKAZE import AKAZE
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
import numpy as np
from dataset.get_pictures import DatasetOperations
import os

if __name__ == "__main__":
    os.makedirs("12345")