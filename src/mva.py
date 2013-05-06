#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2013/05/05 18:58:07. 

import numpy

class MVA(object):
    """MultiVariate Analysis super model
    """
    def __init__(self, features, categories, num_categories):
        self.features = numpy.array(features)
        self.categories = categories
        self.num_samples = self.features.shape[0]
        self.num_dimensions = self.features.shape[1]
        self.num_categories = num_categories

    def get_sigmoid(self, x):
        return 1.0 / (1.0 + numpy.exp(-x))

    def get_softmax(self, x, numerator):
        return numpy.exp(x) / numerator

    def get_RBFkernel(self, x, y, coef):
        return numpy.exp(-numpy.linalg.norm(x-y)**2 / coef**2)

    def get_POLkernel(self, x, y, coef):
        return (numpy.prod(x,y) + coef) ** 2

    def calc_kernel(self, feature_id, eig_vecs, func=get_RBFkernel, coef=1.0):
        res = [0.0 for tmp in range(self.num_dimensions)]
        for i in range(self.num_dimensions):
            temp = [eig_vecs[i,tmp] * func(self.features[tmp],self.features[feature_id]) for tmp in range(self.num_samples)]
            value = reduce((lambda a,b:a+b), temp)
            res[i] = value
        return res

    def get_prod(self, features, eig_vecs):
        new_features = numpy.dot(features, eig_vecs)
        return numpy.array(new_features)

    def get_kernel_prod(self, features, eig_vecs):
        """Get mapped features to the eigen space.
        """
        new_features = [None for tmp in range(numAllSamples)]
        for i in range(num_samples):
            new_features[i] = self.calc_kernel(i, eig_vecs, self.get_RBFkernel)
        return numpy.array(new_features)

    def calc_eigen(self, x, ignore_small_eig=True):
        print "# Calculate eigen decomposition problem for (%d x %d) matrix ..." % (x.shape[0], x.shape[1])
        eigen_value, eigen_vector = numpy.linalg.eig(x)
        eigen_value = numpy.array(eigen_value)
        eigen_vector = numpy.array(eigen_vector)
        if ignore_small_eig: 
            eigen_vector = eigen_vector.T
            temp = [(eigen_value[tmp], eigen_vector[tmp]) for tmp in range(eigen_vector.shape[0])]
            eigen_value = [tmp[0] for tmp in temp]
            eigen_vector = [tmp[1] for tmp in temp]
        return eigen_value, eigen_vector
    
    def plot_samples(self, category_features, dim1, dim2):
        colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
        forms = ["o", "^", "v", "s", "D", "x", "H", "p"]
        for i,j in enumerate(category_features):
            legend = colors[i%len(colors)] + forms[i/len(forms)]
            for k in j: pylab.plot(k[dim1], k[dim2], legend)
        pylab.show()
        return True

