#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2013/05/05 19:07:12. 

import mva
import numpy

class PCA(mva.MVA):
    """Principal Component Analysis model
    """
    def __init__(self, features, categories, num_categories):
        super(PCA, self).__init__(features, categories, num_categories)

    def view_accum_contribution(self, eigen_value):
        normalize = sum(eigen_value)
        rate = [(tmp/normalize*100.0) for tmp in eigen_value]
        print "# Accumulative contribution rate:"
        for i,j in enumerate(rate):
            print "# -- %2d: %3.2f" % (i, j)
        return True

    def run_pca(self):
        features = self.features - numpy.mean(self.features, axis=0)
        cov = numpy.cov(features.T)
        eigen_value, eigen_vector = self.calc_eigen(cov, ignore_small_eig=False)
        self.view_accum_contribution(eigen_value)
        return eigen_value, eigen_vector

    def run_kpca(self, kernel_func):
        mapped_features = numpy.matrix([[kernel_func(tmp,tmp2) for tmp2 in self.features] for tmp in self.features])
        normalize = numpy.matrix(numpy.identity(num_samples)) - (1.0/num_samples)
        prod = numpy.dot(normalize, mapped_features)
        eigen_value, eigen_vector = self.calc_eigen(prod, ignore_small_eig=False)
        self.view_accum_contribution(eigen_value)
        return eigen_value, eigen_vector

