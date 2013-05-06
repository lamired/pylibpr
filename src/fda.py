#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2013/05/05 23:51:34. 

import numpy
import mva, dataaccess

class FDA(mva.MVA):  
    """Fisher Discriminant Analysis model 
    """
    def __init__(self, features, categories, num_categories):
        super(FDA, self).__init__(features, categories, num_categories)

    def calc_cov(self, features):
        """ Calculate following matrices:
        Between covariance (self.num_dimensions x self.num_dimensions),
        Within  covariance (self.num_dimensions x self.num_dimensions)
        """
        category_features = dataaccess.get_category_features(features, self.categories, self.num_categories)
        category_means = [numpy.mean(tmp, axis=0) for tmp in category_features]
        all_sample_mean = numpy.mean(features, axis=0)

        diff_means = [numpy.matrix(all_sample_mean - tmp) for tmp in category_means]
        each_prod = [len(category_features[tmp]) * numpy.dot(diff_means[tmp].T, diff_means[tmp]) for tmp in range(self.num_categories)]
        between_cov = reduce((lambda a,b:a+b), each_prod) / self.num_samples

        diff_means = [[numpy.matrix(category_means[tmp] - tmp2) for tmp2 in category_features[tmp]] for tmp in range(self.num_categories)]
        each_prod = [[numpy.dot(tmp2.T, tmp2) for tmp2 in tmp] for tmp in diff_means]
        within_cov = reduce((lambda a,b:a+b), [reduce((lambda a,b:a+b), tmp) for tmp in each_prod]) / self.num_samples
        return between_cov, within_cov

    def run_fda(self):
        """Run FDA and get its eigen value / vector
        """
        between_cov, within_cov = self.calc_cov(self.features)
        eigen_value, eigen_vector = self.calc_eigen(numpy.dot(within_cov.I, between_cov))
        return eigen_value, eigen_vector

    def run_kfda(self, kernel_func, slack_constant=1.0):
        """Run Kernel-FDA and get its eigen value / vector
        """
        mapped_features = numpy.matrix([[kernel_func(tmp,tmp2) for tmp2 in self.features] for tmp in self.features])
        between_cov, within_cov = self.calc_cov(mapped_features)
        within_cov += slack_constant * self.features
        eigen_value, eigen_vector = self.calc_eigen(numpy.dot(within_cov.I, between_cov))
        return eigen_value, eigen_vector

    def run_dfda(self, num_eigs):
        """Run Detailed-FDA and get its eigen value / vector
        """
        between_cov, within_cov = self.calc_cov(self.features)
        category_features = dataaccess.get_category_features(self.features, self.categories, self.num_categories)
        eigen_vectors = []
        for i in category_features:
            acmat = reduce((lambda a,b:a+b), [numpy.dot(numpy.matrix(tmp).T, numpy.matrix(tmp)) for tmp in i]) / len(i)
            eigen_value, eigen_vector = self.calc_eigen(acmat)
            eigen_vectors.append(eigen_vector.T)
        clafic_cov = numpy.array(numpy.zeros((self.num_dimensions, self.num_dimensions)))
        tmp_eigs = [[(eigen_vectors[tmp2] - eigen_vectors[tmp]) for tmp2 in range(self.num_categories) if tmp2 != tmp] for tmp in range(self.num_categories)]

        # Get covariance for orthogonal eigen vectors in each category (CLAFIC covariance)
        for i in range(self.num_categories):
            tmp_eig = eigen_vectors[i]
            for j in range(self.num_categories):
                if i != j:
                    tmp_eig2 = eigen_vectors[j]
                    claffic_cov += reduce((lambda a,b:a+b), [reduce((lambda a,b:a+b), [numpy.dot((tmp-tmp2), (tmp-tmp2).T) for tmp2 in tmp_eig2]) for tmp in tmp_eig])
        between_cov += clafic_cov
        eigen_value, eigen_vector = self.calc_eigen(numpy.dot(within_cov.I, between_cov))
        return eigen_value, eigen_vector

