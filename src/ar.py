#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2013/05/06 10:50:21. 

import numpy
import mva

class AR(mva.MVA):
    """Auto-regressive model
    """
    def __init__(self, features, categories, num_categories):
        super(AR, self).__init__(features, categories, num_categories)
        self.max_model_order = int(2.0 * numpy.sqrt(self.num_samples))
  
    def calc_coef(self, i, j):
        data1 = self.features[(self.max_model_order - i):(self.num_samples - i),]
        data2 = self.features[(self.max_model_order - j):(self.num_samples - j),]
        temp = min(data1.shape[0], data2.shape[0])
        return reduce((lambda a,b:a+b), [numpy.dot(data1[tmp,], data2[tmp,]) for tmp in range(temp)])
    
    def get_noise(self, n, m):
        if m == 0:
            noise_coef = []
            noise_var = self.calc_coef(0, 0) / n
        else:
            right_matrix = numpy.matrix([[self.calc_coef(tmp, 0)] for tmp in range(1, m+1)])
            left_matrix = numpy.matrix([[self.calc_coef(tmp, tmp2) for tmp2 in range(1, m+1)] for tmp in range(1, m+1)])
            noise_coef = numpy.dot(numpy.linalg.pinv(left_matrix), right_matrix)
            temp = [noise_coef[tmp,] * self.calc_coef(tmp+1, 0) for tmp in range(m)]
            noise_var = (self.calc_coef(0, 0) - reduce((lambda a,b:a+b), temp)) / n
        return noise_coef, noise_var
    
    def run(self):
        """Get MAICE model order (minimum AR model order on Akaike Information Criterion)
        """
        aics = []
        n = self.num_samples - self.max_model_order
        self.features -= numpy.mean(self.features, axis=0)
        for m in range(self.max_model_order + 1):
            noise_coef, noise_var = self.get_noise(n, m)
            aic = n * (numpy.log(2.0 * numpy.pi) + 1.0) + n * numpy.log(noise_var) + 2.0 * (m + 1.0)
            aics.append(aic)
        temp = [(tmp, aics[tmp]) for tmp in range(len(aics))]
        temp.sort()
        return temp[0][0]

    def set_max_model_order(max_model_order):
        self.max_model_order = max_model_order

