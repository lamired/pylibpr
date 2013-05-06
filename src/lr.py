#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2013/05/06 10:29:51. 

import numpy 
import mva

class LR(mva.MVA):
    """Logistic regression model
    """
    def __init__(self, features, categories, num_categories):
        super(LR, self).__init__(features, categories, num_categories)
        self.num_loop_count = 100

    def run(self):
        if num_categories == 2:
            weight = get_weight_2categories()
        else if num_categories > 2:
            weight = get_weight_multi_categories()
        else:
            return False
        return weight

    def get_weight_2categories(self):
        """Get weight via Logistic regression (2 class ver.)
        """
        loop_counter = 0
        converge_flag = False
        converge_cond = 0.001
        pre_weight = numpy.array([0.1 for tmp in range(self.num_dimensions)])

        while (loop_counter < self.num_loop_count) and (not converge_flag):
            output = [self.get_sigmoid(numpy.dot(pre_weight, self.features[tmp,])) for tmp in range(self.num_samples)]
            output_derivative = numpy.zeros((self.num_samples, self.num_samples))
            for tmp in range(self.num_samples): 
                output_derivative[tmp,tmp] = output[tmp] * (1.0 - output[tmp])
            hesse = numpy.dot(self.features.T, numpy.dot(output_derivative, self.features))
            hesse_inverse = numpy.linalg.pinv(hesse)
            weight = pre_weight - numpy.dot(hesse_inverse, numpy.dot(self.features.T, (output - self.categories)))
            deviation = numpy.linalg.norm(weight - pre_weight) / numpy.linalg.norm(pre_weight)
            temp = [self.categories[tmp] * output[tmp] - self.get_sigmoid(-output[tmp]) for tmp in range(self.num_samples)]
            log_posterior = reduce((lambda a,b:a+b), temp)
            if deviation <= converge_cond:
                converge_flag = True
            pre_weight = weight
            loop_counter += 1
        return weight
    
    def get_weight_multi_categories(self):
        """Get weight via Logistic regression (multi-class ver.)
        """
        loop_counter = 0
        converge_flag = False
        converge_cond = 0.001
        pre_weight = numpy.zeros((self.num_categories, self.num_dimensions))
        switch_categories = [[1 if self.categories[tmp]==tmp2 else 0 for tmp2 in range(self.num_categories)] for tmp in range(self.num_samples)]
        switch_categories = numpy.array(switch_categories)

        while (loop_counter < self.num_loop_count) and (not converge_flag):
            output = []
            for i in range(self.num_samples):
                temp = [numpy.exp(numpy.dot(pre_weight[tmp], self.features[i])) for tmp in range(self.num_categories)]
                numerator = reduce((lambda a,b:a+b), temp)
                output.append(temp / numerator)
            output = numpy.array(output)
            output_derivative = [reduce((lambda a,b:a+b), [(output[tmp2,tmp]-switch_categories[tmp2,tmp])*self.features[tmp2] for tmp2 in range(self.num_samples)]) for tmp in range(self.num_categories)]
            output_derivative = numpy.array(output_derivative)
            temp = [numpy.dot(numpy.matrix(tmp).T, numpy.matrix(tmp)) for tmp in self.features]
            weight = numpy.zeros((self.num_categories, self.num_dimensions))
            for i in range(self.num_categories):
                hesse = reduce((lambda a,b:a+b), [switch_categories[tmp,i]*(1.0 - switch_categories[tmp,i])*temp[tmp] for tmp in range(self.num_samples)])
                hesse_inverse = numpy.linalg.pinv(hesse)
                weight[i] = pre_weight[i] - numpy.dot(hesse_inverse, output_derivative[i])
            deviation = numpy.linalg.norm(weight - pre_weight) / numpy.linalg.norm(pre_weight)
            if deviation <= converge_cond:
                converge_flag = True
            pre_weight = weight
            loop_counter += 1
        return weight

    def set_num_loop_count(num_loop_count):
        self.num_loop_count = num_loop_count

