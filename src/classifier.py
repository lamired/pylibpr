#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2013/02/22 18:37:02. 

import dataaccess
import numpy

class Classifier(object):
    def __init__(self, features, categories, num_categories):
        self.features = features
        self.categories = categories
        self.num_samples = len(self.features)
        self.num_dimensions = len(self.features[0])
        self.num_categories = num_categories

    def compute_nn(self, target_features=None, target_categories=None, num_candidate=1):
        if target_features == None:
            target_features = self.features
            target_categories = self.categories
        num_target_samples = len(target_features)
        correct_counter = [0 for tmp in range(self.num_categories)]
        num_category_samples = [0 for tmp in range(self.num_categories)]
        print "# Compute %d-Nearest Neighbor classifier for %d samples ..." % (num_candidate, num_target_samples)
        for i,j in enumerate(target_features):
            distances = [numpy.linalg.norm(numpy.array(j) - numpy.array(tmp)) for tmp in self.features]
            temp = [(tmp, distances[tmp]) for tmp in range(self.num_samples)]
            if target_features == self.features: del temp[i]
            min_id = min(temp, key=(lambda a:a[1]))[0]
            correct_category = target_categories[i]
            if self.categories[min_id] == correct_category: correct_counter[correct_category] += 1
            num_category_samples[correct_category] += 1
        num_correct_samples = reduce((lambda a,b:a+b), correct_counter)
        accuracy = float(num_correct_samples) / num_target_samples * 100.0

        print "# Recognition accuracy: %3.2f (%4d/%4d)" % (accuracy, num_correct_samples, num_target_samples)
        for i,j in enumerate(correct_counter):
            accuracy = float(j) / num_category_samples[i] * 100.0
            print "# -- Category %2d: %3.2f (%4d/%4d)" % (i, accuracy, j, num_category_samples[i])
        return True

