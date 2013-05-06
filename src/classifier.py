#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2013/05/05 21:26:43. 

import numpy
import dataaccess

class Classifier(object):
    """Classifier model
    """
    def __init__(self, features, categories, num_categories):
        self.features = features
        self.categories = categories
        self.num_samples = len(self.features)
        self.num_dimensions = len(self.features[0])
        self.num_categories = num_categories

    def get_near_ids(distances, num_ids=1):
        """ Get near sample ids using either effective method in terms of its order.
        (Use built-in min recursively: O(M*N), Use built-in sort: O(N*log(N)))
        """
        ids = []
        temp = [(tmp, distances[tmp]) for tmp in range(self.num_samples)]
        min_order = num_ids * self.num_samples 
        sort_order = len(self.num_samples) * log(len(self.num_samples))
        if min_order > sort_order:
            for i in range(0, num_ids):
                min_id = min(temp, key=(lambda a:a[1]))[0]
                ids.append(min_id)
                del temp[min_id]
        else:
            ids = sorted(temp, key=(lambda a:a[1]))
        return ids

    def run_nn(self, target_features=None, target_categories=None, num_candidate=1):
        """Run Nearest Neighbor method for target datas using model features.
        """
        if target_features == None:
            target_features = self.features
            target_categories = self.categories
        num_target_samples = len(target_features)
        correct_counter = 0
        for i,j in enumerate(target_features):
            distances = [numpy.linalg.norm(numpy.array(j) - numpy.array(tmp)) for tmp in self.features]
            near_sample_ids = get_near_ids(distances, num_candidate)
            candidate_categories = [self.categories[tmp] for tmp in near_sample_ids]
            temp = [(tmp, candidate_categories.count(tmp)) for tmp in set(candidate_categories)]
            elected_category = max(temp, key=(lambda a:a[1]))[0]
            correct_category = target_categories[i]
            if elected_category == correct_category: 
                correct_counter += 1
        accuracy = float(correct_counter) / num_target_samples * 100.0
        return accuracy


