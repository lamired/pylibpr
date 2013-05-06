#!/usr/bin/python
# coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2013/05/06 00:25:24.

import dataaccess
import math, numpy

class Prototypes(object):
    """Prototype model
    """
    def __init__(self, num_prototypes=3):
        self.num_prototypes = num_prototypes

    def run(self, features, categories, num_categories):
        pass

    # Utility
    def calc_distance_L2(self, vec1, vec2):
        return reduce((lambda a,b:a+b), [tmp**2 for tmp in (vec1-vec2)])
    
    def calc_distance(self, vec1, vec2):
        return math.sqrt(self.calc_distance_L2(vec1, vec2))
    
    def get_nearest_id(self, target, candidates):
        distances = [self.calc_distance(target, tmp) for tmp in candidates]
        temp = [(tmp, distance[tmp]) for tmp in range(len(distance))]
        nearest_id  = min(temp, key=(lambda a:a[1]))
        return nearest_id
 
    def get_tf_prototype_id(self, data, category):
        nearest_ptototype_ids = [self.get_nearest_id(data, tmp) for tmp in self.prototypes]
        temp = [self.prototypes[tmp][nearest_ptototype_ids[tmp]] for tmp in range(len(self.prototypes)) if tmp!=category]
        nearest_category_id = self.get_nearest_id(data, temp)
        if nearest_category_id >= category: 
            nearest_category_id += 1
        return nearest_ptototype_ids[category], nearest_category_id, nearest_ptototype_ids[nearest_category_id]

    def classify(self, data, category):
        nearest_ptototype_ids = [self.get_nearest_id(data, tmp) for tmp in self.prototypes]
        temp = [self.prototypes[tmp][nearest_ptototype_ids[tmp]] for tmp in range(len(self.prototypes))]
        return self.get_nearest_id(data, temp)

    def get_recognition_accuracy(self, datas, categories):
        num_samples = datas.shape[0]
        result = [self.recognition(datas[tmp], categories[tmp]) for tmp in range(num_samples)]
        return (reduce((lambda a,b:a+b), [1.0 if result[tmp]==categories[tmp] else 0.0 for tmp in range(num_samples)]) / num_samples) * 100.0

    # Accessor
    def init_prototypes(self, num_categories, num_prototypes, num_dimensions):
        self.num_prototypes = num_prototypes
        self.prototypes = numpy.zeros((num_categories, num_prototypes, num_dimensions))

    def set_prototypes(self, prototypes):
        self.num_prototypes = prototypes.shape[1]
        self.prototypes = prototypes

    def get_prototypes(self):
        return self.prototypes

