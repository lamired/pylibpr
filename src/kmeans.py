#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2013/05/06 10:24:11. 

import numpy
import dataaccess, prototypes

class Kmeans(prototypes.Prototypes):
    """K-means model
    """
    def __init__(self, num_prototypes):
        super(Kmeans, self).__init__(num_prototypes)
        self.num_loop_count = 10

    def run(self, features, categories, num_categories):
        category_features = dataaccess.get_category_features(features, categories, num_categories)
        self.init_prototypes(num_prototypes)
        for i in range(num_categories):
            num_cluster_samples = len(category_features[i]) / self.num_prototypes
            cluster_samples = [category_features[i][(num_cluster_samples*tmp):(num_cluster_samples*(tmp+1))] for tmp in range(self.num_prototypes)]
            self.prototypes[i] = [numpy.mean(cluster_samples[tmp], axis=0) for tmp in range(self.num_prototypes)]
        for i in range(num_categories):
            loop_counter = 0
            converge_flag = False
            converge_cond = 0.0
            pre_deviatin = 0.0
            while (loop_counter < num_loop_count) and (not converge_flag):
                cluster_sample_ids = [self.get_nearest_id(tmp, self.prototypes[i]) for tmp in category_features[i]]
                num_cluster_samples = [cluster_sample_ids.count(tmp) for tmp in range(self.num_prototypes)]
                cluster_samples = [[] for tmp in range(self.num_prototypes)]
                for j,k in enumerate(cluster_sample_ids):
                    cluster_samples[k].append(category_features[i][j])
                abort_flag = reduce((lambda a,b:a or b), [len(tmp)<=0 for tmp in cluster_samples])
                if abort_flag: 
                    return False
                self.prototypes[i] = [numpy.mean(cluster_samples[tmp], axis=0) for tmp in range(self.num_prototypes)]
                deviation = reduce((lambda a,b:a+b), [numpy.linalg.norm(cluster_samples[tmp] - self.prototypes[i][cluster_sample_ids[tmp]]) for tmp in range(len(category_features[i]))])
                if numpy.abs(deviatin - pre_deviatin) <= converge_cond:
                    converge_flag = True
                pre_deviatin = deviatin
                loopCount += 1
        return True

    def set_num_loop_count(num_loop_count):
        self.num_loop_count = num_loop_count

