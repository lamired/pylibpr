#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2012/08/03 00:22:40. 

if __name__ == "__main__":

import dataaccess

class Kmeans(object):
    def __init__(self, features, categories, num_categories):
        self.features = numpy.array(features)
        self.categories = categories
        self.num_categories = self.num_categories
        self.num_samples = self.features.shape[0]
        self.num_dimensions = self.features.shape[1]

    def init_prototypes(self, num_prototypes):
        self.num_prototypes = num_prototypes
        self.prototypes = numpy.zeros((self.num_categories, self.num_prototypes, self.num_dimensions))

    def get_nearest_id(self, target, candidates):
        distance = [numpy.lialg.norm(target-tmp) for tmp in candidates]
        temp = [(tmp, distance[tmp]) for tmp in range(len(distance))]
        nearest_id = min(temp, key=(lambda a:a[1]))[0]
        return nearest_id

    def learn(self, num_prototypes, max_loop=10):
        category_features = dataaccess.get_category_features(self.features, self.categories, self.num_categories)
        self.init_prototypes(num_prototypes)
        for i in range(self.num_categories):
            num_include_samples = len(category_features[i]) / self.num_prototypes
            include_samples = [category_features[i][(num_include_samples*tmp):(num_include_samples*(tmp+1))] for tmp in range(self.num_prototypes)]
            self.prototypes[i] = [numpy.mean(include_samples[tmp], axis=0) for tmp in range(self.num_prototypes)]
        for i in range(self.num_categories):
            loop_counter = 0
            converge_flag = False
            converge_cond = 0.0
            while (loop_counter < max_loop) and (not converge_flag):
                include_sample_ids = [self.get_nearest_id(tmp, self.prototypes[i]) for tmp in category_features[i]]
                num_include_samples = 
                include_samples = [[None for tmp2 in range()] ]


        


  def learn(self, datas, categories, numCategories, maxLoop=100):
    datas_byCategory = util.getDatas_byCategory(datas, categories, numCategories)
    self.init_prototypes(numCategories, self.numPrototypes, datas.shape[1])
    for i in range(numCategories):
      tmp_numDatas = len(datas_byCategory[i]) / self.numPrototypes
      temp = [datas_byCategory[i][tmp_numDatas*tmp:tmp_numDatas*(tmp+1)] for tmp in range(self.numPrototypes)]
      self.prototypes[i] = [numpy.mean(temp[tmp], axis=0) for tmp in range(self.numPrototypes)]
    for i in range(numCategories):
      loopCount = 0
      convFlag = False
      convFactor = 0.0
      while(loopCount < maxLoop) and not convFlag:
        clusterID_table = [self.findNearestID(tmp, self.prototypes[i]) for tmp in datas_byCategory[i]]
        tmp_datas = [[] for tmp in range(self.numPrototypes)]
        for j,k in enumerate(clusterID_table): tmp_datas[k].append(datas_byCategory[i][j])
        for j in range(self.numPrototypes):
          if len(tmp_datas[j]) == 0: return False
        self.prototypes[i] = [numpy.mean(tmp_datas[tmp], axis=0) for tmp in range(self.numPrototypes)]
        tmp_convFactor = reduce((lambda a,b:a+b), [self.calcDistance(datas_byCategory[i][tmp], self.prototypes[i][clusterID_table[tmp]]) for tmp in range(len(datas_byCategory[i]))])
        if numpy.abs(convFactor - tmp_convFactor) <= 0.001: convFlag = True
        convFactor = tmp_convFactor
        loopCount += 1
    return True


