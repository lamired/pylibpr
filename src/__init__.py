#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2013/04/28 00:05:36. 

import sys
import numpy
import dataaccess, pca, classifier

if __name__ == "__main__":
    argv = sys.argv
    filename = "../data/glass.data"
    accessor = dataaccess.DataAccess(filename)
    datas = accessor.load_data()
    features, categories = accessor.format_datas(datas)
    categories, numCategories, categoryList = accessor.format_categories(categories)
    print accessor.format_info(datas)

#    method = pca.PCA(features, categories, numCategories)
#    eigen_value, eigen_vector = method.calc_pca()
#    features = method.get_prod(features, eigen_vector)
#    classifier = classifier.Classifier(features[:int(argv[1])], categories[:int(argv[1])], numCategories) 
#    classifier.compute_nn()
#    classifier.compute_nn(target_features=features[int(argv[2]):], target_categories=categories[int(argv[2]):])

#    category_features = dataaccess.get_category_features(features, categories, numCategories)
#    method.plot_samples(category_features, 0, 1)
