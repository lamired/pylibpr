#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2012/08/02 02:33:02. 

import random

class DataAccess(object):
    def __init__(self, filename):
        self.filename = filename

    def load_data(self, spliter=" ", seek_offset=0, only_feature=False):
        """Load data, feature and category file.
        
        You can get datas from data and category file in default setting.
        Also you can get it from feature file if you set only_feature=True.
        """
        datas = []
        with open(self.filename, "r") as f:
            f.seek(seek_offset)
            while True:
                line = f.readline().strip()
                if not line: break
                lines = line.split(spliter)
                if len(lines)>1: datas.append([float(tmp) for tmp in lines[:-1]])
                if not only_feature: datas[-1].append(lines[-1])
        return datas

    def format_datas(self, datas):
        """Arrange datas to a set of features and categories.
        """
        numDatas = len(datas)
        features = [datas[tmp][:-1] for tmp in range(numDatas)]
        categories = [datas[tmp][-1] for tmp in range(numDatas)]
        return features, categories

    def format_categories(self, categories):
        """Make categories from unique character style to handy number style.
        And count the number of categories.
        """
        categoryList = []
        new_categories = [0 for tmp in range(len(categories))]
        for i,j in enumerate(categories):
            if not j in categoryList: categoryList.append(j)
            new_categories[i] = categoryList.index(j)
        numCategories = len(categoryList)
        return new_categories, numCategories, categoryList

    def get_info(self, datas):
        features, categories = self.format_datas(datas)
        numDatas = len(features)
        numDimensions = len(features[0])
        categories, numCategories, categoryList = self.format_categories(categories)
        numDatas_byCategory = [0 for tmp in range(numCategories)]
        for i in categories: numDatas_byCategory[i] += 1
        return numDatas, numDimensions, numCategories, categoryList, numDatas_byCategory

    def format_info(self, datas):
        """Give information of the dataset in string type.
        You can see the information just with print it.
        """
        numDatas, numDimensions, numCategories, categoryList, numDatas_byCategory = self.get_info(datas)
        info = [numDatas, numDimensions, numCategories]
        annotation = ["datas", "dimensions", "categories"]
        line = "# Dataset: %s\n" % (self.filename)
        for i,j in enumerate(annotation):
            line += "# Num of %-10s: %d\n" % (j, info[i])
        for i,j in enumerate(categoryList):
            line += "# -- Category %d (%s): %d\n" % (i, j, numDatas_byCategory[i])
        return line

    def data_order_randomize(self, datas):
        """Randomize the order of datas.
        """
        numDatas = len(datas)
        ids = range(numDatas)
        for i in ids:
            tmp_id = random.randint(0, numDatas-1)
            ids[tmp_id], ids[i] = i, tmp_id
        return [datas[tmp] for tmp in ids]

    def random_sampling(self, datas, numDatas):
        """Random sampling datas.
        (Caution: this method doesn't care how many samples is picked up from each category.)
        """
        ids = [-1 for tmp in range(numDatas)]
        while True:
            if len(ids) >= numDatas: break
            tmp_id = random.randint(0, len(datas))
            if not tmp_id in ids: ids.append(tmp_id)
        new_datas = [datas[tmp] for tmp in ids]
        return new_datas

def get_category_features(features, categories, num_categories):
    category_features = [[] for tmp in range(num_categories)]
    for i in range(len(features)):
        category_features[categories[i]].append(features[i])
    return category_features

