#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2013/05/05 23:41:04. 

import random

class DataAccess(object):
    """Dataset accessor
    """
    def __init__(self, filename):
        self.filename = filename

    def load_data(self, spliter=" ", seek_offset=0, only_feature=False):
        """Load data, feature and category file.
        [only_feature] Whether file contains category info or not.
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
        num_samples = len(datas)
        features = [datas[tmp][:-1] for tmp in range(num_samples)]
        categories = [datas[tmp][-1] for tmp in range(num_samples)]
        return features, categories

    def format_categories(self, categories):
        """Change categories from unique character style to handy number style.
        And count the number of categories.
        """
        category_list = []
        new_categories = [0 for tmp in range(len(categories))]
        for i,j in enumerate(categories):
            if not j in category_list: category_list.append(j)
            new_categories[i] = category_list.index(j)
        num_categories = len(category_list)
        return new_categories, num_categories, category_list

    def get_info(self, datas):
        """Get following dataset info:
        number of datas, number of dimensions, number of categories,
        unique categories info, number of datas in each category 
        """
        features, categories = self.format_datas(datas)
        num_samples = len(features)
        num_dimensions = len(features[0])
        categories, num_categories, category_list = self.format_categories(categories)
        num_category_samples = [0 for tmp in range(num_categories)]
        for i in categories: num_category_samples[i] += 1
        return num_samples, num_dimensions, num_categories, category_list, num_category_samples

    def format_info(self, datas):
        """Get dataset info in readable style.
        """
        num_samples, num_dimensions, num_categories, category_list, num_category_samples = self.get_info(datas)
        info = [num_samples, num_dimensions, num_categories]
        annotation = ["datas", "dimensions", "categories"]
        line = "# Dataset: %s\n" % (self.filename)
        for i,j in enumerate(annotation):
            line += "# Num of %-10s: %d\n" % (j, info[i])
        for i,j in enumerate(category_list):
            line += "# -- Category %d (%s): %d\n" % (i, j, num_category_samples[i])
        return line

# Utility
def data_order_randomize(datas):
    """Randomize the order of datas.
    """
    num_samples = len(datas)
    ids = range(num_samples)
    for i in ids:
        tmp_id = random.randint(0, num_samples-1)
        ids[tmp_id], ids[i] = i, tmp_id
    return [datas[tmp] for tmp in ids]

def random_sampling(datas, num_samples):
    """Random sampling datas.
    (Caution: this doesn't care how many samples is picked up from each category.)
    """
    ids = [-1 for tmp in range(num_samples)]
    while True:
        if len(ids) >= num_samples: break
        tmp_id = random.randint(0, len(datas))
        if not tmp_id in ids: ids.append(tmp_id)
    new_datas = [datas[tmp] for tmp in ids]
    return new_datas

def get_category_features(features, categories, num_categories):
    """Get features arranged in each category
    """
    category_features = [[] for tmp in range(num_categories)]
    for i in range(len(features)):
        category_features[categories[i]].append(features[i])
    return category_features

