#!/usr/bin/python
#coding: utf-8

# Author: Tsukasa Ohashi
# Last Change: 2013/05/06 10:24:49.

import math, numpy
import dataaccess, prototypes

class MCEProto(prototypes.Prototypes):
    """Minimum Classification Error model
    """
    def __init__(self, num_prototypes):
        super(MCEProto, self).__init__(num_prototypes)
        self.learning_type = "Batch"
        self.mis_measure_type = "GM"
        self.exp_mis_measure = -1
        self.num_loop_count = 100
        self.init_learning_rate = 0.01
        self.loss_smoothness = 2.0

    def run(self, features, categories, num_categories):
        loop_counter = 0
        converge_flag = False
        converge_cond = 0.0
        pre_empirical_loss = 0.0
        learning_rate = init_learning_rate
        learning_rate_diff = init_learning_rate / (features.shape[0] * num_loop_count)

        while (loop_counter < num_loop_count) and (not converge_flag):
            if self.learning_type == "Batch":
                self.train_batch(features, categories, loss_smoothness, learning_rate)
            elif self.learning_type == "Seq":
                self.train_sequence(features, categories, loss_smoothness, learning_rate, learning_rate_diff)
            learning_rate -= init_learning_rate / num_loop_count
            empirical_loss = self.calc_empirical_loss(features, categories, loss_smoothness)
            if numpy.abs(pre_empirical_loss - empirical_loss) <= converge_cond: 
                converge_flag = True
            pre_empirical_loss = tmp_convFactor
            loop_counter += 1
        return True

    # Misclassification Measure Calculator
    def calc_fm_measure(self, data, category):
        if self.exp_mis_measure > 0:
            nearest_ptototype_ids = [self.get_nearest_id(data, tmp) for tmp in self.prototypes]
            correct_proto_id = nearest_ptototype_ids[category]
            temp = [numpy.exp(-self.exp_mis_measure * self.calc_distance_L2(data, self.prototypes[tmp][nearest_ptototype_ids[tmp]])) for tmp in range(len(self.prototypes)) if tmp!=category]
            best_incorrect_discriminant = numpy.log(reduce((lambda a,b:a+b), temp) / (len(self.prototypes)-1)) / self.exp_mis_measure
        else:
            correct_proto_id, best_incorrect_category, best_incorrect_proto_id = self.get_tf_prototype_id(data, category)
            best_incorrect_discriminant = -self.calc_distance_L2(data, self.prototypes[best_incorrect_category][best_incorrect_proto_id])
        mis_measure = self.calc_distance_L2(data, self.prototypes[category][correct_proto_id]) + best_incorrect_discriminant
        return mis_measure, correct_proto_id, best_incorrect_category, best_incorrect_proto_id

    def calc_fm_measure_diff(self, data, category, loss_smoothness):
        mis_measure, correct_proto_id, best_incorrect_category, best_incorrect_proto_id = self.calc_fm_measure(data, category)
        loss = self.calc_sigmoid_loss(mis_measure, loss_smoothness)
        coef = 2.0 * loss_smoothness * loss * (1.0-loss)
        correct_diff = coef * (data - self.prototypes[category][correct_proto_id])
        best_incorrect_diff = coef * (-data + self.prototypes[best_incorrect_category][best_incorrect_proto_id])
        return correct_diff, best_incorrect_diff, correct_proto_id, best_incorrect_category, best_incorrect_proto_id

    def calc_gm_measure(self, data, category):
        correct_proto_id, best_incorrect_category, best_incorrect_proto_id = self.get_tf_prototype_id(data, category)
        fm_measure = self.calc_distance_L2(data, self.prototypes[category][correct_proto_id]) - self.calc_distance_L2(data, self.prototypes[best_incorrect_category][best_incorrect_proto_id])
        denominator = 2.0 * self.calc_distance(self.prototypes[category][correct_proto_id], self.prototypes[best_incorrect_category][best_incorrect_proto_id])
        return fm_measure, denominator, correct_proto_id, best_incorrect_category, best_incorrect_proto_id

    def calc_gm_measure_diff(self, data, category, loss_smoothness):
        fm_measure, denominator, correct_proto_id, best_incorrect_category, best_incorrect_proto_id = self.calc_gm_measure(data, category)
        mis_measure = fm_measure / denominator
        derivative_coef = fm_measure / (denominator**3)
        derivative_vec = 2.0 * (-self.prototypes[category][correct_proto_id] + self.prototypes[best_incorrect_category][best_incorrect_proto_id])
        loss = self.calc_sigmoid_loss(mis_measure, loss_smoothness)
        coef = loss_smoothness * loss * (1.0 - loss) * 2.0
        correct_diff = -coef * (((self.prototypes[category][correct_proto_id]-data) / denominator) + (derivative_vec*derivative_coef))
        best_incorrect_diff = -coef * (((-self.prototypes[best_incorrect_category][best_incorrect_proto_id]+data) / denominator) - (derivative_vec*derivative_coef))
        return correct_diff, best_incorrect_diff, correct_proto_id, best_incorrect_category, best_incorrect_proto_id

    def calc_sigmoid_loss(self, mis_measure, loss_smoothness):
        return 1.0 / (1.0 + numpy.exp(-loss_smoothness * mis_measure))

    def calc_empirical_loss(self, features, categories, loss_smoothness):
        num_samples = features.shape[0]
        if self.mis_measure_type == "FM":
            temp = [self.calc_fm_measure(features[tmp], categories[tmp]) for tmp in range(num_samples)]
            mis_measures = [temp[tmp][0] for tmp in range(num_samples)]
        elif self.mis_measure_type == "GM":
            temp = [self.calc_gm_measure(features[tmp], categories[tmp]) for tmp in range(num_samples)]
            mis_measures = [(temp[tmp][0]/temp[tmp][1]) for tmp in range(num_samples)]
        return reduce((lambda a,b:a+b), [self.calc_sigmoid_loss(tmp, loss_smoothness) for tmp in mis_measures]) / num_samples

    # Trainer
    def train_batch(self, features, categories, loss_smoothness, learning_rate):
        num_samples = features.shape[0]
        differences = numpy.zeros((len(self.prototypes), self.num_prototypes, features.shape[1]))
        for i in range(num_samples):
            if self.mis_measure_type == "FM":
                correct_diff, best_incorrect_diff, correct_proto_id, best_incorrect_category, best_incorrect_proto_id = self.calc_fm_measure_diff(features[i], categories[i], loss_smoothness)
            elif self.mis_measure_type == "GM":
                correct_diff, best_incorrect_diff, correct_proto_id, best_incorrect_category, best_incorrect_proto_id = self.calc_gm_measure_diff(features[i], categories[i], loss_smoothness)
            differences[categories[i]][correct_proto_id] += correct_diff
            differences[best_incorrect_category][best_incorrect_proto_id] += best_incorrect_diff
            self.prototypes += learning_rate * differences / num_samples
        return True

    def train_sequence(self, features, categories, loss_smoothness, learning_rate, learning_rate_diff):
        num_samples = features.shape[0]
        features = numpy.array(dataaccess.randomize(features))
        for i in range(num_samples):
            if self.mis_measure_type == "FM":
                correct_diff, best_incorrect_diff, correct_proto_id, best_incorrect_category, best_incorrect_proto_id = self.calc_fm_measure_diff(features[i], categories[i], loss_smoothness)
            elif self.mis_measure_type == "GM":
                correct_diff, best_incorrect_diff, correct_proto_id, best_incorrect_category, best_incorrect_proto_id = self.calc_gm_measure_diff(features[i], categories[i], loss_smoothness)
            self.prototypes[categories[i]][correct_proto_id] += learning_rate * correct_diff
            self.prototypes[best_incorrect_category][best_incorrect_proto_id] += learning_rate * best_incorrect_diff
            learning_rate -= learning_rate_diff
        return True

    # Accessor
    def set_learning_type(self, learning_type):
        self.learning_type = learning_type

    def set_mis_measure_type(self, mis_measure_type):
        self.mis_measure_type = mis_measure_type

    def set_exp_mis_measure(self, exp_mis_measure):
        self.exp_mis_measure = exp_mis_measure

    def set_num_loop_count(self, num_loop_count):
        self.num_loop_count = num_loop_count

    def set_init_learning_rate(self, init_learning_rate):
        self.init_learning_rate = init_learning_rate

    def set_loss_smoothness(self, loss_smoothness):
        self.loss_smoothness = loss_smoothness

