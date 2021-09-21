import pickle
import time
import numpy
import theano
from theano import sandbox
import theano.tensor as tensor
import os
#import scipy.io
from collections import defaultdict
from theano.tensor.shared_randomstreams import RandomStreams

import struct


import scipy.io
import datetime
import argparse

dtype=theano.config.floatX

class NeuralHawkesCTLSTM(object):
    '''
    here is the sequence generator
    using Neural Hawkes process with continuous-time LSTM
    '''
    def __init__(self, args):
        #
        print "initializing generator ... "
        
       
        self.args = args
        
        print "read pretrained model ... "
        path_pre_train = os.path.abspath(
            args.FilePretrain
        )
        with open(path_pre_train, 'rb') as f:
            model_pre_train = pickle.load(f)
        self.dim_process = model_pre_train['dim_process']
        self.dim_model = model_pre_train['dim_model']
        self.dim_time = model_pre_train['dim_time']
        #
        self.scale = model_pre_train['scale']
        self.W_alpha = model_pre_train['W_alpha']
        self.Emb_event = model_pre_train['Emb_event']
        self.W_recur = model_pre_train['W_recur']
        self.b_recur = model_pre_train['b_recur']
            #
        #
        #self.intensity = numpy.copy(self.mu)
        self.name = 'NeuralHawkesGenCTLSTM'
        #
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        # initialization for LSTM states
        
        #self.hidden_t = numpy.zeros(
        #    (self.dim_model, ), dtype = dtype
        #)
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_target = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_decay = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.gate_output = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq) )
        print "initialization done "
        #
    #
    def set_args(self, dict_args):
        self.args = dict_args
    #
    def soft_relu(self, x):
        return numpy.log(numpy.float32(1.0)+numpy.exp(x))
    #
    def soft_relu_scale(self, x):
        # last dim of x is dim_process
        x /= self.scale
        y = numpy.log(numpy.float32(1.0)+numpy.exp(x))
        y *= self.scale
        return y
    #
    def hard_relu(self, x):
        return numpy.float32(0.5) * (x + numpy.abs(x) )
        #
    #

    def restart_sequence(self):
        # clear the events memory and reset starting time is 0
        self.intensity_tilde = None
        self.intensity = None
        #
        self.intensity_tilde_ub = None
        self.intensity_ub = None
        #
        self.one_seq = []
        # initialization for LSTM states
        
        #self.hidden_t = numpy.zeros(
        #    (self.dim_model, ), dtype = dtype
        #)
        self.cell_t = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_target = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.cell_decay = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        self.gate_output = numpy.zeros(
            (self.dim_model, ), dtype = dtype
        )
        #self.flag_continue = True
        self.cnt_total_event = numpy.int32(len(self.one_seq) )
        #
    #
    #
    def sigmoid(self, x):
        return 1 / (1+numpy.exp(-x))
    #
    #
    def compute_hidden_states(self):
        # every time it is called,
        # it computes the new hidden states of the LSTM
        # it gets the last event in the sequence
        # which is generated at t_(rec(t))
        # and compute its hidden states
        # Note : for this event, we get its type
        # and time elapsed since last event
        # that is to say, this func is different than
        # rnn_unit in models
        # THERE : event, time_since_this_event_to_next
        # so first update, and then decay
        # HERE : time_since_last_event, event
        # so first decay, and then update
        # Note : this should be called
        # after one event is generated and appended
        # so the state is updated accordingly
        #TODO: decay
        cell_t_after_decay = self.cell_target + (
            self.cell_t - self.cell_target
        ) * numpy.exp(
            -self.cell_decay * self.one_seq[-1][
                'time_since_last_event'
            ]
        )
        hidden_t_after_decay = self.gate_output * numpy.tanh(
            cell_t_after_decay
        )
        #TODO: update
        emb_event_t = self.Emb_event[
            self.one_seq[-1]['type_event'], :
        ]
        post_transform = numpy.dot(
            numpy.concatenate(
                (emb_event_t, hidden_t_after_decay),
                axis = 0
            ), self.W_recur
        ) + self.b_recur
        #
        gate_input = self.sigmoid(
            post_transform[:self.dim_model]
        )
        gate_forget = self.sigmoid(
            post_transform[self.dim_model:2*self.dim_model]
        )
        gate_output = self.sigmoid(
            post_transform[2*self.dim_model:3*self.dim_model]
        )
        gate_pre_c = numpy.tanh(
            post_transform[3*self.dim_model:4*self.dim_model]
        )
        # 2 -- input_bar and forget_bar gates
        gate_input_target = self.sigmoid(
            post_transform[4*self.dim_model:5*self.dim_model]
        )
        gate_forget_target = self.sigmoid(
            post_transform[5*self.dim_model:6*self.dim_model]
        )
        # cell memory decay
        cell_decay = self.soft_relu(
            post_transform[6*self.dim_model:]
        )
        #
        cell_t = gate_forget * cell_t_after_decay + gate_input * gate_pre_c
        cell_target = gate_forget_target * self.cell_target + gate_input_target * gate_pre_c
        #
        self.cell_t = numpy.copy(cell_t)
        self.cell_target = numpy.copy(cell_target)
        self.cell_decay = numpy.copy(cell_decay)
        self.gate_output = numpy.copy(gate_output)
        #
        #
    #
    #
    def compute_intensity_given_past(self, time_current):
        # compute the intensity of current time
        # given the past events
        time_recent = self.one_seq[-1]['time_since_start']
        #
        cell_t_after_decay = self.cell_target + (
            self.cell_t - self.cell_target
        ) * numpy.exp(
            -self.cell_decay * (
                time_current - time_recent
            )
        )
        hidden_t_after_decay = self.gate_output * numpy.tanh(
            cell_t_after_decay
        )
        #
        self.intensity_tilde = numpy.dot(
            hidden_t_after_decay, self.W_alpha
        )
        self.intensity = self.soft_relu_scale(
            self.intensity_tilde
        )
        # intensity computation is finished
        #
    #
    #
    def compute_intensity_upper_bound(self, time_current):
        # compute the upper bound of intensity
        # at the current time
        # Note : this is very tricky !!!
        # in decomposable process, finding upper bound is easy
        # see B.3 in NIPS paper
        # but in neural model
        # it is not a combo of POSITIVE decreasing funcs
        # So how to do this?
        # we find the functon is a sum of temrs
        # some terms are decreasing, we keep them
        # some terms are increasing, we get their upper-limit
        #
        # In detail, we compose it to 4 parts :
        # (dc = c-c_target)
        # w + dc -  increasing
        # w + dc +  decreasing
        # w - dc -  decreasing
        # w - dc +  increasing
        #
        time_recent = self.one_seq[-1]['time_since_start']
        #
        cell_gap = self.cell_t - self.cell_target
        cell_gap_matrix = numpy.outer(
            cell_gap, numpy.ones(
                (self.dim_process, ), dtype=dtype
            )
        )
        # dim * dim_process
        index_increasing_0 = (cell_gap_matrix > 0.0) & (self.W_alpha < 0.0)
        index_increasing_1 = (cell_gap_matrix < 0.0) & (self.W_alpha > 0.0)
        #
        cell_gap_matrix[
            index_increasing_0
        ] = numpy.float32(0.0)
        cell_gap_matrix[
            index_increasing_1
        ] = numpy.float32(0.0)
        #
        cell_t_after_decay = numpy.outer(
            self.cell_target, numpy.ones(
                (self.dim_process, ), dtype=dtype
            )
        ) + cell_gap_matrix * numpy.exp(
            -numpy.outer(
                self.cell_decay, numpy.ones(
                    (self.dim_process, ), dtype=dtype
                )
            ) * (
                time_current - time_recent
            )
        )
        hidden_t_after_decay = numpy.outer(
            self.gate_output, numpy.ones(
                (self.dim_process, ), dtype=dtype
            )
        ) * numpy.tanh(cell_t_after_decay)
        #
        self.intensity_tilde_ub = numpy.sum(
            hidden_t_after_decay * self.W_alpha,
            axis=0
        )
        self.intensity_ub = self.soft_relu_scale(
            self.intensity_tilde_ub
        )
        #
        # intensity computation is finished
    #
    #
    def sample_time_given_type(self, type_event):
        # type_event is the type of event for which we want to sample the time
        # it is the little k in our model formulation in paper
        time_current = numpy.float32(0.0)
        if len(self.one_seq) > 0:
            time_current = self.one_seq[-1]['time_since_start']
        #
        #self.compute_intensity(time_current)
        self.compute_intensity_upper_bound(time_current)
        intensity_hazard = numpy.copy(
            self.intensity_ub[type_event]
        )
        #
        u = 1.5
        while u >= 1.0:
            #print "type is : ", type_event
            E = numpy.random.exponential(
                scale=1.0, size=None
            )
            U = numpy.random.uniform(
                low=0.0, high=1.0, size=None
            )
            #print "E U time_current : "
            #print E, U, time_current
            #print "intensity hazard is : "
            #print intensity_hazard
            time_current += (E / intensity_hazard)
            self.compute_intensity_given_past(time_current)
            u = U * intensity_hazard / self.intensity[type_event]
            #print "new time_current and u : "
            #print time_current, u
            #print "intensity and upper bound is : "
            #print self.intensity
            #print self.intensity_ub
            # use adaptive thinning algorithm
            # that is, decreasing the upper bound
            # to make the sampling quicker
            # use adaptive method by
            # toggling on the following block
            '''
            self.compute_intensity_upper_bound(
                time_current
            )
            intensity_hazard = numpy.copy(
                self.intensity_ub[type_event]
            )
            '''
        return time_current
        #
    #
    #
    
    def run(self):
        N_repeat = 1
        with open(self.args.FileData + "/test.pkl") as f:
            data = pickle.load(f)
        ae = 0
        n = 0
        error_list = list()
        #print(data)
        for sample in data['test']:
            self.restart_sequence() #reset hidden states
            for i in range(1,len(sample)):
                self.one_seq = sample[:i]
                self.compute_hidden_states() # update h
                if int(sample[i]['type_event']) == int(self.args.Target):
                    pred_time = numpy.mean(numpy.array([self.sample_time_given_type(int(self.args.Target)) for _ in range (N_repeat)] )) #predict next event.
                    error = abs(float(sample[i]['time_since_start']) - pred_time)
                    error_list.append(error)
                    ae += error
                    n += 1
            
                    
        MAE = ae / n
        print MAE
        
        with open("./res.pkl", "wb") as f:
            pickle.dump(error_list, f)

def test_survival():
    with open("./res.pkl","rb") as f:
        error_list = pickle.load(f)
    acc = 0
    n = 0
    for e in error_list:
        if e < 4:
            n+=1
            if e < 3:
                acc+=1
    acc = float(acc) / n 
    print acc

def main():

    parser = argparse.ArgumentParser(
        description='Generating sequences... '
    )
    #
    
    parser.add_argument(
        '-fp', '--FilePretrain', required=True,
        help='File of pretrained model (e.g. ./tracks/track_PID=XX_TIME=YY/model.pkl)'
    )
    parser.add_argument(
        '--FileData', required=True,
        help='File of data (e.g. ./data/crime/)'
    )
    parser.add_argument(
        '--Target' ,required=True,
        help='target predicate idx'
    )
    #
    
    #
    args = parser.parse_args()
    
    
   
    
    model = NeuralHawkesCTLSTM(args)
    model.run()
    
    
    

if __name__ == "__main__": 
    #main()
    test_survival()
