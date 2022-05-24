#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import os

__all__ = ['DataLogger']

class DataLogger(object):
    def __init__(self, file_name, path_name='./'):
        # path check
        os.makedirs(path_name, exist_ok=True)
        self.path_name = path_name
        self.f = open(path_name + file_name, 'w')
        self.writer = csv.writer(self.f, lineterminator='\n')

    def write_csv(self, data_list):
        self.writer.writerow(data_list)

    def write_comment(self, data_str):
        self.f.write('# ' + data_str)
        
    def close(self):
        self.f.close()