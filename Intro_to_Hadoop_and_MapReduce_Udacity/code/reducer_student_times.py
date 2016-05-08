#!/usr/bin/python

import sys
from collections import defaultdict

# Loop around the data
# It will be in the format key\tval
# Where key is the author_id, val is the hour

author_dict = {}
oldKey = None
hour_dict = defaultdict(int) # keeps count of hour for the current author_id

for line in sys.stdin:
    data_mapped = line.strip().split('\t')
    
    if len(data_mapped) != 2:
        continue # some issue
        
    author_id, hour = data_mapped
    
    if oldKey and (author_id != oldKey):
        # get the hour(s) whose count is max
        max_count = 0
        max_hour_arr = []
        
        for hr in hour_dict.keys():
            if max_count < hour_dict[hr]:
                max_count = hour_dict[hr]
                max_hour_arr = []
                max_hour_arr.append(hr)
            elif max_count == hour_dict[hr]:
                max_hour_arr.append(hr)
                
        for hr in max_hour_arr:
            print '{0}\t{1}'.format(oldKey, hr)
        
        # reset
        hour_dict = defaultdict(int)
        
    hour_dict[hour] += 1
    oldKey = author_id
    
if oldKey:
    # get the hour(s) whose count is max
    max_count = 0
    max_hour_arr = []
    
    for hr in hour_dict.keys():
        if max_count < hour_dict[hr]:
            max_count = hour_dict[hr]
            max_hour_arr = []
            max_hour_arr.append(hr)
        elif max_count == hour_dict[hr]:
            max_hour_arr.append(hr)
            
    for hr in max_hour_arr:
        print '{0}\t{1}'.format(author_id, hr)
