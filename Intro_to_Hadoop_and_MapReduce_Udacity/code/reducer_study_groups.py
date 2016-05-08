#!/usr/bin/python

import sys

# Loop around the data
# It will be in the format key\tval
# Where key=Question ID, val=author_id

oldKey = None
author_id_arr = []

for line in sys.stdin:
    data_mapped = line.strip().split('\t')
    
    if len(data_mapped) != 2:
        continue # improper line
        
    question_id, author_id = data_mapped
    
    if oldKey and (oldKey != question_id):
        # next series of question id started
        # dump the data corresponding to previous question id
        print '{0}\t{1}'.format(oldKey, author_id_arr)
        # reset author id array
        author_id_arr = []
        
    author_id_arr.append(author_id)
    oldKey = question_id
    
if oldKey != None:
    print '{0}\t{1}'.format(oldKey, author_id_arr)
        
    
