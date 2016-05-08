#!/usr/bin/python

import sys

# Loop around the data
# It will be in the format key\ttype\tval
# Where key=Question ID, type=node type, val=length of post

oldKey = None
sum_answer_len = 0
question_len = 0
count_answer = 0

for line in sys.stdin:
    data_mapped = line.strip().split('\t')
    
    if len(data_mapped) != 3:
        continue # improper line
        
    question_id, node_type, post_len = data_mapped
    post_len = int(post_len)
    
    if oldKey and (question_id != oldKey):
        # print average answer length for previous question i.e. oldKey
        if count_answer > 0:
            print '{0}\t{1}\t{2}'.format(oldKey, question_len, sum_answer_len*1.0/count_answer)
        else:
            print '{0}\t{1}\t{2}'.format(oldKey, question_len, 0)
        
        # re-initialize the values    
        sum_answer_len = 0
        question_len = 0
        count_answer = 0
        
    if node_type == 'A':
        sum_answer_len += post_len
        count_answer += 1
    elif node_type == 'Q':
        question_len = post_len
        
    oldKey = question_id
    
if oldKey != None:
    # print average answer length for previous question i.e. oldKey
    if count_answer > 0:
        print '{0}\t{1}\t{2}'.format(oldKey, question_len, sum_answer_len*1.0/count_answer)
    else:
        print '{0}\t{1}\t{2}'.format(oldKey, question_len, 0)
