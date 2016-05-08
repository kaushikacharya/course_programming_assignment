#!/usr/bin/python

import sys

line_count = 0
flag_fresh_entry = True

for line in sys.stdin:
    data_mapped = line.strip().split('\t')
    
    if line_count == 0:
        # header row
        count_field = len(data_mapped)
        
        # now find out the field indices which we are interested in
        field_index_tagnames = None
        
        for field_i in range(0, count_field):
            # print '{0}:{1}:{2}'.format(field_i, data_mapped[field_i][1:len(data_mapped[field_i])-1], data_mapped[field_i] == 'author_id')
            if (data_mapped[field_i] == '"tagnames"') or (data_mapped[field_i] == 'tagnames'):
                field_index_tagnames = field_i
    else:
        # 'body' column can have newline character
        if flag_fresh_entry:
            data_tagnames = data_mapped[field_index_tagnames].strip('"').split()
            
            for tagname in data_tagnames:
                print '{0}'.format(tagname)
            
            if len(data_mapped) != count_field:
                flag_fresh_entry = False
        else:
            if len(data_mapped) == 1:
                # current line contains only "body"
                pass
            else:
                flag_fresh_entry = True
            
    line_count += 1
