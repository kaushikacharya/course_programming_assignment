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
        field_index_author_id = None
        field_index_body = None
        field_index_added_at = None
        
        for field_i in range(0, count_field):
            # print '{0}:{1}:{2}'.format(field_i, data_mapped[field_i][1:len(data_mapped[field_i])-1], data_mapped[field_i] == 'author_id')
            if (data_mapped[field_i] == '"author_id"') or (data_mapped[field_i] == 'author_id'):
                field_index_author_id = field_i
            if (data_mapped[field_i] == '"body"') or (data_mapped[field_i] == 'body'):
                field_index_body = field_i
            if (data_mapped[field_i] == '"added_at"') or (data_mapped[field_i] == 'added_at'):
                field_index_added_at = field_i
                
    else:
        # 'body' column can have newline character
        if flag_fresh_entry:
            data_author_id = int(data_mapped[field_index_author_id].strip('"'))
            
            if len(data_mapped) == count_field:
                data_added_at = data_mapped[field_index_added_at].strip('"')
            else:
                data_added_at = None
                flag_fresh_entry = False
        else:
            if len(data_mapped) == 1:
                # current line contains only "body"
                data_added_at = None
            else:
                data_added_at = data_mapped[field_index_added_at - field_index_body].strip('"')
                flag_fresh_entry = True
            
        if data_added_at != None:
            hour_added_at = int(data_added_at.split()[1].split(':')[0])
            print '{0}\t{1}'.format(data_author_id, hour_added_at)
        
    line_count += 1
   
# for debugging purpose    
# print '{0}\t{1}\t{2}\t{3}\t{4}'.format(field_index_author_id, field_index_body, field_index_added_at, line_count, count_field)
