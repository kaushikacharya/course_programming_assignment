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
        field_index_id = None
        field_index_author_id = None
        field_index_body = None
        field_index_node_type = None
        field_index_abs_parent_id = None
        
        for field_i in range(0, count_field):
            # print '{0}:{1}:{2}'.format(field_i, data_mapped[field_i][1:len(data_mapped[field_i])-1], data_mapped[field_i] == 'author_id')
            if (data_mapped[field_i] == '"id"') or (data_mapped[field_i] == 'id'):
                field_index_id = field_i
            if (data_mapped[field_i] == '"author_id"') or (data_mapped[field_i] == 'author_id'):
                field_index_author_id = field_i
            if (data_mapped[field_i] == '"body"') or (data_mapped[field_i] == 'body'):
                field_index_body = field_i
            if (data_mapped[field_i] == '"node_type"') or (data_mapped[field_i] == 'node_type'):
                field_index_node_type = field_i
            if (data_mapped[field_i] == '"abs_parent_id"') or (data_mapped[field_i] == 'abs_parent_id'):
                field_index_abs_parent_id = field_i
    else:
        # 'body' column can have newline character
        if flag_fresh_entry:
            data_id = int(data_mapped[field_index_id].strip('"'))
            data_author_id = int(data_mapped[field_index_author_id].strip('"'))
            
            if len(data_mapped) == count_field:
                data_node_type = data_mapped[field_index_node_type].strip('"')
                data_abs_parent_id = data_mapped[field_index_abs_parent_id].strip('"')
            else:
                data_node_type = None
                data_abs_parent_id = None
                flag_fresh_entry = False
        else:
            if len(data_mapped) == 1:
                # current line contains only "body"
                data_node_type = None
                data_abs_parent_id = None
            else:
                # first field is continuation of body from previous line
                data_node_type = data_mapped[field_index_node_type - field_index_body].strip('"')
                data_abs_parent_id = data_mapped[field_index_abs_parent_id - field_index_body].strip('"')
                flag_fresh_entry = True
                
        if data_node_type != None:
            if data_node_type == 'question':
                print '{0}\t{1}'.format(data_id, data_author_id)
            elif data_node_type == 'answer':
                # data_abs_parent_id hasn't been converted to int from string
                print '{0}\t{1}'.format(int(data_abs_parent_id), data_author_id)
                
    line_count += 1
