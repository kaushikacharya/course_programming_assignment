#!/usr/bin/python

import sys

# Loop around the data
# It will be in the format key
# Where key is tagname

tagCount = 0
oldKey = None
topN = 10
tag_dict = {}
# These keeps track of min count in topN
minCountInTopN = float('Inf')
minTagInTopN = None

for line in sys.stdin:
    data_mapped = line.strip().split('\t')
    
    if len(data_mapped) != 1:
        continue # issue in line
        
    tagname = data_mapped[0]
        
    if oldKey and (oldKey != tagname):
        # next tagname's series started
        # print '{0}\t{1}'.format(oldKey, tagCount)
        if len(tag_dict) < topN:
            tag_dict[oldKey] = tagCount
            # check if oldKey has count less than all of already existing tags in tag_dict
            if tagCount < minCountInTopN:
                minTagInTopN = oldKey
                minCountInTopN = tagCount
        else:
            # check if oldKey needs to be in topN elements
            if minCountInTopN < tagCount:
                del tag_dict[minTagInTopN]
                tag_dict[oldKey] = tagCount
                # now update minCountInTopN, minTagInTopN
                minCountInTopN = float('Inf')
                for key in tag_dict.keys():
                    if tag_dict[key] < minCountInTopN:
                        minCountInTopN = tag_dict[key]
                        minTagInTopN = key
        
        # now reset
        tagCount = 0
        
    tagCount += 1
    oldKey = tagname
    
if oldKey != None:
    # next tagname's series started
    # print '{0}\t{1}'.format(oldKey, tagCount)
    if len(tag_dict) < topN:
        tag_dict[oldKey] = tagCount
        # check if oldKey has count less than all of already existing tags in tag_dict
        if tagCount < minCountInTopN:
            minTagInTopN = oldKey
            minCountInTopN = tagCount
    else:
        # check if oldKey needs to be in topN elements
        if minCountInTopN < tagCount:
            del tag_dict[minTagInTopN]
            tag_dict[oldKey] = tagCount
            # now update minCountInTopN, minTagInTopN
            minCountInTopN = float('Inf')
            for key in tag_dict.keys():
                if tag_dict[key] < minCountInTopN:
                    minCountInTopN = tag_dict[key]
                    minTagInTopN = key
    
# now print in reverse count
sorted_dict_arr = sorted(tag_dict.items(), key=lambda x:x[1], reverse=True)

for tag_elem in sorted_dict_arr:
    print '{0}\t{1}'.format(tag_elem[0], tag_elem[1])

