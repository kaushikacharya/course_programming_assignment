#!/usr/bin/python

import sys

for line in sys.stdin:
    data_mapped = line.strip().split()
    
    if len(data_mapped) != 10:
        continue # log sentence isn't for page hit
        
    thisPage = data_mapped[6]
    
    if thisPage[0] != '/':
        # extract the relative path from the absolute path
        # TBD: put regex
        if (len(thisPage) > 7) and (thisPage[0:7] == 'http://'):
            pos = 7
            flag_found = False
            while (pos < len(thisPage)):
                if (thisPage[pos] == '/'):
                    flag_found = True
                    break
                pos += 1
                
            if flag_found:
                thisPage = thisPage[pos:]
                
    print '{0}\t{1}'.format(thisPage,1)
