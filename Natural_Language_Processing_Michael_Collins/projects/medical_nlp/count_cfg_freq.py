#! /usr/bin/python

__author__="Alexander Rush <srush@csail.mit.edu>"
__date__ ="$Sep 12, 2012"

import sys, json

"""
Count rule frequencies in a binarized CFG.
Modified by KA: lower casing the words
"""

class Counts:
  def __init__(self):
    self.unary = {}
    self.binary = {}
    self.nonterm = {}

  def show(self):
    for symbol, count in self.nonterm.iteritems():
      print count, "NONTERMINAL", symbol

    for (sym, word), count in self.unary.iteritems():
      print count, "UNARYRULE", sym, word

    for (sym, y1, y2), count in self.binary.iteritems():
      print count, "BINARYRULE", sym, y1, y2

  def write(self, out_file):
      with open(out_file, 'w') as fd:
          for symbol, count in self.nonterm.iteritems():
              fd.write('{0} {1} {2}\n'.format(count, "NONTERMINAL", symbol))

          for (sym, word), count in self.unary.iteritems():
              fd.write('{0} {1} {2} {3}\n'.format(count, "UNARYRULE", sym, word))

          for (sym, y1, y2), count in self.binary.iteritems():
              fd.write('{0} {1} {2} {3} {4}\n'.format(count, "BINARYRULE", sym, y1, y2))

  def count(self, tree):
    """
    Count the frequencies of non-terminals and rules in the tree.
    """
    if isinstance(tree, basestring): return

    # Count the non-terminal symbol. 
    symbol = tree[0]
    self.nonterm.setdefault(symbol, 0)
    self.nonterm[symbol] += 1
    
    if len(tree) == 3:
      # It is a binary rule.
      y1, y2 = (tree[1][0], tree[2][0])
      key = (symbol, y1, y2)
      self.binary.setdefault(key, 0)
      self.binary[(symbol, y1, y2)] += 1
      
      # Recursively count the children.
      self.count(tree[1])
      self.count(tree[2])
    elif len(tree) == 2:
      # It is a unary rule.
      assert (type(tree[1]) is not list), "{0} should have been string".format(tree[1])
      y1 = tree[1].lower()  # word in the beginning of a sentence starts with capital letters. Hence lower casing.
      key = (symbol, y1)
      self.unary.setdefault(key, 0)
      self.unary[key] += 1

def main(parse_file):
  counter = Counts() 
  for l in open(parse_file):
    t = json.loads(l)
    try:
        counter.count(t)
    except:
        print('Error in counter.count: parse tree: {0}'.format(t))

  counter.show()

def usage():
    sys.stderr.write("""
    Usage: python count_cfg_freq.py [tree_file]
        Print the counts of a corpus of trees.\n""")

if __name__ == "__main__": 
  if len(sys.argv) != 2:
    usage()
    sys.exit(1)
  main(sys.argv[1])
  
