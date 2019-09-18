from __future__ import division
from nltk.corpus import wordnet as wn
import numpy as np
import logging

import random


class WN_Lookup:

    def __init__(self, in_terms=[], threshold=0.025, allow_multiple=False, rseed=1, num_rands=10):
        self.threshold = threshold

        self.synset_counts = {}
        self.synset_set = set()
        self.preceding_words = set([''])
        self.succeeding_words = set([''])
        self.matched_terms = set()
        self.skip_set = set()
        self.max_length = 0
        self.allow_multiple = allow_multiple

        if rseed > 0:
            random.seed(rseed)

        words = []
        for i in range(num_rands):
            word = random.sample(list(wn.words()), 1)[0]
            while wn.synsets(word)[0].pos() != 'n':
                word = random.sample(list(wn.words()), 1)[0]
            print('word {}'.format(word))
            words += [word]

        self.get_synsets_recursively(wn.synsets(words[0]))
        self.overfrequent_synsets = self.skip_set.copy()
        print('overfrequent synsets : {}'.format(self.overfrequent_synsets))
        for word in words[1:]:
            self.skip_set = set()
            self.get_synsets_recursively(wn.synsets(word))
            print('skip_set synsets : {}'.format(self.skip_set))
            self.overfrequent_synsets &= self.skip_set
            print('overfrequent synsets : {}'.format(self.overfrequent_synsets))
        
        for t in in_terms:
            if not self.add_training_term(t):
                logging.debug('no wn vocab found :: {}'.format(t))
        self.make_set()

    def lemmatize_if_needed(self, t):
        if t in wn.words() and wn.synsets(t)[0].pos() == 'n':
            logging.debug('found in wn : {}'.format(t))
            return t
        if t[-3:] == 'ies':
            t = t[:-3] + 'y'
        if t[-1:] == 's':
            t = t[:-1]
        if t in wn.words() and wn.synsets(t)[0].pos() == 'n':
            logging.debug('found in wn : {}'.format(t))
            return t
        return None
            
    def add_training_term(self,term):
        term = term.split()
        for reduction_size in range(len(term)):
            for i in reversed(range(reduction_size + 1)):
                t = '_'.join(term[i:i+len(term) - reduction_size])

                t = self.lemmatize_if_needed(t)
                if t != None:
                    if self.allow_multiple or t not in self.matched_terms:
                        self.get_synsets_recursively(wn.synsets(t))
                        for sn in self.skip_set:
                            if sn in self.overfrequent_synsets:
                                continue
                            if sn not in self.synset_counts:
                                self.synset_counts[sn] = 0
                            self.synset_counts[sn] += 1
                    self.skip_set = set()

                    self.preceding_words |= set(term[:i])
                    self.succeeding_words |= set(term[i+len(term) - reduction_size:])
                    self.matched_terms.add(t)
                    return True
                else:
                    self.preceding_words |= set(term)
                    logging.debug('not in wn : {} {}'.format(term,t))
        return False

    def make_set(self):
        self.max_ct = max(self.synset_counts.values())
        self.synset_set = set([k for k in self.synset_counts.keys()
                               if self.synset_counts[k] / self.max_ct >= self.threshold])
        self.max_length = max([len(k.split('_')) for k in self.synset_counts])

    def get_synsets_recursively(self, ssets):
        for s in ssets:
            sn = s.name()
            if s.pos() == 'n' and sn not in self.skip_set:
                self.skip_set.add(sn)
                self.get_synsets_recursively(s.hypernyms())
                
    def score_match(self, term, prethresh=.95, postthresh=.95, binary=False, show_synsets=False):
        term = term.split()
        score = 0.
        for reduction_size in range(len(term)):
            for i in reversed(range(reduction_size + 1)):
                t = '_'.join(term[i:i+len(term) - reduction_size])
                pre = term[:i]
                post = term[i+len(term) - reduction_size:]
                prematches = 1. if len(pre) == 0 else len([1 for w in pre if w in self.preceding_words]) / len(pre)
                postmatches = 1. if len(post) == 0 else len([1 for w in post if w in self.succeeding_words]) / len(post)
                t = self.lemmatize_if_needed(t)
                if prematches >= prethresh and postmatches >= postthresh and t != None:
                    self.get_synsets_recursively(wn.synsets(t))
                    if show_synsets:
                        print( term, self.skip_set )
                        print( self.synset_set )
                        print( [s in self.synset_set for s in self.skip_set] )
                    if binary == True:
                        this_score = sum([int(s in self.synset_set) for s in self.skip_set]) / len(self.synset_set)
                    else:
                        logvec = [np.log(self.synset_counts[s]) for s in self.skip_set if s in self.synset_counts]
                        if len(logvec) == 0:
                            this_score = 0.
                        else:
                            this_score = np.mean(logvec) / np.log(self.max_ct)
                    self.skip_set = set()
                    # logging.debug('scoring found matching term : {} in : {} score : {}'.format(t,term,this_score))
                    # print('scoring found matching term : {} in : {} score : {}'.format(t,term,this_score))
                    score = max([score,this_score])
        return score
                    
if __name__ == "__main__":
    words = ["angry orange", "mad as hell pineapple", "aloe vera"]
    wnl = WN_Lookup(in_terms=words)
    print(wnl.synset_counts)
    print(wnl.matched_terms)
    print("\nsquash wnl score match : {}".format(wnl.score_match("squash")))
    print("\nraccoon wnl score match : {}\n\n".format(wnl.score_match("raccoon")))
    print("\norange wnl score match : {}\n\n".format(wnl.score_match("orange",binary=True,show_synsets=True)))
    print("\norange wnl score match : {}\n\n".format(wnl.score_match("orange",binary=True,show_synsets=True)))

#     random.seed(1)
#     words = random.sample(list(wn.words()),10)
# for word in words:
#     if wn.synsets(word)[0].pos() == 'n':
#         print("{} wnl score match : {}\n".format(word,wnl.score_match(word, show_synsets=True)))
#         print("{} wnl score match : {}\n".format(word,wnl.score_match(word, show_synsets=True, binary=True)))
