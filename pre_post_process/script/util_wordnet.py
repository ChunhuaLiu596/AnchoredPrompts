from nltk.corpus import wordnet as wn
# from inflection import singularize, pluralize 


def get_sister_terms(word, distance_to_hypernym=1):
    '''
    "Coordinate (sister) terms: share the same hypernym"
    "The sister relation is the usual one encountered when working with tree structures: sisters are word forms (either simple words or collocations) that are both immediate hyponyms of the same node"
    
    Args:
        word: the input word
        hop: the hops to hypernyms, default is 1, which means take the top 1 hypernym of x
    '''
    sister_terms = set()
    for synset in wn.synsets(word ,"n"):
        for hypernym in synset.hypernyms()[:distance_to_hypernym]:
#             print(hypernym)
            sister_synsets = hypernym.hyponyms()
            for sister_synset in sister_synsets:
                sister_names = [x.name() for x in sister_synset.lemmas()]
                sister_names_selected = [name.lower() for name in sister_names if len(name.split("_"))==1 and  len(name.split("-"))==1  and name!=word]
                sister_terms = sister_terms.union(set(sister_names_selected))
    return list(sister_terms)


def test_sister_terms():
    for k in range(1,10):
        print(k, get_sister_terms('dog', k))
        print()
