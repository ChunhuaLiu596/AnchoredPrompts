import pandas as pd
from sklearn.metrics import accuracy_score



import spacy
# import pyinflect
import lemminflect
nlp = spacy.load('en_core_web_sm')


def pluralize(word):
    pl = nlp(word)
    if pl[0].pos_ not in ["NOUN"] or pl[0].tag_ == 'NNS':
        return word
    else:
        pl= pl[0]._.inflect('NNS')
        return pl if pl is not None else word


def singularize(word):
    '''
    sg is None if the word is not in the vocab 
    '''
    sg =  nlp(word) #[0]._.inflect('NN')

    if sg[0].pos_ not in ["NOUN"] or sg[0].tag_ == 'NN':
        return word
    else:
        sg = sg[0]._.inflect('NN') 
        return sg if sg is not None else word
    



# def singularize(word):
#     '''
#     sg is None if the word is not in the vocab 
#     '''
#     sg =  nlp(word)[0]._.inflect('NN')
    
#     return sg if sg is not None else word
    
# def pluralize(word):
#     '''
#     sg is None if the word is not in the vocab 
#     '''
#     pl= nlp(word)[0]._.inflect('NNS')
#     return pl if pl is not None else word



# import inflect
# p = inflect.engine()
# p.classical(all=True)  # USE ALL CLASSICAL PLURALS
# # def singularize(words):
# #     '''
# #     Args: 
# #         words: a list of word or a single words 
# #     '''
# #     words_sg = [p.singular_noun(x) if p.singular_noun(x)!=False else x for x in words]
# #     return words_sg

# # def pluralize(words):
# #     words_pl = [ p.plural_noun(x) if p.plural_noun(x)!=False else x for x in words]
# #     return words_pl

# def singularize(word):
#     '''
#     Args: 
#         words: a list of word or a single words 
#     '''
#     return p.singular_noun(word) if p.singular_noun(word)!=False else word 

# def pluralize(word):
#     return p.plural_noun(word) if p.plural_noun(word)!=False else word 