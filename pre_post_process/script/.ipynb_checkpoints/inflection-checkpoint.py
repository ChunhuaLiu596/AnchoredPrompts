# # -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import accuracy_score



import spacy
import pyinflect

nlp = spacy.load('en_core_web_sm')

def singularize(word):
    '''
    sg is None if the word is not in the vocab 
    '''
    sg =  nlp(word)[0]._.inflect('NN')
    
    return sg if sg is not None else word
    
def pluralize(word):
    '''
    sg is None if the word is not in the vocab 
    '''
    pl= nlp(word)[0]._.inflect('NNS')
    return pl if pl is not None else word





###########################BELOW is anthoer version: less accurate, can manully adjust more
# """
#     inflection
#     ~~~~~~~~~~~~

#     A port of Ruby on Rails' inflector to Python.

#     :copyright: (c) 2012-2020 by Janne Vanhala

#     :license: MIT, see below

#     Copyright (C) 2012-2020 Janne Vanhala

#     Permission is hereby granted, free of charge, to any person obtaining a copy
#     of this software and associated documentation files (the "Software"), to deal
#     in the Software without restriction, including without limitation the rights
#     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#     copies of the Software, and to permit persons to whom the Software is
#     furnished to do so, subject to the following conditions:

#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.

#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.
# """
# import re
# import unicodedata

# __version__ = '0.5.1'

# PLURALS = [
#     (r"(?i)(quiz)$", r'\1zes'),
#     (r"(?i)^(oxen)$", r'\1'),
#     (r"(?i)^(ox)$", r'\1en'),
#     (r"(?i)(m|l)ice$", r'\1ice'),
#     (r"(?i)(m|l)ouse$", r'\1ice'),
#     (r"(?i)(passer)s?by$", r'\1sby'),
#     (r"(?i)(matr|vert|ind)(?:ix|ex)$", r'\1ices'),
#     (r"(?i)(x|ch|ss|sh)$", r'\1es'),
#     (r"(?i)([^aeiouy]|qu)y$", r'\1ies'),
#     (r"(?i)(hive)$", r'\1s'),
#     (r"(?i)([lr])f$", r'\1ves'),
#     (r"(?i)([^f])fe$", r'\1ves'),
#     (r"(?i)sis$", 'ses'),
#     (r"(?i)([ti])a$", r'\1a'),
#     (r"(?i)([ti])um$", r'\1a'),
#     (r"(?i)(buffal|potat|tomat)o$", r'\1oes'),
#     (r"(?i)(bu)s$", r'\1ses'),
#     (r"(?i)(alias|status)$", r'\1es'),
#     (r"(?i)(octop|vir)i$", r'\1i'),
#     (r"(?i)(octop|vir)us$", r'\1i'),
#     (r"(?i)^(ax|test)is$", r'\1es'),
#     (r"(?i)s$", 's'),
#     (r"$", 's'),
# ]

# SINGULARS = [
#     (r"(?i)(database)s$", r'\1'),
#     (r"(?i)(quiz)zes$", r'\1'),
#     (r"(?i)(matr)ices$", r'\1ix'),
#     (r"(?i)(vert|ind)ices$", r'\1ex'),
#     (r"(?i)(passer)sby$", r'\1by'),
#     (r"(?i)^(ox)en", r'\1'),
#     (r"(?i)(alias|status)(es)?$", r'\1'),
#     (r"(?i)(octop|vir)(us|i)$", r'\1us'),
#     (r"(?i)^(a)x[ie]s$", r'\1xis'),
#     (r"(?i)(cris|test)(is|es)$", r'\1is'),
#     (r"(?i)(shoe)s$", r'\1'),
#     (r"(?i)(o)es$", r'\1'),
#     (r"(?i)(bus)(es)?$", r'\1'),
#     (r"(?i)(m|l)ice$", r'\1ouse'),
#     (r"(?i)(x|ch|ss|sh)es$", r'\1'),
#     (r"(?i)(m)ovies$", r'\1ovie'),
#     (r"(?i)(s)eries$", r'\1eries'),
#     (r"(?i)([^aeiouy]|qu)ies$", r'\1y'),
#     (r"(?i)([lr])ves$", r'\1f'),
#     (r"(?i)(tive)s$", r'\1'),
#     (r"(?i)(hive)s$", r'\1'),
#     (r"(?i)([^f])ves$", r'\1fe'),
#     (r"(?i)(t)he(sis|ses)$", r"\1hesis"),
#     (r"(?i)(s)ynop(sis|ses)$", r"\1ynopsis"),
#     (r"(?i)(p)rogno(sis|ses)$", r"\1rognosis"),
#     (r"(?i)(p)arenthe(sis|ses)$", r"\1arenthesis"),
#     (r"(?i)(d)iagno(sis|ses)$", r"\1iagnosis"),
#     (r"(?i)(b)a(sis|ses)$", r"\1asis"),
#     (r"(?i)(a)naly(sis|ses)$", r"\1nalysis"),
#     (r"(?i)([ti])a$", r'\1um'),
#     (r"(?i)(n)ews$", r'\1ews'),
#     (r"(?i)(ss)$", r'\1'),
#     (r"(?i)s$", ''),
# ]


# # nouns that can be countable or uncountable (but uncountable is more often used)
# countunable_and_uncountable = ['acacia',
#  'carp',
#  'corn',
#  'garlic',
#  'lettuce',
#  'lime',
#  'sheep',
#  'strawberry',
#  'wardrobe']



# # list of uncountable nouns aka mass nouns scraped from http://simple.wiktionary.org/wiki/Category%3aUncountable_nouns
# UNCOUNTABLES =('abaft',
#     'abandon',
#     'abasia',
#     'abating',
#     'abbreviated',
#     'abducted',
#     'able-bodiedism',
#     'able-bodism',
#     'ableism',
#     'ablepharia',
#     'ablism',
#     'abolition',
#     'abolitionism',
#     'abracadabra',
#     'absence',
#     'absenteeism',
#     'absorbency',
#     'abstention',
#     'abstinence',
#     'abstractness',
#     'abstruseness',
#     'absurdity',
#     'abulia',
#     'abuse',
#     'Abyssinia',
#     'academe',
#     'academia',
#     'acapnia',
#     'acardia',
#     'acariasis',
#     'acaridiasis',
#     'acariosis',
#     'acarophobia',
#     'acataphasia',
#     'acceleration',
#     'acceptability',
#     'acceptance',
#     'access',
#     'accessibility',
#     'accommodation',
#     'accord',
#     'accordance',
#     'account',
#     'accumulation',
#     'accuracy',
#     'accurateness',
#     'acephalia',
#     'acephalism',
#     'acephaly',
#     'achromatism',
#     'aciclovir',
#     'acidosis',
#     'acorea',
#     'acoustics',
#     'acquisition',
#     'acrocyanosis',
#     'acromegalia',
#     'acromegaly',
#     'acromicria',
#     'acromikria',
#     'acrophobia',
#     'actinium',
#     'actinometry',
#     'actinomycin',
#     'action',
#     'activewear',
#     'activism',
#     'activity',
#     'actuality',
#     'acupressure',
#     'acupuncture',
#     'acyclovir',
#     'adactylia',
#     'adactylism',
#     'adactyly',
#     'adaptation',
#     'addition',
#     'adenomegaly',
#     'adherence',
#     'adiposis',
#     'administration',
#     'admirability',
#     'adolescence',
#     'adorability',
#     'adsorption',
#     'adulthood',
#     'advantageousness',
#     'advertising',
#     'advice',
#     'affect',
#     'affection',
#     'afternoon',
#     'age',
#     'agency',
#     'aggregate',
#     'aggression',
#     'agon',
#     'aid',
#     'ailurophilia',
#     'aim',
#     'air',
#     'alarm',
#     'Albanian',
#     'alcohol',
#     'alcoholism',
#     'algebra',
#     'algophobia',
#     'all',
#     'allocation',
#     'allowance',
#     'alphabetical order',
#     'alto',
#     'amalgamation',
#     'amber',
#     'ambition',
#     'amendment',
#     'americium',
#     'ammunition',
#     'amusement',
#     'anaesthetic',
#     'anal sex',
#     'analysis',
#     'anesthetic',
#     'anilingus',
#     'anime',
#     'anthropolatry',
#     'anthropology',
#     'antidisestablishmentarianism',
#     'antimasonry',
#     'antimony',
#     'antinatalism',
#     'antiseptic',
#     'antivirus',
#     'apathy',
#     'appeal',
#     'appeasement',
#     'appreciation',
#     'appropriation',
#     'arachnophobia',
#     'arborolatry',
#     'archaeology',
#     'archery',
#     'Argentine',
#     'argon',
#     'arithmetic',
#     'armament',
#     'Armenian',
#     'armor',
#     'armour',
#     'arrogance',
#     'arson',
#     'articulation',
#     'asbestos',
#     'ash',
#     'aspect',
#     'aspirin',
#     'assessment',
#     'assistance',
#     'association',
#     'assurance',
#     'astatine',
#     'astrology',
#     'astrophysics',
#     'asylum',
#     'atheism',
#     'atmosphere',
#     'atmospheric pressure',
#     'attention',
#     'attitude',
#     'attraction',
#     'attractiveness',
#     'auburn',
#     'audio',
#     'audition',
#     'authority',
#     'avoidance',
#     'awareness',
#     'awkwardness',
#     'babble',
#     'backbone',
#     'background',
#     'bacon',
#     'badminton',
#     'baggage',
#     'baking soda',
#     'balance',
#     'balefulness',
#     'ballet',
#     'bambakophobia',
#     'bardolatry',
#     'barf',
#     'bark',
#     'barley',
#     'baseball',
#     'basketball',
#     'battery',
#     'bazooka',
#     'beauty',
#     'bedrock',
#     'beef',
#     'beer',
#     'beeswax',
#     'behavior',
#     'behaviour',
#     'berkelium',
#     'best',
#     'bestiality',
#     'bias',
#     'Bible',
#     'bibliolatry',
#     'bingo',
#     'biodiversity',
#     'biology',
#     'bird',
#     'biscuit',
#     'bismuth',
#     'blah',
#     'blame',
#     'blanket',
#     'blond',
#     'blonde',
#     'blood',
#     'bloom',
#     'blues',
#     'bluster',
#     'bohrium',
#     'boilerplate',
#     'bondage',
#     'booze',
#     'botany',
#     'bother',
#     'bounciness',
#     'bowling',
#     'brass',
#     'bravery',
#     'bread',
#     'breakfast',
#     'breast',
#     'breath',
#     'bribery',
#     'brimstone',
#     'bromine',
#     'bulk',
#     'bully',
#     'bunk',
#     'burlesque',
#     'business',
#     'bust',
#     'butter',
#     'buzzing',
#     'cadmium',
#     'caesium',
#     'cake',
#     'calculation',
#     'californium',
#     'calm',
#     'camel',
#     'campus',
#     'canvas',
#     'capacity',
#     'capital punishment',
#     'capitalism',
#     'capture',
#     'carbon',
#     'cardboard',
#     'care',
#     'carelessness',
#     'carnation',
#     'carp',
#     'carriage',
#     'carry',
#     'cash',
#     'cattle',
#     'celebrity',
#     'cement',
#     'censorship',
#     'ceramic',
#     'cerium',
#     'certainty',
#     'chai',
#     'chalk',
#     'champagne',
#     'change',
#     'chaos',
#     'character',
#     'charity',
#     'check',
#     'checkers',
#     'cheer',
#     'cheese',
#     'chemical',
#     'chemistry',
#     'chess',
#     'chicken',
#     'childhood',
#     'chili',
#     'chilli',
#     'china',
#     'Chinese checkers',
#     'chlorine',
#     'chocolate',
#     'choice',
#     'chop',
#     'choreography',
#     'Christendom',
#     'Christianity',
#     'chronology',
#     'church',
#     'circumstance',
#     'clarity',
#     'class',
#     'classification',
#     'clay',
#     'clearance',
#     'clockwork',
#     'cloth',
#     'clothing',
#     'clout',
#     'clumsiness',
#     'coal',
#     'cocaine',
#     'code',
#     'coffee',
#     'coherence',
#     'cold-heartedness',
#     'coloured',
#     'combination',
#     'comfort',
#     'command',
#     'commerce',
#     'commission',
#     'common sense',
#     'communication',
#     'companionship',
#     'company',
#     'compassion',
#     'compensation',
#     'competence',
#     'competition',
#     'complication',
#     'compost',
#     'comprehension',
#     'comprehensiveness',
#     'compromise',
#     'conatus',
#     'concentrate',
#     'concentration',
#     'concern',
#     'concrete',
#     'conduct',
#     'conflict',
#     'conformity',
#     'confusion',
#     'conjugation',
#     'consent',
#     'construction',
#     'consultation',
#     'consumption',
#     'content',
#     'context',
#     'contraction',
#     'contrary',
#     'contrast',
#     'control',
#     'convenience',
#     'convention',
#     'conversation',
#     'conversion',
#     'cooking',
#     'cooler',
#     'coordination',
#     'copernicium',
#     'copper',
#     'coprolite',
#     'copyright',
#     'coral',
#     'cork',
#     'corn',
#     'correspondence',
#     'cosmos',
#     'cotton',
#     'counsel',
#     'country',
#     'country music',
#     'countryside',
#     'courage',
#     'courtship',
#     'cowardice',
#     'crack cocaine',
#     'cream cheese',
#     'creation',
#     'credit',
#     'crime',
#     'criminality',
#     'criticism',
#     'culpability',
#     'cultivation',
#     'cunctation',
#     'curiosity',
#     'curium',
#     'curling',
#     'currency',
#     'current',
#     'cursedness',
#     'custody',
#     'customer',
#     'cyclamen',
#     'damage',
#     'dance',
#     'dark',
#     'darkness',
#     'darmstadtium',
#     'data',
#     'daylight',
#     'daytime',
#     'deafness',
#     'death',
#     'debate',
#     'debris',
#     'decay',
#     'deceit',
#     'deception',
#     'decimal',
#     'decomposition',
#     'deduction',
#     'defeat',
#     'defence',
#     'defense',
#     'deferment',
#     'deism',
#     'delete',
#     'delight',
#     'delivery',
#     'demand',
#     'democracy',
#     'denim',
#     'density',
#     'dental caries',
#     'depravity',
#     'depression',
#     'deprivation',
#     'design',
#     'desolation',
#     'determinism',
#     'deviation',
#     'dew',
#     'dexterity',
#     'diamond',
#     'Dianthus',
#     'dictation',
#     'diction',
#     'diet',
#     'difference',
#     'differentiation',
#     'digestion',
#     'dignity',
#     'direction',
#     'dirt',
#     'disappointment',
#     'disaster',
#     'discomfort',
#     'discord',
#     'discourse',
#     'discretion',
#     'discrimination',
#     'disease',
#     'dishonesty',
#     'disloyalty',
#     'disorganization',
#     'displacement',
#     'disposal',
#     'disregard',
#     'dissatisfaction',
#     'dissimulation',
#     'distinction',
#     'distress',
#     'distribution',
#     'disunity',
#     'diversity',
#     'division',
#     'dole',
#     'doom',
#     'DOS',
#     'doubles',
#     'dough',
#     'down',
#     'draft',
#     'drama',
#     'drill',
#     'drive',
#     'drollery',
#     'drumming',
#     'dubnium',
#     'duck',
#     'duck tape',
#     'duct tape',
#     'dust',
#     'dustiness',
#     'dye',
#     'dynamic',
#     'dynamics',
#     'dynamite',
#     'dyslexia',
#     'e-mail',
#     'earth',
#     'ease',
#     'east',
#     'ecology',
#     'economics',
#     'economy',
#     'efficiency',
#     'egg',
#     'einsteinium',
#     'electricity',
#     'electromotive force',
#     'elevation',
#     'elite',
#     'eloquence',
#     'email',
#     'embroidery',
#     'emergency',
#     'emphasis',
#     'employment',
#     'endeavor',
#     'endurance',
#     'energy',
#     'enforcement',
#     'engineering',
#     'entertainment',
#     'entrails',
#     'entry',
#     'equal',
#     'equality',
#     'equation',
#     'equipment',
#     'erbium',
#     'erosion',
#     'erosiveness',
#     'eruption',
#     'Estonia',
#     'eternity',
#     'ethics',
#     'ethnicity',
#     'etymology',
#     'eugenics',
#     'euphemism',
#     'euphony',
#     'europium',
#     'evaluation',
#     'evening',
#     'evidence',
#     'evil',
#     'evilness',
#     'evolution',
#     'examination',
#     'excellence',
#     'exceptional',
#     'excess',
#     'exercise',
#     'existence',
#     'expenditure',
#     'expense',
#     'experience',
#     'exploitation',
#     'export',
#     'exposure',
#     'extension',
#     'extract',
#     'eyesight',
#     'fact',
#     'faith',
#     'fallow',
#     'fame',
#     'fancy',
#     'farce',
#     'fascism',
#     'fashion',
#     'fat',
#     'fate',
#     'fault',
#     'fealty',
#     'fear',
#     'feed',
#     'feedback',
#     'fellowship',
#     'female genital mutilation',
#     'feminism',
#     'fencing',
#     'fermium',
#     'fiction',
#     'fill',
#     'film',
#     'finance',
#     'finish',
#     'fire',
#     'fish',
#     'flame',
#     'flavor',
#     'flavour',
#     'flerovium',
#     'flesh',
#     'flexibility',
#     'floccinaucinihilipilification',
#     'flour',
#     'flow',
#     'flower',
#     'focus',
#     'fog',
#     'food',
#     'football',
#     'forbearance',
#     'forethought',
#     'foundation',
#     'fresh blood',
#     'friendship',
#     'fruit',
#     'frustration',
#     'fudge',
#     'fuel',
#     'fun',
#     'function',
#     'funnies',
#     'fur',
#     'fusion',
#     'fuss',
#     'futurology',
#     'g-force',
#     'gadolinium',
#     'gaiety',
#     'gallium',
#     'gaming',
#     'garbage',
#     'garlic',
#     'gas',
#     'gaseousness',
#     'gear',
#     'gel',
#     'gender',
#     'generation',
#     'genetics',
#     'genocide',
#     'geography',
#     'geology',
#     'geometry',
#     'germanium',
#     'germination',
#     'gibberish',
#     'ginger',
#     'glamor',
#     'glamour',
#     'glass',
#     'gold',
#     'goo',
#     'goose egg',
#     'gore',
#     'gospel',
#     'government',
#     'grace',
#     'grammar',
#     'gramps',
#     'grass',
#     'gratitude',
#     'gravel',
#     'gravels',
#     'grease',
#     'greed',
#     'grief',
#     'gross pay',
#     'ground',
#     'guardian',
#     'gyneolatry',
#     'hafnium',
#     'hail',
#     'hair',
#     'ham',
#     'hamburger',
#     'handwriting',
#     'happiness',
#     'harm',
#     'harmony',
#     'harvest',
#     'hassium',
#     'haste',
#     'hate',
#     'hatred',
#     'hay',
#     'health',
#     'hearing',
#     'heart',
#     'heat',
#     'heating',
#     'heaven',
#     'helium',
#     'help',
#     'heritage',
#     'heroin',
#     'heroism',
#     'herpetology',
#     'hillbilly music',
#     'hippopotomonstrosesquipedaliophobia',
#     'histology',
#     'history',
#     'hoax',
#     'homecoming',
#     'homework',
#     'honesty',
#     'honey',
#     'hood',
#     'hoodoo',
#     'horror',
#     'horseradish',
#     'hostility',
#     'hour',
#     'house',
#     'humanity',
#     'humiliation',
#     'humour',
#     'hunter',
#     'hurt',
#     'hydrogen',
#     'hygienics',
#     'hyperbole',
#     'hypothesis',
#     'ice',
#     'ice cream',
#     'ice-cream',
#     'identification',
#     'ideology',
#     'imagination',
#     'immigration',
#     'immortality',
#     'impact',
#     'implementation',
#     'importance',
#     'impression',
#     'improvement',
#     'inaccuracy',
#     'inaccurateness',
#     'inattention',
#     'inceldom',
#     'incentive',
#     'income',
#     'incorrectness',
#     'independence',
#     'indigo',
#     'indium',
#     'individuality',
#     'industry',
#     'infinity',
#     'inflation',
#     'info',
#     'information',
#     'infrastructure',
#     'inhibition',
#     'initiative',
#     'injury',
#     'innocence',
#     'innovation',
#     'input',
#     'insecticide',
#     'insight',
#     'insolence',
#     'inspection',
#     'inspiration',
#     'integration',
#     'intelligence',
#     'intensity',
#     'interest',
#     'interests',
#     'interference',
#     'internet',
#     'interpretation',
#     'intrigue',
#     'investigation',
#     'investment',
#     'iridium',
#     'Irish',
#     'iron',
#     'irony',
#     'isolation',
#     'ivy',
#     'jack',
#     'jam',
#     'jealousy',
#     'jean',
#     'jeans',
#     'jelly',
#     'joy',
#     'judgment',
#     'judo',
#     'juice',
#     'junk',
#     'junkfood',
#     'justification',
#     'ketchup',
#     'kindheartedness',
#     'kindness',
#     'kinesiology',
#     'Klingon',
#     'knowledge',
#     'Korean',
#     'krypton',
#     'kung fu',
#     'kung-fu',
#     'labor',
#     'labour',
#     'lack',
#     'lager',
#     'land',
#     'lanthanum',
#     'laughter',
#     'laundry',
#     'lava',
#     'law',
#     'law of identity',
#     'lawrencium',
#     'laziness',
#     'lead',
#     'leather',
#     'left',
#     'legislation',
#     'lemonade',
#     'lettuce',
#     'leukemia',
#     'levity',
#     'light',
#     'lime',
#     'linen',
#     'linguistics',
#     'lip',
#     'lit',
#     'literature',
#     'litter',
#     'livermorium',
#     'livestock',
#     'load',
#     'lobster',
#     'logic',
#     'loneliness',
#     'loss',
#     'loudness',
#     'lubricant',
#     'luck',
#     'luggage',
#     'lumber',
#     'lunch',
#     'lung cancer',
#     'lutetium',
#     'lymph',
#     'magic',
#     'magma',
#     'magnification',
#     'mail',
#     'maintenance',
#     'maize',
#     'management',
#     'manga',
#     'manipulation',
#     'mankind',
#     'manufacture',
#     'Manx',
#     'marijuana',
#     'marketing',
#     'Mary Jane',
#     'master',
#     'material',
#     'maternity',
#     'math',
#     'mathematics',
#     'maths',
#     'matoke',
#     'matter',
#     'meantime',
#     'meanwhile',
#     'meat',
#     'mechanic',
#     'media',
#     'media studies',
#     'mediation',
#     'medicine',
#     'meitnerium',
#     'melody',
#     'membership',
#     'mendelevium',
#     'merchandise',
#     'mercury',
#     'mercy',
#     'metal',
#     'might',
#     'migration',
#     'milk',
#     'minute',
#     'misconception',
#     'misdemeanour',
#     'misery',
#     'misopedia',
#     'moderation',
#     'modesty',
#     'moisture',
#     'moldiness',
#     'molding',
#     'molybdenum',
#     'moment',
#     'money',
#     'monotheism',
#     'morale',
#     'morning',
#     'morphology',
#     'morphosyntax',
#     'moss',
#     'motherhood',
#     'motility',
#     'motion',
#     'motivation',
#     'movement',
#     'muck',
#     'mucus',
#     'mud',
#     'multiplication',
#     'murder',
#     'music',
#     'mustard',
#     'mystery',
#     'mythology',
#     'nachos',
#     'napalm',
#     'national anthem',
#     'nature',
#     'navy',
#     'neglect',
#     'negligence',
#     'neodymium',
#     'neon',
#     'nerve',
#     'nesting instinct',
#     'networking',
#     'neurology',
#     'neuroscience',
#     'New Age',
#     'news',
#     'newspaper',
#     'nickel',
#     'night',
#     'nitrogen',
#     'nitroglycerin',
#     'nobelium',
#     'noise',
#     'nomination',
#     'northwest',
#     'notice',
#     'notoriety',
#     'nudity',
#     'obedience',
#     'objection',
#     'oblation',
#     'observation',
#     'obsessiveness',
#     'occupation',
#     'oceanography',
#     'oceanology',
#     'off-season',
#     'offence',
#     'offense',
#     'offensive',
#     'offset',
#     'oil',
#     'oleo',
#     'ominousness',
#     'omission',
#     'omnipotence',
#     'onanism',
#     'oniochalasia',
#     'opacity',
#     'ophiolatry',
#     'opium',
#     'opposition',
#     'oral sex',
#     'orange juice',
#     'order',
#     'ordnance',
#     'organisation',
#     'organization',
#     'orientation',
#     'origami',
#     'osmium',
#     'osteosarcoma',
#     'output',
#     'overcrowding',
#     'oxygen',
#     'paintball',
#     'pandeism',
#     'panic',
#     'pantheism',
#     'paper',
#     'paperwork',
#     'papyrus',
#     'paralysis',
#     'participation',
#     'partnership',
#     'passion',
#     'past',
#     'paste',
#     'patience',
#     'patronage',
#     'pay',
#     'peace',
#     'pearl',
#     'pederasty',
#     'pedestrianisation',
#     'pedestrianization',
#     'pee',
#     'pegging',
#     'pellet',
#     'penetration',
#     'pepper',
#     'perception',
#     'perfection',
#     'performance',
#     'permission',
#     'persistence',
#     'perspective',
#     'persuasion',
#     'pesticide',
#     'petrichor',
#     'petrol',
#     'pharmacy',
#     'phase',
#     'philanthropy',
#     'philosophy',
#     'phobophobia',
#     'phonetics',
#     'phonology',
#     'photography',
#     'photosynthesis',
#     'physics',
#     'physiology',
#     'piss',
#     'pitch',
#     'pity',
#     'plagiarism',
#     'plastic',
#     'platinum',
#     'pleasantness',
#     'plenty',
#     'plutonium',
#     'poetic justice',
#     'poetry',
#     'poison',
#     'police',
#     'policy',
#     'polish',
#     'politeness',
#     'politics',
#     'pollen',
#     'pollution',
#     'polonium',
#     'polyglotism',
#     'polytheism',
#     'popularity',
#     'pork',
#     'pornography',
#     'portability',
#     'possession',
#     'post',
#     'pot',
#     'potassium',
#     'potential',
#     'poverty',
#     'powder',
#     'power',
#     'practice',
#     'praseodymium',
#     'precipitation',
#     'press',
#     'presumption',
#     'pretence',
#     'pretense',
#     'pride',
#     'principal',
#     'print',
#     'priority',
#     'privacy',
#     'privatisation',
#     'privatization',
#     'probability',
#     'procedure',
#     'produce',
#     'product',
#     'production',
#     'profanity',
#     'progeny',
#     'progress',
#     'promethium',
#     'promiscuity',
#     'promotion',
#     'pronunciation',
#     'propaganda',
#     'property',
#     'property tax',
#     'propulsion',
#     'prosperity',
#     'prostitution',
#     'protactinium',
#     'protein',
#     'providence',
#     'prowess',
#     'psephology',
#     'psychoanalysis',
#     'psychology',
#     'puberty',
#     'public',
#     'public transport',
#     'publication',
#     'publicity',
#     'pudding',
#     'puke',
#     'pulp',
#     'punch',
#     'punctuation',
#     'punishment',
#     'purification',
#     'purity',
#     'pwn',
#     'quantity',
#     'quickness',
#     'quiet',
#     'quinine',
#     'quotation',
#     'rabies',
#     'racism',
#     'radon',
#     'rancor',
#     'random access memory',
#     'rap',
#     'rape',
#     'reaction',
#     'real estate',
#     'reason',
#     'rebellion',
#     'recall',
#     'receipt',
#     'recess',
#     'reciprocity',
#     'recognition',
#     'recursion',
#     'reflection',
#     'refreshment',
#     'refuse',
#     'regard',
#     'regicide',
#     'registration',
#     'regulation',
#     'relativity',
#     'relaxation',
#     'reliance',
#     'relief',
#     'remedilessness',
#     'remorse',
#     'renal cell carcinoma',
#     'renovation',
#     'rent',
#     'repair',
#     'repetition',
#     'reproduction',
#     'research',
#     'reserve',
#     'resolution',
#     'respect',
#     'respiration',
#     'restlessness',
#     'restraint',
#     'retaliation',
#     'return',
#     'revenge',
#     'revenue',
#     'revision',
#     'revolution',
#     'rhenium',
#     'rhodium',
#     'rice',
#     'ridicule',
#     'ripeness',
#     'rivalry',
#     'rock',
#     'roentgenium',
#     'Romance',
#     'room',
#     'rot',
#     'rounders',
#     'rubbish',
#     'rubidium',
#     'rum',
#     'run time',
#     'Russian',
#     'rust',
#     'rustiness',
#     'ruthenium',
#     'rutherfordium',
#     'sack',
#     'sacking',
#     'sadism',
#     'sadness',
#     'sailing',
#     'sake',
#     'sale',
#     'sales',
#     'saliva',
#     'salt',
#     'salt water',
#     'salvation',
#     'samarium',
#     'sand',
#     'sandstone',
#     'santorum',
#     'satire',
#     'savings',
#     'sawdust',
#     'scandium',
#     'scantiness',
#     'scenery',
#     'schedule',
#     'school',
#     'Scientology',
#     'scope',
#     'scorn',
#     'scouting',
#     'seaborgium',
#     'segregation',
#     'selenium',
#     'selenolatry',
#     'semen',
#     'sentience',
#     'sentiency',
#     'sewage',
#     'sex',
#     'sexism',
#     'sexual intercourse',
#     'shadow',
#     'shelter',
#     'shine',
#     'shit',
#     'shorts',
#     'sickle-cell anaemia',
#     'silence',
#     'simplicity',
#     'sin',
#     'single',
#     'singles',
#     'sisterhood',
#     'sky',
#     'slang',
#     'slaughter',
#     'sleep',
#     'sleet',
#     'smoke',
#     'snow',
#     'soap',
#     'society',
#     'sod',
#     'soda',
#     'sodium',
#     'software',
#     'soil',
#     'somebody',
#     'sorrow',
#     'soul',
#     'soup',
#     'south',
#     'southeast',
#     'southwest',
#     'space',
#     'spam',
#     'sparkle',
#     'spawn',
#     'speech',
#     'sperm',
#     'spillage',
#     'spirit',
#     'spiritlessness',
#     'spirituality',
#     'sport',
#     'squash',
#     'stability',
#     'staff',
#     'stamina',
#     'starlight',
#     'statistics',
#     'stats',
#     'status',
#     'stock',
#     'storminess',
#     'stormlessness',
#     'strapping',
#     'strategy',
#     'straw',
#     'strawberry',
#     'strength',
#     'stress',
#     'sturdiness',
#     'submission',
#     'substitution',
#     'subway',
#     'success',
#     'sugar',
#     'suicide',
#     'sulfur',
#     'sunlight',
#     'sunrise',
#     'sunshade',
#     'sunshine',
#     'superposition',
#     'supply',
#     'surfing',
#     'suspicion',
#     'swear word',
#     'swing',
#     'symbiosis',
#     'symbololatry',
#     'symmetry',
#     'syntax',
#     'tackle',
#     'tag',
#     'talent',
#     'talk',
#     'tantalum',
#     'tapioca',
#     'tar',
#     'taste',
#     'tea',
#     'technetium',
#     'technique',
#     'techno',
#     'technobabble',
#     'technology',
#     'tellurium',
#     'temper',
#     'tempestuousness',
#     'tense',
#     'tension',
#     'terbium',
#     'termination',
#     'terrorism',
#     'text',
#     'thallium',
#     'thanks',
#     'theft',
#     'thelarche',
#     'Thelema',
#     'third person',
#     'third-person',
#     'thirst',
#     'thorium',
#     'thought',
#     'throw up',
#     'thulium',
#     'thunder',
#     'thunderousness',
#     'tile',
#     'time',
#     'tin',
#     'tinfoil',
#     'tissue',
#     'toast',
#     'tobacco',
#     'tolerance',
#     'tomato',
#     'tosh',
#     'totalitarianism',
#     'track',
#     'trade',
#     'tradition',
#     'traffic',
#     'transcendence',
#     'transformation',
#     'transition',
#     'translation',
#     'transmission',
#     'transportation',
#     'trap',
#     'trauma',
#     'treachery',
#     'treasure',
#     'treatment',
#     'trial',
#     'trinitrotoluene',
#     'tripe',
#     'trust',
#     'truth',
#     'turkey',
#     'type',
#     'uncompetitiveness',
#     'underclothing',
#     'undergrowth',
#     'underpants',
#     'understandability',
#     'understanding',
#     'unhappiness',
#     'uniform',
#     'universe',
#     'unobtainium',
#     'ununbium',
#     'ununhexium',
#     'ununoctium',
#     'ununpentium',
#     'ununquadium',
#     'ununseptium',
#     'ununtrium',
#     'upkeep',
#     'upset',
#     'uranium',
#     'urgency',
#     'usability',
#     'usefulness',
#     'utility',
#     'utilization',
#     'validity',
#     'vanilla',
#     'vanity',
#     'vapor',
#     'vapour',
#     'variation',
#     'variety',
#     'veal',
#     'vegetarianism',
#     'velocity',
#     'vengeance',
#     'ventilation',
#     'vibration',
#     'victory',
#     'video',
#     'viola',
#     'violation',
#     'violence',
#     'violet',
#     'virginity',
#     'virtuosity',
#     'visibility',
#     'vision',
#     'vocabulary',
#     'volcanicity',
#     'volleyball',
#     'voltage',
#     'volume',
#     'vomit',
#     'voodoo',
#     'voyeurism',
#     'waffle',
#     'wallpaper',
#     'wank',
#     'want',
#     'warmth',
#     'water',
#     'wax',
#     'weather',
#     'web',
#     'weight',
#     'welfare',
#     'well-being',
#     'west',
#     'wheat',
#     'wholesale',
#     'wilderness',
#     'wildlife',
#     'wind',
#     'windbaggery',
#     'wine',
#     'winter',
#     'wire',
#     'wood',
#     'woods',
#     'wool',
#     'Worcestershire sauce',
#     'work',
#     'wretchedness',
#     'xenon',
#     'yogurt',
#     'youth',
#     'ytterbium',
#     'yttrium',
#     'zealousness',
#     'zest',
#     'zilch',
#     'zinc',
#     'zirconium',
#     'zodiac',
#     'zoology',
#     'art', 
#     'furniture', 
#     'love', 
#     'series', 
#     'sheep', 
#     'species', 
#     'travel',
#      'home',
#               )

# def _irregular(singular: str, plural: str) -> None:
#     """
#     A convenience function to add appropriate rules to plurals and singular
#     for irregular words.

#     :param singular: irregular word in singular form
#     :param plural: irregular word in plural form
#     """
#     def caseinsensitive(string: str) -> str:
#         return ''.join('[' + char + char.upper() + ']' for char in string)

#     if singular[0].upper() == plural[0].upper():
#         PLURALS.insert(0, (
#             r"(?i)({}){}$".format(singular[0], singular[1:]),
#             r'\1' + plural[1:]
#         ))
#         PLURALS.insert(0, (
#             r"(?i)({}){}$".format(plural[0], plural[1:]),
#             r'\1' + plural[1:]
#         ))
#         SINGULARS.insert(0, (
#             r"(?i)({}){}$".format(plural[0], plural[1:]),
#             r'\1' + singular[1:]
#         ))
#     else:
#         PLURALS.insert(0, (
#             r"{}{}$".format(singular[0].upper(),
#                             caseinsensitive(singular[1:])),
#             plural[0].upper() + plural[1:]
#         ))
#         PLURALS.insert(0, (
#             r"{}{}$".format(singular[0].lower(),
#                             caseinsensitive(singular[1:])),
#             plural[0].lower() + plural[1:]
#         ))
#         PLURALS.insert(0, (
#             r"{}{}$".format(plural[0].upper(), caseinsensitive(plural[1:])),
#             plural[0].upper() + plural[1:]
#         ))
#         PLURALS.insert(0, (
#             r"{}{}$".format(plural[0].lower(), caseinsensitive(plural[1:])),
#             plural[0].lower() + plural[1:]
#         ))
#         SINGULARS.insert(0, (
#             r"{}{}$".format(plural[0].upper(), caseinsensitive(plural[1:])),
#             singular[0].upper() + singular[1:]
#         ))
#         SINGULARS.insert(0, (
#             r"{}{}$".format(plural[0].lower(), caseinsensitive(plural[1:])),
#             singular[0].lower() + singular[1:]
#         ))


# def camelize(string: str, uppercase_first_letter: bool = True) -> str:
#     """
#     Convert strings to CamelCase.

#     Examples::

#         >>> camelize("device_type")
#         'DeviceType'
#         >>> camelize("device_type", False)
#         'deviceType'

#     :func:`camelize` can be thought of as a inverse of :func:`underscore`,
#     although there are some cases where that does not hold::

#         >>> camelize(underscore("IOError"))
#         'IoError'

#     :param uppercase_first_letter: if set to `True` :func:`camelize` converts
#         strings to UpperCamelCase. If set to `False` :func:`camelize` produces
#         lowerCamelCase. Defaults to `True`.
#     """
#     if uppercase_first_letter:
#         return re.sub(r"(?:^|_)(.)", lambda m: m.group(1).upper(), string)
#     else:
#         return string[0].lower() + camelize(string)[1:]


# def dasherize(word: str) -> str:
#     """Replace underscores with dashes in the string.

#     Example::

#         >>> dasherize("puni_puni")
#         'puni-puni'

#     """
#     return word.replace('_', '-')


# def humanize(word: str) -> str:
#     """
#     Capitalize the first word and turn underscores into spaces and strip a
#     trailing ``"_id"``, if any. Like :func:`titleize`, this is meant for
#     creating pretty output.

#     Examples::

#         >>> humanize("employee_salary")
#         'Employee salary'
#         >>> humanize("author_id")
#         'Author'

#     """
#     word = re.sub(r"_id$", "", word)
#     word = word.replace('_', ' ')
#     word = re.sub(r"(?i)([a-z\d]*)", lambda m: m.group(1).lower(), word)
#     word = re.sub(r"^\w", lambda m: m.group(0).upper(), word)
#     return word


# def ordinal(number: int) -> str:
#     """
#     Return the suffix that should be added to a number to denote the position
#     in an ordered sequence such as 1st, 2nd, 3rd, 4th.

#     Examples::

#         >>> ordinal(1)
#         'st'
#         >>> ordinal(2)
#         'nd'
#         >>> ordinal(1002)
#         'nd'
#         >>> ordinal(1003)
#         'rd'
#         >>> ordinal(-11)
#         'th'
#         >>> ordinal(-1021)
#         'st'

#     """
#     number = abs(int(number))
#     if number % 100 in (11, 12, 13):
#         return "th"
#     else:
#         return {
#             1: "st",
#             2: "nd",
#             3: "rd",
#         }.get(number % 10, "th")


# def ordinalize(number: int) -> str:
#     """
#     Turn a number into an ordinal string used to denote the position in an
#     ordered sequence such as 1st, 2nd, 3rd, 4th.

#     Examples::

#         >>> ordinalize(1)
#         '1st'
#         >>> ordinalize(2)
#         '2nd'
#         >>> ordinalize(1002)
#         '1002nd'
#         >>> ordinalize(1003)
#         '1003rd'
#         >>> ordinalize(-11)
#         '-11th'
#         >>> ordinalize(-1021)
#         '-1021st'

#     """
#     return "{}{}".format(number, ordinal(number))


# def parameterize(string: str, separator: str = '-') -> str:
#     """
#     Replace special characters in a string so that it may be used as part of a
#     'pretty' URL.

#     Example::

#         >>> parameterize(u"Donald E. Knuth")
#         'donald-e-knuth'

#     """
#     string = transliterate(string)
#     # Turn unwanted chars into the separator
#     string = re.sub(r"(?i)[^a-z0-9\-_]+", separator, string)
#     if separator:
#         re_sep = re.escape(separator)
#         # No more than one of the separator in a row.
#         string = re.sub(r'%s{2,}' % re_sep, separator, string)
#         # Remove leading/trailing separator.
#         string = re.sub(r"(?i)^{sep}|{sep}$".format(sep=re_sep), '', string)

#     return string.lower()


# def pluralize(word: str) -> str:
#     """
#     Return the plural form of a word.

#     Examples::

#         >>> pluralize("posts")
#         'posts'
#         >>> pluralize("octopus")
#         'octopi'
#         >>> pluralize("sheep")
#         'sheep'
#         >>> pluralize("CamelOctopus")
#         'CamelOctopi'

#     """
#     if not word or word.lower() in UNCOUNTABLES:
#         return word
#     else:
#         for rule, replacement in PLURALS:
#             if re.search(rule, word):
#                 return re.sub(rule, replacement, word)
#         return word


# def singularize(word: str) -> str:
#     """
#     Return the singular form of a word, the reverse of :func:`pluralize`.

#     Examples::

#         >>> singularize("posts")
#         'post'
#         >>> singularize("octopi")
#         'octopus'
#         >>> singularize("sheep")
#         'sheep'
#         >>> singularize("word")
#         'word'
#         >>> singularize("CamelOctopi")
#         'CamelOctopus'

#     """
#     for inflection in UNCOUNTABLES:
#         if re.search(r'(?i)\b(%s)\Z' % inflection, word):
#             return word

#     for rule, replacement in SINGULARS:
#         if re.search(rule, word):
#             return re.sub(rule, replacement, word)
#     return word


# def tableize(word: str) -> str:
#     """
#     Create the name of a table like Rails does for models to table names. This
#     method uses the :func:`pluralize` method on the last word in the string.

#     Examples::

#         >>> tableize('RawScaledScorer')
#         'raw_scaled_scorers'
#         >>> tableize('egg_and_ham')
#         'egg_and_hams'
#         >>> tableize('fancyCategory')
#         'fancy_categories'
#     """
#     return pluralize(underscore(word))


# def titleize(word: str) -> str:
#     """
#     Capitalize all the words and replace some characters in the string to
#     create a nicer looking title. :func:`titleize` is meant for creating pretty
#     output.

#     Examples::

#       >>> titleize("man from the boondocks")
#       'Man From The Boondocks'
#       >>> titleize("x-men: the last stand")
#       'X Men: The Last Stand'
#       >>> titleize("TheManWithoutAPast")
#       'The Man Without A Past'
#       >>> titleize("raiders_of_the_lost_ark")
#       'Raiders Of The Lost Ark'

#     """
#     return re.sub(
#         r"\b('?\w)",
#         lambda match: match.group(1).capitalize(),
#         humanize(underscore(word)).title()
#     )


# def transliterate(string: str) -> str:
#     """
#     Replace non-ASCII characters with an ASCII approximation. If no
#     approximation exists, the non-ASCII character is ignored. The string must
#     be ``unicode``.

#     Examples::

#         >>> transliterate('älämölö')
#         'alamolo'
#         >>> transliterate('Ærøskøbing')
#         'rskbing'

#     """
#     normalized = unicodedata.normalize('NFKD', string)
#     return normalized.encode('ascii', 'ignore').decode('ascii')


# def underscore(word: str) -> str:
#     """
#     Make an underscored, lowercase form from the expression in the string.

#     Example::

#         >>> underscore("DeviceType")
#         'device_type'

#     As a rule of thumb you can think of :func:`underscore` as the inverse of
#     :func:`camelize`, though there are cases where that does not hold::

#         >>> camelize(underscore("IOError"))
#         'IoError'

#     """
#     word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
#     word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
#     word = word.replace("-", "_")
#     return word.lower()

# if __name__ == '__main__':
#     _irregular('person', 'people')
#     _irregular('man', 'men')
#     _irregular('human', 'humans')
#     _irregular('child', 'children')
#     _irregular('sex', 'sexes')
#     _irregular('move', 'moves')
#     _irregular('cow', 'kine')
#     _irregular('zombie', 'zombies')



#     print(pluralize("posts")) #'posts'
#     print(pluralize("octopus")) #'octopi'
#     print(pluralize("sheep")) #'sheep'
#     print(pluralize("CamelOctopus"))  #'CamelOctopi'