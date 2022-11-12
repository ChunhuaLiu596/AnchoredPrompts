relations = [
    {
        "relation": "AtLocation",
        "template_sap": "[X] are found in [Y] ."
    },
    {
        "relation": "CapableOf",
        "template_sap": "[X] can [Y] ."
    },
    {
        "relation": "Causes",
        "template_sap": "It is typical for [X] to cause [Y] ."
    },
    {
        "relation": "CausesDesire",
        "template_sap": "The [X] makes people want to [Y]r ."
    },
    {
        "relation": "Desires",
        "template_sap": "[X] desire to [Y] ."
    },
    {
        "relation": "HasA",
        "template_sap": "Usually, we would expect [X] to have [Y] ."
    },
    {
        "relation": "HasPrerequisite",
        "template_sap": "In order to [X], one must first [Y] ."
    },
    {
        "relation": "HasProperty",
        "template_sap": "[X] are [Y] ."
    },
    {
        "relation": "HasSubEvent",
        "template_sap": "You [Y] when you [X] ."
    },
    {
        "relation": "IsA",
        "template_sap": "A [X] is a type of [Y] ."
    },
    {
        "relation": "MadeOf",
        "template_sap": "[X] is made of [Y] ."
    },
    {
        "relation": "MotivatedByGoal",
        "template_sap": "Someone [X]s in order to [Y] ."
    },
    {
        "relation": "PartOf",
        "template_sap": "[X] is a part of [Y] ."
    },
    {
        "relation": "ReceivesAction",
        "template_sap": "The [X] can be [Y] ."
    },
    {
        "relation": "UsedFor",
        "template_sap": "The [X] is used for [Y]ing ."
    }
]


# let's assumet the anchor concept is [Z] 
new_relations = []
for relation in relations:
    relation['template_sap'] = relation['template_sap'].replace("[Y]", "[MASK]")
    dap_temp = {"template_dap": relation['template_sap'].replace("[X]", "[Z] or [X]")}
    relation.update(dap_temp)
    new_relations.append(relation)

import json 
save_dict_to_json(new_relations, 'log/relation_sap_dap.json')
# json.dump(x, open('log/relation_sap_dap.json', 'w'), indent=4)
