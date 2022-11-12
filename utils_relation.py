
relations = [
    {
        "relation": "HasProperty",
        "template_sap": "[X] are [Y] ." ,
        "use_dap": False,
        "target_anchor": "[X]",
        "template_anchor": [
            f"[X] and [MASK] have similar properties .",
            f"[X] and [MASK] share similar properties .",
            f"[X] and [MASK] are similar .",
            f"[X] and [MASK] have similar meanings .",
            f"[X] have a similar meaning as [MASK] .",
            f"[X] and [MASK] are synonyms .",
            f"[X] and [MASK] are synonymous .",
            f'[X] and [MASK] mean the same thing .',
            f'[X] and [MASK] are the same thing .',
            f'[X] means the same thing as [MASK] .'
            ],
        "dap_x": "[X] and [Z] are [Y] .",
        "dap_y": "[X] and [Z] are [Y] .",
    }
]


relations_old=[
    {
        "relation": "Causes",
        "template_sap": "It is typical for [X] to cause [Y] .",
        "use_dap": False,
        "target_anchor": "[X]",
        "template_anchor": ", including [X] and [MASK] .",
        "dap_x": "It is typical for [X] and [Z] to cause [Y] .",
        "dap_y": "It is typical for [X] to cause [Y] or [Z] .",
    },
    {
        "relation": "Desires",
        "template_sap": "[X] desire to [Y] .",
        "use_dap": False,
        "target_anchor": "[X]",
        "template_anchor": "Such as [X] and [MASK] .",
        "dap_x": "[X] or [Z] desires to [Y] .",
        "dap_y": "[X] desires to [Y] or [X] .",
    },
    {
        "relation": "HasA",
        "template_sap": "Usually, we would expect [X] to have [Y] .",
        "use_dap": False,
        "target_anchor": "[X]",
        "template_anchor": "Such as [X] and [MASK] .",
        # "template_anchor": "Such [X] as [MASK] .",
        "template_anchor": "Such [X] as [MASK] .",
        "dap_x": "Usually, we would expect [X] or [Z] to have [Y] .",
        "dap_y": "Usually, we would expect [X] to have [Y] or [Z].",
    },
    {
        "relation": "HasPrerequisite",
        "template_sap": "In order to [X], one must first [Y] .",
        "use_dap": False,
        "target_anchor": "[Y]",
        "template_anchor": "Such as [X] and [MASK] .",
        "dap_x": "In order to [X] or [Z], one must first [Y] .",
        "dap_y": "In order to [X], one must first [Y] or [Z] ."
    },
    {
        "relation": "HasSubevent",
        "template_sap": "You [Y] when you [X] .",
        "use_dap": False,
        "target_anchor": "[Y]",
        "template_anchor": "Such as [X] and [MASK] .",
        "dap_x": "You [Y] when you [X] and [Z] .",
        "dap_y": "You [Y] when you [X] or [Z] .",
    },
    {
        "relation": "CapableOf",
        "template_sap": "[X] can [Y] .",
        "use_dap": False,
        "target_anchor": "[X]",
        "template_anchor": "A [X] is a type of [MASK] .",
        "dap_x": "[X] and [Z] can [Y] .",
        "dap_x": "[X] can [Y] or [Z] .",
    },
    {
        "relation": "ReceivesAction",
        "template_sap": "The [X] can be [Y] .",
        "use_dap": False,
        "target_anchor": "[X]",
        "template_anchor": "A [X] is a type of [MASK] .",
        "dap_x": "The [X] or [Z] can be [Y] .",
        "dap_y": "The [X] can be [Y] or [Z] .",
    },
    {
        "relation": "HasProperty",
        "template_sap": "[X] are [Y] .",
        "use_dap": False,
        "target_anchor": "[X]",
        "template_anchor": "[X] and [MASK] have similar features .",
        "dap_x": "[X] and [Z] are [Y] .",
        "dap_y": "[X] and [Z] are [Y] .",
    },
    {
        "relation": "MotivatedByGoal",
        "template_sap": "Someone [X]s in order to [Y] .",
        "use_dap": True,
        "target_anchor": "[X]",
        "template_anchor": "[X] or [MASK] .",
        "dap_x": "Someone [X]s or [Z]s in order to [Y] .",
        "dap_y": "Someone [X]s in order to [Y] or [Z]s.",
    },
    {
        "relation": "AtLocation",
        "template_sap": "[X] are found in [Y] .",
        "use_dap": True,
        "target_anchor": "[X]",
        "template_anchor": "Such as [X] and [MASK] .",
        "dap_x": "[X] or [Z] are found in [Y] .",
        "dap_y": "",
    },
    {
        "relation": "CausesDesire",
        "template_sap": "The [X] makes people want to [Y]r .",
        "use_dap": True,
        "target_anchor": "[X]",
        "template_anchor": "Such as [X] and [MASK] .",
        "dap_x": "The [X] and [Z] make people want to [Y]r .",
        "dap_y": "",
    },
    {
        "relation": "UsedFor",
        "template_sap": "The [X] is used for [Y] .",
        "use_dap": True,
        "target_anchor": "[X]",
        "template_anchor": "Such as [X] and [MASK] .",
        "dap_x": "The [X] and [Z] are used for [Y] .",
        "dap_y": "",
    },
    {
        "relation": "IsA",
        "template_sap": "A [X] is a type of [Y] .",
        "use_dap": True,
        "target_anchor": "[X]",
        "template_anchor": "Such as [X] and [MASK] .",
        "dap_x": "A [X] and [Z] are a type of [Y] .",
        "dap_y": "",
    },
    {
        "relation": "MadeOf",
        "template_sap": "[X] is made of [Y] .",
        "use_dap": True,
        "target_anchor": "[X]",
        "template_anchor": "Such as [X] and [MASK] .",
        "dap_x": "[X] or [Z] is made of [Y] .",
        "dap_y": "",
    },
    {
        "relation": "PartOf",
        "template_sap": "[X] is a part of [Y] .",
        "use_dap": True,
        "target_anchor": "[X]",
        "template_anchor": "Such as [X] and [MASK] .",
        "dap_x": "[X] and [Z] are  part of [Y] .",
        "dap_y": "",
    }
]
