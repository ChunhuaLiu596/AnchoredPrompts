
def_sap_patterns={"IsA": [
    "[X] is a [Y].",
    "[X] is a type of [Y].",
    "[X] is a kind of [Y].",
]}

# def_dap_patterns = {"IsA": [
#     "[X] is a [Y]. So is [Z].",
#     "[X] is a type of [Y]. So is [Z].",
#     "[X] is a kind of [Y]. So is [Z].",
# ]}

def_dap_patterns = {"IsA": [
    "[X] or [Z] is a [Y].",
    "[X] or [Z] is a type of [Y].",
    "[X] or [Z] is a kind of [Y]."
]}

# def_sap_patterns = {"IsA": [
#                             # "[X] is a type of [Y].", #works mostly for plural
#                             # "[X] is [Y].",
#                             # "A [X] is [Y].",
#                             "A [X] is a [Y].",
#                             "An [X] is a [Y].",
#                             # "An [X] is an [Y].",
#                             # "A [X] is a kind of [Y].", #didn't bring improvement
#                             # "A [X] is a type of [Y].",
#                             "A [X] is a type of [Y].",
#                             "An [X] is a type of [Y].",
#                             "A [X] is a kind of [Y].",
#                             "An [X] is a kind of [Y].",
#                             # "An [X] is a type of [Y].",
#                             # "An [X] is a kind of [Y].",
#                             # "The [X] is [Y].", #is is ambiguoure and works mostly for plural
#                             # "Some [X] are [Y]." # most cues in CLSB is singular
#                             ]}



# def_dap_patterns = {"IsA": [
#                             # "[X] or [Z] is a type of [Y].",
#                             # "[X] or [Z] is [Y].",
#                             # "A [X] or [Z] is [Y].",
#                             "A [X] or [Z] is a [Y].", #removed for CLSB
#                             "A [X] or [Z] is an [Y].",
#                             # "A [X] or [Z] is a kind of [Y].",
#                             # "A [X] or [Z] is a type of [Y].",
#                             "An [X] or [Z] is a [Y].",
#                             "An [X] or [Z] is an [Y].", #removed for CLSB
#                             # "An [X] or [Z] is a type of [Y].",
#                             # "An [X] or [Z] is a kind of [Y].",
#                             # "The [X] or [Z] is [Y].",
#                             # "Some [X] or [Z] are [Y]." # most cues in CLSB is singular
#                             ]}


lsp_sap_patterns = {
    "IsA": [
         "[Y] such as [X].", 
         "[Y], including [X].", 
         "[Y], especially [X].", 
         "[X] or other [Y].", 
         "[X] and other [Y].", 
         "such [Y] as [X].", 
         # "[Y] (e.g., [X] )", 
        #  "[Y], such as [X]", 
        ], 
    "IsA_EXTENDED": [
                "[Y] such as [X], ", 
                "[Y], such as [X], ", 
                "[Y], such as [X] and  ", 
                "[Y], such as [X] or ", 
                "such [Y] as [X], ", 
                ", [X] or other [Y]", 
                ", [X] and other [Y]", 
                "[Y], including [X], ", 
                "[Y], including [X] and", 
                "[Y], including [X] or", 
                "[Y], especially [X], ", 
                "[Y], especially [X] and ", 
                "[Y], especially [X] or ", 
                # "[Y] (e.g., [X] )", 
            ], 
    "Coordinate": [
        "such as [X] and [Y].",
        "including [X] and [Y].",
        "especially [X] and [Y].",
        "[X], [Y] or other", 
        "[X], [Y] and other" 
        # "as [X] and [Y]",
    ],
    "Coordinate_keep_Y": [
        '[Z] such as [X] and [Y].',
        'such [Z] as [X] and [Y].',
        '[X], [Y] or other [Z].',
        '[X], [Y] and other [Z].',
        '[Z], including [X] and [Y].',
        '[Z], especially [X] and [Y].'
        ],
    "Coordinate_remove_Y": [
        'such as [X] and [Y].',
        'such as [X] and [Y].',
        '[X], [Y] or other.',
        '[X], [Y] and other.',
        ', including [X] and [Y].',
        ', especially [X] and [Y].'],
    "Coordinate_remove_Y_PUNC": [
        'such as [X] and [Y].',
        'including [X] and [Y].',
        'especially [X] and [Y].',
        '[X], [Y] or other',
        '[X], [Y] and other',
        ],
    "Coordinate_repl_Y_CLS_SEP": [
        '[CLS] such as [X] and [Y].',
        'such [CLS] as [X] and [Y].',
        '[X], [Y] or other [SEP].',
        '[X], [Y] and other [SEP].',
        '[CLS], including [X] and [Y].',
        '[CLS], especially [X] and [Y].'
        ],
    "Coordinate_remove_Y_PUNC_FULL": [
        'such as [X] and [Y].',
        'such as [X] or [Y].',
        'such as [X], [Y],',
        'including [X] and [Y].',
        'including [X] or [Y].',
        'including [X], [Y],',
        'especially [X] and [Y].',
        'especially [X] or [Y].',
        'especially [X], [Y],',
        '[X], [Y] or other',
        '[X], [Y] and other',
        ],
    # Coordinate_FULL=[
    #     'such as [X] and [Y]',
    #     'such as [X] or [Y]',
    #     'such as [X], [Y],',
    #     # 'such as [X], [Y] and',
    #     # 'such as [X], [Y] or',
    #     'including [X] and [Y]',
    #     'including [X] or [Y]',
    #     'including [X], [Y],',
    #     # 'including [X], [Y] and',
    #     # 'including [X], [Y] or',
    #     'especially [X] and [Y]',
    #     'especially [X] or [Y]',
    #     'especially [X], [Y],',
    #     # 'especially [X], [Y] and',
    #     # 'especially [X], [Y] or',
    #     "[X], [Y] or other", 
    #     "[X], [Y] and other",  
    # ],
    "Coordinate_OLD" : [
                "such as [X] and [Y],", 
                "such as [X] or [Y],", 
                "such as [X], [Y],", 
                "such as [X], [Y], and", 
                ", such as [X] and [Y],", 
                ", such as [X] or [Y], ", 
                "as [X] and [Y].", 
                "as [X] or [Y].", 
                "as [X], [Y],", 
                "[X], [Y] or other", 
                "[X], [Y] and other", 
                ", including [X] and [Y]", 
                ", including [X] or [Y]", 
                ", including [X], [Y],", 
                ", including [X], [Y], and", 
                ", including [X], [Y], or", 
                ", especially [X] and [Y],", 
                ", especially [X] or [Y],", 
                ", especially [X], [Y],", 
                ", especially [X], [Y], and", 
            ]
}

lsp_dap_patterns = {
    "IsA": [
         "[Y] such as [X] and [Z].", 
         "[Y], including [X] and [Z].", 
         "[Y], especially [X] and [Z].", 
         "[X], [Z] and other [Y].", 
         "[X], [Z] or other [Y].", 
         "such [Y] as [X] and [Z].", 
         # "[Y] (e.g., [X] )", 
        ],
    "IsA_FULL": [
         "[Y] such as [X] and [Z].", 
         "[Y], including [X] and [Z].", 
         "[Y], especially [X] and [Z].", 
         "[X], [Z] and other [Y].", 
         "such [Y] as [X] and [Z].", 
         "[Y] such as [X] or [Z].", 
         "[Y], including [X] or [Z].", 
         "[Y], especially [X] or [Z].", 
         "[X], [Z] or other [Y].", 
         "such [Y] as [X] or [Z].", 
         "[X], [Z] or other [Y].", 
         # "[Y] (e.g., [X] )", 
        ],
    "IsA_EXTENDED": [
                    "[Y] such as [X] and [Z]",
                    "[Y] such as [X] or [Z]",
                    "such [Y] as [X] and [Z]",
                    "such [Y] as [X] or [Z]",
                    "[X], [Z] or other [Y]",
                    "[X], [Z] and other [Y]",
                    "[Y] including [X] and [Z]",
                    "[Y] including [X] or [Z]",
                    "[Y], especially [X] and [Z] ",
                    "[Y], especially [X] or [Z] ",
    ]}






