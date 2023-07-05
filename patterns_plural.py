
def_sap_patterns = {"IsA": [
                            "[X] are [Y].",
                            "[X] are a type of [Y].",
                            "[X] are a kind of [Y]."
                            ]}



def_dap_patterns = {"IsA": [
                            "[X] or [Z] are [Y].",
                            "[X] or [Z] are a type of [Y].",
                            "[X] or [Z] are a kind of [Y].",
                            ]}



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