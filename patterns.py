
def_sap_patterns = {"IsA": [
                            # "[X] is a type of [Y].", #works mostly for plural
                            # "[X] is [Y].",
                            # "A [X] is [Y].",
                            "A [X] is a [Y].",
                            "A [X] is an [Y].",
                            # "A [X] is a kind of [Y].", #didn't bring improvement
                            # "A [X] is a type of [Y].",
                            "An [X] is a [Y].",
                            "An [X] is an [Y].",
                            # "An [X] is a type of [Y].",
                            # "An [X] is a kind of [Y].",
                            # "The [X] is [Y].", #is is ambiguoure and works mostly for plural
                            # "Some [X] are [Y]." # most cues in CLSB is singular
                            ]}



def_dap_patterns = {"IsA": [
                            # "[X] or [Z] is a type of [Y].",
                            # "[X] or [Z] is [Y].",
                            # "A [X] or [Z] is [Y].",
                            "A [X] or [Z] is a [Y].", #removed for CLSB
                            "A [X] or [Z] is an [Y].",
                            # "A [X] or [Z] is a kind of [Y].",
                            # "A [X] or [Z] is a type of [Y].",
                            "An [X] or [Z] is a [Y].",
                            "An [X] or [Z] is an [Y].", #removed for CLSB
                            # "An [X] or [Z] is a type of [Y].",
                            # "An [X] or [Z] is a kind of [Y].",
                            # "The [X] or [Z] is [Y].",
                            # "Some [X] or [Z] are [Y]." # most cues in CLSB is singular
                            ]}


lsp_sap_patterns = {
    "IsA": [
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
                "[Y], especillay [X], ", 
                "[Y], especillay [X] and ", 
                "[Y], especillay [X] or ", 
                "[Y] (e.g., [X] )", 
            ], 
    "Coordinate" : [
                "such as [X] and [Y],", 
                "such as [X] or [Y],", 
                "such as [X], [Y],", 
                "such as [X], [Y], and", 
                ", such as [X] and [Y],", 
                ", such as [X] or [Y], ", 
                "as [X] and [Y]", 
                "as [X] or [Y]", 
                "as [X], [Y],", 
                "[X], [Y] or other", 
                "[X], [Y] and other", 
                ", including [X] and [Y]", 
                ", including [X] or [Y]", 
                ", including [X], [Y],", 
                ", including [X], [Y], and", 
                ", including [X], [Y], or", 
                ", especially [X] and [Y] ,", 
                ", especially [X] or [Y] ,", 
                ", especially [X], [Y] ,", 
                ", especially [X], [Y] , and", 
            ]
}

lsp_dap_patterns = {"IsA": [
                        "[Y] such as [X] and [Z]", 
                        "such [Y] as [X] and [Z]", 
                        "[X], [Z] or other [Y]", 
                        "[X], [Z] and other [Y]", 
                        "[Y] including [X], [Z]", 
                        "[Y], especillay [X], [Z] "
                ]}


