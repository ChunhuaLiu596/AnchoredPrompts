
class DataFrame2Latex(object):
    def __init__(self, df, label, caption, output_file, adjustbox_width, precision, column_format=None, multicolumn_format=None, hide_index=False):
        self.df = df
        self.label = label 
        self.caption = caption
        self.output_file = output_file
        self.adjustbox = adjustbox_width
        self.precision = precision
        self.column_format = column_format
        self.multicolumn_format = multicolumn_format

        self.convert_df_to_latex_table(self.df, label=self.label, caption=self.caption, 
                                       output_file=self.output_file, adjustbox_width=adjustbox_width, 
                                       precision=precision, column_format=self.column_format, 
                                       multicolumn_format=self.multicolumn_format,
                                       hide_index = hide_index
                                      )


    def add_span_columns(self, table):
        ''' 
        add * after table so that the table can span two columns
        '''
        table_lines = table.split("\n")
        table_lines[0] = table_lines[0].replace("table", "table*")
        table_lines[-2] = table_lines[-2].replace("table", "table*")
        return "\n".join(table_lines)

    def move_caption_to_bottom(self, table):
        '''
        the default caption position is on the top, this function shifts it to the bottom
        '''
        table_lines = table.split("\n")
        caption_line_num, end_tabular_line_num = None, None 
        caption_line = None

        for i, line in enumerate(table_lines):
            if line.startswith(r"\caption"):
                caption_line_num = i 
                caption_line = line
            if line.startswith(r'\end{tabular}'):
                end_tabular_line_num = i
        print(caption_line_num)
        print(end_tabular_line_num)

        table_lines.insert(end_tabular_line_num+1, caption_line)
        del table_lines[caption_line_num]
        return "\n".join(table_lines)

    def get_line_with_startswith(self, table_lines, start_str):
        line_num = None
        for i, line in enumerate(table_lines):
            if line.startswith(start_str):
                line_num  = i 
        return line_num


    def write_latex_table(self, table_latex, output_file):
        ''' 
        table_latex: table represented in a list of string
        '''
        with open(output_file, 'w', encoding='utf-8') as fout:
            for line in table_latex:
                fout.write(line)

        print(f"saved {output_file}")

    def adjust_box(self, table_latex, width='columnwidth'):
        ''' 
        This function add the addjust box outside the tabular

        TODO: fill with more options for width
        '''
        table_lines =  table_latex.split("\n")
        if width == 'columnwidth':
            adjust_box_begin = r"\begin{adjustbox}{width=\columnwidth}"

        if width == 'textwidth':
            adjust_box_begin = r"\begin{adjustbox}{width=\textwidth}"

        adjust_box_end = r" \end{adjustbox}"

        tabular_start_line_num = self.get_line_with_startswith(table_lines, r'\begin{tabular}')
        tabular_end_line_num = self.get_line_with_startswith(table_lines, r'\end{tabular}')

        table_lines.insert(tabular_end_line_num +1, adjust_box_end)
        table_lines.insert(tabular_start_line_num-1, adjust_box_begin)

        return "\n".join(table_lines)

    def convert_df_to_latex_table(self, df, label, caption, output_file=None, adjustbox_width='textwidth', span_two_columns=True, bold_max=True, precision=0, column_format=None, multicolumn_format=None, hide_index=False):
        table_string = df.style.highlight_max(
            # props='cellcolor:[HTML]{FFFF00}; color:{red};'
            # props= 'textit:--rwrap; 
            props='textbf:--rwrap;'
        ).format(precision=precision)
        if hide_index:
            table_string = table_string.hide(axis='index')

        if column_format is None: 
            column_format = 'l'* len(df.columns)
            
        if multicolumn_format is None:
            multicolumn_format = 'l'

        table_latex = table_string.to_latex(
            column_format=column_format,
            multicol_align=multicolumn_format,
            position="!h", 
            position_float="centering",
            hrules=True, 
            multirow_align="t", 
            caption=caption,
            label=label, 
        )  

        if  span_two_columns:
            table_latex = self.add_span_columns(table_latex)

        # if caption is not None:
        table_latex = self.move_caption_to_bottom(table_latex)
        table_latex = self.adjust_box(table_latex, width=adjustbox_width)
        
        if output_file !=None:
            self.write_latex_table(table_latex, output_file)
        print(table_latex)
        
        return table_latex 

