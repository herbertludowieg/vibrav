import numpy as np

def dataframe_to_txt(df, columns=None, ncols=4, fp=None, float_format='{:10.6E}'.format):
    # TODO: implement the columns parameter for more generalization
    nrows = int(df.shape[1]/ncols)
    idx = 0
    text = ''
    if isinstance(float_format, list) and nrows == 1:
        formatters = float_format
    elif isinstance(float_format, list) and nrows > 1:
        raise NotImplementedError("Have not yet implemented support for having column " \
                                  +"specific formatters when we have more than one row " \
                                  +"to write.")
    else:
        formatters = [float_format]*ncols
    if nrows == 1:
        text += df.to_string(formatters=formatters)
    else:
        while idx < nrows:
            to_print = range(idx*ncols, idx*ncols+ncols)
            tmp = df[df.columns[to_print]]
            text += tmp.to_string(formatters=formatters)
            text += '\n\n'
            idx += 1
        else:
            formatters = [float_format]*int(df.shape[1] - idx*ncols)
            to_print = range(idx*ncols, df.shape[1])
            tmp = df[df.columns[to_print]]
            text += tmp.to_string(formatters=formatters)
    if fp is not None:
        with open(fp, 'w') as fn:
            fn.write(text)
    else:
        return text

