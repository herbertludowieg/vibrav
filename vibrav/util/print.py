# This file is part of vibrav.
#
# vibrav is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# vibrav is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with vibrav.  If not, see <https://www.gnu.org/licenses/>.
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

