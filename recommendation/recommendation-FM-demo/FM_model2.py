from sklearn import  preprocessing;
import pandas as pd
import numpy as np


from collections import defaultdict
import csv
import datetime
from io import StringIO
import re
import sys
from textwrap import fill
from typing import Any, Dict, Set
import warnings

import numpy as np

import pandas._libs.lib as lib
import pandas._libs.ops as libops
import pandas._libs.parsers as parsers
from pandas._libs.tslibs import parsing
from pandas.errors import (
    AbstractMethodError,
    EmptyDataError,
    ParserError,
    ParserWarning,
)
from pandas.util._decorators import Appender

from pandas.core.dtypes.cast import astype_nansafe
from pandas.core.dtypes.common import (
    ensure_object,
    ensure_str,
    is_bool_dtype,
    is_categorical_dtype,
    is_dtype_equal,
    is_extension_array_dtype,
    is_float,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_object_dtype,
    is_scalar,
    is_string_dtype,
    pandas_dtype,
)
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.missing import isna

from pandas._typing import FilePathOrBuffer
from pandas.core import algorithms
from pandas.core.arrays import Categorical
from pandas.core.frame import DataFrame
from pandas.core.index import Index, MultiIndex, RangeIndex, ensure_index_from_sequences
from pandas.core.series import Series
from pandas.core.tools import datetimes as tools

from pandas.io.common import (
    _NA_VALUES,
    BaseIterator,
    UnicodeReader,
    UTF8Recoder,
    _get_handle,
    _infer_compression,
    _validate_header_arg,
    get_filepath_or_buffer,
    is_file_like,
)
from pandas.io.date_converters import generic_parser

cols = ['user','item','rating','timestamp']
"加载数据"
train = pd.read_csv('data/ua.base',delimiter='\t',names = cols)
test = pd.read_csv('data/ua.test',delimiter='\t',names = cols)
pd.read_csv
enc =preprocessing.OneHotEncoder;

enc.fit([train["user"],train["item"]]);

train1=enc.transform(train)

print(train1.todense())
