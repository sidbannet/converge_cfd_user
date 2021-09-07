"""
Tool kit for data manipulation.

@author: siddhartha.banerjee
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import warnings


class AttrDict(dict):
    """dict to attributed class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class MetaDataFrame(pd.DataFrame):
    """panda dataframe with metadata"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
        )
        self._unit = {}
        self._desc = {}

    def set_index(
            self,
            keys,
            drop=True,
            append=False,
            inplace=False,
            verify_integrity=False,
    ) -> MetaDataFrame:
        """Set index for the CFD dataframe."""
        super().set_index(
            keys=keys,
            drop=drop,
            append=append,
            inplace=inplace,
            verify_integrity=verify_integrity,
        )
        dataframe = self.copy()
        dataframe = dataframe.set_index(
            keys=keys,
            drop=drop,
            append=append,
            inplace=inplace,
            verify_integrity=verify_integrity,
        )
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
        )
        setattr(dataframe, 'unit_', self.unit_)
        setattr(dataframe, 'desc_', self.desc_)
        return dataframe

    def sort_values(
            self,
            by,
            axis=0,
            ascending=True,
            inplace=False,
            kind='quicksort',
            na_position='last',
    ) -> MetaDataFrame:
        """Sort values using a row"""
        super().sort_values(
            by=by,
            axis=axis,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            na_position=na_position,
        )
        dataframe = self.copy()
        dataframe = dataframe.sort_values(
            by=by,
            axis=axis,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            na_position=na_position,
        )
        warnings.filterwarnings(
            action="ignore",
            category=UserWarning,
        )
        setattr(dataframe, 'unit_', self.unit_)
        setattr(dataframe, 'desc_', self.desc_)
        return dataframe

    def append_with(
            self,
            df: pd.DataFrame = None,
    ) -> None:
        """Appending dataframe to itself without changing id()."""
        for row in df.iterrows():
            try:
                self.loc[row[0]] = row[1]
            except ValueError:
                pass

    @property
    def unit_(self):
        return self._unit

    @property
    def desc_(self):
        return self._desc

def MergeInterpolate(
        first_dataframe: pd.DataFrame = pd.DataFrame(),
        second_dataframe: pd.DataFrame = pd.DataFrame(),
        field_name: str = None,
) -> pd.DataFrame:
    t1 = first_dataframe.index
    t2 = second_dataframe.index
    y1 = first_dataframe[field_name]
    y2 = second_dataframe[field_name]
    second_interp_to_first = np.interp(
        x = t1,
        xp = t2,
        fp = y2,
    )
    df = pd.DataFrame(
        {
            'first': y1,
            'second': pd.Series(second_interp_to_first, index=t1),
        }
    )
    return df
