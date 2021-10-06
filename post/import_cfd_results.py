"""
Tools for importing CFD results.

@author: siddhartha.banerjee
"""

import numpy as np
import pandas as pd
import os
from pandas.io.parsers import TextFileReader
from tqdm import tqdm
from tool.data import MetaDataFrame as CFDDataFrame
from tool.data import AttrDict as CFDDict

LOADCHAR = r'\|/-'


class FileNameFmt:
    """Helper class to breakdown file name"""

    def __init__(
            self,
            file_fmt: str = None,
    ):
        """Intantiate the class."""
        assert (type(file_fmt) is str), 'input argument should be str'
        assert ('*' in file_fmt), 'input argument should have a *'
        file_fmt = file_fmt.replace('.out', '')
        self._file_category_name = file_fmt.split('*')[0].split('#')[-1]
        self._is_file_fmt_has_domain = '#' in file_fmt
        self._is_file_fmt_has_3D_data = '$' in file_fmt
        if self._is_file_fmt_has_domain:
            self._file_domain_type = file_fmt.split('#')[0].split('*')[-1]
        else:
            self._file_domain_type = None

    def is_subdomain_file(
            self,
            file_name: str = None,
    ) -> bool:
        """Answers is the file a domain specific file?"""
        file_name = file_name.replace('.out', '')
        if self._is_file_fmt_has_domain:
            return self._file_domain_type in file_name
        else:
            return False

    def id_subdomain_file(
            self,
            file_name: str = None,
    ):
        """Sub-domain ID number is returned."""
        if self.is_subdomain_file(file_name=file_name):
            iloc = file_name.find(self._file_domain_type) \
                + len(self._file_domain_type)
            try:
                return int(file_name[iloc:].split('_')[0])
            except ValueError:
                return int(file_name[iloc:].split('-')[0])
        else:
            return None

    @property
    def file_category(self):
        return self._file_category_name

    @property
    def file_domain_type(self):
        try:
            return self._file_domain_type.split('_')[-1]
        except AttributeError:
            return None


def organize_cfd_results(
        folder_name: str = None,
        append_folder_name: str = None,
        file_fmt: FileNameFmt = None,
        sorter: str = None,
        indexer: str = None,
):
    """Give dict with CFD data from a given file category of CFD output"""
    if file_fmt.file_domain_type is not None:
        cfd_dict = CFDDict(
            {
                'all': [],
                file_fmt.file_domain_type: [],
            }
        )
    else:
        cfd_dict = CFDDict({})
    files = []
    files_append = []
    num_reg = []
    file_category = file_fmt.file_category
    for _, _, f in os.walk(folder_name):
        for file in f:
            if '.out' in file and file_category in file:
                files.append(file.replace('.out', ''))
    if append_folder_name is not None:
        for _, _, f in os.walk(append_folder_name):
            for file in f:
                if '.out' in file and file_category in file:
                    files_append.append(file.replace('.out', ''))
    for f in files:
        if file_fmt.is_subdomain_file(file_name=f):
            num_reg.append(
                int(file_fmt.id_subdomain_file(file_name=f))
            )
    cfd_data = []
    try:
        cfd_data_reg = [[] for i in range(max(num_reg) + int(1))]
    except ValueError:
        cfd_data_reg = None
    print('Loading ' + file_category + ' files: ', end=' ')
    iload = int(-1)
    for file in files:
        iload += int(1)
        if not file_fmt.is_subdomain_file(file_name=file):
            file_name = file
            print('\b'
                  + LOADCHAR[np.mod(iload, len(LOADCHAR))], end='')
            cfd_data.append(
                __import_cfd_timeseries_result(
                    folder_name=folder_name,
                    file_name=file_name,
                )
            )
        elif file_fmt.is_subdomain_file(file_name=file):
            file_name = file
            print('\b'
                  + LOADCHAR[np.mod(iload, len(LOADCHAR))], end='')
            reg_num = int(file_fmt.id_subdomain_file(file_name=file))
            try:
                cfd_data_reg[reg_num] = pd.concat(
                    [
                        cfd_data_reg[reg_num],
                        __import_cfd_timeseries_result(
                            folder_name=folder_name,
                            file_name=file_name,
                        )
                    ],
                    axis=0,
                    sort=False,
                )
            except TypeError:
                cfd_data_reg[reg_num] = __import_cfd_timeseries_result(
                    folder_name=folder_name, file_name=file_name)
                __unit = cfd_data_reg[reg_num].unit_
        else:
            continue
    if append_folder_name is not None:
        for file in files_append:
            iload += int(1)
            if not file_fmt.is_subdomain_file(file_name=file):
                file_name = file
                print('\b'
                      + LOADCHAR[np.mod(iload, len(LOADCHAR))], end='')
                cfd_data.append(
                    __import_cfd_timeseries_result(
                        folder_name=append_folder_name,
                        file_name=file_name,
                    )
                )
            elif file_fmt.is_subdomain_file(file_name=file):
                file_name = file
                print('\b'
                      + LOADCHAR[np.mod(iload, len(LOADCHAR))], end='')
                reg_num = int(file_fmt.id_subdomain_file(file_name=file))
                try:
                    cfd_data_reg[reg_num] = pd.concat(
                        [
                            cfd_data_reg[reg_num],
                            __import_cfd_timeseries_result(
                                folder_name=append_folder_name,
                                file_name=file_name,
                            )
                        ],
                        axis=0,
                        sort=False,
                    )
                except TypeError:
                    cfd_data_reg[reg_num] = __import_cfd_timeseries_result(
                        folder_name=append_folder_name, file_name=file_name)
            else:
                continue
    try:
        cfd = pd.concat(
            [data for data in cfd_data],
            axis=0,
            sort=False,
        )
    except ValueError:
        cfd = CFDDataFrame([])
    try:
        cfd = CFDDataFrame(cfd.sort_values(by=sorter))
    except KeyError:
        pass
    try:
        cfd = CFDDataFrame(cfd.set_index(keys=indexer))
    except ValueError:
        pass
    except KeyError:
        pass
    try:
        cfd._unit = cfd_data[0].unit_
        cfd._desc = cfd_data[0].desc_
    except IndexError:
        pass

    for ireg in np.unique(num_reg):
        try:
            cfd_data_reg[ireg] = CFDDataFrame(
                cfd_data_reg[ireg].sort_values(by=sorter)
            )
        except KeyError:
            pass
        try:
            cfd_data_reg[ireg] = \
                CFDDataFrame(cfd_data_reg[ireg].set_index(keys=indexer))
        except ValueError:
            pass
        try:
            cfd_data_reg[ireg]._unit = cfd_data[0].unit_
            cfd_data_reg[ireg]._desc = cfd_data[0].desc_
        except IndexError:
            cfd_data_reg[ireg]._unit = __unit

    if 'all' in cfd_dict:
        cfd_dict['all'] = cfd
        cfd_dict[file_fmt.file_domain_type] = cfd_data_reg
        return cfd_dict
    else:
        return cfd


def import_cfd(
        folder_name: str = None,
        file_category: str = None,
        file_type: str = 'region',
        sorter: str = None,
        indexer: str = None,
) -> CFDDict:
    """Give dict with CFD data from a given file category of CFD output"""
    try:
        cfd_dict = CFDDict(
            {
                'all': [],
                file_type: [],
            }
        )
    except ValueError:
        cfd_dict = CFDDict({'all': []})
    files = []
    num_reg = []
    for _, _, f in os.walk(folder_name):
        for file in f:
            if '.out' in file and file_category in file:
                files.append(str.split(file.replace('.out', ''), '_'))
    for f in files:
        if len(f) == 2:
            num_reg.append(
                int(str.replace(f[1], file_type, ''))
            )
    cfd_data = []
    cfd_data_reg = [[] for i in range(max(num_reg) + int(1))]
    print('Loading ' + file_category + ' files: ', end=' ')
    iload = int(-1)
    for file in files:
        iload += int(1)
        if file.__len__() == 1:
            file_name = file[0]
            print('\b'
                  + LOADCHAR[np.mod(iload, len(LOADCHAR))], end='')
            cfd_data.append(
                __import_data(
                    folder_name=folder_name, file_name=file_name)
            )
        elif file.__len__() == 2:
            file_name = file[0] + '_' + file[1]
            print('\b'
                  + LOADCHAR[np.mod(iload, len(LOADCHAR))], end='')
            reg_num = int(str.replace(file[1], file_type, ''))
            try:
                cfd_data_reg[reg_num] = pd.concat(
                    [cfd_data_reg[reg_num],
                     __import_data(
                         folder_name=folder_name, file_name=file_name)],
                    axis=0,
                    sort=False,
                )
            except TypeError:
                cfd_data_reg[reg_num] = __import_data(
                    folder_name=folder_name, file_name=file_name)
        else:
            continue

    cfd = pd.concat(
        [data for data in cfd_data],
        axis=0,
        sort=False,
    )
    try:
        cfd = CFDDataFrame(cfd.sort_values(by=sorter))
    except KeyError:
        pass
    try:
        cfd = CFDDataFrame(cfd.set_index(keys=indexer))
    except ValueError:
        pass
    cfd._unit = cfd_data[0].unit_
    cfd._desc = cfd_data[0].desc_

    for ireg in np.unique(num_reg):
        try:
            cfd_data_reg[ireg] = CFDDataFrame(
                cfd_data_reg[ireg].sort_values(by=sorter)
            )
        except KeyError:
            pass
        try:
            cfd_data_reg[ireg] = \
                CFDDataFrame(cfd_data_reg[ireg].set_index(keys=indexer))
        except ValueError:
            pass
        cfd_data_reg[ireg]._unit = cfd_data[0].unit_
        cfd_data_reg[ireg]._desc = cfd_data[0].desc_

    cfd_dict['all'] = cfd
    cfd_dict[file_type] = cfd_data_reg
    return cfd_dict


def __import_data(
        folder_name: str = None,
        file_name: str = None,
        header: list = [0, 1],
        skiprows: list = [0, 1, 4],
) -> CFDDataFrame:
    """Give panda data frame for the CFD output file."""
    folder = r'' + folder_name
    cfd_data_file = folder + os.sep + file_name + '.out'
    raw_data: TextFileReader = pd.read_csv(
        filepath_or_buffer=cfd_data_file,
        header=header,
        skiprows=skiprows,
    )
    columns = str.split(raw_data.columns[0][0][1:])
    units = str.split(raw_data.columns[0][1][1:])
    ncol = columns.__len__()
    nrow = raw_data.values.__len__()
    values = np.empty([nrow, ncol])
    for irow, row in enumerate(raw_data.values):
        for icol, _ in enumerate(columns):
            values[irow][icol] = np.float(
                (str.split(row[0], ))[icol]
            )
    df = CFDDataFrame([])
    metadata = {}
    for icolumn, column in enumerate(columns):
        df[column] = values[:, icolumn]
        metadata[column] = units[icolumn]
    df._unit = CFDDict(metadata)
    return df


def __import_cfd_timeseries_result(
        folder_name: str = None,
        file_name: str = None,
        skiprows: list = [0, 1],
        header: list = [0, 1, 2],
) -> CFDDataFrame:
    """Give panda data frame for the CFD output file."""
    folder = r'' + folder_name
    cfd_data_file = folder + os.sep + file_name + '.out'
    raw_data: TextFileReader
    try:
        raw_data = pd.read_csv(
            filepath_or_buffer=cfd_data_file,
            header=header,
            skiprows=skiprows,
        )
    except pd.errors.ParserError:
        raw_data = pd.read_csv(
            filepath_or_buffer=cfd_data_file,
            header=header,
            skiprows=skiprows,
            escapechar=',',
        )
    columns = str.split(raw_data.columns.levels[0].values[0][1:], )
    subcolumns = str.split(raw_data.columns.levels[2].values[0][1:], )
    units = str.split(raw_data.columns.levels[1].values[0][1:], )
    ncol = columns.__len__()
    nrow = raw_data.values.__len__()
    values = np.empty([nrow, ncol])
    for irow, row in enumerate(raw_data.values):
        for icol, _ in enumerate(columns):
            try:
                values[irow][icol] = np.float(
                    (str.split(row[0], ))[icol]
                )
            except (IndexError, ValueError):
                values[irow][icol] = np.nan
    metadata = {}
    if subcolumns.__len__() != 0:
        col_arr = [columns, subcolumns]
        columns_tup = list(zip(*col_arr))
        head_idx = pd.MultiIndex.from_tuples(
            tuples=columns_tup,
            names=['header', 'subheader'],
        )
        df = CFDDataFrame(values, columns=head_idx)
    else:
        df = CFDDataFrame(
            values,
            columns=columns,
        )
    for icolumn, column in enumerate(columns):
        metadata[column] = units[icolumn]
    df._unit = CFDDict(metadata)
    return df


class ImportCFDResult:
    """Class to import CFD results."""

    def __init__(
            self,
            proj_dir: str = None,
            proj_name: str = None,
    ):
        """Instantiate the class."""
        self.proj_dir = proj_dir
        self.proj_name = proj_name
        self.data_timeseries = None
        self.data_3d = {}
        self.processed_scav_3d = {}
        self._loaded_timeseries = False
        self._loaded_3d = False
        self._got_processed_3d_scav = False
        self._got_processed_3d_cyl_flow_3d = False

    def load_timeseries(
            self,
            file_category: str = None,
            file_type: str = 'region',
            time_sorter: str = 'Crank',
            indexer: str = 'Crank',
            parsing_func=import_cfd,
    ) -> None:
        """Import time series based results."""
        # //todo: Do assert checks on the parameters
        # //todo: Write a docstring and give examples
        folder_name = self.proj_dir + os.sep + self.proj_name
        self.data_timeseries = parsing_func(
            folder_name=folder_name,
            file_category=file_category,
            file_type=file_type,
            sorter=time_sorter,
            indexer=indexer,
        )
        self._loaded_timeseries = True

    def load_cfd3d(
        self,
        parsing_function=pd,
    ) -> None:
        """Import *.col files from 3D CFD results."""
        folder_name = \
            self.proj_dir + os.sep \
            + self.proj_name + os.sep + \
            'output'
        all_files = [
            f for f in os.listdir(
                folder_name
            ) if os.path.isfile(
                os.path.join(folder_name, f)
            )
        ]
        col_files = [
            file for file in all_files if '.col' in file[-4:]
        ]
        t = tqdm(total=len(col_files))
        for file in col_files:
            file_name = folder_name + os.sep + file
            with open(file_name, "r") as colfile:
                crank_time = colfile.readline().split()[0]
                self.data_3d[crank_time] = \
                    parsing_function.read_csv(
                        filepath_or_buffer=file_name,
                        header=[0],
                        skiprows=[0],
                        delim_whitespace=True,
                )
                t.update()
        t.close()
        self._loaded_3d = True

    def get_processed_scav_3d(
        self,
        cyl_axis: str = 'z',
        density: str = 'density',
        volume: str = 'volume',
        intake_scalar: str = 'INT',
        residual_scalar: str = 'CYL',
        exhaust_scalar: str = 'EXH',
    ) -> None:
        """Get the scavenging front analyzed."""
        assert self._loaded_3d, "3D data not loaded yet."
        cumulative_sum = {
            'cumINT': intake_scalar,
            'cumCYL': residual_scalar,
            'cumEXH': exhaust_scalar,
        }
        for key, value in self.data_3d.items():
            self.processed_scav_3d[key] = value[
                [cyl_axis, density, volume, intake_scalar,
                    residual_scalar, exhaust_scalar]
            ].sort_values(
                by=cyl_axis, ascending=True)
            for k, v in cumulative_sum.items():
                self.processed_scav_3d[key][k] = (
                    self.processed_scav_3d[key][v] *
                    self.processed_scav_3d[key][density] *
                    self.processed_scav_3d[key][volume]
                ).cumsum(axis=0) / (
                    self.processed_scav_3d[key][density] *
                    self.processed_scav_3d[key][volume]
                ).cumsum(axis=0)
        self._got_processed_3d_scav = True

    def get_processed_cyl_flow_3d(
        self,
        cyl_x: str = 'x',
        cyl_y: str = 'y',
        cyl_u: str = 'u',
        cyl_v: str = 'v',
    ) -> tuple:
        """Get in-cylinder flow analysis using 3d CFD data."""
        assert self._loaded_3d, "3d data not loaded yet."

        def transform(x, y, u, v): return (
            np.sqrt(x ** 2.0 + y ** 2.0),
            np.arctan2(y, x),
            u * x / np.sqrt(
                x ** 2 + y ** 2
            ) + v * y / np.sqrt(
                x ** 2 + y ** 2
            ),
            - u * y / np.sqrt(
                x ** 2 + y ** 2
            ) + v * x / np.sqrt(
                x ** 2 + y ** 2
            ),
        )
        t = tqdm(total=self.data_3d.__len__())
        for data3d in self.data_3d.values():
            data3d['r'],\
                data3d['theta'],\
                data3d['V_r'],\
                data3d['V_theta'] = np.vectorize(
                transform
            )(
                x=data3d[cyl_x],
                y=data3d[cyl_y],
                u=data3d[cyl_u],
                v=data3d[cyl_v],
            )
            t.update()
        t.close()
        self._got_processed_3d_cyl_flow_3d = True
