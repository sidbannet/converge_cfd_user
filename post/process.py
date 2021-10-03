"""
Tools to process CFD results.

@author: siddhartha.banerjee
"""

from tool.data import AttrDict as CFDDict
from tool.data import MetaDataFrame as CFDDataFrame
from tool.plot import get_port_grid_lines as GetGridLine
from post.import_cfd_results import FileNameFmt as filefmt
from post.import_cfd_results import organize_cfd_results as cfdread
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

LOADCHAR = r'\|/-'


class Case:
    """Class containing CFD properties and methods."""

    def __init__(
        self,
        proj_dir:str = None,
        proj_name:str = None,
    ):
        """Instantiate the class."""

        self.result_dir = proj_dir + os.sep + proj_name
        self.cyc_freq = self._get_echo_info(
            file_name='engine.echo',
            eng_info='rpm',
        ) / 60.0
        try:
            self._version = int(
                self._get_sim_info(
                    file_name='engine.echo',
                    eng_info='version',
                    info_pos=int(1)
                ).split(sep='.')[0]
            )
        except ValueError:
            self._version = int(-1)
        # Version specific regions_flow header specification
        if self._version >= 3:
            self._header = {
                'rate': 'Rate_',
                'tot': 'Tot_',
            }
        else:
            self._header = {
                'rate': 'Rate-',
                'tot': 'Tot-',
            }

        self._loaded_data = False
        self._got_processed = False
        self._appended_with_other = False
        self._appending_index = []

        self.file_category = CFDDict(
            {
                'region_based': [
                    {'thermo': filefmt('thermo*_region#')},
                    {'turbulence': filefmt('turbulence*_region#')},
                    {'temperature': filefmt('temperature*_region#')},
                    {'react_ratio_bin': filefmt('react_ratio_bin*_region#')},
                    {'passive': filefmt('passive*_region#')},
                    {'mixing': filefmt('mixing*_region#')},
                    {'equiv_ratio_bin': filefmt('equiv_ratio_bin*_region#')},
                    {'cell_count_region': filefmt('cell_count_regions*')},
                    {'dynamic': filefmt('dynamic*_region#')},
                ],
                'rank_based': [
                    {'cell_count_ranks': filefmt('cell_count_ranks*')},
                ],
                'boundary_based': [
                    {'bound': filefmt('bound#-wall*')},
                ],
                'other_based': [
                    {'time': filefmt('time*')},
                    {'memory_usage': filefmt('memory_usage*')},
                    {'amr': filefmt('amr*')},
                ],
                'flow_based': [
                    {'regions_flow': filefmt('regions_flow*')},
                    {'mass_avg_flow': filefmt('mass_avg_flow*')},
                    {'area_avg_flow': filefmt('area_avg_flow*')},
                ],
                'monitor_point_based': [
                    {'mon_pt': filefmt('monitor*_point_#_mass_avg')}
                ]
            }
        )

    def _get_echo_info(
        self,
        file_name: str = 'engine.echo',
        eng_info: str = 'rpm',
        info_pos: int = 0,
    ):
        """Get simulation run parameter from the echo file."""

        line_txt = []
        with open(
            self.result_dir + os.sep + file_name, "r"
        ) as fp:
            for line in fp:
                line_txt.append(line)
        fp.close()
        param = line_txt[
            [
                inum
                for inum, ltxt in enumerate(line_txt)
                if eng_info in ltxt
            ][0]
        ].split()
        try:
            eng_param = float(param[info_pos])
        except ValueError:
            eng_param = float(param[info_pos + 1])
        except:
            eng_param = None

        return eng_param

    def _get_sim_info(
        self,
        file_name: str = 'engine.echo',
        eng_info: str = 'version',
        info_pos: int = int(1),
    ) -> str:
        """Get simulation run parameter from the echo file."""

        line_txt = []
        with open(
            self.result_dir + os.sep + file_name, "r"
        ) as fp:
            for line in fp:
                line_txt.append(line)
        fp.close()
        try:
            param = line_txt[
                [
                    inum
                    for inum, ltxt in enumerate(line_txt)
                    if eng_info in ltxt
                ][0]
            ].split()
        except IndexError:
            return ''
        try:
            eng_param = str(param[info_pos])
        except:
            eng_param = ''

        return eng_param

    def load_cfd_data(
            self,
            append_dir: str = None,
    ) -> None:
        """Load data from CFD out files."""

        if 'region_based' in self.file_category.keys():
            for out_type in self.file_category.region_based:
                try:
                    setattr(
                        self,
                        list(out_type)[0],
                        cfdread(
                            folder_name=self.result_dir,
                            append_folder_name=append_dir,
                            file_fmt=out_type[list(out_type)[0]],
                            indexer='Crank',
                            sorter='Crank',
                        )
                    )
                except (ValueError, KeyError):
                    setattr(
                        self,
                        list(out_type)[0],
                        cfdread(
                            folder_name=self.result_dir,
                            append_folder_name=append_dir,
                            file_fmt=out_type[list(out_type)[0]],
                            indexer=('Crank', '(none)'),
                            sorter=('Crank', '(none)'),
                        )
                    )
                print(' ... Done.')
            print('====================')

        if 'boundary_based' in self.file_category.keys():
            for out_type in self.file_category.boundary_based:
                setattr(
                    self,
                    list(out_type)[0],
                    cfdread(
                        folder_name=self.result_dir,
                        append_folder_name=append_dir,
                        file_fmt=out_type[list(out_type)[0]],
                        indexer='Crank',
                        sorter='Crank',
                    )
                )
                print(' ... Done.')
            print('====================')

        if 'flow_based' in self.file_category.keys():
            for out_type in self.file_category.flow_based:
                tmp = cfdread(
                    folder_name=self.result_dir,
                    append_folder_name=append_dir,
                    file_fmt=out_type[list(out_type)[0]],
                    sorter=('Crank', '(none)'),
                )
                tmp_modified = CFDDataFrame(
                    tmp.set_index(keys=('Crank', '(none)'))
                )
                tmp_modified._unit = tmp.unit_
                tmp_modified.index.name = 'Crank'
                setattr(
                    self,
                    list(out_type)[0],
                    tmp_modified,
                )
                print(' ... Done.')
            print('====================')

        if 'monitor_point_based' in self.file_category.keys():
            for out_type in self.file_category.monitor_point_based:
                setattr(
                    self,
                    list(out_type)[0],
                    cfdread(
                        folder_name=self.result_dir,
                        append_folder_name=append_dir,
                        file_fmt=out_type[list(out_type)[0]],
                        indexer='Crank',
                        sorter='Crank',
                    )
                )
                print(' ... Done.')
            print('====================')

        if 'rank_based' in self.file_category.keys():
            for out_type in self.file_category.rank_based:
                try:
                    setattr(
                        self,
                        list(out_type)[0],
                        cfdread(
                            folder_name=self.result_dir,
                            append_folder_name=append_dir,
                            file_fmt=out_type[list(out_type)[0]],
                            indexer='Crank',
                            sorter='Crank',
                        )
                    )
                except (ValueError, KeyError):
                    setattr(
                        self,
                        list(out_type)[0],
                        cfdread(
                            folder_name=self.result_dir,
                            append_folder_name=append_dir,
                            file_fmt=out_type[list(out_type)[0]],
                            indexer=('Crank', '(none)'),
                            sorter=('Crank', '(none)'),
                        )
                    )
                print(' ... Done.')
            print('====================')

        if 'other_based' in self.file_category.keys():
            for out_type in self.file_category.other_based:
                try:
                    setattr(
                        self,
                        list(out_type)[0],
                        cfdread(
                            folder_name=self.result_dir,
                            append_folder_name=append_dir,
                            file_fmt=out_type[list(out_type)[0]],
                            indexer='Crank',
                            sorter='Crank',
                        )
                    )
                except (ValueError, KeyError):
                    setattr(
                        self,
                        list(out_type)[0],
                        cfdread(
                            folder_name=self.result_dir,
                            append_folder_name=append_dir,
                            file_fmt=out_type[list(out_type)[0]],
                            indexer=('Crank', '(none)'),
                            sorter=('Crank', '(none)'),
                        )
                    )
                print(' ... Done.')
            print('====================')

        self._loaded_data = True

    def append_cfd_data(
            self,
            *args,
    ) -> None:
        """Append CFD results together."""
        assert self._loaded_data, "Data not loaded yet."
        iload = int(-1)
        print('Appending CFD cases: ', end=' ')
        for arg in args:
            assert arg._loaded_data, "Load data before appending."
        for file_sys in self.file_category:
            for file_type in self.file_category[file_sys]:
                for file_key in file_type:
                    try:
                        try:
                            _unit = self.__getattribute__(
                                file_key).unit_
                            _desc = self.__getattribute__(
                                file_key).desc_
                        except AttributeError:
                            _unit = None
                            _desc = None
                        value = self.__getattribute__(file_key)
                        for arg in args:
                            try:
                                self._appending_index.append(
                                    value.index[-1]
                                )
                            except (AttributeError, IndexError):
                                pass
                            value.append_with(
                                arg.__getattribute__(file_key))
                            iload += int(1)
                            print('\b' \
                                  + LOADCHAR[np.mod(iload, len(LOADCHAR))],
                                  end='')
                        value._unit = _unit
                        value._desc = _desc
                    except (TypeError, AttributeError):
                        for file_key_type in self.__getattribute__(
                                file_key):
                            value = self.__getattribute__(
                                file_key)[file_key_type]
                            try:
                                _unit = self.__getattribute__(
                                    file_key)[file_key_type].unit_
                                _desc = self.__getattribute__(
                                    file_key)[file_key_type].desc_
                            except AttributeError:
                                _unit = None
                                _desc = None
                            try:
                                value._unit = _unit
                                value._desc = _desc
                                for arg in args:
                                    try:
                                        self._appending_index.append(
                                            value.index[-1]
                                        )
                                    except (AttributeError, IndexError):
                                        pass
                                    value.append_with(
                                        arg.__getattribute__(
                                            file_key
                                        )[file_key_type]
                                    )
                                    iload += int(1)
                                    print('\b' \
                                          + LOADCHAR[
                                              np.mod(iload, len(LOADCHAR))
                                          ],
                                          end='')
                                value._unit = _unit
                                value._desc = _desc
                            except AttributeError:
                                for id, v in enumerate(value):
                                    if type(v) is list:
                                        continue
                                    try:
                                        _unit = v.unit_
                                        _desc = v.desc_
                                    except AttributeError:
                                        _unit = None
                                        _desc = None
                                    for arg in args:
                                        try:
                                            self._appending_index.append(
                                                v.index[-1]
                                            )
                                        except (AttributeError, IndexError):
                                            pass
                                        v.append_with(
                                            arg.__getattribute__(
                                                file_key
                                            )[file_key_type][id]
                                        )
                                        iload += int(1)
                                        print('\b' \
                                              + LOADCHAR[
                                                  np.mod(iload, len(LOADCHAR))
                                              ],
                                              end='')
                                    v._unit = _unit
                                    v._desc = _desc
        print(' \n ... Done ...')
        self._appending_index = np.unique(self._appending_index)
        self._appended_with_other = True

    def __cumulative_sum(
            self,
            x: CFDDataFrame = None,
    ) -> CFDDataFrame:
        """Static method to cumulative sum data."""
        for idx in self._appending_index:
            if x.ndim == 2:
                for name, data in x.iteritems():
                    d = x[name][x.index == idx].values - \
                        x[name][x.index > idx].values[0]
                    a = np.add(data[x.index > idx].values, d)
                    a = np.append(
                        data[x.index <= idx].values,
                        a
                    )
                    x.loc[:, [name]] = np.array([a]).T
            elif x.ndim == 1:
                d = x[x.index == idx].values - x[x.index > idx].values[0]
                a = np.add(x[x.index > idx].values, d)
                a = np.append(
                    x[x.index <= idx].values,
                    a
                )
                x.loc[:] = (np.array([a]).T).reshape([a.size])
        return x


class SimpleCase(Case):
    """Reduced data from the CFD simulation."""

    def __init__(
        self,
        proj_dir: str = None,
        proj_name: str = None,
    ):
        """Instantiate the subclass."""

        super().__init__(
            proj_dir=proj_dir,
            proj_name=proj_name
        )

        self.file_category = CFDDict(
            {
                'region_based': [
                    {'thermo': filefmt('thermo*_region#')},
                    {'turbulence': filefmt('turbulence*_region#')},
                    {'passive': filefmt('passive*_region#')},
                    {'dynamic': filefmt('dynamic*_region#')},
                    {'emissions': filefmt('emissions*_region#')},
                    {'soot': filefmt('soot_hiroy*_region#')},
                    {'species': filefmt('species_mass*_region#')},
                    {'mixing': filefmt('mixing*_region#')},
                ],
                'boundary_based': [
                ],
                'flow_based': [
                ],
                'monitor_point_based': [
                ],
                'other_based': [
                    {'time': filefmt('time*')},
                    {'memory_usage': filefmt('memory_usage*')},
                ],
            }
        )
        self.mon_pts = CFDDict(
            {
            }
        )
