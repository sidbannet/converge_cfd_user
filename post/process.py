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

LOADCHAR = '\|/-'


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
        self._got_port_timing = False
        self._got_breathing_eff = False
        self._got_mass_flow = False
        self._got_processed = False
        self._appended_with_other = False
        self._appending_index = []
        self._frac_lean = float(0.0)

        self._passive = {
            'int': 'INT',
            'rs_int': 'RS-INT',
            'cyl': 'CYL',
            'exh': 'EXH',
            'rs_exh': 'RS-EXH',
            'fuel': None,
            'lean': None,
        }
        self.region = CFDDict(
            {
                'cyl_comb': [0],
                'int_man': [1],
                'exh_man': [2],
                'int_rs': [3],
                'exh_rs': [4],
                'int_ports': [31, 32, 33, 34, 35, 36, 37, 38],
                'int_port_quad': [11, 12, 13, 14],
                'int_port_top': [15],
                'int_port_bottom': [16],
                'exh_ports': [41, 42, 43, 44, 45, 46, 47, 48],
                'exh_port_quad': [21, 22, 23, 24],
                'exh_port_top': [25],
                'exh_port_bottom': [26],
            }
        )
        self.port_regs = CFDDict(
            {
                'int_ports': [],
                'exh_ports': [],
            }
        )
        try:
            self.port_regs.int_ports.extend(self.region.int_ports)
            self.port_regs.int_ports.extend(self.region.int_port_quad)
            self.port_regs.int_ports.extend(self.region.int_port_top)
            self.port_regs.int_ports.extend(
                self.region.int_port_bottom)
            self.port_regs.exh_ports.extend(self.region.exh_ports)
            self.port_regs.exh_ports.extend(self.region.exh_port_quad)
            self.port_regs.exh_ports.extend(self.region.exh_port_top)
            self.port_regs.exh_ports.extend(
                self.region.exh_port_bottom)
        except AttributeError:
            pass
        self._num_of_ports = int(5)
        self.boundary = CFDDict(
            {
                'int_liner': [1],
                'exh_liner': [2],
                'int_liner_rs': [3],
                'exh_liner_rs': [4],
                'cyl_liner': [5],
                'inlet': [7],
                'outlet': [8],
                'int_man': [10],
                'int_port_quad': [11, 12, 13, 14],
                'int_ports': [51, 52, 53, 54, 55, 56, 57, 58],
                'int_port_top': [15],
                'int_port_bottom': [16],
                'int_PCAP': [17],
                'exh_man': [20],
                'exh_port_quad': [22, 21, 24, 23],
                'exh_ports': [61, 62, 63, 64, 65, 66, 67, 68],
                'exh_port_top': [25],
                'exh_port_bottom': [26],
                'exh_PCAP': [27],
                'int_transtube': [30],
                'int_pist_skirt': [31],
                'int_pist_ring': [32],
                'int_pist_head': [33],
                'int_pist_face': [34],
                'int_pist_crev': [35],
                'int_pist_ring_skirt': [36],
                'int_pist_ring_top': [38],
                'exh_transtube': [40],
                'exh_pist_skirt': [41],
                'exh_pist_ring': [42],
                'exh_pist_head': [43],
                'exh_pist_face': [44],
                'exh_pist_crev': [45],
                'exh_pist_ring_skirt': [46],
                'exh_pist_ring_top': [48],
            }
        )
        self.mon_pts = CFDDict(
            {
                'cyl_center': int(3),
                'int_pist': int(4),
                'exh_pist': int(5),
                'cyl_int_port': int(1),
                'cyl_exh_port': int(2),
                'int_man_measured': int(6),
                'exh_man_measured': int(7),
                'int_pist_crev': [10, 9, 8, 11],
                'exh_pist_crev': [12, 13, 14, 15],
            }
        )
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

    def get_port_time(
        self,
        flow_rate_threshold_open: float = 0.05,
    ) -> None:
        """Get the port timing."""
        assert self._got_mass_flow, "Port flow not calculated yet."
        int_port_flow = self.mass_to_cyl.Flow_Rate.from_int_ports
        exh_port_flow = self.mass_to_cyl.Flow_Rate.from_exh_ports
        reg_cyl_comb = self.region.cyl_comb[0]
        self.ipo_ca = int_port_flow.index[
            [idx for idx, fr in enumerate(
                int_port_flow
            ) if abs(fr) > flow_rate_threshold_open][0]
        ]
        self.ipc_ca = self.thermo.region[
            reg_cyl_comb].index[
            [idx for idx, V in enumerate(
                self.thermo.region[reg_cyl_comb].Volume
            ) if V > self.thermo.region[reg_cyl_comb].Volume[
                self.ipo_ca
            ]][-1]
        ]
        self.epo_ca = exh_port_flow.index[
            [idx for idx, fr in enumerate(
                exh_port_flow
            ) if abs(fr) > flow_rate_threshold_open][0]
        ]
        self.epc_ca = self.thermo.region[
            reg_cyl_comb].index[
            [idx for idx, V in enumerate(
                self.thermo.region[reg_cyl_comb].Volume
            ) if V > self.thermo.region[reg_cyl_comb].Volume[
                 self.epo_ca
             ]][-1]
        ]
        self.bdc_ca = self.thermo.region[
            reg_cyl_comb
        ].index[
            [idx for idx, v in enumerate(
                self.thermo.region[
                    reg_cyl_comb
                ].Volume
            ) if v == max(
                self.thermo.region[
                    reg_cyl_comb
                ].Volume
            )][0]
        ]

        self._ca2t = \
            lambda ca: (1e3 * (ca - self.bdc_ca) / 360.0) \
                      / self.cyc_freq
        self._t2ca = \
            lambda t: (t * self.cyc_freq * 360.0) \
                      / 1e3 + self.bdc_ca

        low_limit = - (self.bdc_ca / 360) * (1e3 / self.cyc_freq)
        high_limit = ((360 - self.bdc_ca) / 360) * (1e3 / self.cyc_freq)
        tlabel = ([i for i in range(
            int(low_limit), int(high_limit), 5
        )])
        try:
            tlabel.remove(0)
        except ValueError:
            pass
        caticks = sorted(
            [
                self._t2ca(t) for t in tlabel
            ] + [
                self.epo_ca,
                self.ipo_ca,
                self.bdc_ca,
                self.ipc_ca,
                self.epc_ca,
            ]
        )
        tticks = [str(round(t)) for t in self._ca2t(caticks)]
        itickEPO = [idx
                    for idx, ca in enumerate(caticks)
                    if ca == self.epo_ca][0]
        itickIPO = [idx
                    for idx, ca in enumerate(caticks)
                    if ca == self.ipo_ca][0]
        itickEPC = [idx
                    for idx, ca in enumerate(caticks)
                    if ca == self.epc_ca][0]
        itickIPC = [idx
                    for idx, ca in enumerate(caticks)
                    if ca == self.ipc_ca][0]
        itickBDC = [idx
                    for idx, ca in enumerate(caticks)
                    if ca == self.bdc_ca][0]
        tticks[itickEPO] = 'EPO'
        tticks[itickIPC] = 'IPC'
        tticks[itickBDC] = 'BDC'
        tticks[itickIPO] = 'IPO'
        tticks[itickEPC] = 'EPC'

        self.tlabel = tlabel
        self.tticks = tticks
        self.caticks = caticks
        self.itick_ipo = itickIPO
        self.itick_ipc = itickIPC
        self.itick_epo = itickEPO
        self.itick_epc = itickEPC
        self.itick_bdc = itickBDC

        self._got_port_timing = True

    def get_breathing_eff(
        self,
        frac_lean: float = None,
    ) -> None:
        """Get breathing efficiencies."""
        assert self._got_mass_flow, "Mass flow not calculated yet."
        if frac_lean is not None:
            self._frac_lean = frac_lean
        assert self._frac_lean <= 1, "Fuel lean portion should be <= 1."
        assert self._frac_lean >= 0, "Fuel lean factor should be >=0."
        reg_cyl_comb = self.region.cyl_comb[0]
        exh_port_flow_to_cyl = \
            self.mass_to_cyl.Tot_INT.from_exh_ports \
            + self.mass_to_cyl.Tot_RS_INT.from_exh_ports \
            - self._frac_lean * self.mass_to_cyl.Tot_LEAN.from_exh_ports
        if self._passive['rs_int'] not in self.passive.region[reg_cyl_comb].keys():
            self.passive.region[reg_cyl_comb][self._passive['rs_int']] = \
                0 * self.passive.region[reg_cyl_comb][self._passive['int']]
        if self._passive['rs_exh'] not in self.passive.region[reg_cyl_comb].keys():
            self.passive.region[reg_cyl_comb][self._passive['rs_exh']] = \
                0 * self.passive.region[reg_cyl_comb][self._passive['exh']]
        if self._passive['lean'] not in self.passive.region[reg_cyl_comb].keys():
            self.passive.region[reg_cyl_comb][self._passive['lean']] = \
                0 * self.passive.region[reg_cyl_comb][self._passive['int']]
        if self._passive['fuel'] not in self.passive.region[reg_cyl_comb].keys():
            self.passive.region[reg_cyl_comb][self._passive['fuel']] = \
                0 * self.passive.region[reg_cyl_comb][self._passive['int']]
        trap_eff = 1 - (
            - exh_port_flow_to_cyl
        ) / (self.passive.region[
            reg_cyl_comb
        ][self._passive['int']] + self.passive.region[
            reg_cyl_comb
        ][self._passive['rs_int']])

        scav_eff = (self.passive.region[
            reg_cyl_comb
        ][self._passive['int']] + self.passive.region[
            reg_cyl_comb
        ][self._passive['rs_int']]) / (self.passive.region[
            reg_cyl_comb
        ][self._passive['int']] + self.passive.region[
            reg_cyl_comb
        ][self._passive['rs_int']] + self.passive.region[
            reg_cyl_comb
        ][self._passive['cyl']])

        # //todo: replace nan with 0 in trap_eff

        self.trap_eff = trap_eff
        self.scav_eff = scav_eff

        self._got_breathing_eff = True

    def get_mass_flow(self) -> None:
        """Get the mass flow rates throught the power cylinder."""
        assert self._loaded_data, 'Data not loaded.'
        int_port_regs = self.port_regs.int_ports
        exh_port_regs = self.port_regs.exh_ports
        cyl_reg = self.region.cyl_comb[0]
        tot_mass_int_to_cyl = float(0.0)
        tot_mass_exh_to_cyl = float(0.0)
        tot_int_int_to_cyl = float(0.0)
        tot_rsint_int_to_cyl = float(0.0)
        tot_cyl_int_to_cyl = float(0.0)
        tot_exh_int_to_cyl = float(0.0)
        tot_rsexh_int_to_cyl = float(0.0)
        tot_lean_int_to_cyl = float(0.0)
        tot_fuel_int_to_cyl = float(0.0)
        tot_int_exh_to_cyl = float(0.0)
        tot_rsint_exh_to_cyl = float(0.0)
        tot_cyl_exh_to_cyl = float(0.0)
        tot_exh_exh_to_cyl = float(0.0)
        tot_rsexh_exh_to_cyl = float(0.0)
        tot_lean_exh_to_cyl = float(0.0)
        tot_fuel_exh_to_cyl = float(0.0)
        rate_mass_int_to_cyl = float(0.0)
        rate_mass_exh_to_cyl = float(0.0)
        rate_int_int_to_cyl = float(0.0)
        rate_rsint_int_to_cyl = float(0.0)
        rate_cyl_int_to_cyl = float(0.0)
        rate_exh_int_to_cyl = float(0.0)
        rate_rsexh_int_to_cyl = float(0.0)
        rate_lean_int_to_cyl = float(0.0)
        rate_fuel_int_to_cyl = float(0.0)
        rate_int_exh_to_cyl = float(0.0)
        rate_rsint_exh_to_cyl = float(0.0)
        rate_cyl_exh_to_cyl = float(0.0)
        rate_exh_exh_to_cyl = float(0.0)
        rate_rsexh_exh_to_cyl = float(0.0)
        rate_lean_exh_to_cyl = float(0.0)
        rate_fuel_exh_to_cyl = float(0.0)

        for port in int_port_regs:
            region_direction_fwd = \
                'Regions_' + str(cyl_reg) + '_to_' + str(port)
            region_direction_bcw = \
                'Regions_' + str(port) + '_to_' + str(cyl_reg)
            try:
                tot_mass_int_to_cyl -= \
                    self.regions_flow.Tot_Mass[region_direction_fwd]
                tot_int_int_to_cyl -= \
                    self.regions_flow[
                        self._header['tot'] + self._passive['int']
                    ][region_direction_fwd]
                try:
                    tot_rsint_int_to_cyl -= \
                        self.regions_flow[
                            self._header['tot'] + self._passive['rs_int']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    tot_rsint_int_to_cyl = 0 * tot_int_int_to_cyl
                try:
                    tot_lean_int_to_cyl -= \
                        self.regions_flow[
                            self._header['tot'] + self._passive['lean']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    tot_lean_int_to_cyl = 0 * tot_int_int_to_cyl
                try:
                    tot_fuel_int_to_cyl -= \
                        self.regions_flow[
                            self._header['tot'] + self._passive['fuel']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    tot_fuel_int_to_cyl = 0 * tot_int_int_to_cyl
                tot_cyl_int_to_cyl -= \
                    self.regions_flow[
                        self._header['tot'] + self._passive['cyl']
                    ][region_direction_fwd]
                tot_exh_int_to_cyl -= \
                    self.regions_flow[
                        self._header['tot'] + self._passive['exh']
                    ][region_direction_fwd]
                try:
                    tot_rsexh_int_to_cyl -= \
                        self.regions_flow[
                            self._header['tot'] + self._passive['rs_exh']
                        ][region_direction_fwd]
                except KeyError:
                    tot_rsexh_int_to_cyl = 0 * tot_exh_int_to_cyl
                rate_mass_int_to_cyl -= \
                    self.regions_flow.Flow_Rate[region_direction_fwd]
                rate_int_int_to_cyl -= \
                    self.regions_flow[
                        self._header['rate'] + self._passive['int']
                    ][region_direction_fwd]
                try:
                    rate_rsint_int_to_cyl -= \
                        self.regions_flow[
                            self._header['rate'] + self._passive['rs_int']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    rate_rsint_int_to_cyl = 0 * rate_int_int_to_cyl
                try:
                    rate_lean_int_to_cyl -= \
                        self.regions_flow[
                            self._header['rate'] + self._passive['lean']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    rate_lean_int_to_cyl = 0 * rate_int_int_to_cyl
                try:
                    rate_fuel_int_to_cyl -= \
                        self.regions_flow[
                            self._header['rate'] + self._passive['fuel']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    rate_fuel_int_to_cyl = 0 * rate_int_int_to_cyl
                rate_cyl_int_to_cyl -= \
                    self.regions_flow[
                        self._header['rate'] + self._passive['cyl']
                    ][region_direction_fwd]
                rate_exh_int_to_cyl -= \
                    self.regions_flow[
                        self._header['rate'] + self._passive['exh']
                    ][region_direction_fwd]
                try:
                    rate_rsexh_int_to_cyl -= \
                        self.regions_flow[
                            self._header['rate'] + self._passive['rs_exh']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    rate_rsexh_int_to_cyl = 0 * rate_exh_int_to_cyl
            except AttributeError:
                tot_mass_int_to_cyl += \
                    self.regions_flow.Tot_Mass[region_direction_bcw]
                tot_int_int_to_cyl += \
                    self.regions_flow[
                        self._header['tot'] + self._passive['int']
                    ][region_direction_bcw]
                try:
                    tot_rsint_int_to_cyl += \
                        self.regions_flow[
                            self._header['tot'] + self._passive['rs_int']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    tot_rsint_int_to_cyl = 0 * tot_int_int_to_cyl
                try:
                    tot_lean_int_to_cyl += \
                        self.regions_flow[
                            self._header['tot'] + self._passive['lean']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    tot_lean_int_to_cyl = 0 * tot_int_int_to_cyl
                try:
                    tot_fuel_int_to_cyl += \
                        self.regions_flow[
                            self._header['tot'] + self._passive['fuel']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    tot_fuel_int_to_cyl = 0 * tot_int_int_to_cyl
                tot_cyl_int_to_cyl += \
                    self.regions_flow[
                        self._header['tot'] + self._passive['cyl']
                    ][region_direction_bcw]
                tot_exh_int_to_cyl += \
                    self.regions_flow[
                        self._header['tot'] + self._passive['exh']
                    ][region_direction_bcw]
                try:
                    tot_rsexh_int_to_cyl += \
                        self.regions_flow[
                            self._header['tot'] + self._passive['rs_exh']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    tot_rsexh_int_to_cyl = 0 * tot_exh_int_to_cyl
                rate_mass_int_to_cyl += \
                    self.regions_flow.Flow_Rate[region_direction_bcw]
                rate_int_int_to_cyl += \
                    self.regions_flow[
                        self._header['rate'] + self._passive['int']
                    ][region_direction_bcw]
                try:
                    rate_rsint_int_to_cyl += \
                        self.regions_flow[
                            self._header['rate'] + self._passive['rs_int']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    rate_rsint_int_to_cyl = 0 * rate_int_int_to_cyl
                try:
                    rate_lean_int_to_cyl += \
                        self.regions_flow[
                            self._header['rate'] + self._passive['lean']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    rate_lean_int_to_cyl = 0 * rate_int_int_to_cyl
                try:
                    rate_fuel_int_to_cyl += \
                        self.regions_flow[
                            self._header['rate'] + self._passive['fuel']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    rate_fuel_int_to_cyl = 0 * rate_int_int_to_cyl
                rate_cyl_int_to_cyl += \
                    self.regions_flow[
                        self._header['rate'] + self._passive['cyl']
                    ][region_direction_bcw]
                rate_exh_int_to_cyl += \
                    self.regions_flow[
                        self._header['rate'] + self._passive['exh']
                    ][region_direction_bcw]
                try:
                    rate_rsexh_int_to_cyl += \
                        self.regions_flow[
                            self._header['rate'] + self._passive['rs_exh']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    rate_rsexh_int_to_cyl = 0 * rate_exh_int_to_cyl
        for port in exh_port_regs:
            region_direction_fwd = \
                'Regions_' + str(cyl_reg) + '_to_' + str(port)
            region_direction_bcw = \
                'Regions_' + str(port) + '_to_' + str(cyl_reg)
            try:
                tot_mass_exh_to_cyl -= \
                    self.regions_flow.Tot_Mass[region_direction_fwd]
                tot_int_exh_to_cyl -= \
                    self.regions_flow[
                        self._header['tot'] + self._passive['int']
                    ][region_direction_fwd]
                try:
                    tot_rsint_exh_to_cyl -= \
                        self.regions_flow[
                            self._header['tot'] + self._passive['rs_int']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    tot_rsint_exh_to_cyl = 0 * tot_int_exh_to_cyl
                try:
                    tot_lean_exh_to_cyl -= \
                        self.regions_flow[
                            self._header['tot'] + self._passive['lean']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    tot_lean_exh_to_cyl = 0 * tot_int_exh_to_cyl
                try:
                    tot_fuel_exh_to_cyl -= \
                        self.regions_flow[
                            self._header['tot'] + self._passive['fuel']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    tot_fuel_exh_to_cyl = 0 * tot_int_exh_to_cyl
                tot_cyl_exh_to_cyl -= \
                    self.regions_flow[
                        self._header['tot'] + self._passive['cyl']
                    ][region_direction_fwd]
                tot_exh_exh_to_cyl -= \
                    self.regions_flow[
                        self._header['tot'] + self._passive['exh']
                    ][region_direction_fwd]
                try:
                    tot_rsexh_exh_to_cyl -= \
                        self.regions_flow[
                            self._header['tot'] + self._passive['rs_exh']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    tot_rsexh_exh_to_cyl = 0 * tot_exh_exh_to_cyl
                rate_mass_exh_to_cyl -= \
                    self.regions_flow.Flow_Rate[region_direction_fwd]
                rate_int_exh_to_cyl -= \
                    self.regions_flow[
                        self._header['rate'] + self._passive['int']
                    ][region_direction_fwd]
                try:
                    rate_rsint_exh_to_cyl -= \
                        self.regions_flow[
                            self._header['rate'] + self._passive['rs_int']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    rate_rsint_exh_to_cyl = 0 * rate_int_exh_to_cyl
                try:
                    rate_lean_exh_to_cyl -= \
                        self.regions_flow[
                            self._header['rate'] + self._passive['lean']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    rate_lean_exh_to_cyl = 0 * rate_int_exh_to_cyl
                try:
                    rate_fuel_exh_to_cyl -= \
                        self.regions_flow[
                            self._header['rate'] + self._passive['fuel']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    rate_fuel_exh_to_cyl = 0 * rate_int_exh_to_cyl
                rate_cyl_exh_to_cyl -= \
                    self.regions_flow[
                        self._header['rate'] + self._passive['cyl']
                    ][region_direction_fwd]
                rate_exh_exh_to_cyl -= \
                    self.regions_flow[
                        self._header['rate'] + self._passive['exh']
                    ][region_direction_fwd]
                try:
                    rate_rsexh_exh_to_cyl -= \
                        self.regions_flow[
                            self._header['rate'] + self._passive['rs_exh']
                        ][region_direction_fwd]
                except (KeyError, TypeError):
                    rate_rsexh_exh_to_cyl = 0 * rate_exh_exh_to_cyl
            except AttributeError:
                tot_mass_exh_to_cyl += \
                    self.regions_flow.Tot_Mass[region_direction_bcw]
                tot_int_exh_to_cyl += \
                    self.regions_flow[
                        self._header['tot'] + self._passive['int']
                    ][region_direction_bcw]
                try:
                    tot_rsint_exh_to_cyl += \
                        self.regions_flow[
                            self._header['tot'] + self._passive['rs_int']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    tot_rsint_exh_to_cyl = 0 * tot_int_exh_to_cyl
                try:
                    tot_lean_exh_to_cyl += \
                        self.regions_flow[
                            self._header['tot'] + self._passive['lean']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    tot_lean_exh_to_cyl = 0 * tot_int_exh_to_cyl
                try:
                    tot_fuel_exh_to_cyl += \
                        self.regions_flow[
                            self._header['tot'] + self._passive['fuel']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    tot_fuel_exh_to_cyl = 0 * tot_int_exh_to_cyl
                tot_cyl_exh_to_cyl += \
                    self.regions_flow[
                        self._header['tot'] + self._passive['cyl']
                    ][region_direction_bcw]
                tot_exh_exh_to_cyl += \
                    self.regions_flow[
                        self._header['tot'] + self._passive['exh']
                    ][region_direction_bcw]
                try:
                    tot_rsexh_exh_to_cyl += \
                        self.regions_flow[
                            self._header['tot'] + self._passive['rs_exh']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    tot_rsexh_exh_to_cyl = 0 * tot_exh_exh_to_cyl
                rate_mass_exh_to_cyl += \
                    self.regions_flow.Flow_Rate[region_direction_bcw]
                rate_int_exh_to_cyl += \
                    self.regions_flow[
                        self._header['rate'] + self._passive['int']
                    ][region_direction_bcw]
                try:
                    rate_rsint_exh_to_cyl += \
                        self.regions_flow[
                            self._header['rate'] + self._passive['rs_int']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    rate_rsint_exh_to_cyl = 0 * rate_int_exh_to_cyl
                try:
                    rate_lean_exh_to_cyl += \
                        self.regions_flow[
                            self._header['rate'] + self._passive['lean']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    rate_lean_exh_to_cyl = 0 * rate_int_exh_to_cyl
                try:
                    rate_fuel_exh_to_cyl += \
                        self.regions_flow[
                            self._header['rate'] + self._passive['fuel']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    rate_fuel_exh_to_cyl = 0 * rate_int_exh_to_cyl
                rate_cyl_exh_to_cyl += \
                    self.regions_flow[
                        self._header['rate'] + self._passive['cyl']
                    ][region_direction_bcw]
                rate_exh_exh_to_cyl += \
                    self.regions_flow[
                        self._header['rate'] + self._passive['exh']
                    ][region_direction_bcw]
                try:
                    rate_rsexh_exh_to_cyl += \
                        self.regions_flow[
                            self._header['rate'] + self._passive['rs_exh']
                        ][region_direction_bcw]
                except (KeyError, TypeError):
                    rate_rsexh_exh_to_cyl = 0 * rate_exh_exh_to_cyl
        tmp = np.array(
            [
                tot_mass_int_to_cyl.values,
                tot_mass_exh_to_cyl.values,
                tot_int_int_to_cyl.values,
                tot_int_exh_to_cyl.values,
                tot_rsint_int_to_cyl.values,
                tot_rsint_exh_to_cyl.values,
                tot_lean_int_to_cyl.values,
                tot_lean_exh_to_cyl.values,
                tot_fuel_int_to_cyl.values,
                tot_fuel_exh_to_cyl.values,
                tot_cyl_int_to_cyl.values,
                tot_cyl_exh_to_cyl.values,
                tot_exh_int_to_cyl.values,
                tot_exh_exh_to_cyl.values,
                tot_rsexh_int_to_cyl.values,
                tot_rsexh_exh_to_cyl.values,
                rate_mass_int_to_cyl.values,
                rate_mass_exh_to_cyl.values,
                rate_int_int_to_cyl.values,
                rate_int_exh_to_cyl.values,
                rate_rsint_int_to_cyl.values,
                rate_rsint_exh_to_cyl.values,
                rate_lean_int_to_cyl.values,
                rate_lean_exh_to_cyl.values,
                rate_fuel_int_to_cyl.values,
                rate_fuel_exh_to_cyl.values,
                rate_cyl_int_to_cyl.values,
                rate_cyl_exh_to_cyl.values,
                rate_exh_int_to_cyl.values,
                rate_exh_exh_to_cyl.values,
                rate_rsexh_int_to_cyl.values,
                rate_rsexh_exh_to_cyl.values,
            ]
        ).T
        self.mass_to_cyl = CFDDataFrame(
            tmp,
            columns=[
                [
                    'Tot_Mass',
                    'Tot_Mass',
                    'Tot_INT',
                    'Tot_INT',
                    'Tot_RS_INT',
                    'Tot_RS_INT',
                    'Tot_LEAN',
                    'Tot_LEAN',
                    'Tot_FUEL',
                    'Tot_FUEL',
                    'Tot_CYL',
                    'Tot_CYL',
                    'Tot_EXH',
                    'Tot_EXH',
                    'Tot_RS_EXH',
                    'Tot_RS_EXH',
                    'Flow_Rate',
                    'Flow_Rate',
                    'Rate_INT',
                    'Rate_INT',
                    'Rate_RS_INT',
                    'Rate_RS_INT',
                    'Rate_LEAN',
                    'Rate_LEAN',
                    'Rate_FUEL',
                    'Rate_FUEL',
                    'Rate_CYL',
                    'Rate_CYL',
                    'Rate_EXH',
                    'Rate_EXH',
                    'Rate_RS_EXH',
                    'Rate_RS_EXH',
                ],
                [
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                    'from_int_ports',
                    'from_exh_ports',
                ]
            ],
            index=tot_mass_int_to_cyl.index,
        )
        replace_keys = {
            self._header['tot'] + self._passive['rs_int']:
                self._header['tot'] + self._passive['int'],
            self._header['tot'] + self._passive['rs_exh']:
                self._header['tot'] + self._passive['exh'],
            self._header['rate'] + self._passive['rs_int']:
                self._header['rate'] + self._passive['int'],
            self._header['rate'] + self._passive['rs_exh']:
                self._header['rate'] + self._passive['exh'],
        }
        for replace_key, replace_with in replace_keys.items():
            try:
                if replace_key not in self.regions_flow.unit_.keys():
                    self.regions_flow._unit[
                        replace_key
                    ] = self.regions_flow.unit_[replace_with]
            except AttributeError:
                continue
        try:
            self.mass_to_cyl._unit = CFDDict(
                {
                    'Tot_Mass': self.regions_flow.unit_.Tot_Mass,
                    'Tot_INT': self.regions_flow.unit_[
                        self._header['tot'] + self._passive['int']],
                    'Tot_RS_INT': self.regions_flow.unit_[
                        self._header['tot'] + self._passive['rs_int']],
                    'Tot_CYL': self.regions_flow.unit_[
                        self._header['tot'] + self._passive['cyl']],
                    'Tot_EXH': self.regions_flow.unit_[
                        self._header['tot'] + self._passive['exh']],
                    'Tot_RS_EXH': self.regions_flow.unit_[
                        self._header['tot'] + self._passive['rs_exh']],
                    'Flow_Rate': self.regions_flow.unit_.Flow_Rate,
                    'Rate_INT': self.regions_flow.unit_[
                        self._header['rate'] + self._passive['int']],
                    'Rate_RS_INT': self.regions_flow.unit_[
                        self._header['rate'] + self._passive['rs_int']],
                    'Rate_CYL': self.regions_flow.unit_[
                        self._header['rate'] + self._passive['cyl']],
                    'Rate_EXH': self.regions_flow.unit_[
                        self._header['rate'] + self._passive['exh']],
                    'Rate_RS_EXH': self.regions_flow.unit_[
                        self._header['rate'] + self._passive['rs_exh']],
                }
            )
        except AttributeError:
            self.mass_to_cyl._unit = CFDDict(
                {
                    'Tot_Mass': '[kg]',
                    'Tot_INT': '[kg]',
                    'Tot_RS_INT': '[kg]',
                    'Tot_CYL': '[kg]',
                    'Tot_EXH': '[kg]',
                    'Tot_RS_EXH': '[kg]',
                    'Flow_Rate': '[kg/s]',
                    'Rate_INT': '[kg/s]',
                    'Rate_RS_INT': '[kg/s]',
                    'Rate_CYL': '[kg/s]',
                    'Rate_EXH': '[kg/s]',
                    'Rate_RS_EXH': '[kg/s]',
                }
            )
        if self._passive['lean'] is not None:
            try:
                self.mass_to_cyl._unit[
                    'Tot_LEAN'
                ] = self.regions_flow.unit_[
                    self._header['tot'] + self._passive['lean'],
                ]
                self.mass_to_cyl._unit[
                    'Rate_LEAN'
                ] = self.regions_flow.unit_[
                    self._header['rate'] + self._passive['lean']
                ]
            except (KeyError, TypeError, AttributeError):
                self.mass_to_cyl._unit[
                    'Tot_LEAN'
                ] = '[kg]'
                self.mass_to_cyl._unit[
                    'Rate_LEAN'
                ] = '[kg/s]'
        else:
            pass
        if self._passive['fuel'] is not None:
            try:
                self.mass_to_cyl._unit[
                    'Tot_FUEL'
                ] = self.regions_flow.unit_[
                    self._header['tot'] + self._passive['fuel'],
                ]
                self.mass_to_cyl._unit[
                    'Rate_FUEL'
                ] = self.regions_flow.unit_[
                    self._header['rate'] + self._passive['fuel']
                ]
            except (KeyError, TypeError, AttributeError):
                self.mass_to_cyl._unit[
                    'Tot_FUEL'
                ] = '[kg]'
                self.mass_to_cyl._unit[
                    'Rate_FUEL'
                ] = '[kg/s]'
        else:
            pass
        self._got_mass_flow = True

    def get_processed(self) -> None:
        """Get all data processed."""
        pd.set_option("mode.chained_assignment", None)
        if not self._appended_with_other:
            self.get_mass_flow()
            self.get_port_time()
            self.get_breathing_eff()
            self._got_processed = True
            return
        unique_idx = ~self.regions_flow.index.duplicated(keep='first')
        self.regions_flow['Tot_Mass'] = self.__cumulative_sum(
            self.regions_flow['Tot_Mass'][unique_idx]
        )
        self.regions_flow[
            self._header['tot'] + self._passive['int']
        ] = self.__cumulative_sum(
            self.regions_flow[
                self._header['tot'] + self._passive['int']
            ][unique_idx]
        )
        self.regions_flow[
            self._header['tot'] + self._passive['cyl']
        ] = self.__cumulative_sum(
            self.regions_flow[
                self._header['tot'] + self._passive['cyl']
            ][unique_idx]
        )
        self.regions_flow[
            self._header['tot'] + self._passive['exh']
        ] = self.__cumulative_sum(
            self.regions_flow[
                self._header['tot'] + self._passive['exh']
            ][unique_idx]
        )
        try:
            self.regions_flow[
                self._header['tot'] + self._passive['rs_int']
            ] = self.__cumulative_sum(
                self.regions_flow[
                    self._header['tot'] + self._passive['rs_int']
                ][unique_idx]
            )
        except KeyError:
            pass
        try:
            self.regions_flow[
                self._header['tot'] + self._passive['rs_exh']
            ] = self.__cumulative_sum(
                self.regions_flow[
                    self._header['tot'] + self._passive['rs_exh']
                ][unique_idx]
            )
        except KeyError:
            pass
        for _, value in self.boundary.items():
            for id in value:
                try:
                    self.bound.bound[id].Tot_HT_xfer = self.__cumulative_sum(
                        self.bound.bound[id].Tot_HT_xfer[unique_idx]
                    )
                except AttributeError:
                    pass
        self.get_mass_flow()
        self.get_port_time()
        self.get_breathing_eff()
        self._got_processed = True

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


class Plots:
    """Class for plotting CFD results."""

    def __init__(
        self,
        color: str = 'k',
        style: str = '-',
        label: str = '',
        marker: str = '*',
        size: int = 200,
    ):
        """Instantiate the class."""
        self.color = color
        self.style = style
        self.label = label
        self.marker = marker
        self.size = size

    @staticmethod
    def port_flow(
        rslt: Case = None,
        *args,
    ) -> tuple:
        """Plot port flow rates."""
        assert rslt._got_port_timing, 'Port data not loaded yet.'
        try:
            fig, ax = args
            user_given_fig = True
        except ValueError:
            fig = plt.figure('Fig: Port flow rates')
            ax = fig.subplots(
                nrows=2, ncols=1, sharex=True, sharey=True)
            user_given_fig = False
        for inum, port in enumerate(rslt.region.int_ports):
            flow_direction = 'Regions_' \
                             + str(rslt.region.cyl_comb[0]) \
                             + '_to_' \
                             + str(port)
            ax[0].plot(
                - rslt.regions_flow.Flow_Rate[
                    flow_direction
                ] * 1e3 / rslt._num_of_ports,
                '.-',
                label=str(inum),
            )
        for inum, port in enumerate(rslt.region.exh_ports):
            flow_direction = 'Regions_' \
                             + str(rslt.region.cyl_comb[0]) \
                             + '_to_' \
                             + str(port)
            ax[1].plot(
                - rslt.regions_flow.Flow_Rate[
                    flow_direction
                ] * 1e3 / rslt._num_of_ports,
                '.-',
                label=str(inum),
            )
        for axes in ax.flat:
            axes.legend()
            axes.set_ylabel('Mass flow rate per port [g/s]')
            axes.grid(True)
            axes.set_xticks(rslt.caticks)
            axes.set_xticklabels(rslt.tticks)
            if not user_given_fig:
                GetGridLine(
                    ax=axes,
                    rslt=rslt,
                )
        ax[0].set_xlim(rslt.epo_ca - 10, rslt.epc_ca + 10)
        ax[0].set_title('Intake ports')
        ax[1].set_title('Exhaust ports')
        ax[-1].set_xlabel('Time after BDC [ms]')
        fig.suptitle('CFD: Mass flow rates through ports')

        return fig, ax

    @staticmethod
    def port_flow_tot(
            rslt: Case = None,
            *args,
    ) -> tuple:
        """Plot port flow rates."""
        assert rslt._got_port_timing, 'Port data not loaded yet.'
        try:
            fig, ax = args
            user_given_fig = True
        except ValueError:
            fig = plt.figure('Fig: Total port flow')
            ax = fig.subplots(
                nrows=2, ncols=1, sharex=True, sharey=True)
            user_given_fig = False
        for inum, port in enumerate(rslt.region.int_ports):
            flow_direction = 'Regions_' \
                             + str(rslt.region.cyl_comb[0]) \
                             + '_to_' \
                             + str(port)
            ax[0].plot(
                - rslt.regions_flow.Tot_Mass[
                    flow_direction
                ] * 1e3 / rslt._num_of_ports,
                '.-',
                label=str(inum),
            )
        for inum, port in enumerate(rslt.region.exh_ports):
            flow_direction = 'Regions_' \
                             + str(rslt.region.cyl_comb[0]) \
                             + '_to_' \
                             + str(port)
            ax[1].plot(
                - rslt.regions_flow.Tot_Mass[
                    flow_direction
                ] * 1e3 / rslt._num_of_ports,
                '.-',
                label=str(inum),
            )
        for axes in ax.flat:
            axes.legend()
            axes.set_ylabel('Mass flow per port [g]')
            axes.grid(True)
            axes.set_xticks(rslt.caticks)
            axes.set_xticklabels(rslt.tticks)
            if not user_given_fig:
                GetGridLine(
                    ax=axes,
                    rslt=rslt,
                )
        ax[0].set_xlim(rslt.epo_ca - 10, rslt.epc_ca + 10)
        ax[0].set_title('Intake ports')
        ax[1].set_title('Exhaust ports')
        ax[-1].set_xlabel('Time after BDC [ms]')
        fig.suptitle('CFD: Mass flow through ports')

        return fig, ax

    @staticmethod
    def cyl_pressure(
        rslt: Case = None,
        *args
    ) -> tuple:
        """Plot power cylinder pressure traces."""
        assert rslt._got_port_timing, 'Port data not loaded.'
        inlet_id = 'bound_id_' + str(rslt.boundary.inlet[0])
        outlet_id = 'bound_id_' + str(rslt.boundary.outlet[0])
        try:
            fig, ax = args
            user_given_fig = True
        except ValueError:
            fig = plt.figure('Fig: Pressure trace')
            ax = fig.subplots()
            user_given_fig = False
        ax.plot(
            rslt.mon_pt[''][
                rslt.mon_pts.cyl_center
            ].Pressure * 10,
            '-k',
            label='Cyl center',
        )
        ax.plot(
            rslt.mon_pt[''][
                rslt.mon_pts.int_man_measured
            ].Pressure * 10,
            '-b',
            label='Int Manifold',
        )
        ax.plot(
            rslt.mon_pt[''][
                rslt.mon_pts.exh_man_measured
            ].Pressure * 10,
            '-r',
            label='Exh Manifold',
        )
        ax.plot(
            rslt.mass_avg_flow.Total_Pres[inlet_id] / 1e5,
            '--b',
            label='Inlet',
        )
        ax.plot(
            rslt.mass_avg_flow.Total_Pres[outlet_id] / 1e5,
            '--r',
            label='Outlet',
        )
        ax.set_ylim([0.5, 2.5])
        ax.set_xticks(rslt.caticks)
        ax.legend()
        ax.grid(True)
        ax.set_ylabel('Pressure [bar]')
        ax.set_xlabel('Time after BDC [ms]')
        fig.suptitle('CFD: Pressure traces')
        ax.set_xticklabels(rslt.tticks)
        if not user_given_fig:
            GetGridLine(
                ax=ax,
                rslt=rslt,
            )
        ax.set_xlim(rslt.epo_ca - 10, rslt.epc_ca + 10)

        return fig, ax

    def breathing_eff(
        self,
        rslt: Case = None,
        *args,
    ) -> tuple:
        """Get plots for breathing efficiencies."""
        assert rslt._got_breathing_eff, 'Data not available.'
        assert rslt._got_port_timing, 'Port timing not available.'
        try:
            fig, ax = args
        except ValueError:
            fig = plt.figure('Fig: Breathing efficiency')
            ax = fig.subplots()
        ax.plot(
            rslt.scav_eff[rslt.epo_ca : rslt.epc_ca] * 100,
            rslt.trap_eff[rslt.epo_ca : rslt.epc_ca] * 100,
            self.style + self.color,
            label=self.label,
        )
        ax.scatter(
            rslt.scav_eff[rslt.epc_ca] * 100,
            rslt.trap_eff[rslt.epc_ca] * 100,
            marker=self.marker,
            c=self.color,
            s=self.size,
        )
        ax.set_xlim(0, 105)
        ax.set_ylim(85, 105)
        ax.grid(True)
        ax.set_ylabel('Trapping efficiency [%]')
        ax.set_xlabel('Scavenging efficiency [%]')
        fig.suptitle('CFD: Breathing efficiency')

        return fig, ax

    @staticmethod
    def crevice_pts(
        rslt: Case = None,
        *args,
    ) -> tuple:
        """Plots time series from the monitor points."""
        assert rslt._got_port_timing, 'Port timing is not available.'
        try:
            fig, ax = args
        except ValueError:
            fig = plt.figure('Fig: Crevice points')
            ax = fig.subplots(nrows=5, ncols=2, sharex=True)
        for inum, pt in enumerate(rslt.mon_pts.int_pist_crev):
            label_st = str(inum)
            ax[0][0].plot(
                rslt.mon_pt[''][pt].Temperature,
                label=label_st,
            )
            ax[1][0].plot(
                rslt.mon_pt[''][pt].Pressure,
                label=label_st,
            )
            ax[2][0].plot(
                rslt.mon_pt[''][pt].passive_cyl,
                label=label_st,
            )
            try:
                ax[3][0].plot(
                    (
                        rslt.mon_pt[''][pt].passive_int
                        + rslt.mon_pt[''][pt]['passive_rs-int']
                    ),
                    label=label_st,
                )
            except KeyError:
                ax[3][0].plot(
                    (
                            rslt.mon_pt[''][pt].passive_int
                    ),
                    label=label_st,
                )
            try:
                ax[4][0].plot(
                    (
                        rslt.mon_pt[''][pt].passive_exh
                        + rslt.mon_pt[''][pt]['passive_rs-exh']
                    ),
                    label=label_st,
                )
            except KeyError:
                ax[4][0].plot(
                    (
                            rslt.mon_pt[''][pt].passive_exh
                    ),
                    label=label_st,
                )
        try:
            ax[0][0].set_ylabel(
                'T '
                + rslt.mon_pt[''][pt].unit_.Temperature
            )
            ax[1][0].set_ylabel(
                'P '
                + rslt.mon_pt[''][pt].unit_.Pressure
            )
        except AttributeError:
            ax[0][0].set_ylabel('T [K]')
            ax[1][0].set_ylabel('P [MPa]')
        ax[2][0].set_ylabel('CYL [-]')
        ax[3][0].set_ylabel('INT [-]')
        ax[4][0].set_ylabel('EXH [-]')
        for inum, pt in enumerate(rslt.mon_pts.exh_pist_crev):
            label_st = str(inum)
            ax[0][1].plot(
                rslt.mon_pt[''][pt].Temperature,
                label=label_st,
            )
            ax[1][1].plot(
                rslt.mon_pt[''][pt].Pressure,
                label=label_st,
            )
            ax[2][1].plot(
                rslt.mon_pt[''][pt].passive_cyl,
                label=label_st,
            )
            try:
                ax[3][1].plot(
                    (
                        rslt.mon_pt[''][pt].passive_int
                        + rslt.mon_pt[''][pt]['passive_rs-int']
                    ),
                    label=label_st,
                )
            except KeyError:
                ax[3][1].plot(
                    (
                            rslt.mon_pt[''][pt].passive_int
                    ),
                    label=label_st,
                )
            try:
                ax[4][1].plot(
                    (
                        rslt.mon_pt[''][pt].passive_exh
                        + rslt.mon_pt[''][pt]['passive_rs-exh']
                    ),
                    label=label_st,
                )
            except KeyError:
                ax[4][1].plot(
                    (
                            rslt.mon_pt[''][pt].passive_exh
                    ),
                    label=label_st,
                )
        _ = [ax[inum][0].set_ylim(-0.05, 1.05) for inum in [2, 3, 4]]
        _ = [ax[inum][1].set_ylim(-0.05, 1.05) for inum in [2, 3, 4]]
        _ = [axes.grid(True) for axes in ax.flat]
        ax[0][0].legend()
        ax[0][1].legend()
        ax[0][0].set_title('Intake Piston Crevice')
        ax[0][1].set_title('Exh Piston Crevice')
        ax[-1][0].set_xlabel('Time after BDC [ms]')
        ax[-1][1].set_xlabel('Time after BDC [ms]')
        fig.suptitle('CFD: Crevice thermochemistry')
        for axes in ax.flat:
            axes.set_xticks(rslt.caticks)
            axes.set_xticklabels(rslt.tticks, rotation=45)
            GetGridLine(ax=axes, rslt=rslt, linewidth=2)

        return fig, ax

    @staticmethod
    def heat_transfer(
        rslt: Case = None,
        *args,
    ) -> tuple:
        """Plot Heat loss from power chamber surfaces."""
        assert rslt._loaded_data, 'Data not loaded.'
        try:
            fig, ax = args
            user_given_fig = True
        except ValueError:
            fig = plt.figure('Fig: Heat Transfer')
            ax = fig.subplots(nrows=2, ncols=4, sharex=True)
            user_given_fig = False
        int_trans_tube_ids = rslt.boundary.int_transtube
        exh_trans_tube_ids = rslt.boundary.exh_transtube
        cyl_ids = []
        cyl_ids.extend(rslt.boundary.cyl_liner)
        cyl_ids.extend(rslt.boundary.int_liner)
        cyl_ids.extend(rslt.boundary.exh_liner)
        cyl_ids.extend(rslt.boundary.int_liner_rs)
        cyl_ids.extend(rslt.boundary.exh_liner_rs)
        int_trans_tube_Tot_HTxfer = float(0.0)
        int_trans_tube_HT_xfer_Rate = float(0.0)
        exh_trans_tube_Tot_HTxfer = float(0.0)
        exh_trans_tube_HT_xfer_Rate = float(0.0)
        cyl_Tot_HTxfer = float(0.0)
        cyl_HT_xfer_Rate = float(0.0)
        for id in int_trans_tube_ids:
            int_trans_tube_Tot_HTxfer += \
                rslt.bound.bound[id].Tot_HT_xfer
            int_trans_tube_HT_xfer_Rate += \
                rslt.bound.bound[id].HT_xfer_Rate
        for id in exh_trans_tube_ids:
            exh_trans_tube_Tot_HTxfer += \
                rslt.bound.bound[id].Tot_HT_xfer
            exh_trans_tube_HT_xfer_Rate += \
                rslt.bound.bound[id].HT_xfer_Rate
        for id in cyl_ids:
            cyl_Tot_HTxfer += \
                rslt.bound.bound[id].Tot_HT_xfer
            cyl_HT_xfer_Rate += \
                rslt.bound.bound[id].HT_xfer_Rate
        ax[0][0].plot(
            int_trans_tube_Tot_HTxfer,
            '-b',
            label='Int Trans Tube',
        )
        ax[0][0].plot(
            exh_trans_tube_Tot_HTxfer,
            '-r',
            label='Exh Trans Tube',
        )
        ax[1][0].plot(
            int_trans_tube_HT_xfer_Rate,
            '-b',
            label='Int Trans Tube',
        )
        ax[1][0].plot(
            exh_trans_tube_HT_xfer_Rate,
            '-r',
            label='Exh Trans Tube',
        )
        ax[0][1].plot(
            cyl_Tot_HTxfer,
            '-k',
            label='Liner',
        )
        ax[1][1].plot(
            cyl_HT_xfer_Rate,
            '-k',
            label='Liner',
        )
        ax[0][2].plot(
            rslt.bound.bound[
                rslt.boundary.int_pist_face[0]
            ].Tot_HT_xfer,
            label='Face',
        )
        ax[0][2].plot(
            rslt.bound.bound[
                rslt.boundary.int_pist_head[0]
            ].Tot_HT_xfer,
            label='Head',
        )
        ax[0][2].plot(
            rslt.bound.bound[
                rslt.boundary.int_pist_ring_top[0]
            ].Tot_HT_xfer,
            label='Ring Top',
        )
        ax[0][2].plot(
            rslt.bound.bound[
                rslt.boundary.int_pist_ring[0]
            ].Tot_HT_xfer,
            label='Ring',
        )
        ax[0][2].plot(
            rslt.bound.bound[
                rslt.boundary.int_pist_crev[0]
            ].Tot_HT_xfer,
            label='Crevice',
        )
        ax[0][2].plot(
            rslt.bound.bound[
                rslt.boundary.int_pist_ring_skirt[0]
            ].Tot_HT_xfer,
            label='Ring Skirt',
        )
        ax[1][2].plot(
            rslt.bound.bound[
                rslt.boundary.int_pist_face[0]
            ].HT_xfer_Rate,
            label='Face',
        )
        ax[1][2].plot(
            rslt.bound.bound[
                rslt.boundary.int_pist_head[0]
            ].HT_xfer_Rate,
            label='Head',
        )
        ax[1][2].plot(
            rslt.bound.bound[
                rslt.boundary.int_pist_ring_top[0]
            ].HT_xfer_Rate,
            label='Ring Top',
        )
        ax[1][2].plot(
            rslt.bound.bound[
                rslt.boundary.int_pist_ring[0]
            ].HT_xfer_Rate,
            label='Ring',
        )
        ax[1][2].plot(
            rslt.bound.bound[
                rslt.boundary.int_pist_crev[0]
            ].HT_xfer_Rate,
            label='Crevice',
        )
        ax[1][2].plot(
            rslt.bound.bound[
                rslt.boundary.int_pist_ring_skirt[0]
            ].HT_xfer_Rate,
            label='Ring Skirt',
        )
        ax[0][3].plot(
            rslt.bound.bound[
                rslt.boundary.exh_pist_face[0]
            ].Tot_HT_xfer,
            label='Face',
        )
        ax[0][3].plot(
            rslt.bound.bound[
                rslt.boundary.exh_pist_head[0]
            ].Tot_HT_xfer,
            label='Head',
        )
        ax[0][3].plot(
            rslt.bound.bound[
                rslt.boundary.exh_pist_ring_top[0]
            ].Tot_HT_xfer,
            label='Ring Top',
        )
        ax[0][3].plot(
            rslt.bound.bound[
                rslt.boundary.exh_pist_ring[0]
            ].Tot_HT_xfer,
            label='Ring',
        )
        ax[0][3].plot(
            rslt.bound.bound[
                rslt.boundary.exh_pist_crev[0]
            ].Tot_HT_xfer,
            label='Crevice',
        )
        ax[0][3].plot(
            rslt.bound.bound[
                rslt.boundary.exh_pist_ring_skirt[0]
            ].Tot_HT_xfer,
            label='Ring Skirt',
        )
        ax[1][3].plot(
            rslt.bound.bound[
                rslt.boundary.exh_pist_face[0]
            ].HT_xfer_Rate,
            label='Face',
        )
        ax[1][3].plot(
            rslt.bound.bound[
                rslt.boundary.exh_pist_head[0]
            ].HT_xfer_Rate,
            label='Head',
        )
        ax[1][3].plot(
            rslt.bound.bound[
                rslt.boundary.exh_pist_ring_top[0]
            ].HT_xfer_Rate,
            label='Ring Top',
        )
        ax[1][3].plot(
            rslt.bound.bound[
                rslt.boundary.exh_pist_ring[0]
            ].HT_xfer_Rate,
            label='Ring',
        )
        ax[1][3].plot(
            rslt.bound.bound[
                rslt.boundary.exh_pist_crev[0]
            ].HT_xfer_Rate,
            label='Crevice',
        )
        ax[1][3].plot(
            rslt.bound.bound[
                rslt.boundary.exh_pist_ring_skirt[0]
            ].HT_xfer_Rate,
            label='Ring Skirt',
        )
        for axis in ax.flat:
            axis.set_xticks(rslt.caticks)
            axis.set_xticklabels(rslt.tticks, rotation=90)
            axis.grid(True)
            axis.legend()
        [axis.set_xlabel('Time after BDC [ms]')
         for axis in ax[1].flat]
        ax[0][0].set_ylabel('Cumulative heat loss [J]')
        ax[1][0].set_ylabel('Heat loss rate [J/s]')
        ax[0][0].set_title('Trans-tube')
        ax[0][1].set_title('Cylinder Liner')
        ax[0][2].set_title('Intake Piston')
        ax[0][3].set_title('Exhaust Piston')
        if not user_given_fig:
            [GetGridLine(
                ax=axis, rslt=rslt, linewidth=2
            ) for axis in ax.flat]
        fig.suptitle('CFD: Heat transfer out of surfaces')

        return fig, ax


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
                ],
                'boundary_based': [
                    {'bound': filefmt('bound#-wall*')},
                ],
                'flow_based': [
                    {'regions_flow': filefmt('regions_flow*')},
                    {'mass_avg_flow': filefmt('mass_avg_flow*')},
                ],
                'monitor_point_based': [
                    {'mon_pt': filefmt('monitor*_point_#_mass_avg')}
                ]
            }
        )
        self.mon_pts = CFDDict(
            {
                'cyl_center': int(3),
                'int_pist': int(4),
                'exh_pist': int(5),
                'cyl_int_port': int(1),
                'cyl_exh_port': int(2),
                'int_man_measured': int(6),
                'exh_man_measured': int(7),
                'int_pist_crev': [10, 9, 8, 11],
                'exh_pist_crev': [12, 13, 14, 15],
            }
        )
