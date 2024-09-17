import numpy as np
import time; import datetime
import argparse
import sys

import data
from cross_sec import CrossSection
from figures import Figures

if __name__ == '__main__':

    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_file', type=str, #required=True, 
        help='Name of input file (e.g. input.example_ExoMol)', 
        )

    parser.add_argument('--cross_sections', '-cs', action='store_true')
    parser.add_argument('--save', '-s', action='store_true')
    parser.add_argument('--plot', action='store_true')

    parser.add_argument('--convert_to_pRT2', action='store_true')
    parser.add_argument('--append_to_existing', action='store_true')

    # Index to read if multiple .trans files are given
    parser.add_argument('--trans_idx_min', '-i_min', default=0, type=int)
    parser.add_argument('--trans_idx_max', '-i_max', default=1, type=int)
    parser.add_argument('--show_pbar', action='store_true')
    
    # optionally, overwrite the P and T values of conf
    parser.add_argument('--P', type=float, default=None)
    parser.add_argument('--T', type=float, default=None)
    parser.add_argument('--combine_grid', action='store_true')
    parser.add_argument('--ncpus', type=int, default=1)
    parser.add_argument('--species', type=str, default=None)
    args = parser.parse_args()

    # Import input file as 'conf'
    # input_string = str(args.input_file).replace('.py', '').replace('/', '.')
    input_string = str(args.input_file).split('.py')[0].replace('/', '.')
    species = str(input_string.split('-s')[-1])
    conf = __import__(input_string, fromlist=[''])
    
    if args.species is not None:
        from input.atomic_mass import atomic_mass_dict
        conf.species = args.species
        conf.mass = atomic_mass_dict[args.species]
        print(f' Updated species: {conf.species} | Mass: {conf.mass}')
        
        conf.input_dir = str(conf.input_dir).replace('Fe', conf.species)
        for key, val in conf.files.items():
            conf.files[key] = val.replace('Fe', conf.species)
        conf.pRT['out_dir'] = conf.pRT['out_dir'].replace('Fe', conf.species)
        
        if args.species in ['Na', 'K']:
            conf.max_nu_sep = 4500.0 # [cm^-1]
            conf.wing_cutoff = lambda gamma_V, P: conf.max_nu_sep
            
        # manual fix for species with VALD data instead of Kurucz
        VALD_species = ['Cs', 'Rb']
        if args.species in VALD_species:
            conf.database = 'VALD'
            conf.files['transitions'] = conf.files['transitions'].replace('Kurucz', 'VALD')
    
    if (args.P is not None) and (args.T is not None):
        conf.P_grid = np.array([args.P])
        conf.T_grid = np.array([args.T])
        # update tmp file to avoid overwriting
        conf.files['tmp_output'] = conf.files['tmp_output'].replace('.hdf5', f'_P{args.P:.0e}_T{args.T:.0f}.hdf5')
        
    
        print(f' Updated values of PT grid, saving to file: {conf.files["tmp_output"]}')
        # check im tmp_output file exists, skip if it does
        import pathlib
        if pathlib.Path(conf.files['tmp_output']).exists():
            print(' File already exists, skipping...')
            sys.exit()
            
    # Load data
    D = data.load_data(conf)
    trans_file      = conf.files['transitions']
    tmp_output_file = conf.files['tmp_output']
        
    if args.cross_sections:
        show_pbar = True
        if args.show_pbar:
            show_pbar = True

        time_start = time.time()
        # Compute + save cross-sections
        CS = CrossSection(conf, Q=D.Q, mass=D.mass)
        CS = D.get_cross_sections(CS, trans_file, show_pbar=show_pbar)
        CS.save_cross_sections(tmp_output_file)
        
        time_finish = time.time()
        time_elapsed = time_finish - time_start
        if not show_pbar:
            print('Time elapsed (total): {}'.format(str(datetime.timedelta(seconds=time_elapsed))))

    if args.combine_grid:
        # Combine cross-sections from different (P,T) files into final output file
        D.combine_cross_sections_grid()
        
    if args.plot:
        F = Figures(
            D, wave_range=[(1/3,50.0), (1.05,1.35), (1.9,2.5), (2.29,2.4), (2.332,2.339)]
            )
        # F.plot_P(
        #     T=1000, P=10**np.array([-4,-2,0,2], dtype=np.float64), 
        #     ylim=(1e-28,1e-16)
        #     )
        F.plot_T(
            P=1, T=np.array([2000,2250,2500,3000,4000]), 
            ylim=(1e-28,1e-16)
            )
        
    if args.convert_to_pRT2:
        D.convert_to_pRT2_format(
            out_dir=conf.pRT['out_dir'], 
            pRT_wave_file=conf.pRT['wave'], 
            make_short_file=conf.pRT['make_short_file'],
            ncpus=args.ncpus
        )