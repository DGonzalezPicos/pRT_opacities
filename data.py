import numpy as np

from pandas import read_fwf, read_csv
import bz2
import re
import h5py

from scipy.interpolate import interp1d
import scipy.constants as sc

c2 = 1.438777      # cgs: [cm K]
e = 4.80320425e-10 # cgs: [cm^3/2 g^1/2 s^-1]

import itertools
import pathlib
import wget
import json

from tqdm import tqdm

def load_data(conf):
    if conf.database.lower() in ['hitemp', 'hitran']:
        return HITEMP(conf)
    if conf.database.lower() == 'exomol':
        return ExoMol(conf)
    if conf.database.lower() in ['vald', 'kurucz']:
        return VALD_Kurucz(conf)

    raise Exception(f'Database \"{conf.database}\" not recognised.')

def wget_if_not_exist(url, out_dir, out_name=None):
    
    if out_name is None:
        out_name = out_dir + url.split('/')[-1]

    if pathlib.Path(out_name).is_file():
        print(f'File \"{out_name}\" already exists, skipping download')
        return out_name
    
    # Download and rename
    tmp_file = wget.download(url, out=out_dir)
    tmp_file = pathlib.Path(tmp_file)

    out_name = tmp_file.rename(out_name)
    return str(out_name)

class LineList:
    
    @classmethod
    def load_hdf5_output(cls, file, datasets_to_read=['wave','cross_sec','P','T']):

        data = []
        with h5py.File(file, 'r') as f:
            for key_i in datasets_to_read:
                
                # Make an array [...]
                if key_i in ['wave', 'T']:
                    # [m] or [K]
                    data.append(f[key_i][...])
                elif key_i in ['cross_sec', 'P']:
                    # [m^2] or [Pa]
                    data.append(10**f[key_i][...]-1e-250)
                else:
                    raise KeyError(f'Dataset \"{key_i}\" not in \"{file}\"')

        return data
    
    def __init__(self, conf):

        # Atomic mass [kg]
        self.mass = conf.mass * 1.0e-3/sc.N_A

        self.final_output_file = conf.files['final_output']
        self.N_lines_max = getattr(conf, 'N_lines_max', 10_000_000)           

        # Pressure-broadening parameters
        self.broad = {}
        if hasattr(conf, 'broadening'):
            self._load_pressure_broad(conf.broadening)

    def _load_pressure_broad(self, broadening):

        for species_i, input_i in broadening.items():
            
            # Volume-mixing ratio
            self.broad[species_i] = input_i.copy()

            # Use gamma and T-exponent from input-dictionary
            # gamma * (T0/T)^n * (P/1bar)
            gamma_i = input_i.get('gamma', 0.0)
            if callable(gamma_i):
                self.broad[species_i]['gamma'] = gamma_i
            else:
                self.broad[species_i]['gamma'] = np.array([gamma_i])
            self.broad[species_i]['n'] = np.array([input_i.get('n', 0.0)])

            file_i = input_i.get('file')
            try:
                # Read broadening parameters from file
                br = read_fwf(file_i, header=None)
            except (FileNotFoundError, ValueError) as e:
                # No broadening parameters in file
                continue

            label_i = np.array(br[0], dtype=str)
            gamma_i, n_i = np.array(br[[1,2]]).T

            # Currently only handles these 2 broadening diets
            mask_label = (label_i == 'a0')
            if not mask_label.any():
                mask_label = (label_i == 'm0')
                self.broad[species_i]['label'] = 'm0'

            self.broad[species_i]['gamma'] = gamma_i[mask_label]
            self.broad[species_i]['n']     = n_i[mask_label]

            if (br.shape[0]==1) or (br.shape[1]==3):
                # Single row or no quantum-number columns, 
                # ignore quantum-number dependency
                self.broad[species_i]['gamma'] = np.nanmean(
                    self.broad[species_i]['gamma'], keepdims=True
                    )
                self.broad[species_i]['n'] = np.nanmean(
                    self.broad[species_i]['n'], keepdims=True
                    )
                continue

            # Total angular momentum quantum number
            self.broad[species_i]['J'] = np.array(br[3])[mask_label]
            if br.shape[1] == 4:
                # No (more) quantum numbers in file
                continue

            # TODO: additional quantum numbers
            ## Rotational quantum number
            #self.broad[species_i]['K'] = np.array(br[4])

    def load_final_output(self):
        # Convert to [um], [cm^2/molecule], [bar], [K]
        wave, sigma, P, T = self.load_hdf5_output(self.final_output_file)
        return wave*1e6, sigma*1e4, P*1e-5, T

    def convert_to_pRT2_format(self, out_dir, pRT_wave_file, make_short_file, debug=False, ncpus=1):

        def process_PT_point(P, T, wave_cm, sigma_PT, pRT_wave, out_dir, debug=False):
            """
            Helper function to parallelize the interpolation to pRT2 and saving for each (P,T) point.
            """
            pRT_file = f'{out_dir}/sigma_{T:.0f}.K_{P:.06f}bar.dat'
            if debug:
                print(pRT_file)

            # Interpolate onto pRT's wavelength-grid
            interp_func = interp1d(wave_cm, sigma_PT, bounds_error=False, fill_value=0.0)
            interp_sigma = interp_func(pRT_wave)

            # Save the result to the file
            np.savetxt(pRT_file, np.column_stack((pRT_wave, interp_sigma)))

            return P, T, pRT_file.split('/')[-1]

        print('\nConverting to pRT2 opacity format')

        # Load output
        wave_micron, sigma, P_grid, T_grid = self.load_final_output()
        wave_cm = wave_micron * 1e-4

        # Load pRT wavelength-grid
        pRT_wave = np.genfromtxt(pRT_wave_file)

        # Create directory if not exist
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

        PTpaths = []

        # Prepare the list of tasks
        tasks = []
        for idx_P, P in enumerate(P_grid):
            for idx_T, T in enumerate(T_grid):
                sigma_PT = sigma[:, idx_P, idx_T]
                tasks.append((P, T, wave_cm, sigma_PT, pRT_wave, out_dir, debug))

        # Make a nice progress bar
        pbar_kwargs = dict(
            total=len(tasks),
            bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
        )

        if ncpus > 1:
            # Use multiprocessing with the specified number of CPUs
            import multiprocessing as mp
            with mp.Pool(ncpus) as pool:
                with tqdm(**pbar_kwargs) as pbar:
                    for result in pool.starmap(process_PT_point, tasks):
                        PTpaths.append(result)
                        pbar.update(1)
        else:
            # Run sequentially without multiprocessing
            with tqdm(**pbar_kwargs) as pbar:
                for task in tasks:
                    result = process_PT_point(*task)
                    PTpaths.append(result)
                    pbar.update(1)

        # Create directory if not exist
        short_stream_dir = f'{out_dir}/short_stream/'
        pathlib.Path(short_stream_dir).mkdir(parents=True, exist_ok=True)

        # Save pressures/temperatures corresponding to each file
        np.savetxt(
            f'{short_stream_dir}/PTpaths.ls', np.array(PTpaths), delimiter=' ', fmt='%s'
        )

        # Create pRT input-files
        short_stream_lambs_mass = [
            '# Minimum wavelength in cm', '0.3d-4', 
            '# Maximum wavelength in cm', '28d-4', 
            '# Molecular mass in amu', 
            '{:.3f}d0'.format(self.mass*sc.N_A*1e3), 
        ]
        with open(f'{out_dir}/short_stream_lambs_mass.dat', 'w') as f:
            f.write('\n'.join(short_stream_lambs_mass))

        sigma_list = list(np.array(PTpaths)[:,2])
        with open(f'{out_dir}/sigma_list.ls', 'w') as f:
            f.write('\n'.join(sigma_list))

        molparam_id = [
            '#### Species ID (A2) format', '06', 
            '#### molparam value', '1', 
        ]
        with open(f'{short_stream_dir}/molparam_id.txt', 'w') as f:
            f.write('\n'.join(molparam_id))

        # Copy the make_short.f90 fortran-script over
        import shutil
        shutil.copy(make_short_file, dst=f'{out_dir}/make_short.f90')

        # Make executable and run
        import subprocess
        subprocess.run(['gfortran', '-o', f'{out_dir}/make_short', f'{out_dir}/make_short.f90'])
        subprocess.call('./make_short', cwd=f'{out_dir}')

        # Remove temporary files (on pRT's low-res wavelengths)
        for file_to_remove in np.array(PTpaths)[:,2]:
            file_to_remove = pathlib.Path(f'{out_dir}/{file_to_remove}')
            file_to_remove.unlink()

        if isinstance(self, HITEMP):
            print('\n'+'#'*50)
            print('Database is HITRAN/HITEMP, so line-strengths are scaled by solar isotope ratio.\n\
You may need to change the \"molparam\" value in \"molparam_id.txt\".')
            print('#'*50)
            
    def convert_to_pRT3_format(self,
                               out_dir, 
                               pRT_wave_file,
                               isotopologue_id={'C': 12, 'O': 16}, # --> "12C-O16"
                               debug=False,
                               **kwargs):
        
        print('\nConverting to pRT3 opacity format')

        # Load output: wave [um], sigma [cm^2/molecule], P [bar], T [K]
        wave_um, sigma, P_bar, T = self.load_final_output()
        wave_cm = wave_um * 1e-4

        # Load pRT wavelength-grid
        pRT_wave = np.genfromtxt(pRT_wave_file) # [cm]
        if kwargs.get('crop_to_28um', False): # Set maximum wavelength to 28um, common for pRT3
            pRT_wave = pRT_wave[pRT_wave <= 28e-4]
            

        # Create directory if not exist
        out_dir = out_dir.replace('pRT2', 'pRT3')
        pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)


        # (wave.size, P.size, T.size) -> (P.size, T.size, wave.size)
        sigma = np.moveaxis(sigma, 0, -1)
        
        # Interpolate onto pRT's wavelength-grid
        new_sigma = np.zeros((sigma.shape[0], sigma.shape[1], pRT_wave.size))
        for i in range(sigma.shape[0]):
            for j in range(sigma.shape[1]):
                new_sigma[i,j] = np.interp(pRT_wave, wave_cm, sigma[i,j], left=0.0, right=0.0)
                
        # 2) Convert sigma to variable `xsecarr` 
        wavenumbers = 1 / pRT_wave
        wavenumbers = wavenumbers[::-1]
        wave_um_pRT = 1e4 * (1 / wavenumbers) # [cm^-1] -> [um]
        
        # check valid wavelength range
        wave_um_min = max(wave_um.min(), wave_um_pRT.min())
        wave_um_max = min(wave_um.max(), wave_um_pRT.max())
        
        xsecarr = new_sigma[:,:,::-1] # (P, T, wavenumber)
        del sigma
        
        if debug:
            print(f'[convert_to_pRT3_format] wave_um min: {wave_um.min():.2e} um wave_um max: {wave_um.max():.2e} um')
            print(f'[convert_to_pRT3_format] xsecarr shape: {xsecarr.shape}')
        
        
        # Save in pRT3 format following:
        # https://gitlab.com/mauricemolli/petitRADTRANS/-/blob/master/petitRADTRANS/__file_conversion.py

        # examples: 
        # 1) isotopologue_id={'K':39}           --> '39K'
        # 2) isotopologue_id={'C': 12, 'O': 16} --> '12C-16O'
        # 3) isotopologue_id={'H2':1, 'O':18}   --> '1H2-18O'
        # 4) isotopologue_id={'C':12, 'H4':1}   --> '12C-1H4'
        species = "".join(list(isotopologue_id.keys()))
        species_isotopologue_name = "-".join([f"{v}{k}" for k,v in isotopologue_id.items()])
        mass = self.atoms_info.loc[species,'mass']
        if debug:
            print(f"[convert_to_pRT3_format] Atomic mass of {species_isotopologue_name}: {mass}")
        
        source = getattr(self, 'database', 'UNKNOWN')
        if source == 'UNKNOWN':
            print('WARNING: No database source found, set using "database" attribute. Defaulting to "UNKNOWN"')
            
        resolving_power = kwargs.get('resolving_power', 1e6)
        file_pRT3 = get_opacity_filename(resolving_power=resolving_power,
                                            wavelength_boundaries=[wave_um_min, wave_um_max],
                                            species_isotopologue_name=species_isotopologue_name,
                                            source=source,
        )

        # output_dir = pathlib.Path("/net/lem/data2/pRT3/input_data/opacities/lines/line_by_line/") / species / species_isotopologue_name
        # assert output_dir.exists(), f"Output directory {output_dir} does not exist"
        out_dir_species = pathlib.Path(out_dir) / species / species_isotopologue_name
        out_dir_species.mkdir(parents=True, exist_ok=True)
        
        hdf5_opacity_file = out_dir_species / f'{file_pRT3}.xsec.petitRADTRANS.h5'
        if debug:
            print(f'[convert_to_pRT3_format] Saving to file: {hdf5_opacity_file}...')

        print(hdf5_opacity_file)
        doi = kwargs.get('doi', 'None')
        contributor = kwargs.get('contributor', 'LEM') # default to LEM (Leiden Exoplanet Machine)
        description = kwargs.get('description', 'Converted from `cs_package` format to `pRT3` format')

        write_line_by_line(hdf5_opacity_file, 
                        doi,
                        wavenumbers, 
                        xsecarr, 
                        mass,
                        species, 
                        np.unique(P_bar),
                        np.unique(T), 
                        wavelengths=None, 
                        contributor=contributor,
                        description=description)
        
        

    def combine_cross_sections(self, tmp_dir):

        print(f'\nCombining temporary cross-sections to file \"{self.final_output_file}\"')

        # Check compatibility of files before combining
        print(f'Looking in directory \"{tmp_dir}\"')
        tmp_files = sorted(pathlib.Path(tmp_dir).glob('*.hdf5'))

        all_PT = []
        for i, tmp_file_i in enumerate(tmp_files):
            
            wave_i, P_i, T_i = self.load_hdf5_output(
                tmp_file_i, datasets_to_read=['wave','P','T']
                )
            
            for P_ij, T_ij in itertools.product(P_i, T_i):
                all_PT.append([P_ij,T_ij])
            
            if i == 0:
                wave = wave_i.copy()            
            assert (wave == wave_i).all(), 'Wavelengths do not match'

        # Check if PT grid is rectangular
        all_PT = np.array(all_PT)
        unique_P = np.unique(all_PT[:,0])
        for i, P_i in enumerate(unique_P):
            unique_T_i = all_PT[all_PT[:,0]==P_i,1]
            if i == 0:
                unique_T = unique_T_i.copy()
            
            assert (unique_T == unique_T_i).all(), 'PT-grid is not rectangular'

        unique_P = np.sort(unique_P)
        unique_T = np.sort(unique_T)

        # Combine all files into a single cross-section array
        sigma_tot = np.zeros((len(wave),len(unique_P),len(unique_T)), dtype=np.float64)
        for i, tmp_file_i in enumerate(tmp_files):
            
            # Load the opacity of each file
            sigma_i, P_i, T_i = self.load_hdf5_output(
                tmp_file_i, datasets_to_read=['cross_sec','P','T']
                )
            
            # Insert this opacity-array in the total array
            for j, P_ij in enumerate(P_i):
                idx_P = np.argwhere((unique_P == P_ij)).flatten()[0]

                for k, T_ik in enumerate(T_i):
                    idx_T = np.argwhere((unique_T == T_ik)).flatten()[0]
                    sigma_tot[:,idx_P,idx_T] += sigma_i[:,j,k]


        # Create directory if not exist
        pathlib.Path(self.final_output_file).parent.mkdir(parents=True, exist_ok=True)

        # Save in a single file
        with h5py.File(self.final_output_file, 'w') as f:
            # Rounding to save memory
            dat1 = f.create_dataset(
                'cross_sec', compression='gzip', 
                data=np.around(np.log10(sigma_tot+1e-250),decimals=3)
                )
            dat1.attrs['units'] = 'log(m^2/molecule)'
            
            dat2 = f.create_dataset('wave', compression='gzip', data=wave)
            dat2.attrs['units'] = 'm'

            # Add 1e-250 to allow zero pressure
            dat3 = f.create_dataset('P', compression='gzip', data=np.log10(unique_P+1e-250))
            dat3.attrs['units'] = 'log(Pa)'

            dat4 = f.create_dataset('T', compression='gzip', data=unique_T)
            dat4.attrs['units'] = 'K'


class ExoMol(LineList):

    database = 'exomol'
    
    @classmethod
    def download_data(cls, conf):

        url_def_json = conf.url_def_json
        url_broad = getattr(conf, 'url_broad', None)
        input_dir = conf.input_dir

        # Create destination directory
        pathlib.Path(input_dir).mkdir(parents=True, exist_ok=True)

        # Download the definition file
        def_file = wget_if_not_exist(url_def_json, input_dir)
        url_base = url_def_json.replace('.json', '')
        print()

        # Read definition file
        with open(def_file) as f:
             d = json.load(f)

        # Download partition-function and states files
        print('\nPartition file:')
        partition_file = wget_if_not_exist(f'{url_base}.pf', input_dir)
        print()
        print(pathlib.Path(partition_file).absolute())

        print('\nStates file:')
        states_file = wget_if_not_exist(f'{url_base}.states.bz2', input_dir)
        print()
        print(pathlib.Path(states_file).absolute())
        
        trans_files = []
        print('\nTransition file(s):')
        # Download transition files
        N_trans = d['dataset']['transitions']['number_of_transition_files']
        if N_trans == 1:
            trans_file_i = wget_if_not_exist(f'{url_base}.trans.bz2', input_dir)
            print()
            print(pathlib.Path(trans_file_i).absolute())
        else:
            nu_min = np.linspace(
                0, d['dataset']['transitions']['max_wavenumber'], N_trans, 
                endpoint=False, dtype=int
                )
            delta_nu = nu_min[1]-nu_min[0]

            for nu_min_i in nu_min:
                nu_max_i = nu_min_i + delta_nu

                trans_file_i = wget_if_not_exist(
                    '{}__{:05d}-{:05d}.trans.bz2'.format(url_base, nu_min_i, nu_max_i), 
                    out_dir=input_dir
                    )
                trans_files.append(pathlib.Path(trans_file_i).absolute())
            
            print()
            for trans_file_i in trans_files:
                print(trans_file_i)

        if url_broad is None:
            return
        
        print('\nBroadening files:')
        broad_files = []
        # Download broadening files
        for url_broad_i in url_broad:
            broad_file_i = wget_if_not_exist(url_broad_i, input_dir)
            broad_files.append(pathlib.Path(broad_file_i).absolute())
        
        print()
        for broad_file_i in broad_files:
            print(broad_file_i)
        print()

    def __init__(self, conf):

        assert('H2_broadening' not in conf.files)
        assert('He_broadening' not in conf.files)
        assert(hasattr(conf, 'broadening'))

        # Instantiate the parent class
        super().__init__(conf)
        #print(self.broad)

        # Partition function
        self.Q = np.loadtxt(conf.files['partition_function'])
        
        # Load states-info (ExoMol-specific)
        self.states = self._read_states(conf.files['states'])

    def _read_states(self, file):

        print(f'Reading states from \"{file}\"')

        # How pandas should handle compression
        comp = pathlib.Path(file).suffix.replace('.', '')
        if comp != 'bz2':
            comp = 'infer'
            f = open(file)
        else:
            f = bz2.open(file)

        # Infer column-widths
        col_0 = re.findall('\s+\S+', str(f.readline()))
        col_widths = [len(col) for col in col_0]
        f.close()

        # Load states (ID, E, g, J)
        states = read_fwf(
            file, widths=col_widths[:4], header=None, compression=comp
            )
        states = np.array(states)

        # Check that all states are read
        #assert(np.all(np.diff(states[:,0])==1))
        #assert(states[0,0]==1)

        return states
    
    def _get_pressure_broad(self, J_lower, J_upper, chunk_size=100_000):

        broad_per_trans = {}
        for species_i, broad_i in self.broad.items():

            broad_per_trans[species_i] = dict(VMR=broad_i['VMR'])

            label_i = broad_i.get('label')
            gamma_i = broad_i['gamma']
            n_i     = broad_i['n']

            if callable(gamma_i):
                # User-provided function
                broad_per_trans[species_i]['gamma'] = gamma_i(J_lower)
            else:
                # Mean gamma if not in broadening table
                broad_per_trans[species_i]['gamma'] = np.nanmean(gamma_i)*np.ones_like(J_lower)

            if callable(n_i):
                # User-provided function
                broad_per_trans[species_i]['n'] = n_i(J_lower)
            else:
                # Mean gamma if not in broadening table
                broad_per_trans[species_i]['n'] = np.nanmean(n_i)*np.ones_like(J_lower)

            #print(gamma_i)
            #print(broad_per_trans[species_i])

            if callable(gamma_i) or callable(n_i):
                continue

            J_i = broad_i.get('J')
            if J_i is None:
                # No quantum-number dependency
                continue
            
            # Read in chunks to avoid memory overload
            for idx_l in range(0, len(J_lower), chunk_size):
                idx_h = min([idx_l+chunk_size,len(J_lower)])

                J_to_match = J_lower[idx_l:idx_h]

                if label_i == 'm0':
                    # Check if transition in R-branch (i.e. lower J quantum 
                    # number is +1 higher than upper state).
                    # In that case, 4th column in .broad is |m|=J_l+1.
                    delta_J = J_lower[idx_l:idx_h]-J_upper[idx_l:idx_h]
                    J_to_match[(delta_J == +1)] += 1
                
                # Indices in .broad table corresponding to each transition
                indices_J = np.argwhere(J_i[None,:]==J_to_match[:,None])

                # Update each transition's broadening parameter
                broad_per_trans[species_i]['gamma'][idx_l+indices_J[:,0]] = gamma_i[indices_J[:,1]]
                broad_per_trans[species_i]['n'][idx_l+indices_J[:,0]]     = n_i[indices_J[:,1]]
                
        return broad_per_trans
    
    def get_cross_sections(self, CS, file, show_pbar=True, debug=False):

        print(f'\nComputing cross-sections from \"{file}\"')

        i = 0
        state_ID_u = []; state_ID_l = []; A = []

        with bz2.open(file) as f:

            while True:
                # Compute opacities in chunks to prevent memory-overload
                line = f.readline()

                if i == 0:
                    # Infer column-widths in .trans file (only 1st line)
                    sep_line_0 = re.findall('\s+\S+', str(line))
                    col_widths = [len(col) for col in sep_line_0]
                    col_idx = np.cumsum(col_widths)

                elif (i%self.N_lines_max == 0) or (not line):
                    
                    # Compute when N_lines_max have been read, or end-of-file
                    
                    if len(A) == 0:
                        # Last line was included in previous chunk 
                        # (i.e. len(file) == X*N_lines_max)
                        return CS

                    if not line:
                        print(f'Computing for {len(A)} transitions in final chunk')
                    else:
                        print(f'Computing for {len(A)} transitions in chunk')
                        
                    idx_u = np.searchsorted(self.states[:,0], np.array(state_ID_u, dtype=int))
                    idx_l = np.searchsorted(self.states[:,0], np.array(state_ID_l, dtype=int))

                    E_l = self.states[idx_l,1].astype(np.float64)
                    g_u = self.states[idx_u,2]
                    
                    J_l = self.states[idx_l,3].astype(int)
                    J_u = self.states[idx_u,3].astype(int)

                    E_u = self.states[idx_u,1].astype(np.float64)
                    nu_0 = E_u - E_l
                    nu_0 = np.abs(nu_0.astype(np.float64))

                    # Compute line-strength at reference temperature
                    term1 = np.array(A, dtype=np.float64)*g_u / (8*np.pi*(100*sc.c)*nu_0**2)
                    term2 = np.exp(-c2*E_l/CS.T_0) / CS.q_0
                    term3 = (1-np.exp(-c2*nu_0/CS.T_0))
                    S_0 = term1 * term2 * term3

                    # Add to lines to compute at the next iteration
                    nu_0 *= (100*sc.c) # Unit conversion
                    E_l  *= sc.h * (100*sc.c)
                    S_0  *= (100*sc.c)

                    idx_sort = np.argsort(nu_0)
                    nu_0 = nu_0[idx_sort]
                    E_l  = E_l[idx_sort].astype(np.float64)
                    S_0  = S_0[idx_sort].astype(np.float64)

                    # Get J-specific broadening parameters
                    broad_per_trans = self._get_pressure_broad(J_lower=J_l, J_upper=J_u)

                    # Next iteration, compute opacities
                    CS.loop_over_PT_grid(
                        func=CS.get_cross_sections, show_pbar=show_pbar, 
                        nu_0_static=nu_0, E_low=E_l, S_0=S_0, 
                        broad_per_trans=broad_per_trans, 
                        )
                    
                    if not line:
                        # End-of-file
                        return CS
                    
                    # Empty lists
                    state_ID_u = []; state_ID_l = []; A = []

                # Read transition-info for each 
                # Access info on upper and lower states
                state_ID_u.append(line[0:col_idx[0]])
                state_ID_l.append(line[col_idx[0]:col_idx[1]])
                A.append(line[col_idx[1]:col_idx[2]])

                i += 1

class HITEMP(LineList):
    database = 'hitemp'
    
    @classmethod
    def download_data(cls, conf):
        
        urls = conf.urls
        input_dir = conf.input_dir

        # Create destination directory
        pathlib.Path(input_dir).mkdir(parents=True, exist_ok=True)

        files = []
        for url_i in urls:
            file_i = wget_if_not_exist(url_i, input_dir)
            files.append(pathlib.Path(file_i).absolute())

        print()
        for file_i in files:
            print(file_i)
        print()

    def __init__(self, conf):

        assert('H2_broadening' not in conf.files)
        assert('He_broadening' not in conf.files)
        assert(hasattr(conf, 'broadening'))

        # Instantiate the parent class
        super().__init__(conf)
        
        # Remove any quantum-number dependency as we cannot 
        # match ExoMol state IDs with HITRAN/HITEMP
        for species_i, broad_i in self.broad.items():
            self.broad[species_i]['gamma'] = np.nanmean([broad_i.get('gamma', 0.0)])
            self.broad[species_i]['n']     = np.nanmean([broad_i.get('n', 0.0)])
        #print(self.broad)

        # Partition function
        self.Q = np.loadtxt(conf.files['partition_function'])

        # Isotope index (HITEMP/HITRAN-specific)
        self.isotope_idx = getattr(conf, 'isotope_idx', 1)

    def get_cross_sections(self, CS, file, show_pbar=True):

        print(f'\nComputing cross-sections from \"{file}\"')

        # How pandas should handle compression
        comp = pathlib.Path(file).suffix
        if comp != 'bz2':
            comp = 'infer'

        # Read only N_lines_max lines to prevent memory-overload
        zip_file = read_fwf(
            file, widths=(2,1,12,10,10,5,5,10,4,8), 
            header=None, chunksize=self.N_lines_max, 
            compression=comp
        )
        for trans_i in zip_file:
            trans_i = np.array(trans_i)

            if self.isotope_idx is not None:
                # Only lines from one isotope
                print(f'Reading isotope-index {self.isotope_idx}')
                trans_i = trans_i[trans_i[:,1]==self.isotope_idx]

            # Sort lines by central wavenumber (!! might only work if delta_P==0 !!)
            trans_i = trans_i[np.argsort(trans_i[:,2]),:]

            # Unit conversion
            nu_0  = trans_i[:,2] * (100*sc.c)        # [cm^-1] -> [s^-1]
            E_low = trans_i[:,7] * sc.h * (100*sc.c) # [cm^-1] -> [kg m^2 s^-2] or [J]
            S_0   = trans_i[:,3] * (100*sc.c)        # [cm^-1/(molec. cm^-2)] -> [s^-1/(molec. cm^-2)]
            
            # Compute opacities
            CS.loop_over_PT_grid(
                func=CS.get_cross_sections, show_pbar=show_pbar, 
                nu_0_static=nu_0, E_low=E_low, S_0=S_0, 
                broad_per_trans=self.broad, 
                delta_P=np.zeros_like(nu_0) # !! for P-dependent shifts !!
                )
        
        return CS

class VALD_Kurucz(LineList):

    directory_path = pathlib.Path(__file__).parent.resolve()
    atoms_info = read_csv(directory_path/'atoms_info.csv', index_col=0)

    @classmethod
    def download_data(cls, conf):

        # Create destination directory
        input_dir = conf.input_dir
        pathlib.Path(input_dir).mkdir(parents=True, exist_ok=True)

        # Download the NIST energy levels
        # Validate the element input
        element = conf.species
        assert(element in cls.atoms_info.index)
        
        # Construct the NIST URL for downloading energy levels
        url = f'https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0&\
spectrum={element}+I&submit=Retrieve+Data&units=0&format=3&output=0&page_size=15&\
multiplet_ordered=0&conf_out=on&term_out=on&level_out=on&unc_out=1&j_out=on&g_out=on&\
lande_out=on&biblio=on&temp='
        
        # If no filename is provided, use the default naming convention
        states_file = conf.files.get(
            'states', f'{input_dir}/NIST_levels_tab_delimited.tsv'
            )
        
        states_file = wget_if_not_exist(url, out_dir=input_dir, out_name=states_file)
        print()
        print('States file:')
        print(pathlib.Path(states_file).absolute())
        print()

        if conf.database.lower() == 'vald':
            return
        
        # Download the Kurucz transitions
        atomic_number    = cls.atoms_info.loc[element, 'number']
        ionisation_state = getattr(conf, 'ionisation_state', 0)

        # Format the atomic number and ionization state for constructing the URL
        atom_id = f'{atomic_number:02d}{ionisation_state:02d}'
        print(f'{element} has ID {atom_id}')

        # If no filename is provided, use the default naming convention
        trans_file = conf.files.get(
            'trans', f'{input_dir}/Kurucz_transitions.txt'
            )
        
        # Try different extension
        for extension in ['pos','all']:
            url = f'http://kurucz.harvard.edu/atoms/{atom_id}/gf{atom_id}.{extension}'
            try:
                trans_file = wget_if_not_exist(url, out_dir=input_dir, out_name=trans_file)
                continue
            except:
                pass
        print()
        print('Transitions file:')
        print(pathlib.Path(trans_file).absolute())
        print()
    
    def __init__(self, conf):

        species = getattr(conf, 'species', None)
        if species in self.atoms_info.index:
            conf.mass = self.atoms_info.loc[species,'mass']
            print(f'Using mass from \"atoms_info.csv\": {conf.mass}')

        # Instantiate the parent class
        super().__init__(conf)

        # Different file-formats ['vald', 'kurucz']
        self.database = conf.database.lower()

        # Compute partition function from state-info
        self._get_partition_from_NIST_states(conf.files['states'])

        # If alkali, provide E_ion and Z for different vdW-broadening
        self.E_ion = getattr(conf, 'E_ion', None) # [cm^-1]
        self.Z     = getattr(conf, 'Z', None)

        # Transition-energies to ignore
        self.nu_0_to_ignore = getattr(conf, 'nu_0_to_ignore', None)

    def _get_partition_from_NIST_states(self, file, T=np.arange(1,5001+1e-6,1)):

        print(f'Reading states from \"{file}\"')

        # Load states
        states = read_csv(
            file, sep='\t', engine='python', 
            header=0, #skipfooter=1
            )
        g = np.array(states['g'])
        E = np.array(states['Level (cm-1)'])

        # (higher) ions beyond this index
        idx_u = np.min(np.argwhere(np.isnan(g)))
        g = g[:idx_u]
        E = E[:idx_u]

        # Partition function
        self.Q = np.sum(
            g[None,:] * np.exp(-c2*E[None,:]/T[:,None]), 
            axis=-1 # Sum over states, keep T-axis
            )
        self.Q = np.concatenate((T[:,None], self.Q[:,None]), axis=-1)

    def _read_VALD_transitions(self, file):
        
        # Read all transitions at once
        with open(file, 'r') as f:
            trans = np.array([
                [
                    float(line[7:23].strip()),
                    float(line[24:36].strip()),
                    float(line[37:44].strip()),
                    float(line[45:51].strip()),
                    float(line[59:65].strip())
                ]
                for line in f.readlines()[2:]  # Skip the header line and column names
                if len(line.strip()) > 65  # Filter out footer or irrelevant lines
            ], dtype=np.float64)

        # Oscillator strength
        gf = 10**trans[:,2]

        # Transition energies
        nu_0  = trans[:,0] # [cm^-1]
        E_low = trans[:,1] # [cm^-1]

        # Single vdW-damping given
        gamma_vdW = 10**trans[:,4] # [s^-1 cm^3]
        
        # Log radiative/natural damping
        log_gamma_N = trans[:,3]

        return nu_0, E_low, gf, gamma_vdW, log_gamma_N

    def _read_Kurucz_transitions(self, file):
        
        # Read all transitions at once
        trans = read_fwf(
            file, widths=(11,7,6,12,5,1,10,12,5,1,10,6,6,6), header=None, 
            )
        trans = np.array(trans)

        # Oscillator strength
        gf = 10**trans[:,1].astype(np.float64)

        # Transition energies
        E_low  = trans[:,3].astype(np.float64) # [cm^-1]
        E_high = trans[:,7].astype(np.float64) # [cm^-1]
        
        # Kurucz line list are sorted by parity?
        nu_0  = np.abs(E_high-E_low)
        E_low = np.min(np.concatenate((E_low[None,:],E_high[None,:])), axis=0)

        # Single vdW-damping given
        gamma_vdW = 10**trans[:,13].astype(np.float64) # [s^-1 cm^3]
        
        # Log radiative/natural damping
        log_gamma_N = trans[:,11].astype(np.float64)

        return nu_0, E_low, gf, gamma_vdW, log_gamma_N

    def get_cross_sections(self, CS, file, show_pbar=True):

        print(f'\nComputing cross-sections from \"{file}\"')

        if self.database == 'vald':
            nu_0, E_low, gf, gamma_vdW, log_gamma_N = \
                self._read_VALD_transitions(file)
        elif self.database == 'kurucz':
            nu_0, E_low, gf, gamma_vdW, log_gamma_N = \
                self._read_Kurucz_transitions(file)
       
        # Sort by increasing wavenumber
        idx = np.argsort(nu_0)

        nu_0  = nu_0[idx]
        E_low = E_low[idx]
        gf    = gf[idx]
        gamma_vdW   = gamma_vdW[idx]
        log_gamma_N = log_gamma_N[idx]
                
        # Compute line-strength at reference temperature
        term1 = (gf * np.pi*e**2) / ((1e3*sc.m_e)*(100*sc.c)**2)
        term2 = np.exp(-c2*E_low/CS.T_0) / CS.q_0
        term3 = (1-np.exp(-c2*nu_0/CS.T_0))
        S_0 = term1 * term2 * term3

        # Unit conversion
        nu_0  *= (100*sc.c)        # [cm^-1] -> [s^-1]
        E_low *= sc.h * (100*sc.c) # [cm^-1] -> [kg m^2 s^-2] or [J]
        S_0   *= (100*sc.c)        # [cm^-1/(molec. cm^-2)] -> [s^-1/(molec. cm^-2)]

        mask_valid = np.ones(len(S_0), dtype=bool)
        
        if self.nu_0_to_ignore is not None:
            print('Masking transitions at:')
            # And remove any transitions on request
            for nu_0_i in np.array([self.nu_0_to_ignore]).flatten():
                print(f'{nu_0_i} cm^-1 ({1e4/nu_0_i} um)')
                # Set matching wavenumbers to 'invalid'
                mask_nu_0_i = np.isclose(nu_0, nu_0_i*(100*sc.c), rtol=1e-8)
                mask_valid[mask_nu_0_i] = False

        if None not in [self.E_ion, self.Z]:
            # If alkali (and E_ion/Z given), use Schweitzer+ (1996) vdW
            E_H     = 13.6 * 8065.73 # [cm^-1]
            alpha_H = 0.666793e-24   # [cm^3]
            
            E_low_cm  = E_low/(sc.h*(100*sc.c))    # [cm^-1]
            E_high_cm = E_low_cm + nu_0/(100*sc.c) # [cm^-1]

            # Schweitzer et al. (1996) [cm^6 s^-1]
            for species_i, broad_i in self.broad.items():
                
                alpha_i = broad_i['alpha'] # [cm^3]

                # vdW interaction constant [cm^6 s^-1]
                self.broad[species_i]['C_6'] = np.abs(
                    1.01e-32 * alpha_i/alpha_H * (self.Z+1)**2 * (
                        (E_H/(self.E_ion-E_low_cm))**2 - (E_H/(self.E_ion-E_high_cm))**2
                        )
                )[mask_valid]

        else:
            # If not alkali, use only transitions with valid vdW-damping
            mask_valid = mask_valid & (gamma_vdW != 1.0)

        nu_0  = nu_0[mask_valid]
        E_low = E_low[mask_valid]
        S_0   = S_0[mask_valid]

        gamma_vdW   = gamma_vdW[mask_valid]
        log_gamma_N = log_gamma_N[mask_valid]

        # Compute opacities
        CS.loop_over_PT_grid(
            func=CS.get_cross_sections, show_pbar=show_pbar, 
            nu_0_static=nu_0, E_low=E_low, S_0=S_0, 
            broad_per_trans=self.broad, 
            gamma_vdW=gamma_vdW, 
            log_gamma_N=log_gamma_N
            )
            
        return CS
    
    
## Helper functions (move somewhere else?) ##

# adapted from https://gitlab.com/mauricemolli/petitRADTRANS/-/blob/master/petitRADTRANS/__file_conversion.py
def write_line_by_line(file, doi, wavenumbers, opacities, mol_mass, species,
                       opacities_pressures, opacities_temperatures, wavelengths=None,
                       contributor=None, description=None,
                       pRT_version="3.0.7"):
    import datetime
    
    if wavelengths is None:
        wavelengths = np.array([1 / wavenumbers[0], 1 / wavenumbers[-1]])

    with h5py.File(file, "w") as fh5:
        dataset = fh5.create_dataset(
            name='DOI',
            shape=(1,),
            data=doi
        )
        dataset.attrs['long_name'] = 'Data object identifier linked to the data'
        dataset.attrs['contributor'] = str(contributor)
        dataset.attrs['additional_description'] = str(description)

        dataset = fh5.create_dataset(
            name='Date_ID',
            shape=(1,),
            data=f'petitRADTRANS-v{pRT_version}_{datetime.datetime.now(datetime.timezone.utc).isoformat()}'
        )
        dataset.attrs['long_name'] = 'ISO 8601 UTC time (https://docs.python.org/3/library/datetime.html) ' \
                                     'at which the table has been created, ' \
                                     'along with the version of petitRADTRANS'

        dataset = fh5.create_dataset(
            name='bin_edges',
            data=wavenumbers
        )
        dataset.attrs['long_name'] = 'Wavenumber grid'
        dataset.attrs['units'] = 'cm^-1'

        dataset = fh5.create_dataset(
            name='xsecarr',
            data=opacities
        )
        dataset.attrs['long_name'] = 'Table of the cross-sections with axes (pressure, temperature, wavenumber)'
        dataset.attrs['units'] = 'cm^2/molecule'

        dataset = fh5.create_dataset(
            name='mol_mass',
            shape=(1,),
            data=float(mol_mass)
        )
        dataset.attrs['long_name'] = 'Mass of the species'
        dataset.attrs['units'] = 'AMU'

        dataset = fh5.create_dataset(
            name='mol_name',
            shape=(1,),
            data=species.split('_', 1)[0]
        )
        dataset.attrs['long_name'] = 'Name of the species described'

        dataset = fh5.create_dataset(
            name='p',
            data=opacities_pressures
        )
        dataset.attrs['long_name'] = 'Pressure grid'
        dataset.attrs['units'] = 'bar'

        dataset = fh5.create_dataset(
            name='t',
            data=opacities_temperatures
        )
        dataset.attrs['long_name'] = 'Temperature grid'
        dataset.attrs['units'] = 'K'

        dataset = fh5.create_dataset(
            name='temperature_grid_type',
            shape=(1,),
            data='regular'
        )
        dataset.attrs['long_name'] = 'Whether the temperature grid is "regular" ' \
                                     '(same temperatures for all pressures) or "pressure-dependent"'

        dataset = fh5.create_dataset(
            name='wlrange',
            data=np.array([wavelengths.min(), wavelengths.max()]) * 1e4  # cm to um
        )
        dataset.attrs['long_name'] = 'Wavelength range covered'
        dataset.attrs['units'] = 'µm'

        dataset = fh5.create_dataset(
            name='wnrange',
            data=np.array([wavenumbers.min(), wavenumbers.max()])
        )
        dataset.attrs['long_name'] = 'Wavenumber range covered'
        dataset.attrs['units'] = 'cm^-1'
        
    print(f"Saved to {file}")
    return None


def get_opacity_filename(resolving_power, wavelength_boundaries, species_isotopologue_name,
                         source):
    if resolving_power < 1e6:
        resolving_power = f"{resolving_power:.0f}"
    else:
        decimals = np.mod(resolving_power / 10 ** np.floor(np.log10(resolving_power)), 1)

        if decimals >= 1e-3:
            resolving_power = f"{resolving_power:.3e}"
        else:
            resolving_power = f"{resolving_power:.0e}"

    spectral_info = (f"R{resolving_power}_"
                     f"{wavelength_boundaries[0]:.1f}-{wavelength_boundaries[1]:.1f}mu")

    return join_species_all_info(
        name=species_isotopologue_name,
        source=source,
        spectral_info=spectral_info
    )
    
def join_species_all_info(name, natural_abundance='', charge='', cloud_info='', source='', spectral_info=''):
    if natural_abundance != '':
        name += '-' + natural_abundance

    name += charge + cloud_info

    if source != '':
        name += '__' + source

    if spectral_info != '':
        name += '.' + spectral_info

    return name
