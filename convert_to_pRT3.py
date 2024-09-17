import numpy as np
from pandas import read_csv
import h5py
import pathlib
import datetime

path = pathlib.Path("/net/lem/data2/picos/cs_package/")
species = "K"
atoms_info = read_csv(path/'atoms_info.csv', index_col=0)
mass = atoms_info.loc[species,'mass'] # [u]
print(f"Atomic mass of {species}: {mass}")


file = path / "cross_sec_outputs" / species / f"{species}.hdf5"

def load_hdf5_output(file):

    with h5py.File(file, 'r') as f:
        # Make an array [...]
        wave  = f['wave'][...] # [m]
        sigma = 10**f['cross_sec'][...] - 1e-250 # [m^2]
        P = 10**f['P'][...] - 1e-250 # [Pa]
        T = f['T'][...] # [K]

    return wave, sigma, P, T

wave, sigma, P, T = load_hdf5_output(file)
# Convert to [um], [cm^2/molecule], [bar], [K]
wave_um=wave*1e6
sigma=sigma*1e4
P_bar=P*1e-5

# COnvert to pRT3 format
# 1) Convert wave to variable wavenumbers in cm-1

print(f' wave min: {wave_um.min():.2e} um wave max: {wave_um.max():.2e} um')
wave_cm = wave_um * 1e-4 # [um] -> [cm]


# Load pRT wavelength-grid
def interp_sigma_to_pRT_wave(wave, sigma, pRT_wave):
    
    # FIXME: is linear interpolation good enough?
    new_sigma = np.zeros((sigma.shape[0], sigma.shape[1], pRT_wave.size))
    for i in range(sigma.shape[0]):
        for j in range(sigma.shape[1]):
            new_sigma[i,j] = np.interp(pRT_wave, wave, sigma[i,j], left=0.0, right=0.0)
            
    return new_sigma


        

# (wave.size, P.size, T.size) -> (P.size, T.size, wave.size)
sigma = np.moveaxis(sigma, 0, -1)

# IMPORTANT: invert the wavenumbers array to be in increasing order

wavenumbers = 1 / wave_cm

resample = True
if resample:
    pRT_wave_file = path / 'input_data/wlen_petitRADTRANS.dat'
    pRT_wave = np.genfromtxt(str(pRT_wave_file)) # cm
    print(f' pRT_wave min: {pRT_wave.min():.2e} cm pRT_wave max: {pRT_wave.max():.2e} cm')
    wavenumbers = 1 / pRT_wave # [cm^-1]

    sigma = interp_sigma_to_pRT_wave(wave_cm, sigma, pRT_wave)
    print(f' pRT wave. shape: {pRT_wave.shape} sigma shape: {sigma.shape}')

# 2) Convert sigma to variable `xsecarr` 
wavenumbers = wavenumbers[::-1]
wave_um_pRT = 1e4 * (1 / wavenumbers) # [cm^-1] -> [um]
wave_um_min = max(wave_um.min(), wave_um_pRT.min())
wave_um_max = min(wave_um.max(), wave_um_pRT.max())

xsecarr = sigma[:,:,::-1] # (P, T, wavenumber)
print(xsecarr.shape)

def write_line_by_line(file, doi, wavenumbers, opacities, mol_mass, species,
                       opacities_pressures, opacities_temperatures, wavelengths=None,
                       contributor=None, description=None,
                       pRT_version="3.0.7"):
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

# Save to file with suffix `pRT3`
# file_pRT3 = file.with_name(file.stem + '_pRT3.hdf5')
iso_number = 39
species_isotopologue_name=f'{iso_number}{species}'
file_pRT3 = get_opacity_filename(resolving_power=1e6,
                                    wavelength_boundaries=[wave_um_min, wave_um_max],
                                    species_isotopologue_name=species_isotopologue_name,
                                    source="Kurucz",
)

output_dir = pathlib.Path("/net/lem/data2/pRT3/input_data/opacities/lines/line_by_line/") / species / species_isotopologue_name
assert output_dir.exists(), f"Output directory {output_dir} does not exist"

hdf5_opacity_file = output_dir / f'{file_pRT3}.xsec.petitRADTRANS.h5'

print(hdf5_opacity_file)
doi = 'None'

write_line_by_line(hdf5_opacity_file, 
                   doi,
                   wavenumbers, 
                   xsecarr, 
                   mass,
                   species, 
                   np.unique(P_bar),
                   np.unique(T), 
                   wavelengths=None, 
                   contributor='Dario Gonzalez Picos', 
                   description='Converted from `cs_package` format to `pRT3` format')
