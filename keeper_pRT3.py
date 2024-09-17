# compare opacities from the KEEPER database and the ones computed here

import numpy as np
import h5py
import pathlib
c_cms = 2.99792458e10  # cm/s
amu_g = 1.66053906660e-24  # g

# function from v3 petitRADTRANS/radtrans.py
def load_hdf5_line_opacity_table(file_path_hdf5, frequencies, line_by_line_opacity_sampling=1):
    """Load opacities (cm2.g-1) tables in HDF5 format, based on petitRADTRANS pseudo-ExoMol setup."""
    frequencies_relative_tolerance = 1e-12  # allow for a small relative deviation from the wavenumber grid

    with h5py.File(file_path_hdf5, 'r') as f:
        frequency_grid = c_cms * f['bin_edges'][:]  # cm-1 to s-1

        selection = np.nonzero(np.logical_and(
            np.greater_equal(frequency_grid, np.min(frequencies) * (1 - frequencies_relative_tolerance)),
            np.less_equal(frequency_grid, np.max(frequencies) * (1 + frequencies_relative_tolerance))
        ))[0]
        selection = np.array([selection[0], selection[-1]])

        if line_by_line_opacity_sampling > 1:
            # Ensure that down-sampled wavelength upper bound >= requested wavelength upper bound
            selection[0] -= line_by_line_opacity_sampling - 1  # array is ordered by increasing wavenumber

        # print(f' f[xsecarr].shape: {f["xsecarr"].shape}')
        line_opacities_grid = f['xsecarr'][:, :, selection[0]:selection[-1] + 1]

        # Divide by mass to convert cross-sections to opacities
        mol_mass_inv = 1 / (f['mol_mass'][()] * amu_g) # g-1
        line_opacities_grid *= mol_mass_inv

    line_opacities_grid = line_opacities_grid[:, :, ::-1]

    if line_by_line_opacity_sampling > 1:
        line_opacities_grid = line_opacities_grid[:, :, ::line_by_line_opacity_sampling]

    if line_opacities_grid.shape[-1] != frequencies.size:
        frequency_grid = frequency_grid[selection[0]:selection[-1] + 1]
        frequency_grid = frequency_grid[::-1]

        if line_by_line_opacity_sampling > 1:
            frequency_grid = frequency_grid[::line_by_line_opacity_sampling]

        raise ValueError(
            f"file selected frequencies size is "
            f"{line_opacities_grid.shape[-1]} ({np.min(frequency_grid)}--{np.max(frequency_grid)}), "
            f"but frequency grid size is "
            f"{frequencies.size} ({np.min(frequencies)}--{np.max(frequencies)})\n"
            f"This may be caused by loading opacities of different resolving power "
            f"or from too different wavenumber grids "
            f"(frequencies relative tolerance was {frequencies_relative_tolerance:.0e})"
        )

    line_opacities_grid = np.swapaxes(line_opacities_grid, 0, 1)  # (t, p, wvl)
    line_opacities_grid = line_opacities_grid.reshape(
        (line_opacities_grid.shape[0] * line_opacities_grid.shape[1], line_opacities_grid.shape[2])
    )  # (tp, wvl)
    line_opacities_grid = np.swapaxes(line_opacities_grid, 0, 1)  # (wvl, tp)
    line_opacities_grid = line_opacities_grid[np.newaxis, :, np.newaxis, :]  # (g, wvl, species, tp)

    return line_opacities_grid

def load_line_opacities_pressure_temperature_grid(hdf5_file):
    """Load line opacities temperature grids."""
    with h5py.File(hdf5_file, 'r') as f:
        pressure_grid = f['p'][:] # bar
        temperature_grid = f['t'][:]
    print(f' pressure_grid: {pressure_grid}')
    print(f' temperature_grid: {temperature_grid}')
    ret_val = np.zeros((temperature_grid.size * pressure_grid.size, 2))

    for i_t in range(temperature_grid.size):
        for i_p in range(pressure_grid.size):
            ret_val[i_t * pressure_grid.size + i_p, 1] = pressure_grid[i_p] * 1e6  # bar to cgs
            ret_val[i_t * pressure_grid.size + i_p, 0] = temperature_grid[i_t]

    line_opacities_temperature_pressure_grid = ret_val
    line_opacities_temperature_grid_size = temperature_grid.size
    line_opacities_pressure_grid_size = pressure_grid.size

    return line_opacities_temperature_pressure_grid, line_opacities_temperature_grid_size, \
        line_opacities_pressure_grid_size
        
def init_frequency_grid_from_frequency_grid(frequency_grid, wavelength_boundaries, sampling=1):
    # Get frequency boundaries
    frequency_min = c_cms / wavelength_boundaries[1] * 1e4  # um to cm
    frequency_max = c_cms / wavelength_boundaries[0] * 1e4  # um to cm

    # Check if the requested wavelengths boundaries are within the file boundaries
    bad_boundaries = False

    if frequency_min < frequency_grid[0]:
        bad_boundaries = True

    if frequency_max > frequency_grid[-1]:
        bad_boundaries = True

    if bad_boundaries:
        raise ValueError(f"Requested wavelength interval "
                            f"({wavelength_boundaries[0]}--{wavelength_boundaries[1]}) "
                            f"is out of opacities table wavelength grid "
                            f"({1e4 * c_cms / frequency_grid[-1]}--{1e4 * c_cms / frequency_grid[0]})")

    # Get the freq. corresponding to the requested boundaries, with the request fully within the selection
    selection = np.nonzero(np.logical_and(
        np.greater_equal(frequency_grid, frequency_min),
        np.less_equal(frequency_grid, frequency_max)
    ))[0]
    selection = np.array([selection[0], selection[-1]])

    if frequency_grid[selection[0]] > frequency_min:
        selection[0] -= 1

    if frequency_grid[selection[-1]] < frequency_max:
        selection[-1] += 1

    if sampling > 1:
        # Ensure that down-sampled wavelength upper bound >= requested wavelength upper bound
        selection[0] -= sampling - 1

    frequencies = frequency_grid[selection[0]:selection[-1] + 1]
    frequencies = frequencies[::-1]

    # Down-sample frequency grid in lbl mode if requested
    if sampling > 1:
        frequencies = frequencies[::sampling]

    # frequency_bins_edges = np.array(c_cms / Radtrans.compute_bins_edges(c_cms / frequencies), dtype='d', order='F')

    # return frequencies, frequency_bins_edges     
    return frequencies

path = pathlib.Path("/net/lem/data2/picos/cs_package/")


def wrap(file, lbl=1):
    with h5py.File(file, 'r') as f:
        frequencies = c_cms * f['bin_edges'][:]  # cm-1 to s-1
        
    frequencies = init_frequency_grid_from_frequency_grid(frequencies, [0.5, 12], sampling=lbl)
        
    tp_grid, _, _ = load_line_opacities_pressure_temperature_grid(file)
        
    line_opacities_grid = load_hdf5_line_opacity_table(file, frequencies, line_by_line_opacity_sampling=lbl)
    return frequencies, line_opacities_grid, tp_grid


species = "K"
# file = path / "cross_sec_outputs" / f"{species}" / f"{species}_pRT3.hdf5"
file = "/net/lem/data2/pRT3/input_data/opacities/lines/line_by_line/K/39K/39K__Kurucz.R1e+06_0.3-50.0mu.xsec.petitRADTRANS.h5"
lbl = 1
frequencies, line_opacities_grid, tp_grid = wrap(file, lbl=lbl)

keeper = False
if keeper:
    file_keeper = path / "cross_sec_outputs" / "Ca" / "Ca/40Ca" / "40Ca__Kurucz.R1e6_0.3-28mu.xsec.petitRADTRANS.h5"
    frequencies_keeper, line_opacities_grid_keeper, tp_grid_keeper = wrap(file_keeper, lbl=lbl)


p = 1e6 # cgs
t = 3000.0 # K

tp_id = np.argmin(np.abs(tp_grid[:, 0] - t) + np.abs(tp_grid[:, 1] - p))

print(f' tp_id: {tp_id} --> (t, p) = ({tp_grid[tp_id]})')

plot = True
if plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    wave_um = 1e4 * c_cms / frequencies
    ax.plot(wave_um, line_opacities_grid[0, :, 0, tp_id], label='pRT3', color='orange')
    if keeper:
        tp_id_keeper = np.argmin(np.abs(tp_grid_keeper[:, 0] - t) + np.abs(tp_grid_keeper[:, 1] - p))
        print(f' tp_id_keeper: {tp_id_keeper} --> (t, p) = ({tp_grid_keeper[tp_id_keeper]})')
        ax.plot(1e4*c_cms/frequencies_keeper, line_opacities_grid_keeper[0, :, 0, tp_id_keeper], label='KEEPER', color='blue',
                alpha=0.5)
    # ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Wavelength [um]')
    ax.set_ylabel('Opacity [cm^2/g]')
    # ax.set_xlim(1.6, 3.3)
    ax.set_xlim(1.65, 3.18)
    ax.set_ylim(1e-6, 1e10)
    ax.legend()
    # plt.show()
    fig.savefig(f'opacity_pRT3_vs_KEEPER_{species}.png')
    plt.close(fig)