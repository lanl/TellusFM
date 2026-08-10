import h5py
import glob
import os

path_to_files = 'PHASE-FIELD'
materials = ['steel','aluminum','tungsten','pbx','shale']
bcs = ['horizontal_bc','combined_bc']
force_dirs = ['z', 'xz']
for material in materials:
    for i,bc in enumerate(bcs):
        print(os.getcwd())
        file_directory = path_to_files + '/' + material + '/' + bc
        print(file_directory)
        h5_files = glob.glob(file_directory + f'/frac_pull_{force_dirs[i]}_*.h5', recursive=False)
        print(h5_files)
        # Define the archive name
        archive_name = f'{material}_frac_{force_dirs[i]}.h5'

        # Create the HDF5 file with external links
        with h5py.File(archive_name, mode='w') as h5fw:
            for h5name in h5_files:
                full_path = os.path.abspath(h5name)  # Get the absolute path of the file
                link_name = os.path.basename(h5name)[:-3]  # Remove the ".h5" extension
                h5fw['link_' + link_name] = h5py.ExternalLink(full_path, '/')  # Use full path

        print(f"External links HDF5 file created: {archive_name}")