'''
--load_beam.py-- 
Script by Amelie Sas
2019

This script runs an FEA simulation on a h5-inputfile of a simple beam.
The simple beam simulations were used to test the nonlinear framework implementation.
Either a linear or a nonlinear simulation can be applied.

Input parameters:
      - file_name: name of the h5-inputfile (string). File should be located in the 'data' directory.
      - displacement_magnitude: magnitude of the displacement applied at the top of the beam (in mm) (float)
      - material_nonlinearity: choose between 'linear' or 'nonlinear' simulation (string)
      - material_type: choose between 'trabecular' or 'cortical' bone (string)
      - step_size: if material_type is 'nonlinear', a step_size needs to be defined (in mm) (float)
      (adapt these settings in main() )
Output:
      - h5-file with strains and stresses assigned to the elements; displacement and forces assigned to the nodes
      - values of the total applied force, the total applied force projected on the loading direction and the applied displacement
      - text-file with force-displacement data of the full beam (for nonlinear simulations)

The h5-inputfile of the beam should include the following data:
      - Segmentation
          * Density: 3D matrix of the density values of the beam (single datatype).   
      - Image_Data
          * Fixed_Displacement_Coordinates: 4-by-k array with in each column the indices of the nodes where a displacement is applied,
              followed by the direction in which the displacement is applied
          * Fixed_Displacement_Values: k-by-1 array of the magnitudes of the applied displacements
          * Voxelsize: size of the voxels in the CT scan
An example h5-file is provided in the data folder ('Beam_example.h5').
'''

def load_beam_linear(h5_inputfile,h5_outputfolder,displacement_magnitude,material_type):
    '''
    Linear FEA simulation
    '''
    import h5py
    import numpy as np
    from nonlinear_FEA import Quantity
    from nonlinear_FEA.beam_loading import BeamLoading

    # Read h5 input file
    h5 = h5py.File(h5_inputfile)
    fixed_displacement_coordinates = h5['Image_Data/Fixed_Displacement_Coordinates']
    fixed_displacement_coordinates = np.asarray(fixed_displacement_coordinates)
    fixed_displacement_values = h5['Image_Data/Fixed_Displacement_Values']
    fixed_displacement_values = np.asarray(fixed_displacement_values)
    voxel_size = h5['Image_Data/Voxelsize']
    voxel_size = Quantity(np.asarray(voxel_size)[0],'mm')
    density = h5['Segmentation/Density']
    density = np.asarray(density)
    h5.close()

    # Apply load
    model = BeamLoading(
         stiffness = None, # stiffness will be extracted based on the density
         fixed_displacement_coordinates = fixed_displacement_coordinates,
         fixed_displacement_values = fixed_displacement_values,
         voxel_size = voxel_size,
         output = ['effective strain','von Mises stress','displacements','forces'],
         results_folder = h5_outputfolder,
         command_line_options = ['--tol', '1e-11'], # set tolerance for convergence
         parallel_options = {'number of nodes': 1, 'time limit': '01:00:00',
                             'number of cores per node': 4},
         density = density,
         material_type = material_type)

    force, proj_force, applied_disp, results = model.apply_load_linear(displacement_magnitude=displacement_magnitude)

    print('')
    print('Results: ')
    print('Total force: {}'.format(force))
    print('Projected force: {}'.format(proj_force))
    print('Applied displacement: {}'.format(applied_disp))

def load_beam_nonlinear(h5_inputfile,h5_outputfolder,displacement_magnitude,material_type,step_size):
    '''
    Nonlinear FEA simulation
    '''
    import h5py
    import numpy as np
    from nonlinear_FEA import Quantity
    from nonlinear_FEA.beam_loading import BeamLoading
    
    # Read h5 input file
    h5 = h5py.File(h5_inputfile)
    fixed_displacement_coordinates = h5['Image_Data/Fixed_Displacement_Coordinates']
    fixed_displacement_coordinates = np.asarray(fixed_displacement_coordinates)
    fixed_displacement_values = h5['Image_Data/Fixed_Displacement_Values']
    fixed_displacement_values = np.asarray(fixed_displacement_values)
    voxel_size = h5['Image_Data/Voxelsize']
    voxel_size = Quantity(np.asarray(voxel_size)[0],'mm')
    density = h5['Segmentation/Density']
    density = np.asarray(density)
    h5.close()

    # Apply incremental load
    model = BeamLoading(
         stiffness = None, # stiffness will be extracted based on the density
         fixed_displacement_coordinates = fixed_displacement_coordinates,
         fixed_displacement_values = fixed_displacement_values,
         voxel_size = voxel_size,
         output = ['effective strain','von Mises stress','displacements','forces'],
         results_folder = h5_outputfolder,
         command_line_options = ['--tol', '1e-11'], # set tolerance for convergence
         parallel_options = {'number of nodes': 1, 'time limit': '01:00:00',
                             'number of cores per node': 4},
         density = density,
         material_type = material_type)

    force, proj_force, applied_disp, results = model.apply_load_nonlinear(increment=step_size,
                                                                          displacement_magnitude=displacement_magnitude)

    print('')
    print('Results: ')
    print('Total force: {}'.format(force))
    print('Projected force: {}'.format(proj_force))
    print('Applied displacement: {}'.format(applied_disp))

def _main():
    '''
    Set up the parameters to run the FEA simulation.
    '''
    
    import os

    # Set input parameters (adapt to needs!)
    file_name = 'Beam_example.h5' # Name of the h5 file in the data folder, e.g.: 'Beam_example.h5' (string)
    displacement_magnitude = 12 # Displacement magnitude (in mm) to apply, e.g.: 4 (float)
    material_nonlinearity = 'nonlinear' # Linear or nonlinear material properties, e.g.: 'linear' or 'nonlinear' (string)
    material_type = 'trabecular' # Trabecular or cortical bone, e.g.: 'trabecular' or 'cortical' (string)
    step_size = 0.025  # Step size (in mm) for incremental load application in nonlinear FEA, e.g.= 0.025 (float)

    # Set path
    cwd = os.getcwd()
    inputpath = os.path.join(cwd,'data')
    outputpath = os.path.join(cwd,'results')
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)

    # Run simulation
    if material_nonlinearity == 'linear':
        full_file_name = os.path.join(inputpath,file_name)
        model_specific_outputpath = os.path.join(outputpath,file_name[:-3])
        load_beam_linear(full_file_name,model_specific_outputpath,float(displacement_magnitude),material_type)
    else:
        if step_size.is_integer():
            step_size = int(step_size)
        full_file_name = os.path.join(inputpath, file_name)
        model_specific_outputpath = os.path.join(outputpath,file_name[:-3])
        load_beam_nonlinear(full_file_name,model_specific_outputpath,float(displacement_magnitude),material_type,float(step_size))

if __name__ == '__main__':
    _main()
