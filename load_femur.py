'''
--load_femur.py-- 
Script by Amelie Sas
2019

This script runs an FEA simulation on a h5-inputfile of a voxel-based FE model of a femur.
Either a linear or a nonlinear simulation can be applied.

Input parameters:
      - file_name: name of the h5-inputfile (string). File should be located in the 'data' directory.
      - displacement_magnitude: magnitude of the displacement applied at the femur head (in mm) (float)
      - material_nonlinearity: choose between 'linear' or 'nonlinear' simulation (string)
      - step_size: if material_type is 'nonlinear', a step_size needs to be defined (in mm) (float)
      (adapt these settings in main() )
Output:
      - h5-file with strains and stresses assigned to the elements; displacement and forces assigned to the nodes
      - values of the total applied force, the total applied force projected on the loading direction and the applied displacement
      - text-file with force-displacement data of the total femur (for nonlinear simulations)

The h5-inputfile should include the following data:
      - Image_Data
          * Fixed_Displacement_Coordinates: 4-by-k array with in each column the indices of the nodes [i_z,i_y,i_x] where a 
              displacement is applied, followed by the direction in which the displacement is applied (0 = x-direction,
              1 = y-direction, 2 = z-direction)
         * Fixed_Displacement_Values: k-by-1 array of the magnitudes of the applied displacements
         * Voxelsize: size of the voxels in the CT scan
      - Segmentation
         * Density: 3D matrix of the ash-density values of the femur (single datatype).
         * Femur: 3D matrix of the femur mask (logical datatype)
         * Trabecular: 3D matrix of the trabecular mask (logical datatype)
         * Influence_region: 3D matrix of the mask of the influence region (logical datatype). This influence region is the region on
              the femur head where the load is applied. These elements are assigned a higher Young's modulus and strength to prevent
              severe distortions in this region.
An example h5-file is provided in the data folder ('Femur_example.h5').
In this example file, the ash-density values of the femur are assigned based on the CT scan and a calibration phantom.
The density matrix has the same dimensions as the CT images (1st and 2nd dimension = in-plane width/height, 3rd dimension = number
of slices). All elements outside the femur are assigned a density of 0. Additional structures for the boundary conditions can also
be defined within this density matrix. In the example file, a load_cup and a distal_fixation area are defined. For these parts, the
Young's moduli (2 GPa and 200 GPa respectively) are directly assigned to the corresponding voxels.
'''

def load_femur_linear(h5_inputfile,h5_outputfolder,displacement_magnitude):
    '''
    Linear FEA simulation
    '''
    import h5py
    import numpy as np
    from nonlinear_FEA import Quantity
    from nonlinear_FEA.femur_loading import FemurLoading

    # Read h5 input file
    h5 = h5py.File(h5_inputfile)
    fixed_displacement_coordinates = h5['Image_Data/Fixed_Displacement_Coordinates']
    fixed_displacement_coordinates = np.asarray(fixed_displacement_coordinates)
    fixed_displacement_values = h5['Image_Data/Fixed_Displacement_Values']
    fixed_displacement_values = np.asarray(fixed_displacement_values)
    voxel_size = h5['Image_Data/Voxelsize']
    voxel_size = Quantity(np.asarray(voxel_size)[0],'mm')
    trabecular_mask = h5['Segmentation/Trabecular']
    trabecular_mask = np.asarray(trabecular_mask,dtype=bool)
    femur_mask = h5['Segmentation/Femur']
    femur_mask = np.asarray(femur_mask,dtype=bool)
    BC_influence_region = h5['Segmentation/Influence_region']
    BC_influence_region = np.asarray(BC_influence_region,dtype=bool)
    density = h5['Segmentation/Density']
    density = np.asarray(density)
    h5.close()
    
    # Apply load
    model = FemurLoading(
         stiffness = None, # stiffness will be extracted based on the density
         fixed_displacement_coordinates = fixed_displacement_coordinates,
         fixed_displacement_values = fixed_displacement_values,
         voxel_size = voxel_size,
         output = ['effective strain','von Mises stress','displacements','forces'],
         results_folder = h5_outputfolder,
         command_line_options = ['--tol', '1e-11'], # set tolerance for convergence
         parallel_options = {'number of nodes': 1, 'time limit': '01:00:00',
                             'number of cores per node': 8},
         density = density,
         femur_mask = femur_mask,
         trabecular_mask = trabecular_mask,
         BC_influence_region = BC_influence_region)

    force, proj_force, applied_disp, results = model.apply_load_linear(displacement_magnitude=displacement_magnitude)

    print('')
    print('Results: ')
    print('Total force: {}'.format(force))
    print('Projected force: {}'.format(proj_force))
    print('Applied displacement: {}'.format(applied_disp))

def load_femur_nonlinear(h5_inputfile,h5_outputfolder,displacement_magnitude,step_size):
    '''
    Nonlinear FEA simulation
    '''
    import h5py
    import numpy as np
    from nonlinear_FEA import Quantity
    from nonlinear_FEA.femur_loading import FemurLoading
    
    # Read h5 input file
    h5 = h5py.File(h5_inputfile)
    fixed_displacement_coordinates = h5['Image_Data/Fixed_Displacement_Coordinates']
    fixed_displacement_coordinates = np.asarray(fixed_displacement_coordinates)
    fixed_displacement_values = h5['Image_Data/Fixed_Displacement_Values']
    fixed_displacement_values = np.asarray(fixed_displacement_values)
    voxel_size = h5['Image_Data/Voxelsize']
    voxel_size = Quantity(np.asarray(voxel_size)[0],'mm')
    trabecular_mask = h5['Segmentation/Trabecular']
    trabecular_mask = np.asarray(trabecular_mask,dtype=bool)
    femur_mask = h5['Segmentation/Femur']
    femur_mask = np.asarray(femur_mask,dtype=bool)
    BC_influence_region = h5['Segmentation/Influence_region']
    BC_influence_region = np.asarray(BC_influence_region,dtype=bool)
    density = h5['Segmentation/Density']
    density = np.asarray(density)
    h5.close()

    # Apply incremental load
    model = FemurLoading(
         stiffness = None, # stiffness will be extracted based on the density
         fixed_displacement_coordinates = fixed_displacement_coordinates,
         fixed_displacement_values = fixed_displacement_values,
         voxel_size = voxel_size,
         output = ['effective strain','von Mises stress','displacements','forces'],
         results_folder = h5_outputfolder,
         command_line_options = ['--tol', '1e-11'], # set tolerance for convergence
         parallel_options = {'number of nodes': 1, 'time limit': '01:00:00',
                             'number of cores per node': 8},
         density = density,
         femur_mask = femur_mask,
         trabecular_mask= trabecular_mask,
         BC_influence_region = BC_influence_region)

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
    file_name = 'Femur_example.h5'  # Name of the h5 file in the data folder, e.g.: 'Femur_example.h5' (string)
    displacement_magnitude = 4  # Displacement magnitude (in mm) to apply, e.g.: 4 (float)
    material_nonlinearity = 'nonlinear'  # Linear or nonlinear material properties, e.g.: 'linear' or 'nonlinear' (string)
    step_size = 0.025  # Step size (in mm) for incremental load application in nonlinear FEA, e.g.= 0.025 (float)

    # Set path
    cwd = os.getcwd()
    inputpath = os.path.join(cwd, 'data')
    outputpath = os.path.join(cwd, 'results')
    if not os.path.exists(outputpath):
        os.mkdir(outputpath)

    # Run simulation
    if material_nonlinearity == 'linear':
        full_file_name = os.path.join(inputpath, file_name)
        model_specific_outputpath = os.path.join(outputpath, file_name[:-3])
        load_femur_linear(full_file_name, model_specific_outputpath, float(displacement_magnitude))
    else:
        if step_size.is_integer():
            step_size = int(step_size)
        full_file_name = os.path.join(inputpath, file_name)
        model_specific_outputpath = os.path.join(outputpath, file_name[:-3])
        load_femur_nonlinear(full_file_name, model_specific_outputpath, float(displacement_magnitude), float(step_size))

if __name__ == '__main__':
    _main()
