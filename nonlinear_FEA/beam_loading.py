'''
--beam_loading.py-- 
Script by Amelie Sas
2019

This script implements an iterative routine to enable nonlinear FE analyses in a linear FE solver (in this case ParOSol).
A loop is placed over the FE solver, such that the material properties can be adapted at every iteration.
This script is applied on a simple beam for testing purposes.
The implemented nonlinear material model is described by Keyak et al. 2005, J Clinical Orthopaedics and Related Research
no. 437, pp. 219-228.
'''

import os
import shutil
import logging
import time
import math
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from nonlinear_FEA import Quantity
from nonlinear_FEA.parosol_AS import ParosolSolver #This script is not available! See remark in 'user information'.

def set_logger(filename,print_debug):
    if print_debug == 1:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(filename)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

class BeamLoading:
    '''
    '''

    def __init__(
        self,
        stiffness,
        fixed_displacement_coordinates,
        fixed_displacement_values,
        voxel_size,
        output,
        results_folder,
        command_line_options=None,
        parallel_options=None,
        density=None,
        femur_mask=None,
        trabecular_mask=None,
        BC_influence_region=None,
        material_type=None):
        '''
        '''
        self.stiffness = stiffness
        boundary_conditions = {'coordinates': fixed_displacement_coordinates,
                               'values': fixed_displacement_values}
        self.boundary_conditions = boundary_conditions
        self.voxel_size = voxel_size
        self.output = output
        self.foldername = results_folder
        self.filename = os.path.split(self.foldername)[1]
        self.command_line_options = command_line_options
        self.parallel_options = parallel_options
        self.density = density
        self.femur_mask = femur_mask
        self.trabecular_mask = trabecular_mask
        self.BC_influence_region = BC_influence_region
        self.material_type = material_type
        self.results = None
        self.total_force = {'x':[],'y':[],'z':[],'magn':[]}
        self.projected_force = {'x':[],'y':[],'z':[],'magn':[]}
        self.applied_displacement = {'x':[],'y':[],'z':[],'magn':[]}
        self.fix_force = {'x':[],'y':[],'z':[],'magn':[]}

    def _calculate_displacement(self):
        '''
        Calculate the applied displacement
        '''
        ind_z = np.where(self.boundary_conditions['coordinates'][:,3]==2)
        applied_disp = {}
        z_max = np.argmax(np.abs(np.unique(self.boundary_conditions['values'][ind_z])))
        applied_disp['x'] = 0
        applied_disp['y'] = 0
        applied_disp['z'] = np.unique(self.boundary_conditions['values'][ind_z])[z_max]
        applied_disp['magn'] = math.sqrt(applied_disp['z']**2)

        return applied_disp

    def _calculate_force(self,logger):
        '''
        Calculate the resulting force
        '''
        k = [i for i,v in enumerate(self.boundary_conditions['values']) if np.abs(v) > 0]
        logger.debug('Number of load nodes: {}'.format(len(k)/3))
        load_nodes = np.unique(self.boundary_conditions['coordinates'][k,0:3],axis=0)
        load_nodeset = {}
        load_nodeset['x'] = load_nodes[:,2]
        load_nodeset['y'] = load_nodes[:,1]
        load_nodeset['z'] = load_nodes[:,0]

        total_force = {}
        total_force['x'] = np.sum(self.results['forces']['x'].magnitude[
                                    load_nodeset['z'],
                                    load_nodeset['y'],
                                    load_nodeset['x']])
        total_force['y'] = np.sum(self.results['forces']['y'].magnitude[
                                    load_nodeset['z'],
                                    load_nodeset['y'],
                                    load_nodeset['x']])
        total_force['z'] = np.sum(self.results['forces']['z'].magnitude[
                                    load_nodeset['z'],
                                    load_nodeset['y'],
                                    load_nodeset['x']])
        total_force['magn'] = math.sqrt(total_force['x']**2+total_force['y']**2+
                                        total_force['z']**2)
        F = np.array([total_force['x'],total_force['y'],total_force['z']])

        projected_force = {}
        projected_force['magn'] = np.dot(F,self._disp_direction)
        projected_force['x'] = self._disp_direction[0]*projected_force['magn']
        projected_force['y'] = self._disp_direction[1]*projected_force['magn']
        projected_force['z'] = self._disp_direction[2]*projected_force['magn']

        k = [i for i,v in enumerate(self.boundary_conditions['values']) if np.abs(v) == 0]
        logger.debug('Number of fix nodes: {}'.format(len(k)/3))
        fix_nodes = np.unique(self.boundary_conditions['coordinates'][k,0:3],axis=0)
        fix_nodeset = {}
        fix_nodeset['x'] = fix_nodes[:,2]
        fix_nodeset['y'] = fix_nodes[:,1]
        fix_nodeset['z'] = fix_nodes[:,0]

        fix_force = {}
        fix_force['x'] = np.sum(self.results['forces']['x'].magnitude[
                                    fix_nodeset['z'],
                                    fix_nodeset['y'],
                                    fix_nodeset['x']])
        fix_force['y'] = np.sum(self.results['forces']['y'].magnitude[
                                    fix_nodeset['z'],
                                    fix_nodeset['y'],
                                    fix_nodeset['x']])
        fix_force['z'] = np.sum(self.results['forces']['z'].magnitude[
                                    fix_nodeset['z'],
                                    fix_nodeset['y'],
                                    fix_nodeset['x']])
        fix_force['magn'] = math.sqrt(fix_force['x']**2+fix_force['y']**2+
                                        fix_force['z']**2)

        return total_force, projected_force, fix_force

    def _update_force_and_displacement(self,logger,displacement_vector,displacement_magnitude):
        '''
        Update the force and displacement array with the values at the new iteration
        '''
        F, F_proj,F_fix = self._calculate_force(logger)
        self.total_force['x'].append(F['x'])
        self.total_force['y'].append(F['y'])
        self.total_force['z'].append(F['z'])
        self.total_force['magn'].append(F['magn'])
        self.projected_force['x'].append(F_proj['x'])
        self.projected_force['y'].append(F_proj['y'])
        self.projected_force['z'].append(F_proj['z'])
        self.projected_force['magn'].append(F_proj['magn'])
        self.applied_displacement['x'].append(displacement_vector[0])
        self.applied_displacement['y'].append(displacement_vector[1])
        self.applied_displacement['z'].append(displacement_vector[2])
        self.applied_displacement['magn'].append(displacement_magnitude)
        self.fix_force['x'].append(F_fix['x'])
        self.fix_force['y'].append(F_fix['y'])
        self.fix_force['z'].append(F_fix['z'])
        self.fix_force['magn'].append(F_fix['magn'])

    def plot_force_displacement_curve(self,force_disp_file):
        '''
        Plot the final force-displacement curve
        '''
        with open(force_disp_file) as f:
            lines = f.read().splitlines()

        disp = []
        force_tot = []
        force_proj = []

        for line in lines[1:len(lines)]:
            [d, f, fp] = line.split(" ")
            disp.append(float(d))
            force_tot.append(float(f))
            force_proj.append(float(fp) / 1000)

        plt.plot(disp, force_proj)
        plt.xlabel('Displacement')
        plt.ylabel('Force')
        plt.show()

    def _initialize_material_properties(self,logger):
        '''
        Initialize the material properties assigned to the beam.
        The implemented nonlinear material model is described by Keyak et al. 2005, J Clinical Orthopaedics and Related Research
        no. 437, pp. 219-228.
        '''
        
        density_full_model = self.density
        self.stiffness = density_full_model.copy()
        mask = density_full_model.astype('bool')
        logger.debug('Mask: '.format(mask))
        density = density_full_model[mask]
        self.S = float('inf')*np.ones(self.stiffness.shape)
        self.eps_A = np.zeros(self.stiffness.shape)
        self.eps_B = np.zeros(self.stiffness.shape)
        self.Ep = np.zeros(self.stiffness.shape)
        self.sigma_min = np.zeros(self.stiffness.shape)
        self.eps_C = np.zeros(self.stiffness.shape)
        
        # Trabecular
        if self.material_type == "trabecular":
            E = 14900*density**1.86 # MPa
            S = 102*density**1.8 # MPa
            eps_A = S/E
            eps_AB = (0.00189*np.ones(density.shape) + 0.0241*density)*15/3
            eps_B = eps_A + eps_AB
            Ep = -2080*density**1.45 
            Ep = 3*E*Ep/(15*E-(15-3)*Ep) # MPa
            sigma_min = 43.1*density**1.81 # MPa
            eps_C = eps_B-(S-sigma_min)/Ep

            self.stiffness[mask] = E/1000 # GPa
            self.S[mask] = S
            self.eps_A[mask] = eps_A
            self.eps_B[mask] = eps_B
            self.Ep[mask] = Ep
            self.sigma_min[mask] = sigma_min
            self.eps_C[mask] = eps_C

            logger.debug('Stiffness: '.format(self.stiffness))

        # Cortical
        if self.material_type == "cortical":     
            E = 14900*density**1.86 # MPa
            S = 102*density**1.8 # MPa
            eps_A = S/E
            eps_AB = (0.0184*np.ones(density.shape)-0.0100*density)*5/3
            eps_B = eps_A + eps_AB
            Ep = -1000*np.ones(density.shape)
            Ep = 3*E*Ep/(5*E-(5-3)*Ep) # MPa
            sigma_min = 43.1*density**1.81 # MPa
            eps_C = eps_B-(S-sigma_min)/Ep

            self.stiffness[mask] = E/1000 # GPa       
            self.S[mask] = S
            self.eps_A[mask] = eps_A
            self.eps_B[mask] = eps_B
            self.Ep[mask] = Ep
            self.sigma_min[mask] = sigma_min
            self.eps_C[mask] = eps_C

            logger.debug('Stiffness: '.format(self.stiffness))

        # Assign all elements to the current phase (at initialization all elements are in phase 1, linear elastic phase)
        self.elem_phase1 = [] # linear elastic phase
        for elem in np.argwhere(mask==True):
            self.elem_phase1.append(elem)
        logger.debug('Elem_phase_1: '.format(self.elem_phase1))
        self.elem_phase2 = [] # yield stress, perfectly plastic phase
        self.elem_phase3 = [] # strain softening
        self.elem_phase4 = [] # minimal stress, perfectly plastic phase
        
    def _update_material_properties(self,logger):
        '''
        Update the material properties if needed at every iteration
        '''

        # Initialize arrays
        from_1_to_2 = []
        from_2_to_3 = []
        from_2_to_4 = []
        from_3_to_4 = []
        self.results['von Mises stress'].ito('MPa')

        # Update elements in phase 1, linear elastic phase
        for i in range(len(self.elem_phase1)):
            [x,y,z] = [self.elem_phase1[i][0],self.elem_phase1[i][1],self.elem_phase1[i][2]]
            stress = self.results['von Mises stress'][x,y,z].magnitude
            E = self.stiffness[x,y,z]*1000
            eps = stress/E
            if eps >= self.eps_A[x,y,z]:
                yield_stress = self.S[x,y,z]
                new_E = yield_stress/eps
                self.stiffness[x,y,z] = new_E/1000
                from_1_to_2.append(i)

        # Update elements in phase 2, perfectly plastic phase
        for i in range(len(self.elem_phase2)):
            [x,y,z] = [self.elem_phase2[i][0],self.elem_phase2[i][1],self.elem_phase2[i][2]]
            stress = self.results['von Mises stress'][x,y,z].magnitude
            E = self.stiffness[x,y,z]*1000
            eps = stress/E
            yield_stress = self.S[x,y,z]
            eps_B = self.eps_B[x,y,z]
            eps_C = self.eps_C[x,y,z]
            if eps >= eps_C:
                new_E = self.sigma_min[x,y,z]/eps
                self.stiffness[x,y,z] = new_E/1000
                from_2_to_4.append(i)
                logger.debug('Element from phase 2 directly to phase 4: {}'.format([x,y,z]))
            elif eps_B <= eps < eps_C:
                Ep = self.Ep[x,y,z]
                new_stress = yield_stress+Ep*(eps-eps_B)
                new_E = new_stress/eps
                self.stiffness[x,y,z] = new_E/1000
                from_2_to_3.append(i)
            else:
                new_E = yield_stress/eps
                if new_E/1000 < self.stiffness[x,y,z]:
                    self.stiffness[x,y,z] = new_E/1000
                else:
                    continue

        # Update elements in phase 3, strain softening
        for i in range(len(self.elem_phase3)):
            [x,y,z] = [self.elem_phase3[i][0],self.elem_phase3[i][1],self.elem_phase3[i][2]]
            stress = self.results['von Mises stress'][x,y,z].magnitude
            E = self.stiffness[x,y,z]*1000
            eps = stress/E
            eps_C = self.eps_C[x,y,z]
            if eps >= eps_C:
                new_E = self.sigma_min[x,y,z]/eps
                self.stiffness[x,y,z] = new_E/1000
                from_3_to_4.append(i)
                logger.debug('Element to phase 4: {}'.format([x,y,z]))
                logger.debug('Element has density {}'.format(self.density[x,y,z]))
            else:
                yield_stress = self.S[x,y,z]
                Ep = self.Ep[x,y,z]
                eps_B = self.eps_B[x,y,z]
                new_stress = yield_stress+Ep*(eps-eps_B)
                new_E = new_stress/eps
                if new_E/1000 < self.stiffness[x,y,z]:
                    self.stiffness[x,y,z] = new_E/1000
                else:
                    continue

        # Update elements in phase 4, perfectly plastic phase      
        for i in range(len(self.elem_phase4)):
            [x,y,z] = [self.elem_phase4[i][0],self.elem_phase4[i][1],self.elem_phase4[i][2]]
            stress = self.results['von Mises stress'][x,y,z].magnitude
            E = self.stiffness[x,y,z]*1000
            eps = stress/E
            new_E = self.sigma_min[x,y,z]/eps
            if new_E/1000 < self.stiffness[x,y,z]:
                self.stiffness[x,y,z] = new_E/1000
            else:
                continue

        # Assign elements to new phase if needed
        for i in from_1_to_2:
            self.elem_phase2.append(self.elem_phase1[i])
        for i in from_2_to_3:
            self.elem_phase3.append(self.elem_phase2[i])
        for i in from_2_to_4:
            self.elem_phase4.append(self.elem_phase2[i])
        for i in from_3_to_4:
            self.elem_phase4.append(self.elem_phase3[i])
        
        remove_from_1 = from_1_to_2
        remove_from_2 = from_2_to_3 + from_2_to_4
        remove_from_3 = from_3_to_4

        for i in sorted(remove_from_1,reverse=True):
            self.elem_phase1.pop(i)
        for i in sorted(remove_from_2,reverse=True):
            self.elem_phase2.pop(i)
        for i in sorted(remove_from_3,reverse=True):
            self.elem_phase3.pop(i)
            
    def apply_load_linear(self,displacement_magnitude=None):
        '''
        Apply linear FE simulation
        '''
        # Find the displacement direction
        h5_disp = self._calculate_displacement()
        h5_disp_vector = np.array([h5_disp['x'],h5_disp['y'],h5_disp['z']])
        self._disp_direction = h5_disp_vector/h5_disp['magn']
        
        # Set paths
        results_folder = self.foldername+"_linear"
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)
        if displacement_magnitude is not None:
            if displacement_magnitude.is_integer():
                displacement_magnitude = int(displacement_magnitude)
        h5_file_results = os.path.join(results_folder,self.filename+'.h5')
        if os.path.exists(h5_file_results):
            os.remove(h5_file_results)
        results_log_file = h5_file_results[:-3] + '_log.txt'
        if os.path.exists(results_log_file):
            os.remove(results_log_file)

        # Set debugger
        logger = set_logger(results_log_file,print_debug=0)
        logger.info('Start linear load application')

        # Adapt displacement vectors
        normalized_disp_values = self.boundary_conditions['values']/h5_disp['magn']
        new_displacement_values = normalized_disp_values*displacement_magnitude
        self.boundary_conditions['values'] = new_displacement_values
        displacement_vector = self._disp_direction*displacement_magnitude
        self.applied_displacement['x'] = displacement_vector[0]
        self.applied_displacement['y'] = displacement_vector[1]
        self.applied_displacement['z'] = displacement_vector[2]
        self.applied_displacement['magn'] = displacement_magnitude

        # Calculate stiffness from density
        stiffness = self.density
        stiffness[self.femur_mask] = 14.9*self.density[self.femur_mask]**1.86 #GPa
        self.stiffness = stiffness
        
        # Call ParOSol for FEA
        start_wall_time = time.time()
        start_cpu_time = time.process_time()
        logger.debug('FEA running...')

        with ParosolSolver(
                stiffness = Quantity(self.stiffness,'GPa'),
                boundary_conditions = self.boundary_conditions,
                voxel_size = self.voxel_size,
                output = self.output,
                parallel_options = self.parallel_options,
                boundary_conditions_already_in_h5format = True,
                h5_file = h5_file_results) as solver:

                self.results = solver.get_results2()
        
        logger.info('Total wall clock time: {}'.format(str(timedelta(seconds=time.time() - start_wall_time))))
        logger.info('Total CPU time: {}'.format(str(timedelta(seconds=time.process_time() - start_cpu_time))))

        # Calculate total applied force
        self.total_force, self.projected_force, fix_force = self._calculate_force(logger)
        logger.debug('Displacement: {} {} {}'.format(str(self.applied_displacement['x']),
                            str(self.applied_displacement['y']),str(self.applied_displacement['z'])))
        logger.debug('Displacement magnitude: {}'.format(str(self.applied_displacement['magn'])))
        logger.debug('Force: {} {} {}'.format(str(self.projected_force['x']),
                            str(self.projected_force['y']),str(self.projected_force['z'])))
        logger.debug('Force magnitude: {}'.format(str(self.projected_force['magn'])))

        return self.total_force, self.projected_force, self.applied_displacement, self.results
    
    def apply_load_nonlinear(self,increment=0.1,displacement_magnitude=None):
        '''
        Apply nonlinear FE simulation through an iterative routine
        '''
        # Find the displacement direction
        h5_disp = self._calculate_displacement()
        h5_disp_vector = np.array([h5_disp['x'],h5_disp['y'],h5_disp['z']])
        self._disp_direction = h5_disp_vector/h5_disp['magn']
        
        # Set the number of iterations
        if displacement_magnitude is None:
            displacement_magnitude = h5_disp['magn']
        no_iterations = int(np.round(displacement_magnitude/increment))
        
        # Set paths
        results_folder = self.foldername+"_nonlinear"
        if os.path.exists(results_folder):
            shutil.rmtree(results_folder)
        os.mkdir(results_folder)
        results_log_file = os.path.join(results_folder,self.filename + '_log.txt')
        force_disp_txt_file = os.path.join(results_folder,self.filename + '_force-disp.txt')
        elem_phase_txt_file = os.path.join(results_folder,self.filename + '_elem-phase.txt')
        convergence_txt_file = os.path.join(results_folder,self.filename + '_convergence.txt')
        if os.path.exists(results_log_file):
            os.remove(results_log_file)
        force_disp_output = open(force_disp_txt_file, "w")
        force_disp_output.close()
        elem_phase_output = open(elem_phase_txt_file, "w") # keep track of element material phases (for debugging purposes)
        elem_phase_output.close()
        convergence_output = open(convergence_txt_file, "w") # keep track of convergence (for debugging purposes)
        convergence_output.close()

        # Set debugger
        logger = set_logger(results_log_file,print_debug=0)
        logger.debug('Run function apply_load_nonlinear')
        logger.info('Start nonlinear load application')
        force_disp = []
        force_disp.append('Displacement Force Projected_force\n')
        elem_phase = []
        elem_phase.append('Iteration Phase1 Phase2 Phase3 Phase4\n')
        convergence = []
        convergence.append('Iteration Force_load Force_fix\n')

        # Initialize FEA
        normalized_disp_values = self.boundary_conditions['values']/h5_disp['magn']
        self._initialize_material_properties(logger)

        # Start iterative FEA
        start_wall_time = time.time()
        start_cpu_time = time.process_time()
        start_system_time = time.perf_counter()
        total_solving_time = 0
        
        for i in range(1,no_iterations+1):
            logger.info('Iteration {} started.'.format(i))
            st = time.time()

            # Define file to save results
            h5_file_results = os.path.join(results_folder,self.filename + '_it{}'.format(str(i)) + '.h5')

            # Adapt displacement magnitude
            displacement_magn_i = i*increment
            new_displacement_values = normalized_disp_values*displacement_magn_i
            self.boundary_conditions['values'] = new_displacement_values
            displacement_vector_i = self._disp_direction*displacement_magn_i

            # Update material properties
            if i > 1:
                st_mat = time.time()
                self._update_material_properties(logger)
                logger.debug('Material properties updated after {} s'.format(str(timedelta(seconds=(time.time()-st_mat)))))

            # Clear up results
            self.results = None

            # Call ParOSol for FEA
            logger.debug('FEA running...')
            st_FEA_wall_time = time.time()
            st_FEA_cpu_time = time.process_time()
            st_FEA_system_time = time.perf_counter()
                    
            if round(displacement_magn_i,2).is_integer(): # Only save results to h5-file at limited number of iterations
                with ParosolSolver(
                    stiffness = Quantity(self.stiffness,'GPa'),
                    boundary_conditions = self.boundary_conditions,
                    voxel_size = self.voxel_size,
                    output = self.output,
                    parallel_options = self.parallel_options,
                    command_line_options = self.command_line_options,
                    boundary_conditions_already_in_h5format = True,
                    h5_file = h5_file_results) as solver:
                
                    self.results = solver.get_results2(image_data=True)
            else:
                with ParosolSolver(
                    stiffness = Quantity(self.stiffness,'GPa'),
                    boundary_conditions = self.boundary_conditions,
                    voxel_size = self.voxel_size,
                    output = self.output,
                    parallel_options = self.parallel_options,
                    command_line_options = self.command_line_options,
                    boundary_conditions_already_in_h5format = True) as solver:
                
                    self.results = solver.get_results2()

            logger.info('Iteration {} wall clock time: {}'.format(i,str(timedelta(seconds=(time.time() - st_FEA_wall_time)))))
            logger.info('Iteration {} CPU time: {}'.format(i,str(timedelta(seconds=(time.process_time() - st_FEA_cpu_time)))))
            logger.info('Iteration {} system time: {}'.format(i,str(timedelta(seconds=(time.perf_counter() - st_FEA_system_time)))))

            # Calculate total applied force
            self._update_force_and_displacement(logger,displacement_vector_i,displacement_magn_i)
            logger.debug('Displacement: {} {} {}'.format(str(self.applied_displacement['x'][i-1]),
                            str(self.applied_displacement['y'][i-1]),str(self.applied_displacement['z'][i-1])))
            logger.debug('Displacement magnitude: {}'.format(str(self.applied_displacement['magn'][i-1])))
            logger.debug('Force: {} {} {}'.format(str(self.projected_force['x'][i-1]),
                            str(self.projected_force['y'][i-1]),str(self.projected_force['z'][i-1])))
            logger.debug('Force magnitude: {}'.format(str(self.projected_force['magn'][i-1])))
          
            # Update force-displacement file
            force_disp.append(str(self.applied_displacement['magn'][i-1])+" "+str(self.total_force['magn'][i-1])+" "+
                              str(self.projected_force['magn'][i-1])+"\n")
            os.remove(force_disp_txt_file)
            force_disp_output = open(force_disp_txt_file, "w")
            force_disp_output.writelines(force_disp)
            force_disp_output.close()

            # Update elem-phase file
            elem_phase.append(str(i)+" "+str(len(self.elem_phase1))+" "+str(len(self.elem_phase2))+" "+str(len(self.elem_phase3))+" "+
                              str(len(self.elem_phase4))+"\n")
            os.remove(elem_phase_txt_file)
            elem_phase_output = open(elem_phase_txt_file, "w")
            elem_phase_output.writelines(elem_phase)
            elem_phase_output.close()

            # Update convergence file
            convergence.append(str(self.applied_displacement['magn'][i-1])+" "+str([self.total_force['x'][i-1],self.total_force['y'][i-1],
                                self.total_force['z'][i-1]])+"\n"+"    "+str([self.fix_force['x'][i-1],self.fix_force['y'][i-1],
                                self.fix_force['z'][i-1]])+"\n")
            os.remove(convergence_txt_file)
            convergence_output = open(convergence_txt_file, "w")
            convergence_output.writelines(convergence)
            convergence_output.close()

        # Report analysis time of the total simulation
        logger.info('Total wall clock time: {}'.format(str(timedelta(seconds=time.time() - start_wall_time))))
        logger.info('Total CPU time: {}'.format(str(timedelta(seconds=time.process_time() - start_cpu_time))))
        logger.info('Total system time: {}'.format(str(timedelta(seconds=time.perf_counter() - start_system_time))))

        # Plot the force-displacement curve
        self.plot_force_displacement_curve(force_disp_txt_file)

        return self.total_force, self.projected_force, self.applied_displacement, self.results