import os, sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

sys.path.append('../wakis')

from tqdm import tqdm
from scipy.constants import c

from wakis import SolverFIT3D
from wakis import GridFIT3D 
from wakis import WakeSolver
from wakis.sources import Beam

import pytest 

# Turn true when running local
flag_plot_3D = False 

@pytest.mark.slow
class TestMPILossyCavity:
    # Regression data
    WP = np.array([-1.20518048e-18, -3.43166952e-14, -9.12281264e-11, -5.97035942e-08,
       -1.14192306e-05, -6.67821318e-04, -1.20415976e-02, -6.56115048e-02,
       -9.42241777e-02,  2.56742595e-02,  1.24781721e-01,  7.29968593e-02,
       -1.90301651e-02, -4.85604681e-02, -3.20302973e-02, -1.41630567e-02,
        6.08328447e-02,  8.46606533e-02, -5.47161983e-02, -1.16187054e-01,
        1.96150543e-02,  9.92691760e-02,  2.71860910e-02, -5.20061910e-02,
       -4.06093371e-02, -1.13177884e-02,  1.23466617e-02,  7.68570069e-02,
        2.25885825e-02, -9.37491042e-02, -5.75274767e-02,  6.39914398e-02,
        8.01287490e-02, -2.25733886e-02, -5.80093503e-02, -1.67528287e-02,
        1.87308868e-03,  4.10270452e-02,  5.36752361e-02, -3.30700962e-02,
       -8.35256382e-02, -3.49639450e-03,  8.57982574e-02,  3.52534514e-02,
       -5.59244132e-02, -3.74186918e-02, -1.57839607e-03,  2.16473120e-02,
        4.65718960e-02,  1.13331989e-02, -5.61960931e-02, -5.45019141e-02,
        4.42007208e-02,  7.38889175e-02, -1.60853246e-02, -5.54862915e-02,
       -1.77176103e-02,  1.61884285e-02,  3.33189414e-02,  2.63165025e-02,
       -1.83747643e-02, -5.94742659e-02, -1.15440054e-02,  6.90333378e-02,
        3.30370914e-02, -4.32087757e-02, -4.14847169e-02,  2.95619519e-03,
        3.11636613e-02,  2.43553113e-02,  5.16939737e-03, -3.70587116e-02,
       -4.24802683e-02,  3.09667053e-02,  5.76909730e-02, -4.78650193e-03,
       -5.02596085e-02, -2.09847948e-02,  2.46891158e-02,  2.77035180e-02,
        1.12378750e-02, -1.34129526e-02, -4.22348173e-02, -8.34181401e-03,
        4.81705970e-02,  3.05816259e-02, -2.98194687e-02, -4.38109787e-02,
        6.51710442e-03,  3.11656615e-02,  1.61470910e-02, -1.03388750e-03,
       -2.84571945e-02, -2.64685962e-02,  1.92416497e-02,  4.34041310e-02,
        3.00991809e-03, -4.36368319e-02, -2.09623698e-02,  2.48778148e-02,
        2.47296803e-02,  5.34349674e-03, -1.56807252e-02, -2.75519142e-02])

    Z = np.array([ 9.47509993e+00   -0.j        , -2.15919753e+00  +14.71883444j,
       -1.91733650e+00   +4.44948078j,  1.28440721e+01  +16.82326223j,
        2.08485569e-01  +30.97332066j, -7.33298758e-01  +20.00319681j,
        1.49516134e+01  +32.02189369j,  1.76404185e+00  +47.78480632j,
       -3.62936081e-01  +35.84459845j,  1.72535098e+01  +48.08448111j,
        3.04828054e+00  +66.52697928j, -5.26680373e-01  +53.02686698j,
        2.02908651e+01  +66.08138383j,  4.31307804e+00  +88.81026936j,
       -1.29489146e+00  +72.83357603j,  2.48045309e+01  +87.61209285j,
        5.80483916e+00 +117.5679589j , -3.03027460e+00  +97.61960575j,
        3.23904476e+01 +115.93394774j,  8.03126208e+00 +159.57208335j,
       -6.75009170e+00 +133.07726839j,  4.77988269e+01 +159.81292951j,
        1.30322590e+01 +236.10780148j, -1.57096479e+01 +199.62556782j,
        9.49879835e+01 +256.8795857j ,  4.34787106e+01 +474.2606175j ,
       -3.12909870e+01 +497.30323191j,  1.47315324e+03 +895.76636849j,
        4.48248106e+02-1075.94784917j,  1.49875710e+02 -152.20119808j,
       -7.50787065e+01  -96.69205009j,  4.27773743e+01 -116.86898698j,
        6.11554771e+01  +62.99346182j, -4.02916518e+01  +64.46673526j,
        4.30681823e+01  +43.93756493j,  5.36697200e+01 +181.20496927j,
       -2.28378682e+01 +181.85768871j,  8.69832235e+01 +205.13283157j,
        9.91399674e+01 +442.50957163j,  8.39538715e+01 +558.77780899j,
        1.44544640e+03 +763.92492065j,  7.31144725e+02-1098.00387535j,
        1.19236876e+02 -319.2572357j , -4.43981312e+01 -279.36838792j,
        7.66442044e+01 -202.58986802j,  2.32370752e+01  -38.2338223j ,
       -3.83082870e+01  -83.64248558j,  6.19353789e+01  -55.58739159j,
        2.07975113e+01  +62.02539709j, -1.72130461e+01   +2.47782512j])

    Ez = np.array([  22.29305573,   -5.82227204,   -6.110679  ,   -1.49713877,
        2.05901056,    0.8483035 ,   -9.88525861,  -25.489663  ,
        -47.77037626,  -71.31731543,  -93.72391558, -112.00479779,
        -128.66387341, -141.15872256, -148.13577743, -151.73277772,
        -147.98286971, -137.3735954 , -119.93100115,  -92.03520264,
        -60.36238734,  -30.38879522,  -11.74210147,   -3.59764528,
        -0.32953207,    0.64311415,    0.71992362,   -0.31853229])

    img_folder = 'tests/007_img/'
    if not os.path.exists(img_folder): 
            os.mkdir(img_folder)

    def test_mpi_import(self):
        # ---------- MPI setup ------------
        global use_mpi
        try:
            # can be skipped since it is handled inside GridFIT3D
            from mpi4py import MPI 

            comm = MPI.COMM_WORLD  # Get MPI communicator
            rank = comm.Get_rank()  # Process ID
            size = comm.Get_size()  # Total number of MPI processes
            use_mpi = True
        except:
            use_mpi = False
        
        print(f"Using mpi: {use_mpi}")

    def test_mpi_simulation(self):
        # ---------- Domain setup ---------

        # Geometry & Materials
        solid_1 = 'tests/stl/007_vacuum_cavity.stl'
        solid_2 = 'tests/stl/007_lossymetal_shell.stl'

        stl_solids = {'cavity': solid_1, 
                    'shell': solid_2
                    }

        stl_materials = {'cavity': 'vacuum', 
                        'shell': [30, 1.0, 30] #[eps_r, mu_r, sigma[S/m]]
                        }

        # Extract domain bounds from geometry
        solids = pv.read(solid_1) + pv.read(solid_2)
        xmin, xmax, ymin, ymax, ZMIN, ZMAX = solids.bounds

        # Number of mesh cells
        Nx = 60
        Ny = 60
        NZ = 140
        global use_mpi
        grid = GridFIT3D(xmin, xmax, ymin, ymax, ZMIN, ZMAX, 
                        Nx, Ny, NZ, 
                        use_mpi=use_mpi, # Enables MPI subdivision of the domain
                        stl_solids=stl_solids, 
                        stl_materials=stl_materials,
                        stl_scale=1.0,
                        stl_rotate=[0,0,0],
                        stl_translate=[0,0,0],
                        verbose=1)

        if use_mpi:
            print(f"Process {grid.rank}: Handling Z range {grid.zmin} to {grid.zmax} with {grid.Nz} cells")

        # ------------ Beam source & Wake ----------------
        # Beam parameters
        sigmaz = 10e-2      #[m] -> 2 GHz
        q = 1e-9            #[C]
        beta = 1.0          # beam beta 
        xs = 0.             # x source position [m]
        ys = 0.             # y source position [m]
        ti = 3*sigmaz/c     # injection time [s]

        beam = Beam(q=q, sigmaz=sigmaz, beta=beta,
                    xsource=xs, ysource=ys, ti=ti)

        # ----------- Solver & Simulation ----------
        # boundary conditions
        bc_low=['pec', 'pec', 'pec']
        bc_high=['pec', 'pec', 'pec']

        # Solver setup
        global solver
        solver = SolverFIT3D(grid,
                            bc_low=bc_low, 
                            bc_high=bc_high, 
                            use_stl=True, 
                            use_mpi=use_mpi, # Activate MPI
                            bg='pec' # Background material
                            )

        # -------------- Custom time loop  -----------------
        if use_mpi:
            Nt = 3000
            for n in tqdm(range(Nt)):

                beam.update(solver, n*solver.dt)
                solver.mpi_one_step()

            Ez = solver.mpi_gather('Ez', x=int(Nx/2), y=int(Ny/2), z=np.s_[::5])
            if solver.rank == 0:
                assert np.allclose(Ez, self.Ez), "Electric field Ez samples MPI failed"
        else:
            Nt = 3000
            for n in tqdm(range(Nt)):

                beam.update(solver, n*solver.dt)
                solver.one_step()
            
            Ez = solver.E[int(Nx/2), int(Ny/2), np.s_[::5], 'z']
            assert np.allclose(Ez, self.Ez), "Electric field Ez samples failed"


    def test_mpi_gather_asField(self):
        # Plot inspect after mpi gather
        global solver
        if use_mpi:
            E = solver.mpi_gather_asField('E')
            if solver.rank == 0: #custom plots go in rank 0
                fig, ax = E.inspect(figsize=[20,6], plane='YZ', show=False, handles=True)
                fig.savefig(self.img_folder+'Einspect_'+str(3000).zfill(4)+'.png')
                plt.close(fig)
        else:
            fig, ax = solver.E.inspect(figsize=[20,6], plane='YZ', show=False, handles=True)
            fig.savefig(self.img_folder+'Einspect_'+str(3000).zfill(4)+'.png')
            plt.close(fig)

    def test_mpi_plot2D(self):
        # Plot E abs in 2D every 20 timesteps 
        global solver
        solver.plot2D(field='E', component='Abs', 
                    plane='YZ', pos=0.5, 
                    cmap='rainbow', vmin=0, vmax=500., interpolation='hanning',
                    off_screen=True, title=self.img_folder+'Ez2d', n=3000)

    def test_mpi_plot1D(self):
        # Plot E z in 1D at diferent transverse positions `pos` every 20 timesteps
        global solver
        solver.plot1D(field='E', component='z', 
                line='z', pos=[0.45, 0.5, 0.55], 
                xscale='linear', yscale='linear',
                off_screen=True, title=self.img_folder+'Ez1d', n=3000)
    
    @pytest.mark.skipif(not flag_plot_3D, reason="Requires interactive plotting")
    def test_mpi_plot3D(self):
        # Plot Abs Electric field on domain
        # disabled when mpi = True
        global solver
        solver.plot3D('E', component='Abs', 
                cmap='rainbow', clim=[0, 500],
                add_stl=['cavity', 'shell'], stl_opacity=0.1,
                clip_interactive=True, clip_normal='-y')
        
    @pytest.mark.skipif(not flag_plot_3D, reason="Requires interactive plotting")    
    def test_mpi_plot3DonSTL(self):
        # Plot Abs Electric field on STL solid `cavity`
        # disabled when mpi = True
        global solver
        solver.plot3DonSTL('E', component='Abs', 
                        cmap='rainbow', clim=[0, 500],
                        stl_with_field='cavity', field_opacity=1.0,
                        stl_transparent='shell', stl_opacity=0.1, stl_colors='white',
                        #clip_plane=True, clip_normal='-y', clip_origin=[0,0,0], #coming in v0.5.0
                        off_screen=False, zoom=1.2, title=self.img_folder+'Ez3d')
            
    
    def test_mpi_wakefield(self):
        # Reset fields
        global solver
        solver.reset_fields()

        # ------------ Beam source ----------------
        # Beam parameters
        sigmaz = 10e-2      #[m] -> 2 GHz
        q = 1e-9            #[C]
        beta = 1.0          # beam beta 
        xs = 0.             # x source position [m]
        ys = 0.             # y source position [m]
        xt = 0.             # x test position [m]
        yt = 0.             # y test position [m]
        # [DEFAULT] tinj = 8.53*sigmaz/c_light  # injection time offset [s] 

        # ----------- Wake Solver  setup  ----------
        # Wakefield post-processor
        wakelength = 10. # [m] -> Partially decayed
        skip_cells = 10  # no. cells to skip at zlo/zhi for wake integration
        results_folder = 'tests/007_results/'

        global wake
        wake = WakeSolver(q=q, sigmaz=sigmaz, beta=beta,
                        xsource=xs, ysource=ys, xtest=xt, ytest=yt,
                        skip_cells=skip_cells,
                        results_folder=results_folder,
                        Ez_file=results_folder+'Ez.h5',)

        # Run simulation
        solver.wakesolve(wakelength=wakelength, 
                         wake=wake)
    
    def test_long_wake_potential(self):
        global wake
        global solver
        if use_mpi:
            if solver.rank == 0:
                assert np.allclose(wake.WP[::50], self.WP), "Wake potential samples failed"
                assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(184.43818552913254, 0.1), "Wake potential cumsum MPI failed"
        else:
            print(np.cumsum(np.abs(wake.WP))[-1])
            assert np.allclose(wake.WP[::50], self.WP), "Wake potential samples failed"
            assert np.cumsum(np.abs(wake.WP))[-1] == pytest.approx(184.43818552913254, 0.1), "Wake potential cumsum MPI failed"

    def test_long_impedance(self):
        global wake
        global solver
        if use_mpi:
            if solver.rank == 0:
                assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z)), "Abs Impedance samples MPI failed"
                assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z)), "Real Impedance samples MPI failed"
                assert np.allclose(np.imag(wake.Z)[::20], np.imag(self.Z)), "Imag Impedance samples MPI failed"
                assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(250910.51090497518, 0.1), "Abs Impedance cumsum MPI failed"
        else:
            print(np.cumsum(np.abs(wake.Z))[-1])
            assert np.allclose(np.abs(wake.Z)[::20], np.abs(self.Z)), "Abs Impedance samples failed"
            assert np.allclose(np.real(wake.Z)[::20], np.real(self.Z)), "Real Impedance samples failed"
            assert np.allclose(np.imag(wake.Z)[::20], np.imag(self.Z)), "Imag Impedance samples failed"
            assert np.cumsum(np.abs(wake.Z))[-1] == pytest.approx(250910.51090497518, 0.1), "Abs Impedance cumsum failed"
    