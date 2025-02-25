import sys
sys.path.append('../wakis')
                
from wakis import geometry

import pytest 


class TestGeometryUtils:
    #Compare data
    STP_FILE = 'tests/stl/006_muonCavity.stp'
    
    COLORS = {'Vacuum|Half_cell_dx': [0.5, 0.800000011920929, 1.0],
            'Be windows|Be window left': [0.752941012382507, 0.752941012382507,0.752941012382507],
            'Be windows|Be window right': [0.752941012382507,0.752941012382507,0.752941012382507],
            'Walls|Cavity walls': [1.0, 0.615685999393463, 0.235293999314308],
            'Vacuum|Half_cell_sx': [0.5, 0.800000011920929, 1.0]}
    
    MATERIALS = {'Walls|Cavity walls': 'Copper (annealed)',
                'Be windows|Be window left': 'Berillium',
                'Be windows|Be window right': 'Berillium',
                'Vacuum|Half_cell_dx': 'Vacuum',
                'Vacuum|Half_cell_sx': 'Vacuum'}
    
    SOLIDS = {'Vacuum|Half_cell_dx': '000_Vacuum-Half_cell_dx_Vacuum.stl',
            'Be windows|Be window left': '001_Be-windows-Be-window-left_Berillium.stl',
            'Be windows|Be window right': '002_Be-windows-Be-window-right_Berillium.stl',
            'Walls|Cavity walls': '003_Walls-Cavity-walls_Copper--annealed-.stl',
            'Vacuum|Half_cell_sx': '004_Vacuum-Half_cell_sx_Vacuum.stl'}
    
    def test_colors(self):
        colors = geometry.extract_colors_from_stp(self.STP_FILE)
        assert colors == self.COLORS
    
    def test_materials(self):
        materials = geometry.extract_materials_from_stp(self.STP_FILE)
        assert materials == self.MATERIALS

    def test_solids(self):
        solids = geometry.extract_solids_from_stp(self.STP_FILE)
        assert solids == self.SOLIDS