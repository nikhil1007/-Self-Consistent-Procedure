import pytest
import SCF
from main import main


def test_calc_nuclear_repulsion_energy(mol_h2o):
   assert SCF.calc_nuclear_repulsion_energy(mol_h2o) == 8.00236706181077, "Nuclear Repulsion Energy Test (H2O) Failed"

def test_calc_initial_density(mol_h2o):
   Duv = SCF.calc_initial_density(mol_h2o)
   assert Duv.sum() == 0.0
   assert Duv.shape == (mol_h2o.nao,mol_h2o.nao)

# def test_calc_hcore_matrix(get_Tuv, get_Huv):
#     Huv = SCF.calc_hcore_matrix(get_Tuv, get_Huv)
#     assert Huv[0,0] == -32.57739541261037
#     assert Huv[3,4] == 0.0
#
# def test_calc_fock_matrix()
#     Duv = SCF.calc_initial_density(mol_h2o)
#     Huv = SCF.calc_hcore_matrix(get_Tuv, get_Huv)
#     assert SCF.calc_fock_matrix(mol_h2o,Huv, main.eri, Duv)