import pytest
import SCF
import main
import pickle


def test_calc_nuclear_repulsion_energy(mol_h2o):
    assert SCF.calc_nuclear_repulsion_energy(mol_h2o) == 8.00236706181077, "Nuclear Repulsion Energy Test (H2O) Failed"


def test_calc_initial_density(mol_h2o):
    Duv = SCF.calc_initial_density(mol_h2o)
    assert Duv.sum() == 0.0
    assert Duv.shape == (mol_h2o.nao, mol_h2o.nao)


def test_calc_hcore_matrix():
    Tuv = pickle.load(open("tuv.pkl", "rb"))
    Vuv = pickle.load(open("vuv.pkl", "rb"))
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    assert Huv[0, 0] == -32.57739541261037
    assert Huv[3, 4] == 0.0
