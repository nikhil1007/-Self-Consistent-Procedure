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


def test_calc_fock_matrix(mol_h2o):
    Tuv = pickle.load(open("tuv.pkl", "rb"))
    Vuv = pickle.load(open("vuv.pkl", "rb"))
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = pickle.load(open("eri.pkl", "rb"))
    Duv = SCF.calc_initial_density(mol_h2o)
    assert SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)[0, 0] == -32.57739541261037, "Fock matrix test failed"


def test_Roothan_equations_total_energy(mol_h2o):
    Tuv = pickle.load(open("tuv.pkl", "rb"))
    Vuv = pickle.load(open("vuv.pkl", "rb"))
    Suv = pickle.load(open("suv.pkl", "rb"))
    Huv = SCF.calc_hcore_matrix(Tuv, Vuv)
    eri = pickle.load(open("eri.pkl", "rb"))
    Duv = SCF.calc_initial_density(mol_h2o)
    Fuv = SCF.calc_fock_matrix(mol_h2o, Huv, eri, Duv)
    Enuc_ = SCF.calc_nuclear_repulsion_energy(mol_h2o)
    mo_energies, mo_coeffs = SCF.solve_Roothan_equations(Fuv, Suv)
    assert mo_coeffs[0, :] == pytest.approx([-1.00154358e+00, -2.33624458e-01,
                                        4.97111543e-16, -8.56842145e-02,
                                        2.02299681e-29, 4.82226067e-02,
                                        -4.99600361e-16]), \
        "Roothan Equation Test Failed"
    assert mo_energies == pytest.approx([-32.5783029, -8.08153571, -7.55008599,
                                  -7.36396923, -7.34714487, -4.00229867,
                                  -3.98111115]), \
        "Roothan Equation Test Failed"

    Etot = (0.5 * (Duv * (Huv + Fuv)).sum()) + Enuc_
    assert Etot == pytest.approx(8.0023670618), \
        "total energy is not as expected"
