"""
SCF.py is a module that contains all of the functions
for the HF SCF Procedure
"""

import numpy as np
import scipy.linalg


def calc_nuclear_repulsion_energy(mol_):
    """
    calc_nuclear_repulsion_energy - calculates the n-e repulsion energy of a
                                    molecule

    Arguments:
        mol_: the PySCF molecule data structure created from Input

    Returns:
        Enuc: The n-e repulsion energy
    """

    # Formula to calculate nuclear repulsion in molecule = product of charges of atoms / distance between them
    # Calculating the distance between the atoms using np.linalg.norm
    coordinates_atom = mol_.atom_coords()  # (x,y,z)
    charges = mol_.atom_charges()

    # did not construct a matrix so commenting it out
    # distance_matrix = np.zeros((3, 3), dtype=np.double)

    Enuc = 0

    # Calculating Nuclear Repulsion
    # index_A, index_B are used to select atom charges from charges list
    # atom_A, atom_B are used to get numpy nd-arrays
    # enumerate ref: https://www.geeksforgeeks.org/enumerate-in-python/
    for index_A, atom_A in enumerate(coordinates_atom):
        for index_B, atom_B in enumerate(coordinates_atom):
            if index_A == index_B:  # pointing to same atom
                continue
            za = charges[index_A]
            zb = charges[index_B]
            za_zb = za * zb
            Ra = atom_A
            Rb = atom_B
            R = np.linalg.norm(Ra - Rb)
            Enuc += za_zb / R

    return Enuc * 0.5  # because B > A condition


def calc_initial_density(mol_):
    """
    calc_initial_density - Function to calculate the initial guess density

    Arguments
        mol_: the PySCF molecule data structure created from Input

    Returns:
        Duv: the (mol.nao x mol.nao) Guess Density Matrix
    """

    num_aos = mol_.nao  # Number of atomic orbitals, dimensions of the mats
    Duv = np.zeros((num_aos, num_aos), dtype=np.double)  # num_aos X num_aos zero matrix as initial guess density matrix

    return Duv


def calc_hcore_matrix(Tuv_, Vuv_):
    """
    calc_hcore_matrix - Computes the 1 electron core matrix

    Arguments:
        Tuv_: The Kinetic Energy 1e integral matrix
        Vuv_: The Nuclear Repulsion 1e integrals matrix

    Returns:
        h_core: The one electron hamiltonian matrix
    """

    """
    Replace with your implementation

    Per the readme, this is a simple addition of the two matrices
    """
    h_core = np.add(Tuv_, Vuv_)  # addition of 2 numpy nd arrays
    return h_core


def calc_fock_matrix(mol_, h_core_, er_ints_, Duv_):
    """
    calc_fock_matrix - Calculates the Fock Matrix of the molecule

    Arguments:
        mol_: the PySCF molecule data structure created from Input
        h_core_: the one electron hamiltonian matrix
        er_ints_: the 2e electron repulsion integrals
        Duv_: the density matrix

    Returns:
        Fuv: The fock matrix

    """

    Fuv = h_core_.copy()  # Takes care of the Huv part of the fock matrix
    num_aos = mol_.nao  # Number of atomic orbitals, dimension of the mats

    # Calculating the Coulomb term and Exchange Term
    for mu in range(num_aos):
        for nu in range(num_aos):
            Fuv[mu, nu] += np.sum(np.multiply(Duv_, er_ints_[mu, nu]), dtype=np.double) - \
                           np.sum(np.multiply(Duv_, er_ints_[mu, :, nu] * 0.5), dtype=np.double)
    return Fuv


def solve_Roothan_equations(Fuv_, Suv_):
    """
    solve_Roothan_equations - Solves the matrix equations to determine
                              the MO coefficients

    Arguments:
        Fuv_: The Fock matrix
        Suv_: The overlap matrix

    Returns:
        mo_energies: an array that contains eigenvalues of the solution
        mo_coefficients: a matrix of the eigenvectors of the solution

    """
    mo_energies, mo_coeffs = scipy.linalg.eigh(Fuv_, Suv_)
    print(mo_coeffs[0,:])
    return mo_energies.real, mo_coeffs.real


def form_density_matrix(mol_, mo_coeffs_):
    """
    form_dentsity_matrix - forms the density matrix from the eigenvectors

    Note: the loops are over the number of electrons / 2, not all of the
    atomic orbitals

    Arguments:
        mol_: the PySCF molecule data structure created from Input
        mo_coefficients: a matrix of the eigenvectors of the solution

    Returns:
        Duv: the density matrix
    """

    nelec = mol_.nelec[0]  # Number of occupied orbitals
    num_aos = mol_.nao  # Number of atomic orbitals, dimensions of the mats
    Duv = np.zeros((mol_.nao, mol_.nao), dtype=np.double)

    for mu in range(num_aos):
        for nu in range(num_aos):
            for a in range(nelec):
                Duv[mu, nu] += 2 * (mo_coeffs_[mu, a] * mo_coeffs_[nu, a])
    """
    Replace with your implementation

    This involves basically a computation of each density matrix element
    that is a sum over the produces of the mo_coeffs.

    """

    return Duv


def calc_total_energy(Fuv_, Huv_, Duv_, Enuc_):
    """
    calc_total_energy - This function calculates the total energy of the
    molecular system

    Arguments:
        Fuv_: the current Fock Matrix
        Huv_: the core Hamiltonian Matrix
        Duv_: the Density Matrix that corresponds to Fuv_
        Enuc: the Nuclear Repulsion Energy

    Returns:
        Etot: the total energy of the molecule
    """

    Etot = (0.5 * (Duv_ * (Huv_ + Fuv_)).sum()) + Enuc_

    return Etot
