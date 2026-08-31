.. index:: compute sans

compute sans command
====================

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID sans keyword value ...

* ID, group-ID are documented in :doc:`compute <compute>` command
* sans = style name of this compute command
* zero or more keyword/value pairs may be appended
* keyword = *kmin* or *kmax* or *ikmax* or *nk* or *dR_Ewald* or *maxdeg* or *logdist* or *lengthpath*

  .. parsed-literal::

       *kmin* value = minimum wave vector magnitude to calculate (inverse length units)
                      (default: .. math:: 1.0 \, \text{Å}^{-1})
       *kmax* value = maximum wave vector magnitude to calculate (inverse length units)
                      (default: .. math:: 30.0 \, \text{Å}^{-1})
       *ikmax* value = maximum number of periods in each dimension of the wavevector
                      (default: 50)
       *nk* value = number of wave vectors distributed between kmin and kmax
                    (default: 100)
       *dR_Ewald* value = thickness of Ewald sphere slice around target q values
                          (default: (kmax-kmin)/nk or (log10(kmax)-log10(kmin))/nk for logdist)
       *maxdeg* value = maximum degeneracy allowed per value of *k* selection
                     (default: 100)
       *logdist* = flag to use logarithmic distribution of wave vectors instead
                   of linear distribution
       *lengthpath* file = path to file containing custom neutron scattering lengths

Examples
""""""""

.. code-block:: LAMMPS

   compute 1 all sans
   compute 2 all sans kmin 1.0 kmax 30.0 nk 50
   compute 3 all sans kmax 25.0 dR_Ewald 0.15 logdist
   compute 4 all sans nk 100 lengthpath scattering_lengths.txt

Description
"""""""""""

Define a computation that calculates scattering intensity as a function of wave vector magnitude (*q*).
The scattering intensities of each atom type can be weighted to compare molecular dynamics simulations with neutron diffraction experiments.


The scattering intensity S(k) at each wave vector magnitude is computed from:

.. math::

   S(k) = \frac{1}{N} \left| \sum_{i}^{N} b_i exp(2\pi i \mathbf{k} \cdot \mathbf{r}_i) \right|^2
   
where N is the number of atoms in the group, :math:`\mathbf{r}_i` is the
position of atom i, :math:`b_i` is a weighting factor of atom type i (often the neutron scattering length for atom)
type i, and :math:`\mathbf{k}` is the scattering wave vector with magnitude k.

**Wave Vector Generation**

The compute generates a set of wave vectors distributed within reciprocal
space. By default, we sample *nk* intensities distributed evenly between *kmin*
and *kmax*. If the logdist flag is used they are distributed logarithmically betweek *kmin* and *kmax*.

At each value of *k* a number of wavevectors *\textbf{k} = (kx, ky, kz)* with magnitude *k* = 2*\pi/L (\textbf{k}) are selected at random.
The parameter *kmax* determines the maximum value of *kx*, *ky*, and *kz* explored. 

Wave vectors are selected to lie approximately at the specified k magnitudes, within a
tolerance defined by *dR_Ewald* (the thickness of the Ewald sphere slice).

**Scattering Lengths**

By default, scattering lengths are set to 1.0 for all atom types. For comparison with actual neutron
scattering experiments, custom neutron scattering lengths can be provided
via the *lengthpath* keyword, which specifies a file containing scattering
length values for each atom type. A table of neutron scattering lengths can be found on the NIST website: https://www.ncnr.nist.gov/resources/n-lengths/list.html

The scattering lengths file should contain one line per atom type with the
format:

.. parsed-literal::

   type_number scattering_length

where type_number is the LAMMPS atom type (starting from 1) and
scattering_length is the neutron scattering length in inverse length
units (typically in fm or $10^{-15}$ m for neutrons).
The first line is reserved as a comment line and should not be used for data input.

Note: This file does \textbf{not} need to include a value for every atom type, but does need to include a value for every atom type in the group specified for the compute.

**Degeneracy and Ewald Sphere**

The *dR_Ewald* parameter controls the thickness of the slice in reciprocal space.
A smaller value results in fewer selected wave vectors but more precise targeting of specific k values.
A larger value samples a thicker sphere in reciprocal space around each k value. 
If the value of *dR_Ewald* is large enough, points at adjacent values of *k* will overlap and they may sample from the same set of wavevectors.

The *maxdeg* parameter can be used to limit the degeneracy (number of equivalent wave vectors) selected for each k value.
If there are fewer than *maxdeg* wavevectors, they will all be used.

Output info
"""""""""""

This compute calculates a global array with dimensions of *nk* rows and
2 columns. Each row contains:

* Column 1: Wave vector magnitude (k) in inverse length units
* Column 2: Normalized scattering intensity S(k)/N

If a value of S(k) you are expecting is missing, it is because the compute has failed to find any wave vectors at that magnitude, given your input parameters.

The array can be accessed by any command that uses global values from
a compute as input. See the :doc:`Howto output <Howto_output>` doc page
for an overview of LAMMPS output options.

Restrictions
""""""""""""

This compute is part of the DIFFRACTION package. It is only enabled if
LAMMPS was built with that package. See the :doc:`Build package
<Build_package>` page for more info.

The compute_sans command does not work for triclinic cells.

The compute_sans command only works with 3-dimensional systems.

The compute_sans command only works with cubic simulation boxes (Lx = Ly = Lz).

The compute_sans command does work with simulations changing their box sized, however the wavevectors are initialised using the box sized defined at the time of the compute. 
If the simulation box changes size significantly during this time, the results will be rubbish and unphysical.

Related commands
""""""""""""""""

:doc:`compute saed <compute_saed>`, :doc:`compute xrd <compute_xrd>`

Default
"""""""

The option defaults are kmin = 1.0, kmax = 30.0, ikmax = 50, nk = 100,
maxdeg = 100, dR_Ewald = (kmax-kmin)/nk, logdist = off (linear distribution)
