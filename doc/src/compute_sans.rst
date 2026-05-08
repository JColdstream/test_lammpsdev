.. index:: compute sans

compute sans command
====================

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID sans kmax keyword value ...

* ID, group-ID are documented in :doc:`compute <compute>` command
* sans = style name of this compute command
* kmax = maximum wave vector magnitude to explore (inverse length units)
* zero or more keyword/value pairs may be appended
* keyword = *qmin* or *qmax* or *Nq* or *dR_Ewald* or *maxdeg* or *logdist* or *lengthpath*

  .. parsed-literal::

       *qmin* value = minimum wave vector magnitude to calculate (inverse length units)
                      (default: 1.0 Å⁻¹)
       *qmax* value = maximum wave vector magnitude to calculate (inverse length units)
                      (default: 30.0 Å⁻¹)
       *Nq* value = number of wave vectors distributed between qmin and qmax
                    (default: 100)
       *dR_Ewald* value = thickness of Ewald sphere slice around target q values
                          (inverse length units)
                          (default: 0.2 Å⁻¹)
       *maxdeg* value = maximum degeneracy allowed for wave vector selection
                        (default: varies)
       *logdist* = flag to use logarithmic distribution of wave vectors instead
                   of linear distribution
       *lengthpath* file = path to file containing custom neutron scattering lengths

Examples
""""""""

.. code-block:: LAMMPS

   compute 1 all sans 10.0 qmin 1.0 qmax 30.0 Nq 50
   compute 2 all sans 8.0 qmax 25.0 dR_Ewald 0.15 logdist
   compute 3 all sans 12.0 Nq 100 lengthpath scattering_lengths.txt

Description
"""""""""""

Define a computation that calculates small-angle neutron scattering (SANS)
intensity as a function of wave vector magnitude (q), based on the atomic
structure of the system. This computation is particularly useful for
comparing molecular dynamics simulations with neutron diffraction
experiments.

The SANS intensity I(q) at each wave vector magnitude is computed from
the structure factor F(q) using:

.. math::

   I(q) = \frac{1}{N} \left| \sum_{i}^{N} b_i exp(2\pi i \mathbf{q} \cdot \mathbf{r}_i) \right|^2

where N is the number of atoms in the group, :math:`\mathbf{r}_i` is the
position of atom i, :math:`b_i` is the neutron scattering length for atom
type i, and :math:`\mathbf{q}` is the scattering wave vector with magnitude q.

**Wave Vector Generation**

The compute generates a set of wave vectors distributed within reciprocal
space. By default, wave vectors are distributed linearly between *qmin*
and *qmax* in steps determined by *Nq*. If the *logdist* flag is specified,
wave vectors are instead distributed on a logarithmic scale, which is often
more suitable for exploring a wide range of length scales.

The parameter *kmax* determines the maximum distance explored from the
origin of reciprocal space in units of inverse length. Wave vectors are
selected to lie approximately at the specified q magnitudes, within a
tolerance defined by *dR_Ewald* (the thickness of the Ewald sphere slice).

**Scattering Lengths**

By default, scattering lengths are set to 1.0 for all atom types, which
gives an unweighted calculation. For comparison with actual neutron
scattering experiments, custom neutron scattering lengths can be provided
via the *lengthpath* keyword, which specifies a file containing scattering
length values for each atom type.

The scattering lengths file should contain one line per atom type with the
format:

.. parsed-literal::

   type_number scattering_length

where type_number is the LAMMPS atom type (starting from 1) and
scattering_length is the neutron scattering length in inverse length
units (typically in fm or 10⁻¹⁵ m for neutrons).

**Degeneracy and Ewald Sphere**

The *dR_Ewald* parameter controls how many wave vectors are selected
around each target q value. A smaller value results in fewer selected
wave vectors but more precise targeting of specific q values. A larger
value samples a thicker sphere in reciprocal space around each q value.

The *maxdeg* parameter can be used to limit the degeneracy (number of
equivalent wave vectors) selected for each q value.

**Parallel Calculation**

This compute uses MPI to efficiently distribute the scattering calculation
across multiple processors. The cos and sin components of the structure
factor are accumulated separately on each processor and then reduced
globally before computing the final intensity.

Output info
"""""""""""

This compute calculates a global array with dimensions of Nq rows and
3 columns. Each row contains:

* Column 1: Index (i) of the wave vector
* Column 2: Wave vector magnitude (q) in inverse length units
* Column 3: Normalized SANS intensity I(q)/N

The array can be accessed by any command that uses global values from
a compute as input. See the :doc:`Howto output <Howto_output>` doc page
for an overview of LAMMPS output options.

All array values calculated by this compute are "intensive".

Restrictions
""""""""""""

This compute is part of the DIFFRACTION package. It is only enabled if
LAMMPS was built with that package. See the :doc:`Build package
<Build_package>` page for more info.

The compute_sans command does not work for triclinic cells.

The compute_sans command only works with 3-dimensional systems.

Related commands
""""""""""""""""

:doc:`compute saed <compute_saed>`, :doc:`compute xrd <compute_xrd>`

Default
"""""""

The option defaults are qmin = 1.0, qmax = 30.0, Nq = 100, dR_Ewald = 0.2,
logdist = off (linear distribution)
