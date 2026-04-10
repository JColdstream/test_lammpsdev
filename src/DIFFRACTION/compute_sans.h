/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(sans,ComputeSANS);
// clang-format on
#else

#ifndef LMP_COMPUTE_SANS_H
#define LMP_COMPUTE_SANS_H

#include "compute.h"

namespace LAMMPS_NS {

class ComputeSANS : public Compute {
 public:
  ComputeSANS(class LAMMPS *, int, char **);
  ~ComputeSANS() override;
  void init() override;
  void compute_array() override;
  //double memory_usage() override;
  //testing
  //double sans_var[10];

 private:
  int me;
  int *ztype;           // Atomic number of the different atom types
  double dR_Ewald;      // Thickness of Ewald sphere slice
  bool echo;            // echo compute_array progress
  bool manual;          // Turn on manual recpiprocal map

  double R_Ewald;    // Radius of Ewald sphere (distance units)
  double qmin, qmax;     // min and max Radiation frequency (inverse distance units)
  int Nq;      // maximum integer value for K points in each dimension
  double kmax;       // Maximum reciprocal distance to explore
  int ksqmin, ksqmax;
  int maxdeg;
  double logqmin, logqmax;

  double mypi = 3.141592653589;

  bool logdist;

  int ntypes, nk;
  int nlocalgroup;
  int nRows, nCols;
  int *iksq, *ksq;
  double *k, *q, *skdeg, *skproc, *sktotal;
};

}    // namespace LAMMPS_NS

#endif
#endif

