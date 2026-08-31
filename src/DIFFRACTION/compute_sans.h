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
  double memory_usage() override;

 private:
  int me;
  double dR_Ewald;      // Thickness of Ewald sphere slice

  double kmin, kmax;    // min and max wave vector magnitude (inverse distance units)
  int nk;          // number of wave vectors distributed between kmin and kmax
  int ikmax;       // maximum number of periods in each dimension of the wavevector
  int maxdeg;      // maximum degeneracy allowed per value of k

  double mypi = 3.141592653589;
  double scatteringsum;

  bool logdist;
  bool scatteringlengths;

  const char *filename = nullptr;

  int ntypes;
  int nkvec;
  int nlocalgroup;
  int nRows, nCols;
  int *iksq;
  double *kvec, *k, *skdeg, *b;

  // persistent per-call scratch buffers, reused across invocations of
  // compute_array() instead of being allocated and freed every timestep
  int max_nlocalgroup;               // largest nlocalgroup seen so far (capacity of xlocal/typelocal)
  double *xlocal;                    // positions of local atoms in the group, compacted
  int *typelocal;                    // atom types of local atoms in the group, compacted
  double *cossinsum_ksq, *cossinsum_total;    // cos/sin accumulators, sized 2*nk (fixed after construction)
};

}    // namespace LAMMPS_NS

#endif
#endif

