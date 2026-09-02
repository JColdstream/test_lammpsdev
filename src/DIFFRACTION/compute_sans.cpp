// clang-format off /* ---------------------------------------------------------------------- LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator https://www.lammps.org/, Sandia National Laboratories LAMMPS development team: developers@lammps.org Copyright (2003) Sandia Corporation.  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software.  This software is distributed under the GNU General Public License.  See the README file in the top-level LAMMPS directory.  ------------------------------------------------------------------------- */ /* ---------------------------------------------------------------------- Contributing authors: Jonathan Coldstream (Edinburgh), based off code from Shawn Coleman & Douglas Spearot (Arkansas) ------------------------------------------------------------------------- */ 

#include "compute_sans.h"

#include "atom.h" 
#include "citeme.h" 
#include "comm.h" 
#include "domain.h" 
#include "error.h" 
#include "group.h" 
#include "math_const.h"
#include "memory.h"
#include "update.h"
#include "text_file_reader.h"

#include <cmath>
#include <cstring>

#include <vector>
#include <algorithm>
#include <random>

#include "omp_compat.h"
using namespace LAMMPS_NS;
using namespace MathConst;

static const char cite_compute_saed_c[] =
"Test citation!!"
  "\n\n";

// Combined sin+cos: on glibc this is one call instead of two separate
// transcendental function calls, which matters here since it runs once per
// (wavevector, atom) pair. Falls back to plain sin()/cos() elsewhere.
static inline void sans_sincos(double x, double &s, double &c)
{
#if defined(__GLIBC__)
  ::sincos(x, &s, &c);
#else
  s = std::sin(x);
  c = std::cos(x);
#endif
}

ComputeSANS::ComputeSANS(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg), k(nullptr), kvec(nullptr), iksq(nullptr),
  b(nullptr), skdeg(nullptr), max_nlocalgroup(0), xlocal(nullptr),
  typelocal(nullptr), cossinsum_ksq(nullptr), cossinsum_total(nullptr)
{

  if (lmp->citeme) lmp->citeme->add(cite_compute_saed_c);

  int ntypes = atom->ntypes;
  int natoms = group->count(igroup);
  int dimension = domain->dimension;
  int triclinic = domain->triclinic;
  me = comm->me;
  nprocs = comm->nprocs;

  // Checking errors specific to the compute
  if (dimension == 2)
    error->all(FLERR,"Compute SANS does not work with 2d structures");
  if (narg < 3)
    error->all(FLERR,"Illegal Compute SANS Command");
  if (triclinic == 1)
    error->all(FLERR,"Compute SANS does not work with triclinic structures");

  array_flag = 1;
  extarray = 0;

  // Define atom types for atomic scattering factor coefficients
  // first arg after required
  int iarg = 3;
  
  // Set defaults for optional args
  kmax = 0.5;
  kmin = 0.001;
  ikmax = 50;
  maxdeg = 100;
  nk = 100;
  dR_Ewald = (kmax - kmin) / nk;
  logdist = 0;
  scatteringlengths=0;

  // utils::logmesg(lmp,"arg[0] = {}\n", arg[0]);

  // Process optional args
  while (iarg < narg) {

    if (strcmp(arg[iarg],"kmin") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal Compute SANS Command");
      kmin = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg],"kmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal Compute SANS Command");
      kmax = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (kmax < kmin)
        error->all(FLERR,"Compute SANS: kmax must be greater than kmin");
      iarg += 2;

    } else if (strcmp(arg[iarg],"ikmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal Compute SANS Command");
      ikmax = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (ikmax < 0)
        error->all(FLERR,"Compute SANS: ikmax must be greater than zero");
      iarg += 2;

    } else if (strcmp(arg[iarg],"nk") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal Compute SANS Command");
      nk = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (nk < 0)
        error->all(FLERR,"number of wavevectors to calculate must be greater than 0");
      iarg += 2;

    } else if (strcmp(arg[iarg],"dR_Ewald") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal Compute SANS Command");
      dR_Ewald = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (dR_Ewald < 0)
        error->all(FLERR,"Compute SANS: dR_Ewald slice must be greater than or equal to 0");
      iarg += 2;

    } else if (strcmp(arg[iarg],"maxdeg") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal Compute SANS Command");
      maxdeg = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (maxdeg < 1)
        error->all(FLERR,"Compute SANS: maxdeg must be greater than 0");
      iarg += 2;

    } else if (strcmp(arg[iarg],"logdist") == 0) {
      logdist = true;
      iarg += 1;
      // dR_Ewald = log10(kmax/kmin) / nk;

    } else if (strcmp(arg[iarg],"lengthpath") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal Compute SANS Command");
      scatteringlengths = true;
      filename = arg[iarg+1];
      iarg += 2;

    } else error->all(FLERR,"Illegal Compute SANS Command");
  }

  // Read custom scattering lengths if required and assign them to the array b.
  // value i holds ith scattering length density, value ntypes+i holds a flag to check if it has been assigned
  memory->create(b, 2*ntypes, "sans:b");

  // initialise array to -1.0
  for (int i = 0; i < 2*ntypes; i++) {
    b[i] = -1.0;
  }

  //if (comm->me == 0) {
    if (scatteringlengths) {
      int typeindex;
      // utils::logmesg(lmp, "DEBUG :: Reading scattering lengths from {}\n", filename);

      FILE *fp = fopen(filename, "rb");
      if (fp == nullptr) error->one(FLERR, "Failed to open {}. Check your file path.", filename);

      TextFileReader reader(fp, "Scattering Lengths");
      reader.skip_line();

      bool eof = false;
      while (!eof){

        char *line = reader.next_line();
        // check to see if we are at end of file
        if (line == nullptr) {
          eof = true;
          break;
        }

        std::vector<std::string> values = utils::split_words(line);

        if (values.size() < 2) {
          error->one(FLERR, "Invalid line in scattering lengths file: {}", line);
        } else if (values.size() > 2) {
          error->one(FLERR, "Invalid line in scattering lengths file: {}", line);
        } else if (!utils::is_integer(values[0])) {
          error->one(FLERR, "Invalid atom type in scattering lengths file: {}", line);
        } else if (!utils::is_double(values[1])) {
          error->one(FLERR, "Invalid scattering length in scattering lengths file: {}", line);
        } else
  
        typeindex = utils::numeric(FLERR, values[0], false, lmp)-1;
        if (typeindex >= ntypes) {
          error->one(FLERR, "Invalid atom type in scattering lengths file: {}", line);
        }

        b[typeindex] = utils::numeric(FLERR, values[1], false, lmp);
        b[ntypes+typeindex] = 1.0; // mark as assigned

      } 


    if (fp) fclose(fp);

    // if we don't have custom scattering lengths set all to 1.0.
    } else {
      for (int i = 0; i < 2*ntypes; i++) {
        b[i] = 1.0;
      }
    }


  // check total number of wavevectors to calculate, discarding duplicate values

  // nsamples = 0;

  // allocate memory for q and results
  const double* boxlo = domain->boxlo;
  const double* boxhi = domain->boxhi;
  auto boxdim = new double [3];

  // calculate box lengths
  for (int i = 0; i < 3; i++){
    boxdim[i] = boxhi[i] - boxlo[i];
  }

  // Check if box is cubic
  if (fabs(boxdim[0]-boxdim[1]) > 1.0e-6*boxdim[0] ||
      fabs(boxdim[0]-boxdim[2]) > 1.0e-6*boxdim[0])
    error->all(FLERR,"Compute SANS requires a cubic simulation box (Lx = Ly = Lz)");

  double twopi_L = 2.0*mypi/boxdim[0];

  // tempk to count initial wavevectors
  auto tempk = new double[nk];
  if (logdist) {
    double logkmin = log10(kmin);
    double logkmax = log10(kmax);
    for (int i = 0; i < nk; i++){
      tempk[i] = pow(10, (logkmax-logkmin)*i/nk + logkmin);
    }
  } else {
    for (int i = 0; i < nk; i++){
      tempk[i] = (kmax-kmin)*i/nk + kmin;
      }
    }
  
  auto tempskdeg = new int[nk];
  for (int i = 0; i < nk; i++) {
    tempskdeg[i] = 0;
  }

  int tempnkvec = 0;
  int nullcount = 0;
  int tempksq;
  double tempmodk;

  for (int ik = 0; ik < nk; ik++){
    // distribute shells between ranks
    if (ik % nprocs != me) continue;
     for (int ix = 0; ix <= ikmax; ix++) {
      for (int iy = -ikmax; iy <= ikmax; iy++) {
        for (int iz = -ikmax; iz <= ikmax; iz++) {
            tempksq = ix*ix + iy*iy + iz*iz;
            tempmodk = twopi_L * sqrt((double)tempksq);
            if (fabs(tempmodk - tempk[ik]) < dR_Ewald/2) {
              tempskdeg[ik] = tempskdeg[ik] + 1;
            }
          }
        }
      }
    }

  // each rank contains only its own shells' counts
  // other ranks are 0 so the reduce works properly
  MPI_Allreduce(MPI_IN_PLACE, tempskdeg, nk, MPI_INT, MPI_SUM, world);

  // check which values of k have no valid combinations
  for (int ik = 0; ik < nk; ik++){
    // keep track of any values of k that have no valid combinations
    if (tempskdeg[ik] == 0) {
      nullcount++;
    } else if (tempskdeg[ik] > maxdeg) {
      tempnkvec = tempnkvec + maxdeg;
      tempskdeg[ik] = maxdeg;
    } else {
      tempnkvec = tempnkvec + tempskdeg[ik];
    }
  }

  memory->create(k, nk-nullcount,"sans:k");
  memory->create(skdeg, nk-nullcount,"sans:skdeg");

  nkvec = tempnkvec;

  nRows = nk - nullcount;
  nCols = 2;

  size_array_rows = nRows;
  size_array_cols = nCols;

  // allocate memory 4 fat arrays //
  memory->create(kvec,3*nkvec,"sans:kvec");
  memory->create(iksq, nkvec,"sans:iksq");
  memory->create(array, nRows, nCols, "sans:array");

  // offset[i] keeps track of where shell i wavevectors begin
  std::vector<int> offset(nk - nullcount);

  int kcount = 0;
  int running_offset = 0;
  for (int i = 0; i < nk; i++){
    if (tempskdeg[i] > 0) {
      k[kcount] = tempk[i];
      skdeg[kcount]=tempskdeg[i];
      offset[kcount] = running_offset;
      running_offset += tempskdeg[i];
      kcount++;
    }
  }

  // update number of wavevectors to avoid values of k with no valid combinations
  nk = nk - nullcount;

  // check that number of wavevectors is consistent
  if (nk != kcount) {
    error->all(FLERR, "Compute SANS: Inconsistent number of wavevectors");
  }

  if (me == 0) {
    utils::logmesg(lmp,"-----\nComputing wavevectors for computeSANS.\n");
  }

  for (int i = 0; i < 3*nkvec; i++) {
    kvec[i] = 0.0;
  }
  for (int i = 0; i < nkvec; i++) {
    iksq[i] = 0;
  }

  std::vector<std::vector<int>> tempkvec;
  for (int ik = 0; ik < nk; ik++){
    if (ik % nprocs != me) continue;
     for (int ix = 0; ix <= ikmax; ix++) {
      for (int iy = -ikmax; iy <= ikmax; iy++) {
        for (int iz = -ikmax; iz <= ikmax; iz++) {
            tempksq = ix*ix + iy*iy + iz*iz;
            tempmodk = twopi_L * sqrt((double)tempksq);
            if (fabs(tempmodk - k[ik]) < dR_Ewald/2) {
              tempkvec.push_back({ix, iy, iz});
            }
          }
        }
      }

      if ((int) tempkvec.size() < skdeg[ik]) {
        error->one(FLERR,"ComputeSANS: Number of wavevectors is inconsistent. Contact the developers.");
      }

      int base = offset[ik];
      if (skdeg[ik] == maxdeg) {
        // select a random subset of wavevectors from the allowed list if there are more than maxdeg
        std::shuffle(tempkvec.begin(), tempkvec.end(), std::default_random_engine{});
        tempkvec.resize(maxdeg);
      }
      for (int j = 0; j < skdeg[ik]; j++) {
        kvec[3*(base+j)+0] = twopi_L*tempkvec[j][0];
        kvec[3*(base+j)+1] = twopi_L*tempkvec[j][1];
        kvec[3*(base+j)+2] = twopi_L*tempkvec[j][2];
        iksq[base+j] = ik;
      }
      tempkvec.clear();
    }

  // merge every rank's owned-shell contributions into the complete
  // kvec[]/iksq[] arrays -- the same "zero-fill and sum" trick as above
  MPI_Allreduce(MPI_IN_PLACE, kvec, 3*nkvec, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(MPI_IN_PLACE, iksq, nkvec, MPI_INT, MPI_SUM, world);

  if (me == 0) {
    utils::logmesg(lmp,"\nFound {} wavevectors.\n", nkvec);
  }

  memory->create(cossinsum_ksq, 2*nk, "sans:cossinsum_ksq");
  memory->create(cossinsum_total, 2*nk, "sans:cossinsum_total");

  // calculate sum of all scattering lengths to normalise at the end
  // equal to number of atoms if lengths are not set by the user
  if (scatteringlengths) {
    double proc_scatteringsum = 0.0;
    ntypes = atom->ntypes;
    const auto nlocal = atom->nlocal;
    const auto *type  = atom->type;
    const auto *mask = atom->mask;

    // checks to see if atoms are included in group for compute
    // types are type-1 so they correspond to the correct index
    for (int ii = 0; ii < nlocal; ii++) {
      if (mask[ii] & groupbit) {
        if (b[ntypes+type[ii]-1] < 0.0) {
          error->all(FLERR,"COMPUTE SANS: atom type {} has no scattering length assigned\n", type[ii]);
        }
        proc_scatteringsum += b[type[ii]-1];
      }
    }

    // calculate total scattering sum and broadcast to all processes
    MPI_Allreduce(&proc_scatteringsum, &scatteringsum, 1, MPI_DOUBLE, MPI_SUM, world);
  } else {
    scatteringsum = natoms;
  }

  delete[] boxdim;
  delete[] tempk;
  delete[] tempskdeg;
}

/* ---------------------------------------------------------------------- */

ComputeSANS::~ComputeSANS()
{

  memory->destroy(k);
  memory->destroy(kvec);
  memory->destroy(iksq);
  memory->destroy(b);
  memory->destroy(skdeg);
  memory->destroy(array);
  memory->destroy(xlocal);
  memory->destroy(typelocal);
  memory->destroy(cossinsum_ksq);
  memory->destroy(cossinsum_total);
}

/* ---------------------------------------------------------------------- */

void ComputeSANS::init()
{

}


void ComputeSANS::compute_array()
{
  invoked_array = update->ntimestep;

  double t0 = platform::walltime();

  ntypes = atom->ntypes;
  const auto nlocal = atom->nlocal;
  const auto *type  = atom->type;
  const auto natoms = group->count(igroup);
  const auto *mask = atom->mask;

  // checks to see if atoms are included in group for compute
  nlocalgroup = 0;
  for (int ii = 0; ii < nlocal; ii++) {
    if (mask[ii] & groupbit) {
     nlocalgroup++;
    }
  }

  // xlocal and typelocal are persistent and grow as needed if the number of atoms on the process increases
  if (nlocalgroup > max_nlocalgroup) {
    memory->grow(xlocal, 3*nlocalgroup, "sans:xlocal");
    memory->grow(typelocal, nlocalgroup, "sans:typelocal");
    max_nlocalgroup = nlocalgroup;
  }

  // populate positions and types
  nlocalgroup = 0;
  for (int ii = 0; ii < nlocal; ii++) {
    if (mask[ii] & groupbit) {
     xlocal[3*nlocalgroup+0] = atom->x[ii][0];
     xlocal[3*nlocalgroup+1] = atom->x[ii][1];
     xlocal[3*nlocalgroup+2] = atom->x[ii][2];
     typelocal[nlocalgroup] = type[ii];
     nlocalgroup++;
    }
  }

  for (int i = 0; i < 2*nk; i++) {
    cossinsum_ksq[i] = 0.0;
  }

  // vars for scattering
  double kx, ky, kz;
  double cossum, sinsum;
  double kdotr;
  double sk, ck;

for (int ik = 0; ik < nkvec; ik++){
  // set up wavevectors
  kx = kvec[3*ik+0];
  ky = kvec[3*ik+1];
  kz = kvec[3*ik+2];
  cossum=0.0;
  sinsum=0.0;
  // compute the dot product
    for (int ii=0; ii < nlocalgroup; ii++) {
      kdotr = (kx*xlocal[3*ii+0] + ky*xlocal[3*ii+1] + kz*xlocal[3*ii+2]);
      sans_sincos(kdotr, sk, ck);
      cossum += b[typelocal[ii]-1]*ck;
      sinsum += b[typelocal[ii]-1]*sk;
    }

    cossinsum_ksq[2*iksq[ik]+0] += cossum;
    cossinsum_ksq[2*iksq[ik]+1] += sinsum;
}

  // sum up cos/sin sums across processes
  MPI_Allreduce(cossinsum_ksq, cossinsum_total, 2*nk, MPI_DOUBLE, MPI_SUM, world);

  for (int i = 0; i < nk; i++) {
    array[i][0] = k[i];
    array[i][1] = (cossinsum_total[2*i+0]*cossinsum_total[2*i+0]+cossinsum_total[2*i+1]*cossinsum_total[2*i+1])*natoms/skdeg[i]/scatteringsum/scatteringsum;
  }

  double t1 = platform::walltime();

  // timer if needed
  if (me == 0) {
    // utils::logmesg(lmp,"Scattering sum = {}\n", scatteringsum);
    utils::logmesg(lmp,"Time for SANS calculation: {} seconds\n", t1-t0);
  }

}

/* ----------------------------------------------------------------------
 memory usage of arrays
 ------------------------------------------------------------------------- */

double ComputeSANS::memory_usage()
{
  double bytes = 0.0;
  
  bytes += (double) 3 * nkvec * sizeof(double);      // kvec
  bytes += (double) nkvec * sizeof(int);             // iksq
  bytes += (double) nRows * nCols * sizeof(double);  // array
  bytes += (double) nk * sizeof(double);             // k
  bytes += (double) nk * sizeof(double);             // skdeg
  bytes += (double) 2 * ntypes * sizeof(double);     // b (scattering lengths + flags)
  
  return bytes;
}



