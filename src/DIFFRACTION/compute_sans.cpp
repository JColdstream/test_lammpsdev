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

ComputeSANS::ComputeSANS(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg),
  k(nullptr)
{

  if (lmp->citeme) lmp->citeme->add(cite_compute_saed_c);

  int ntypes = atom->ntypes;
  int natoms = group->count(igroup);
  int dimension = domain->dimension;
  int triclinic = domain->triclinic;
  me = comm->me;

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
  // utils::logmesg(lmp,"READ INPUT VALUES");
  
  // Set defaults for optional args
  kmax = 30;
  kmin = 1;
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


  // // Get all npt fixes style
  // auto npt_fixes = modify->get_fix_by_style("npt");
  // if (!npt_fixes.empty()) {
  //   error->warning(FLERR, "NPT barostat in use. If your box size changes significantly during the simulation, the results will be rubbish.");
  // }

  // utils::logmesg(lmp, "DEBUG :: Reading scattering lengths \n");
  // utils::logmesg(lmp, "DEBUG :: scatteringlengths = {} \n", scatteringlengths);

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
      utils::logmesg(lmp, "DEBUG :: Reading scattering lengths from {}\n", filename);

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

        // utils::logmesg(lmp, "DEBUG :: typeindex = {}\n", typeindex);
        // utils::logmesg(lmp, "DEBUG :: ntypes = {}\n", ntypes);
        // utils::logmesg(lmp, "DEBUG :: b[{}] = {}\n", typeindex, b[typeindex]);
        // utils::logmesg(lmp, "DEBUG :: bflag[{}] = {}\n", ntypes+typeindex, b[ntypes+typeindex]);
      } 


    if (fp) fclose(fp);

    // if we don't have custom scattering lengths set all to 1.0.
    } else {
      for (int i = 0; i < 2*ntypes; i++) {
        b[i] = 1.0;
      }
    }

  utils::logmesg(lmp,"-----\nComputing SANS things\n");

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

  double twopi_L = 2.0*mypi/boxdim[0];

  // tempk to count initial wavevectors
  // we will populate the final k vector later, excluding invalid values of k
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
  
    // if (dR_Ewald > tempk[1]-tempk[0]){
    //   utils::logmesg(lmp, "k1-k0 = {}, dR_Ewald = {}\n", tempk[1]-tempk[0], dR_Ewald);
    //   error->all(FLERR,"Compute SANS: dR_Ewald must be smaller than the smallest difference between k values");
    // }

  auto tempskdeg = new int[nk];
  for (int i = 0; i < nk; i++) {
    tempskdeg[i] = 0;
  }

  int tempnkvec = 0;
  int nullcount = 0;
  int tempksq;
  double tempmodk;
  // for scaling dR_Ewald
  double logkmin = log10(kmin);
  double logkmax = log10(kmax);
  double scale;
  double start_dR_Ewald = dR_Ewald;
  // calculate the number of vectors to allocate arrays
  for (int ik = 0; ik < nk; ik++){
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
      // count the number of values of q that don't have any valid (kx, ky, kz) combinations
      // the number of forbidden q values will depend on the input parameters, as well as the box geometry
      if (tempskdeg[ik] == 0) {
        nullcount++;
      } else if (tempskdeg[ik] > maxdeg) {
        tempnkvec = tempnkvec + maxdeg;
        tempskdeg[ik] = maxdeg;
      } else {
        tempnkvec = tempnkvec + tempskdeg[ik];
      }
      utils::logmesg(lmp, "DEBUG :: ik = {}, skdeg = {}\n", ik, tempskdeg[ik]);
    }
  
  memory->create(k, nk-nullcount,"sans:k");
  memory->create(skdeg, nk-nullcount,"sans:skdeg");
  
  int kcount = 0;
  for (int i = 0; i < nk; i++){
    if (tempskdeg[i] > 0) {
      k[kcount] = tempk[i];
      skdeg[kcount]=tempskdeg[i];
      kcount++;
    }
  }

  // update number of wavevectors to avoid NULL values
  nk = nk - nullcount;

  if (nk != kcount) {
    error->all(FLERR, "Compute SANS: Inconsistent number of wavevectors");
  }

  delete[] boxdim;
  delete[] tempk;
  delete[] tempskdeg;


  
  int myrank;
  MPI_Comm_rank(world, &myrank);
  //utils::logmesg(lmp, "DEBUG :: PROCESS NAME: {}\n", myrank);
  //utils::logmesg(lmp,"DEBUG :: starting wavevectors\n");
  

  //utils::logmesg(lmp,"DEBUG :: tempnkvec = {}\n", tempnkvec);  
  //utils::logmesg(lmp,"DEBUG :: kmax = {}\n", kmax);
  //utils::logmesg(lmp,"DEBUG :: maxdeg = {}\n", maxdeg);


  nkvec = tempnkvec;

  int nRows = nk;
  int nCols = 2;

  size_array_rows = nRows;
  size_array_cols = nCols;

  // utils::logmesg(lmp,"DEBUG :: nkvec = {}\n", nkvec);
  // utils::logmesg(lmp,"DEBUG :: nCols = {}\n", nCols);
  // utils::logmesg(lmp,"DEBUG :: nRows = {}\n", nRows);

  ///// CHECK THE WAVEVECTORS /////
  // utils::logmesg(lmp,"DEBUG :: number of wavevectors calculated\n"); 

  // allocate memory 4 fat arrays //
  memory->create(kvec,3*nkvec,"sans:kvec");
  memory->create(iksq, nkvec,"sans:iksq");
  memory->create(array, nRows, nCols, "sans:array");

  for (int i = 0; i < nRows; i++) {
    for (int j = 0; j < nCols; j++) {
      array[i][j] = 0.0;
    }
  }
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
  //memory->destroy(store_tmp);
}

/* ---------------------------------------------------------------------- */

void ComputeSANS::init()
{

  const double* boxlo = domain->boxlo;
  const double* boxhi = domain->boxhi;
  auto boxdim = new double [3];

  // calculate box lengths
  for (int i = 0; i < 3; i++){
    boxdim[i] = boxhi[i] - boxlo[i];
  }

  double twopi_L = 2.0*mypi/boxdim[0];

  // utils::logmesg(lmp,"DEBUG :: kmax = {}\n", kmax); 

  int initnkvec;
  int tempksq;
  double tempmodk;
  // declare incase we need for logdist
  double logkmin = log10(kmin);
  double logkmax = log10(kmax);
  double scale;
  double start_dR_Ewald = dR_Ewald;
  std::vector<std::vector<int>> tempkvec;
  // calculate the number of vectors to allocate arrays
  initnkvec = 0;
  for (int ik = 0; ik < nk; ik++){
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
      if (skdeg[ik] == maxdeg) {
        std::shuffle(tempkvec.begin(), tempkvec.end(), std::default_random_engine{});
        tempkvec.resize(maxdeg);
          for (int j = 0; j < skdeg[ik]; j++) {
            kvec[3*initnkvec+0] = twopi_L*tempkvec[j][0];
            kvec[3*initnkvec+1] = twopi_L*tempkvec[j][1];
            kvec[3*initnkvec+2] = twopi_L*tempkvec[j][2];
            iksq[initnkvec] = ik;
            initnkvec++;
          }
        } else if (skdeg[ik] > 0) {

          for (int j = 0; j < skdeg[ik]; j++) {
            kvec[3*initnkvec+0] = twopi_L*tempkvec[j][0];
            kvec[3*initnkvec+1] = twopi_L*tempkvec[j][1];
            kvec[3*initnkvec+2] = twopi_L*tempkvec[j][2];
            iksq[initnkvec] = ik;
            initnkvec++;
          }
      }
      tempkvec.clear();
      utils::logmesg(lmp, "DEBUG :: ik = {}, skdeg = {}\n", scale, skdeg[ik]);
    }

    if (initnkvec != nkvec) {
      // utils::logmesg(lmp,"DEBUG :: initnkvec = {}, nkvec = {}\n", initnkvec, nkvec);
      error->all(FLERR,"ComputeSANS: Number of wavevectors is inconsistent. Contact the developers.");
    }

  const auto natoms = group->count(igroup);

  // calculate sum of all scattering lengths to normalise at the end
  // equal to number of atoms if lengths are not set by the user
  if (scatteringlengths) {
    double proc_scatteringsum = 0.0;
    ntypes = atom->ntypes;
    const auto nlocal = atom->nlocal;
    const auto *type  = atom->type;
    const auto *mask = atom->mask;

    // checks to see if atoms are included in group for compute
    // types are -1 so they correspond to the correct index
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
    // utils::logmesg(lmp,"DEBUG :: scatteringsum = {}\n", scatteringsum);
  } else {
    scatteringsum = natoms;
  }
}


void ComputeSANS::compute_array()
{
  invoked_array = update->ntimestep;

  // if (me == 0 && echo)
  //   utils::logmesg(lmp,"-----\nComputing SANS intensities\n");

  double t0 = platform::walltime();

  ntypes = atom->ntypes;
  const auto nlocal = atom->nlocal;
  const auto *type  = atom->type;
  const auto natoms = group->count(igroup);
  const auto *mask = atom->mask;

  const double* boxlo = domain->boxlo;
  const double* boxhi = domain->boxhi;
  auto boxdim = new double [3];

  // calculate box lengths
  for (int i = 0; i < 3; i++){
    boxdim[i] = boxhi[i] - boxlo[i];
  }

  // checks to see if atoms are included in group for compute
  nlocalgroup = 0;
  for (int ii = 0; ii < nlocal; ii++) {
    if (mask[ii] & groupbit) {
     nlocalgroup++;
    }
  }

  // positions and types for local atoms
  auto xlocal = new double [3*nlocalgroup];
  //auto *blocal = new double [nlocalgroup];

  // populate positions and types
  nlocalgroup = 0;
  for (int ii = 0; ii < nlocal; ii++) {
    if (mask[ii] & groupbit) {
     xlocal[3*nlocalgroup+0] = atom->x[ii][0];
     xlocal[3*nlocalgroup+1] = atom->x[ii][1];
     xlocal[3*nlocalgroup+2] = atom->x[ii][2];
     //blocal[nlocalgroup]=b[type[ii]-1];
     nlocalgroup++;
    }
  }
  // utils::logmesg(lmp,"DEBUG :: nk = {}\n", nk); 


//if (me == 0 && echo) utils::logmesg(lmp,"\n");

  // array for accumulating cos/sin components
  // cos elements are in (2*i) and sin are in (2*i)+1
  auto cossinsum_ksq = new double[2*nk];
  for (int i = 0; i < 2*nk; i++) {
    cossinsum_ksq[i] = 0.0;
  }

  // vars for scattering
  double kx, ky, kz;
  double cossum, sinsum;
  double kdotr;
  double templength;

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

      cossum += b[type[ii]-1]*cos(kdotr);
      sinsum += b[type[ii]-1]*sin(kdotr);

      // unweighted calculation
      // cossum += cos(kdotr);
      // sinsum += sin(kdotr);
    }

    cossinsum_ksq[2*iksq[ik]+0] += cossum;
    cossinsum_ksq[2*iksq[ik]+1] += sinsum;
}

  // sum up cos/sin sums across processes
  auto cossinsum_total = new double[2*nk];
  MPI_Allreduce(cossinsum_ksq, cossinsum_total, 2*nk, MPI_DOUBLE, MPI_SUM, world);
  
  for (int i = 0; i < nk; i++) {
    array[i][0] = k[i];
    array[i][1] = (cossinsum_total[2*i+0]*cossinsum_total[2*i+0]+cossinsum_total[2*i+1]*cossinsum_total[2*i+1])*natoms/skdeg[i]/scatteringsum/scatteringsum;
  }

  // // normalise the output and assign to array
  // for (int i = 0; i < Nq; i++){
  //   array[i][0] = q[i];
  //   array[i][1] = sktotal[i]/skdeg[i]/scatteringsum;
  // }

  // free local memory
  delete[] xlocal;
  //delete[] blocal;
  delete[] cossinsum_ksq;
  delete[] cossinsum_total;
  // delete[] boxdim;

  double t1 = platform::walltime();

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



