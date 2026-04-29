// clang-format off /* ---------------------------------------------------------------------- LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator https://www.lammps.org/, Sandia National Laboratories LAMMPS development team: developers@lammps.org Copyright (2003) Sandia Corporation.  Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains certain rights in this software.  This software is distributed under the GNU General Public License.  See the README file in the top-level LAMMPS directory.  ------------------------------------------------------------------------- */ /* ---------------------------------------------------------------------- Contributing authors: Jonathan Coldstream (Edinburgh), based off code from Shawn Coleman & Douglas Spearot (Arkansas) ------------------------------------------------------------------------- */ 

#include "compute_sans.h"

#include "atom.h" 
#include "citeme.h" 
#include "comm.h" 
#include "compute_sans_consts.h" 
#include "domain.h" 
#include "error.h" 
#include "group.h" 
#include "math_const.h"
#include "memory.h"
#include "update.h"

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
  k(nullptr), sktotal(nullptr)
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
  if (narg < 4+ntypes)
    error->all(FLERR,"Illegal Compute SANS Command");
  if (triclinic == 1)
    error->all(FLERR,"Compute SANS does not work with triclinic structures");

  array_flag = 1;
  extarray = 0;

  /// MY COMMENTS ///
  // Need to get from input, kmax, delta mod(q), ewald sphere thickness, scattering lengths

  // gets kmax
  kmax = utils::numeric(FLERR,arg[3],false,lmp);
  if (kmax < 0)
    error->all(FLERR,"Compute SANS: kmax must be greater than zero");

  // Define atom types for atomic scattering factor coefficients
  // first arg after required
  int iarg = 4;
  ztype = new int[ntypes];
  for (int i = 0; i < ntypes; i++) {
    ztype[i] = SANSmaxType + 1;
  }
  // checks to see if the type in the argument is the same as any in the saved SANStypeList, and sets the atom type in the simulation equal to the one in the SANStypeList for future reference.
  for (int i=0; i<ntypes; i++) {
     for (int j = 0; j < SANSmaxType; j++) {
       if (utils::lowercase(arg[iarg]) == utils::lowercase(SANStypeList[j])) {
         ztype[i] = j;
       }
     }
     // if index goes above number of saved types then it isn't included and throws an error.
     if (ztype[i] == SANSmaxType + 1)
       error->all(FLERR,"Compute SANS: Invalid ASF atom type");
    iarg++;
  }

  utils::logmesg(lmp,"READ INPUT VALUES");
  
  // Set defaults for optional args
  qmax = 2;
  qmin = -1;
  Nq = 50;
  dR_Ewald = 0.0;
  logdist = 0;


  // Process optional args
  while (iarg < narg) {

    if (strcmp(arg[iarg],"qmin") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal Compute SANS Command");
      qmin = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg],"qmax") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal Compute SANS Command");
      qmax = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (qmax < qmin)
        error->all(FLERR,"Compute SANS: qmax must be greater than qmin");
      iarg += 2;

    } else if (strcmp(arg[iarg],"Nq") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal Compute SAED Command");
      Nq = utils::numeric(FLERR,arg[iarg+1],false,lmp);
      if (Nq < 0)
        error->all(FLERR,"number of wavevectors to calculate must be greater than 0");
      iarg += 2;

    } else if (strcmp(arg[iarg],"dR_Ewald") == 0) {
      if (iarg+2 > narg) error->all(FLERR,"Illegal Compute SAED Command");
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

    } else error->all(FLERR,"Illegal Compute SANS Command");
  }



  utils::logmesg(lmp,"-----\nComputing SANS things\n");

  // check total number of wavevectors to calculate, discarding duplicate values

  // nsamples = 0;

  // allocate memory for q and results
  memory->create(q, Nq,"sans:q");
  memory->create(skdeg, Nq,"sans:skdeg");
  memory->create(sktotal,Nq,"sans:sktotal");

  const double* boxlo = domain->boxlo;
  const double* boxhi = domain->boxhi;
  auto boxdim = new double [3];

  // calculate box lengths
  for (int i = 0; i < 3; i++){
    boxdim[i] = boxhi[i] - boxlo[i];
  }

  double twopi_L = 2.0*mypi/boxdim[0];

  if (logdist) {
    double logqmin = log10(qmin);
    double logqmax = log10(qmax);
    for (int i = 0; i < Nq; i++){
      q[i] = pow(10, (logqmax-logqmin)*i/Nq + logqmin);
    }
  } else {
    for (int i = 0; i < Nq; i++){
      q[i] = (qmax-qmin)*i/Nq + qmin;
      }
    }
  
    if (dR_Ewald > q[1]-q[0]){
      utils::logmesg(lmp, "q1-q0 = {}, dR_Ewald = {}\n", q[1]-q[0], dR_Ewald);
      error->all(FLERR,"Compute SANS: dR_Ewald must be smaller than the smallest difference between q values");
    }

  auto tempskdeg = new int[Nq];
  for (int i = 0; i < Nq; i++) {
    tempskdeg[i] = 0;
  }

  int tempnk = 0;
  int tempksq;
  double tempmodk;
  // calculate the number of vectors to allocate arrays
  for (int iq = 0; iq < Nq; iq++){
     for (int ix = 0; ix <= kmax; ix++) {
      for (int iy = -kmax; iy <= kmax; iy++) {
        for (int iz = -kmax; iz <= kmax; iz++) {
            tempksq = ix*ix + iy*iy + iz*iz;
            tempmodk = twopi_L * sqrt((double)tempksq);
            if (fabs(tempmodk - q[iq]) < dR_Ewald/2) {
              tempskdeg[iq] = tempskdeg[iq] + 1;
            }
          }
        }
      }
      if (tempskdeg[iq] > maxdeg) {
        tempnk = tempnk + maxdeg;
        tempskdeg[iq] = maxdeg;
      } else {
        tempnk = tempnk + tempskdeg[iq];
      }
    }
  
  for (int i = 0; i < Nq; i++){
    skdeg[i]=tempskdeg[i];
  }

  delete[] boxdim;
  delete[] tempskdeg;

  
  // nk = 0;
  // for (int ix = 0; ix <= kmax; ix++) {
  //   for (int iy = -kmax; iy <= kmax; iy++) {
  //     for (int iz = -kmax; iz <= kmax; iz++) {
  //       if (abs(ix)+abs(iy)+abs(iz) != 0) {
  //         nk++;
  //       }
  //     }
  //   }
  // }
  int myrank;
  MPI_Comm_rank(world, &myrank);
  utils::logmesg(lmp, "DEBUG :: PROCESS NAME: {}\n", myrank);
  utils::logmesg(lmp,"DEBUG :: starting wavevectors\n");
  

  utils::logmesg(lmp,"DEBUG :: tempnk = {}\n", tempnk);  
  utils::logmesg(lmp,"DEBUG :: kmax = {}\n", kmax);
  utils::logmesg(lmp,"DEBUG :: maxdeg = {}\n", maxdeg);


  // allocate memory 4 fat arrays //
  ncombinations = tempnk;

  int nRows = Nq; // = 300
  int nCols = 2;

  size_array_rows = nRows;
  size_array_cols = nCols;

  utils::logmesg(lmp,"DEBUG :: ncombinations = {}\n", ncombinations);
  utils::logmesg(lmp,"DEBUG :: nCols = {}\n", nCols);
  utils::logmesg(lmp,"DEBUG :: nRows = {}\n", nRows);

  ///// CHECK THE WAVEVECTORS /////
  utils::logmesg(lmp,"DEBUG :: number of wavevectors calculated\n"); 

  memory->create(k,3*ncombinations,"sans:k");
  memory->create(iksq, ncombinations,"sans:iksq");
  memory->create(array, nRows, nCols, "sans:array");

  
  //sans_var[0] = logqmax;
  //sans_var[1] = logqmin;
  //sans_var[2] = Nq;
  //sans_var[3] = kmax;

}

/* ---------------------------------------------------------------------- */

ComputeSANS::~ComputeSANS()
{

  memory->destroy(k);
  memory->destroy(sktotal);
  memory->destroy(array);
  //memory->destroy(store_tmp);
  delete[] ztype;
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

  int initnk;
  int tempksq;
  double tempmodk;
  std::vector<std::vector<int>> tempk;
  // calculate the number of vectors to allocate arrays
  initnk = 0;
  for (int iq = 0; iq < Nq; iq++){
     for (int ix = 0; ix <= kmax; ix++) {
      for (int iy = -kmax; iy <= kmax; iy++) {
        for (int iz = -kmax; iz <= kmax; iz++) {
            tempksq = ix*ix + iy*iy + iz*iz;
            tempmodk = twopi_L * sqrt((double)tempksq);
            if (fabs(tempmodk - q[iq]) < dR_Ewald/2) {
              tempk.push_back({ix, iy, iz});
            }
          }
        }
      }
      if (skdeg[iq] == maxdeg) {
        std::shuffle(tempk.begin(), tempk.end(), std::default_random_engine{});
        tempk.resize(maxdeg);
          for (int j = 0; j < skdeg[iq]; j++) {
            k[3*initnk+0] = twopi_L*tempk[j][0];
            k[3*initnk+1] = twopi_L*tempk[j][1];
            k[3*initnk+2] = twopi_L*tempk[j][2];
            iksq[initnk] = iq;
            initnk++;
          }
        } else if (skdeg[iq] > 0) {

          for (int j = 0; j < skdeg[iq]; j++) {
            k[3*initnk+0] = twopi_L*tempk[j][0];
            k[3*initnk+1] = twopi_L*tempk[j][1];
            k[3*initnk+2] = twopi_L*tempk[j][2];
            iksq[initnk] = iq;
            initnk++;
          }
      }
      tempk.clear();
    }

    if (initnk != ncombinations) {
      utils::logmesg(lmp,"DEBUG :: initnk = {}, ncombinations = {}\n", initnk, ncombinations);
      error->all(FLERR,"ComputeSANS: Number of wavevectors is inconsistent. Contact the developers.");
    }
  // for (int i=0; i<Nq; i++) {
  //   utils::logmesg(lmp,"DEBUG :: q = {}, skdeg[{}] = {}\n", q[i], i, skdeg[i]);
  // }

  // old wavevector setup

  // // setup wavevectors
  // int nk = 0;
  // for (int ix = 0; ix <= kmax; ix++) {
  //   for (int iy = -kmax; iy <= kmax; iy++) {
  //     for (int iz = -kmax; iz <= kmax; iz++) {
  //       if (abs(ix)+abs(iy)+abs(iz) != 0) {
  //         //utils::logmesg(lmp,"debug :: nk = {} \n", nk); 
  //         //utils::logmesg(lmp,"debug :: ix = {}, iy = {}, iz = {}\n", ix, iy, iz); 
  //         //utils::logmesg(lmp,"DEBUG :: ix = {}, iy = {}, iz = {}\n", ix, iy, iz); 
  //         k[3*nk+0] = twopi_L*ix;
  //         k[3*nk+1] = twopi_L*iy;
  //         k[3*nk+2] = twopi_L*iz;
  //         //utils::logmesg(lmp,"DEBUG :: kx = {}, ky = {}, kz = {}\n", k[3*nk+0], k[3*nk+1], k[3*nk+2]); 
  //         //utils::logmesg(lmp,"DEBUG :: kx = {}, ky = {}, kz = {}\n", k[3*nk+0], k[3*nk+1], k[3*nk+2]); 
  //         ksq[nk] = ix*ix + iy*iy + iz*iz;
  //         //utils::logmesg(lmp,"DEBUG :: ksq = {}\n", ksq[nk]); 
  //         skdeg[ksq[nk]] = skdeg[ksq[nk]] + 1.0;
  //         modk[ksq[nk]] = twopi_L*sqrt(1.0*ksq[nk]);

  //          // if (ix==0 || iy ==0 || iz==0){
  //          //   utils::logmesg(lmp,"DEBUG :: ix = {}, iy = {}, iz = {}\n", ix, iy, iz); 
  //          //   utils::logmesg(lmp,"DEBUG :: kx = {}, ky = {}, kz = {}\n", k[3*nk+0], k[3*nk+1], k[3*nk+2]); 
  //          //   utils::logmesg(lmp,"DEBUG :: ksq = {}\n", ksq[nk]); 
  //          // }

  //         nk++;
  //       }
  //     }
  //   }
  // }


// utils::logmesg(lmp,"DEBUG :: proc nk = {}\n", nk);  
// utils::logmesg(lmp,"DEBUG :: proc wavevectors calculated\n");

//  ink = 0;
//
//  for (int ix = 0; ix <= kmax; ix++) {
//    for (int iy = -kmax; iy <= kmax; iy++) {
//      for (int iz = -kmax; iz <= kmax; iz++) {
//        if (abs(ix)+abs(iy)+abs(iz) != 0) {
//           utils::logmesg(lmp,"DEBUG :: kx = {}, ky = {}, kz = {}\n",k[3*ink+0],k[3*ink+1],k[ink+2]); 
//           ink++;
//        }
//      }
//    }
//  }


//utils::logmesg(lmp,"-----\nComputing :{} vectors, # of atoms:{}, # for sans\n", 
//    nk,natoms);

}


void ComputeSANS::compute_array()
{
  invoked_array = update->ntimestep;

  // if (me == 0 && echo)
  //   utils::logmesg(lmp,"-----\nComputing SANS intensities\n");

  double t0 = platform::walltime();

  // Initialize sktotal for this compute step
  for (int i = 0; i < Nq; i++) {
    sktotal[i] = 0.0;
  }

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
  // utils::logmesg(lmp,"DEBUG :: boxdim calc done\n");

  //if ((boxdim[0] != boxdim[1]) || (boxdim[1] != boxdim[2]))
  //  utils::logmesg(lmp,"dim 1 = {}, dim2 = {}\n",boxdim[0]-boxdim[1],boxdim[1]-boxdim[2]);
  //  error->all(FLERR,"COMPUTE SANS only suitable for cubic boxes");

  // utils::logmesg(lmp,"DEBUG :: nlocal = {}\n", nlocal);


  // checks to see if atoms are included in group for compute
  nlocalgroup = 0;
  for (int ii = 0; ii < nlocal; ii++) {
    if (mask[ii] & groupbit) {
     nlocalgroup++;
    }
  }

  // utils::logmesg(lmp,"DEBUG :: nlocalgroup = {}\n", nlocalgroup);

  // utils::logmesg(lmp,"DEBUG :: nlocal calc\n"); 

  // positions and types for local atoms
  auto xlocal = new double [3*nlocalgroup];
  int *typelocal = new int [nlocalgroup];

  // populate positions and types
  nlocalgroup = 0;
  for (int ii = 0; ii < nlocal; ii++) {
    if (mask[ii] & groupbit) {
     xlocal[3*nlocalgroup+0] = atom->x[ii][0];
     xlocal[3*nlocalgroup+1] = atom->x[ii][1];
     xlocal[3*nlocalgroup+2] = atom->x[ii][2];
     typelocal[nlocalgroup]=type[ii];
     nlocalgroup++;
    }
  }
  // utils::logmesg(lmp,"DEBUG :: nk = {}\n", nk); 


//if (me == 0 && echo) utils::logmesg(lmp,"\n");

  // Create separate arrays for cos and sin components
  // We accumulate these separately, then square AFTER MPI_Allreduce (not before you numpty)
  auto cossinsum_ksq = new double[2*Nq];
  for (int i = 0; i < 2*Nq; i++) {
    cossinsum_ksq[i] = 0.0;
  }

  // vars for scattering
  double kx, ky, kz;
  double cossum, sinsum;
  double kdotr;

for (int ik = 0; ik < ncombinations; ik++){
  // set up wavevectors, check to see if reassigning these slows performance
  kx = k[3*ik+0];
  ky = k[3*ik+1];
  kz = k[3*ik+2];
  // utils::logmesg(lmp,"DEBUG :: kx = {}, ky = {}, kz = {}, ksq = {}\n", kx, ky, kz, iksq[ik]); 
  // utils::logmesg(lmp,"DEBUG :: Nq = {}\n", Nq); 
  cossum=0.0;
  sinsum=0.0;
  // compute the dot product
    for (int ii=0; ii < nlocalgroup; ii++) {
      kdotr = (kx*xlocal[3*ii+0] + ky*xlocal[3*ii+1] + kz*xlocal[3*ii+2]);

      // if (iksq[ik] == 3){
      //   utils::logmesg(lmp,"DEBUG :: kx = {}, ky = {}, kz = {}\n", kx, ky, kz); 
      //   utils::logmesg(lmp,"DEBUG :: x = {}, y = {}, z = {}\n", xlocal[3*ii+0], xlocal[3*ii+1], xlocal[3*ii+2]); 
      //   utils::logmesg(lmp,"DEBUG :: kx*x = {}, ky*y = {}, kz*z = {}\n", kx*xlocal[3*ii+0], ky*xlocal[3*ii+1], kz*xlocal[3*ii+2]);
      //   utils::logmesg(lmp,"DEBUG :: kdotr = {}\n", kdotr);
      //   utils::logmesg(lmp,"DEBUG :: cos(kdotr) = {}, sin(kdotr) = {}\n", cos(kdotr), sin(kdotr));
      // }

      // unweighted calculation, multiply by b for neutron scattering
      cossum += cos(kdotr);
      sinsum += sin(kdotr);
    }

    // Accumulate cos and sin components separately (not squared yet)
    // This will be reduced across all MPI ranks, then squared after reduction
    cossinsum_ksq[2*iksq[ik]+0] += cossum;
    cossinsum_ksq[2*iksq[ik]+1] += sinsum;
}

  // utils::logmesg(lmp,"DEBUG :: dot prod calc done\n"); 

  // utils::logmesg(lmp,"DEBUG :: MPI_Allreduce starting\n"); 

  // Reduce cos and sin components separately across all MPI ranks
  auto cossinsum_total = new double[2*Nq];

  for (int i = 0; i < 2*Nq; i++) {
    cossinsum_total[i] = 0.0;
  }

  MPI_Allreduce(cossinsum_ksq, cossinsum_total, 2*Nq, MPI_DOUBLE, MPI_SUM, world);
  
  // Now compute intensity from reduced cos/sin components
  // All ranks will compute the same result from the same global sums
  for (int i = 0; i < Nq; i++) {
    sktotal[i] = cossinsum_total[2*i+0]*cossinsum_total[2*i+0] + cossinsum_total[2*i+1]*cossinsum_total[2*i+1];
  }
  
  // utils::logmesg(lmp,"DEBUG :: MPI_Allreduce done\n"); 
  
  // utils::logmesg(lmp,"DEBUG :: ksqmax = {}\n", ksqmax);

  // utils::logmesg(lmp,"DEBUG :: WRITING FINAL ARRAY\n");

  // normalise the output
  for (int i = 0; i < Nq; i++){
    array[i][0] = q[i];
    array[i][1] = sktotal[i]/skdeg[i]/natoms;
    // utils::logmesg(lmp,"iksq = {}, sktotal = {}\n",i, sktotal[i]);
    // utils::logmesg(lmp,"modk = {}, skdeg = {}\n", modk[i], skdeg[i]);
    // utils::logmesg(lmp,"result = {}\n", sktotal[i]/skdeg[i]/natoms);
  }

  // utils::logmesg(lmp,"DEBUG :: normalisation done\n"); 

  delete[] xlocal;
  delete[] typelocal;
  delete[] cossinsum_ksq;
  delete[] cossinsum_total;
  //delete[] boxdim;

  // utils::logmesg(lmp,"DEBUG :: delete memory done\n"); 

  double t1 = platform::walltime();

  if (me == 0) {
    utils::logmesg(lmp,"Time for SANS calculation: {} seconds\n", t1-t0);
  }

}

/* ----------------------------------------------------------------------
 memory usage of arrays
 ------------------------------------------------------------------------- */

// double ComputeSANS::memory_usage()
// {
  // double bytes = 0.0;

  // return bytes;
// }

