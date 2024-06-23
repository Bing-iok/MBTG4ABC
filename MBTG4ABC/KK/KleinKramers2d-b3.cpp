// ==============================================================================
//
//  KleinKramers2d.cpp
//  QTR
//
//  Created by Albert Lu on 8/31/19.
//  alu@tacc.utexas.edu
//
//  Last modified on 2/13/24
//
//  Note:
//
// ==============================================================================

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <omp.h>
#include <vector>
#include <parallel/algorithm>
#include <new>

#include "Constants.h"
#include "Containers.h"
#include "Error.h"
#include "Log.h"
#include "Parameters.h"
#include "KleinKramers2d.h"

using namespace QTR_NS;
using namespace std;
using std::vector;
using std::cout;
using std::endl;
using std::nothrow;

#define BIG_NUMBER 2147483647

#define SMALL_B 3.0

#define KK2DPOT_DW2

/* ------------------------------------------------------------------------------- */

// DEFINE POTENTIAL TYPE

#if defined KK2DPOT_DW1

#define WAVEFUNCTION(x1,x2) Wavefunction_DW1(x1,x2)
#define POTENTIAL(x1,x2) Potential_DW1(x1,x2)
#define POTENTIAL_X(x1,x2) Vx_DW1(x1,x2)
#define POTNAME "DoubleWell-1"

#elif defined KK2DPOT_DW2

#define WAVEFUNCTION(x1,x2) Wavefunction_DW2(x1,x2)
#define POTENTIAL(x1,x2) Potential_DW2(x1,x2)
#define POTENTIAL_X(x1,x2) Vx_DW2(x1,x2)
#define POTNAME "DoubleWell-2"

#elif defined KK2DPOT_DW3

#define WAVEFUNCTION(x1,x2) Wavefunction_DW3(x1,x2)
#define POTENTIAL(x1,x2) Potential_DW3(x1,x2)
#define POTENTIAL_X(x1,x2) Vx_DW3(x1,x2)
#define POTNAME "MSW-1"

#else // Default KK2DPOT_DW2

#define WAVEFUNCTION(x1,x2) Wavefunction_DW2(x1,x2)
#define POTENTIAL(x1,x2) Potential_DW2(x1,x2)
#define POTENTIAL_X(x1,x2) Vx_DW2(x1,x2)
#define POTNAME "DoubleWell-2"

#endif
/* ------------------------------------------------------------------------------- */

KleinKramers2d::KleinKramers2d(class QTR *q)
{
    qtr = q;
    err = qtr->error;
    log = qtr->log;
    parameters = qtr->parameters;
    init();
} 
/* ------------------------------------------------------------------------------- */

KleinKramers2d::~KleinKramers2d()
{     
    return;
}
/* ------------------------------------------------------------------------------- */

void KleinKramers2d::init()
{
    log->log("\n\n[KleinKramers2d] INIT starts ...\n");
    log->log("\n\n[KleinKramers2d] Potential type: %s\n", POTNAME);

    // General parameters
    I = {0,1}; // sqrt(-1)
    PI_INV = 1.0 / PI; // 1/PI
    xZERO = {0,0}; // complex zero
    DIMENSIONS = parameters->scxd_dimensions;
    EDGE = parameters->scxd_edge;
    PERIOD = parameters->scxd_period;
    SORT_PERIOD = parameters->scxd_sortperiod;
    PRINT_PERIOD = parameters->scxd_printperiod;
    PRINT_WAVEFUNC_PERIOD = parameters->scxd_printwavefuncperiod;
    TIME = parameters->scxd_Tf;
    QUIET = parameters->quiet;
    TIMING = parameters->timing;
    isTrans = parameters->scxd_isTrans;
    isCorr = parameters->scxd_isAcf;
    isSwitch = parameters->scxd_isSwitch;
    isPrintEdge = parameters->scxd_isPrintEdge;
    isPrintDensity = parameters->scxd_isPrintDensity;
    isPrintWavefunc = parameters->scxd_isPrintWavefunc;

    log->log("[KleinKramers2d] DIMENSIONS: %d\n", DIMENSIONS);
    log->log("[KleinKramers2d] EDGE: %d\n", EDGE);

    // Grid size
    H.resize(DIMENSIONS);
    Hi.resize(DIMENSIONS);
    Hisq.resize(DIMENSIONS);
    S.resize(DIMENSIONS);  
    kk = parameters->scxd_k;
    H[0] = parameters->scxd_h1;
    H[1] = parameters->scxd_h2;

    log->log("[KleinKramers2d] kk: %lf\n", kk);
    log->log("[KleinKramers2d] H[0]: %lf\n", H[0]);
    log->log("[KleinKramers2d] H[1]: %lf\n", H[1]);

    for (int i = 0; i < DIMENSIONS; i ++)  {
        Hi[i] = 1 / H[i];
        Hisq[i] = 1 / pow(H[i],2);
        S[i] = kk * Hisq[i];
    }

    // Domain size and # grids
    Box.resize(DIMENSIONS * 2);
    Box[0] = parameters->scxd_xi1;
    Box[1] = parameters->scxd_xf1;
    Box[2] = parameters->scxd_xi2;
    Box[3] = parameters->scxd_xf2; 
    BoxShape.resize(DIMENSIONS);

    GRIDS_TOT = 1;
    log->log("[KleinKramers2d] Number of grids = (");

    for (int i = 0; i < DIMENSIONS; i ++)  {

        BoxShape[i] = (int)std::round((Box[2 * i + 1] - Box[2 * i]) / H[i]) + 1;
        GRIDS_TOT *= BoxShape[i];

        if ( i < DIMENSIONS - 1 )
            log->log("%d, ", BoxShape[i]);
        else
            log->log("%d)\n", BoxShape[i]);
    }
    M1 = BoxShape[1];
    W1 = BoxShape[1];
    O1 = BoxShape[0] * BoxShape[1];

    // Parameters
    hb = parameters->scxd_hb;
    m  = parameters->scxd_m;
    log->log("[KleinKramers2d] hb: %lf\n", hb);
    log->log("[KleinKramers2d] m: %lf\n", m);

    kb = parameters->scxd_kb;
    temp = parameters->scxd_temp;
    gamma = parameters->scxd_gamma;
    log->log("[KleinKramers2d] kb: %lf\n", kb);
    log->log("[KleinKramers2d] temp: %lf\n", temp);
    log->log("[KleinKramers2d] gamma: %lf\n", gamma);

    HBSQ_INV = 1.0 / (hb * hb);
    PIHB_INV = 1.0 / (PI * hb);

    // Wavefunction parameters
    Wave0.resize(DIMENSIONS);
    Wave0[0] = parameters->scxd_x01;
    Wave0[1] = parameters->scxd_x02;
    log->log("[KleinKramers2d] Wave0[0]: %lf\n", Wave0[0]);
    log->log("[KleinKramers2d] Wave0[1]: %lf\n", Wave0[1]);

    A.resize(DIMENSIONS);
    A[0] = parameters->scxd_a1;
    A[1] = parameters->scxd_a2;
    log->log("[KleinKramers2d] A[0]: %lf\n", A[0]);
    log->log("[KleinKramers2d] A[1]: %lf\n", A[1]);

    // Truncate parameters
    isFullGrid = parameters->scxd_isFullGrid;
    TolH = parameters->scxd_TolH;    // Tolerance of probability density for Zero point Cutoff
    TolL = parameters->scxd_TolL;    // Tolerance of probability density for Edge point
    TolHd = parameters->scxd_TolHd;  // Tolerance of probability first diff for Zero point Cutoff
    TolLd = parameters->scxd_TolLd;  // Tolerance of probability density for Edge point
    ExReduce = parameters->scxd_ExReduce; //Extrapolation reduce factor
    ExLimit = parameters->scxd_ExLimit;   //Extrapolation counts limit
    SwitchLevel = parameters->scxd_SwitchLevel;

    log->log("[KleinKramers2d] isFullGrid: %d\n", (int)isFullGrid);

    // Transition position
    trans_x0 = parameters->scxd_trans_x0;
    idx_x0 = (int) std::round( ( trans_x0 - Box[0] ) / H[0] );

    log->log("[KleinKramers2d] TolH: %e\n", TolH);
    log->log("[KleinKramers2d] TolL: %e\n", TolL);
    log->log("[KleinKramers2d] TolHd: %e\n", TolHd);
    log->log("[KleinKramers2d] TolLd: %e\n", TolLd);
    log->log("[KleinKramers2d] ExReduce: %lf\n", ExReduce);
    log->log("[KleinKramers2d] ExLimit: %d\n", ExLimit);
    log->log("[KleinKramers2d] INIT done.\n\n");
}
/* ------------------------------------------------------------------------------- */

void KleinKramers2d::Evolve()
{
    #pragma omp declare reduction (merge : MeshIndex : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    log->log("[KleinKramers2d] Evolve starts ...\n");

    // Files
    FILE *pfile;

    // Variables 
    int index;
    int count;
    int n1, n2;
    int ta_size, tb_size;
    int x1_max, x1_min;  // TA box range
    int x2_max, x2_min;  // TA box range
    double sum;
    double coeff1 = 0.0;
    double coeff2 = 0.0;
    double norm;   // normalization factor
    double density;
    double corr;
    double corr_0;
    bool b1, b2, b3, b4, b5;

    // Timing variables
    double t_0_begin, t_1_begin, t_2_begin;    
    double t_0_end, t_1_end, t_2_end;
    double t_0_elapsed = 0.0;
    double t_1_elapsed = 0.0;
    double t_2_elapsed = 0.0;

    // Core computation time (RK4, normalization, initialization, etc)
    double t_full = 0.0;
    double t_truncate = 0.0;

    // Overhead time (truncate)
    double t_overhead = 0.0;

    // Constants
    double k2h0m = kk / (2.0 * H[0] * m);
    double k2h1 = kk / (2.0 * H[1]);
    double i2h1 = 1.0 / (2.0 * H[1]);
    double kgamma = kk * gamma;
    double mkT2h1sq = m * kb * temp / (H[1] * H[1]);
    double TolHdX1 = 2.0 * TolHd * H[0];
    double TolHdX2 = 2.0 * TolHd * H[1];
    double TolLdX1 = 2.0 * TolLd * H[0];
    double TolLdX2 = 2.0 * TolLd * H[1];

    // temporary index container
    MeshIndex tmpVec; 

    // Boundary layer container for extrapolation loop
    MeshIndex ExBD;     
 
    //  2d Grid vector and indices
    VectorXi grid;
    int g1, g2;
    double xx1, xx2;
    double f0, kk0;
    double f1p, f1m;
    double f2p, f2m;
    double kk1p, kk1m;
    double kk2p, kk2m;

    // Vector iterater
    vector<int>::iterator it;

    // Extrapolation 
    int min_dir;
    bool isFirstExtrp;
    double val, val_min;
    double val_min_abs;
    int Excount;

    // Neighborlist
    int nneigh = 0;
    vector<vector<int>> neighlist;
    vector<int> neighs(DIMENSIONS);

    // PF_trans
    double pftrans;
    vector<double> PF_trans;
    PF_trans.push_back(0.0);

    log->log("[KleinKramers2d] Initializing containers ...\n");

    // Initialize containers

    t_0_begin = omp_get_wtime();

    bool *TAMask;

    if ( !isFullGrid ) 
        TAMask = new bool[O1];
    
    double *F = new double[O1];
    double *FF = new double[O1];
    double *PF = new double[O1];
    double *KK1 = new double[O1];
    double *KK2 = new double[O1];
    double *KK3 = new double[O1];
    double *KK4 = new double[O1];

    double *F0;
    double *Ft;

    if ( isCorr )  {
        F0 = new double[BoxShape[0]];
        Ft = new double[BoxShape[0]];

        #pragma omp parallel for
        for (int i1 = 0; i1 < BoxShape[0]; i1 ++)  {
            F0[i1] = 0.0;
            Ft[i1] = 0.0;
        }
    }

    #pragma omp parallel for schedule(runtime)
    for (int i1 = 0; i1 < BoxShape[0]; i1 ++)  {
        for (int i2 = 0; i2 < BoxShape[1]; i2 ++)  {
            F[i1*W1+i2] = 0.0;
            PF[i1*W1+i2] = 0.0;
            FF[i1*W1+i2] = 0.0;
            KK1[i1*W1+i2] = 0.0;
            KK2[i1*W1+i2] = 0.0;
            KK3[i1*W1+i2] = 0.0;
            KK4[i1*W1+i2] = 0.0;
        }
    }

    if ( !isFullGrid )  {

        t_1_begin = omp_get_wtime();
        
        #pragma omp parallel for schedule(runtime)
        for (int i1 = 0; i1 < BoxShape[0]; i1 ++)  {
            for (int i2 = 0; i2 < BoxShape[1]; i2 ++)  {
                TAMask[i1*W1+i2] = 0;
            }
        }

        nneigh = 0;

        for (int d = 1; d <= 4; d ++)  {
            for (n1 = -d; n1 <= d; n1 ++)  {

                n2 = d - abs(n1);

                if (n2 != 0)  {
                    neighs = {n1,n2};
                    neighlist.push_back(neighs);
                    neighs = {n1,-n2};
                    neighlist.push_back(neighs);
                    nneigh += 2;
                }
                else  {
                    neighs = {n1,0};
                    neighlist.push_back(neighs);
                    nneigh += 1;
                }
            }
        }
        log->log("[KleinKramers2d] nneigh = %d\n", nneigh); 
        t_1_end = omp_get_wtime();
        t_1_elapsed = t_1_end - t_1_begin;
        t_overhead += t_1_elapsed;
    }
    t_0_end = omp_get_wtime();
    t_0_elapsed = t_0_end - t_0_begin;
    t_full += t_0_elapsed;

    if ( !isFullGrid )
        t_truncate += t_0_elapsed - t_1_elapsed; // subtract overhead

    if (!QUIET && TIMING) log->log("[KleinKramers2d] Elapsed time (initializing containers) = %lf sec\n\n", t_0_elapsed); 

    // .........................................................................................

    log->log("[KleinKramers2d] Initializing wavefunction ...\n");  

    t_1_begin = omp_get_wtime();

    // Initialize wavefunction

    #pragma omp parallel for schedule(runtime)
    for (int i1 = EDGE; i1 < BoxShape[0]-EDGE ; i1 ++)  {
        for (int i2 = EDGE; i2 < BoxShape[1]-EDGE ; i2 ++)  {
            F[i1*W1+i2] = WAVEFUNCTION(Box[0]+i1*H[0], Box[2]+i2*H[1]);
        }
    }

    // Normalization
    norm = 0.0;

    #pragma omp parallel for reduction (+:norm) schedule(runtime)
    for (int i1 = EDGE; i1 < BoxShape[0]-EDGE ; i1 ++)  {
        for (int i2 = EDGE; i2 < BoxShape[1]-EDGE ; i2 ++)  {
            norm += F[i1*W1+i2];
        }
    }
    norm *= H[0] * H[1];
    log->log("[KleinKramers2d] Normalization factor = %.16e\n",norm);
    norm = 1.0 / norm;
    log->log("[KleinKramers2d] Normalization factor 1/N = %.16e\n",norm);

    #pragma omp parallel for schedule(runtime)
    for (int i1 = EDGE; i1 <  BoxShape[0]-EDGE; i1 ++)  {
        for (int i2 = EDGE; i2 <  BoxShape[1]-EDGE; i2 ++)  {
            F[i1*W1+i2] = norm * F[i1*W1+i2];
            PF[i1*W1+i2] = F[i1*W1+i2];
        }
    }
    t_1_end = omp_get_wtime();
    t_1_elapsed = t_1_end - t_1_begin;
    t_full += t_1_elapsed;
    t_truncate += t_1_elapsed;
    if (!QUIET && TIMING) log->log("[KleinKramers2d] Elapsed time (initializing wavefunction) = %.4e sec\n\n", t_1_elapsed); 
    

    // Initial density
    if ( isCorr )   {
        #pragma omp parallel for private(density)
        for (int i1 = EDGE; i1 < BoxShape[0] - EDGE; i1 ++)  {
            density = 0.0;
            for (int i2 = EDGE; i2 < BoxShape[1] - EDGE; i2 ++)  {
                density += PF[i1*W1+i2]; 
            }
            F0[i1] = density * H[1];
        }

        corr_0 = 0.0;
        for (int i1 = EDGE + 1; i1 < BoxShape[0] - EDGE - 1; i1 ++)  {
            corr_0 += F0[i1] * F0[i1];
        }

        corr_0 = corr_0 * H[0];

        log->log("[KleinKramers2d] corr_0 = %.16e\n",corr_0);
        log->log("[KleinKramers2d] Time %lf, Corr = %.16e\n",0.0,1.0);
    }

    // .........................................................................................

    // Initial truncation & edge point check

    if ( !isFullGrid )
    {
        t_2_begin = omp_get_wtime();

        log->log("[KleinKramers2d] Initial truncation ...\n");

        ta_size = 0;
        t_1_begin = omp_get_wtime();

        // Truncation and TA
        #pragma omp parallel for reduction(+: ta_size) private(b1, b2, b3) schedule(runtime)
        for (int i1 = EDGE; i1 < BoxShape[0] - EDGE; i1 ++)  {
            for (int i2 = EDGE; i2 < BoxShape[1] - EDGE; i2 ++)  {

                b1 = PF[i1*W1+i2] < TolH;
                b2 = std::abs(PF[(i1+1)*W1+i2] - PF[(i1-1)*W1+i2]) < TolHdX1;
                b3 = std::abs(PF[i1*W1+(i2+1)] - PF[i1*W1+(i2-1)]) < TolHdX2;
        
                if (b1 && b2 && b3)
                    F[i1*W1+i2] = 0.0;
                else {
                    TAMask[i1*W1+i2] = 1;
                    ta_size += 1;
                }
            }
        }   
        t_1_end = omp_get_wtime();
        t_1_elapsed = t_1_end - t_1_begin;
        if (!QUIET && TIMING) log->log("[KleinKramers2d] Elapsed time (initializing truncation A-1) = %.4e sec\n\n", t_1_elapsed); 

        log->log("[KleinKramers2d] TA size = %d\n", ta_size);

        // TA box
        x1_min = BoxShape[0];
        x2_min = BoxShape[1];
        x1_max = 0;
        x2_max = 0;

        t_1_begin = omp_get_wtime();

        #pragma omp parallel for reduction(min: x1_min, x2_min) reduction(max: x1_max, x2_max) schedule(runtime)
        for (int i1 = EDGE; i1 < BoxShape[0]-EDGE; i1 ++)  {
            for (int i2 = EDGE; i2 < BoxShape[1]-EDGE; i2 ++)  {
                if ( TAMask[i1*W1+i2] )  {
                    if (i1 < x1_min)  x1_min = i1;
                    if (i1 > x1_max)  x1_max = i1;
                    if (i2 < x2_min)  x2_min = i2;
                    if (i2 > x2_max)  x2_max = i2;
                }
                else {
                    F[i1*W1+i2] = 0.0;
                }
            }
        }
        t_1_end = omp_get_wtime();
        t_1_elapsed = t_1_end - t_1_begin;
        if (!QUIET && TIMING) log->log("[KleinKramers2d] Elapsed time (initializing truncation A-2) = %.4e sec\n\n", t_1_elapsed); 
        t_1_begin = omp_get_wtime();

        // TB
        tmpVec.clear();

        #pragma omp parallel for reduction(merge: tmpVec) schedule(runtime)
        for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
            for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                if (TAMask[i1*W1+i2] && (\
                !TAMask[(i1-1)*W1+i2] || !TAMask[(i1+1)*W1+i2] || \
                !TAMask[i1*W1+(i2-1)] || !TAMask[i1*W1+(i2+1)]))
                tmpVec.push_back(i1*M1+i2);
            }
        }
        tmpVec.swap(TB);
        tmpVec.clear();
        tb_size = TB.size();
        log->log("[KleinKramers2d] TB size = %d\n", tb_size);

        t_1_end = omp_get_wtime();
        t_1_elapsed = t_1_end - t_1_begin;
        if (!QUIET && TIMING) log->log("[KleinKramers2d] Elapsed time (initializing truncation A-3) = %.4e sec\n\n", t_1_elapsed); 
        t_1_begin = omp_get_wtime();

        // `````````````````````````````````````````````````````````````````

        // TA expansion

        #pragma omp parallel for reduction(merge: tmpVec) private(g1, g2) schedule(runtime)
        for (int i = 0; i < TB.size(); i++)
        {
            g1 = (int)(TB[i] / M1);
            g2 = (int)(TB[i] % M1);

            if ( g1+1 < BoxShape[0]-EDGE-1 && !TAMask[(g1+1)*W1+g2])
                tmpVec.push_back(GridToIdx(g1+1,g2));
            if ( g1-1 > EDGE && !TAMask[(g1-1)*W1+g2])
                tmpVec.push_back(GridToIdx(g1-1,g2));

            if ( g2+1 < BoxShape[1]-EDGE-1 && !TAMask[g1*W1+(g2+1)])
                tmpVec.push_back(GridToIdx(g1,g2+1));
            if ( g2-1 > EDGE && !TAMask[g1*W1+(g2-1)])
                tmpVec.push_back(GridToIdx(g1,g2-1));
        }

        t_1_end = omp_get_wtime();
        t_1_elapsed = t_1_end - t_1_begin;
        if (!QUIET && TIMING) log->log("[KleinKramers2d] Elapsed time (initializing truncation A-4) = %.4e sec\n\n", t_1_elapsed); 
        t_1_begin = omp_get_wtime();

        // Find unique elements
        __gnu_parallel::sort(tmpVec.begin(),tmpVec.end());
        it = std::unique (tmpVec.begin(), tmpVec.end()); 
        tmpVec.resize(std::distance(tmpVec.begin(),it));

        t_1_end = omp_get_wtime();
        t_1_elapsed = t_1_end - t_1_begin;
        if (!QUIET && TIMING) log->log("[KleinKramers2d] Elapsed time (initializing truncation A-5) = %.4e sec\n\n", t_1_elapsed);
        t_1_begin = omp_get_wtime();

        #pragma omp parallel for reduction(min: x1_min,x2_min) \
                                 reduction(max: x1_max,x2_max) \
                                 private(g1,g2) schedule(runtime)
        for (int i = 0; i < tmpVec.size(); i ++)  {

            g2 = (int)(tmpVec[i] % M1);
            g1 = (int)(tmpVec[i] / M1);

            if (!TAMask[g1*W1+g2])
                TAMask[g1*W1+g2] = 1;

            // Update TA box
            x1_min = (g1 < x1_min) ? g1 : x1_min;
            x2_min = (g2 < x2_min) ? g2 : x2_min;
            x1_max = (g1 > x1_max) ? g1 : x1_max;
            x2_max = (g2 > x2_max) ? g2 : x2_max;
        }
        tmpVec.clear();

        t_1_end = omp_get_wtime();
        t_1_elapsed = t_1_end - t_1_begin;
        if (!QUIET && TIMING) log->log("[KleinKramers2d] Elapsed time (initializing truncation A-6) = %.4e sec\n\n", t_1_elapsed); 
        t_1_begin = omp_get_wtime();

        // Update ta_size

        ta_size = 0;
        #pragma omp parallel for reduction(+: ta_size) schedule(runtime)
        for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
            for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                if (TAMask[i1*W1+i2])
                    ta_size += 1;    
            }
        }
        log->log("[KleinKramers2d] TA size = %d, TB size = %d\n", ta_size, tb_size);
        log->log("[KleinKramers2d] TA Range [%d, %d][%d, %d]\n", x1_min, x1_max, x2_min, x2_max);

        t_2_end = omp_get_wtime();
        t_2_elapsed = t_2_end - t_2_begin;
        t_overhead += t_2_elapsed;

        if (!QUIET && TIMING)  {
            log->log("[KleinKramers2d] Elapsed time (initial truncation) = %lf sec\n\n", t_1_elapsed);
            log->log("[KleinKramers2d] Initialization core computation time: %lf sec\n", t_truncate);            
            log->log("[KleinKramers2d] Initialization overhead: %lf sec\n", t_overhead); 
        }
    }
    else  // Full grid approach
    {
        log->log("[KleinKramers2d] Initialization core computation time: %lf sec\n", t_full);   
    }
    // .........................................................................................

    // Time iteration 

    log->log("=======================================================\n\n"); 
    log->log("[KleinKramers2d] Time interation starts ...\n"); 
    log->log("[KleinKramers2d] Number of steps = %d\n\n", (int)(TIME / kk)); 
    log->log("=======================================================\n\n"); 

    for (int tt = 0; tt < (int)(TIME / kk); tt ++)
    {
        t_0_begin = omp_get_wtime(); 
        Excount = 0;

        if ( isPrintWavefunc && tt % PRINT_WAVEFUNC_PERIOD == 0 )
        {
            pfile = fopen("wave.dat","a");

            if ( !isFullGrid )  {
                fprintf(pfile, "%d %d\n", tt, ta_size);
                for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                    for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                        if (TAMask[i1*W1+i2])
                            fprintf(pfile, "%d %d %.8e\n", i1, i2, F[i1*W1+i2]);  
                    }
                }
            }
            else  {
                fprintf(pfile, "%d %d\n", tt, GRIDS_TOT);
                for (int i1 = 0; i1 < BoxShape[0]; i1 ++)  {
                    for (int i2 = 0; i2 < BoxShape[1]; i2 ++)  {
                        fprintf(pfile, "%d %d %.8e\n", i1, i2, F[i1*W1+i2]);
                    }
                }
            }
            fclose(pfile);
        }

        if ( tt % PRINT_PERIOD == 0 )
        {
            if ( isPrintEdge  && !isFullGrid )  {

                pfile = fopen ("edge.dat","a");
                fprintf(pfile, "%d %lf %lu\n", tt, tt * kk, TB.size());

                for (int i = 0; i < TB.size(); i++)
                {
                    g2 = (int)(TB[i] % M1);
                    g1 = (int)(TB[i] / M1);
                    xx1 = Box[0] + g1 * H[0];
                    xx2 = Box[2] + g2 * H[1];
                    fprintf(pfile, "%d %d %lf %lf\n", g1, g2, xx1, xx2);          
                }
                fclose(pfile);
            }
            if ( isPrintDensity && !isFullGrid )  {

                pfile = fopen ("density.dat","a");
                fprintf(pfile, "%d %lf %d\n", tt, tt * kk, ta_size);

                for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                    for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                        if (TAMask[i1*W1+i2])  {
                            xx1 = Box[0] + i1 * H[0];
                            xx2 = Box[2] + i2 * H[1];
                            fprintf(pfile, "%d %d %lf %lf %.16e\n", i1, i2, xx1, xx2, PF[i1*W1+i2]);
                        }
                    }
                }
                fclose(pfile);
            }
            if ( isPrintDensity && isFullGrid )  {

                pfile = fopen ("density.dat","a");
                fprintf(pfile, "%d %lf %d\n", tt, tt * kk, BoxShape[0] * BoxShape[1] );
                //fprintf(pfile, "%d %lf %d\n", tt, tt * kk, BoxShape[0] );

                for (int i1 = 0; i1 < BoxShape[0]; i1 ++)  {

                    //density = 0.0;
                    for (int i2 = 0; i2 < BoxShape[1]; i2 ++)  {
                        g1 = i1;
                        g2 = i2;
                        xx1 = Box[0] + g1 * H[0];
                        xx2 = Box[2] + g2 * H[1];
                        fprintf(pfile, "%d %d %lf %lf %.16e\n", g1, g2, xx1, xx2, PF[g1*W1+g2]);
                        //density += PF[i1*W1+i2];
                    }
                    //xx1 = Box[0] + i1 * H[0];
                    //fprintf(pfile, "%d %lf %.16e\n", i1, xx1, density);
                }
                fclose(pfile);      
            }
        }

        // Check if TB of f is higher than TolL
        
        if ( !isFullGrid )
        {
            t_1_begin = omp_get_wtime();
            t_truncate = 0.0;
            t_overhead = 0.0;

            // TBL = Index of Extrapolating Edge points.
            // TBL_P = Index history of TBL of this iteration. To Prevent from extrapolating the same points multiple times.

            TBL.clear();
            tmpVec.clear();

            #pragma omp parallel for reduction(merge: tmpVec) private(g1, g2, b1, b2, b3, b4, b5)
            for (int i = 0; i < TB.size(); i++)
            {
                g1 = (int)(TB[i] / M1);
                g2 = (int)(TB[i] % M1);
                b1 = PF[g1*W1+g2] >= TolL;
                b2 = std::abs(PF[(g1+1)*W1+g2]*TAMask[(g1+1)*W1+g2] - PF[(g1-1)*W1+g2]*TAMask[(g1-1)*W1+g2]) >= TolLdX1;
                b3 = std::abs(PF[g1*W1+(g2+1)]*TAMask[g1*W1+(g2+1)] - PF[g1*W1+(g2-1)]*TAMask[g1*W1+(g2-1)]) >= TolLdX2;
                // Not in DBi2
                b4 = g1 > EDGE && g2 > EDGE;
                b5 = g1 < BoxShape[0] - EDGE - 1 && g2 < BoxShape[1] - EDGE - 1;

                if ( (b1 || b2 || b3 ) && b4 && b5 )
                    tmpVec.push_back(TB[i]);
            }
            tmpVec.swap(TBL);
            tmpVec.clear();
            TBL_P = TBL;

            t_1_end = omp_get_wtime();
            t_1_elapsed = t_1_end - t_1_begin;
            t_overhead += t_1_elapsed;
            if (!QUIET && TIMING) log->log("Elapsed time (omp-a-1: TBL) = %lf sec\n", t_1_elapsed);   
            //if (!QUIET) log->log("TBL size = %d TBL_P size = %d\n", TBL.size(), TBL_P.size());
        }
        else  
        {
            t_full = 0.0;
        }
        isExtrapolate = false;
        isFirstExtrp = true;
        // .........................................................................................

        // CASE 1: Truncating with extrapolation

        while ( TBL.size() != 0 && !isFullGrid && Excount < ExLimit)
        {
            isExtrapolate = true;

            // Extrapolation
            // .............................................................................................

            t_1_begin = omp_get_wtime();

            // Avoid unexpected arrangement of TBL
            __gnu_parallel::sort(TBL.begin(),TBL.end());
            it = std::unique (TBL.begin(), TBL.end()); 
            TBL.resize(std::distance(TBL.begin(),it)); 

            // Find extrapolation target
            // ExFF: Index of Extrapolated points
            ExFF.clear();
            tmpVec.clear();

            #pragma omp parallel for reduction(merge: tmpVec) private(g1,g2) schedule(runtime)
            for (int i = 0; i < TBL.size(); i++)  {

                g1 = (int)(TBL[i] / M1);
                g2 = (int)(TBL[i] % M1);

                if ( g1-1 > EDGE && !TAMask[(g1-1)*W1+g2] )
                    tmpVec.push_back(GridToIdx(g1-1,g2));
                if ( g1+1 < BoxShape[0]-EDGE-1 && !TAMask[(g1+1)*W1+g2] )
                    tmpVec.push_back(GridToIdx(g1+1,g2));
                if ( g2-1 > EDGE && !TAMask[g1*W1+(g2-1)] )
                    tmpVec.push_back(GridToIdx(g1,g2-1));
                if ( g2+1 < BoxShape[1]-EDGE-1 && !TAMask[g1*W1+(g2+1)] )
                    tmpVec.push_back(GridToIdx(g1,g2+1));
            }
            tmpVec.swap(ExFF);
            tmpVec.clear();

            t_1_end = omp_get_wtime();
            t_1_elapsed = t_1_end - t_1_begin;
            if (!QUIET && TIMING) log->log("Elapsed time (omp-b-1: ExFF) = %.4e sec\n", t_1_elapsed); 
            
            if ( ExFF.size() > 0 )  {

                t_1_begin = omp_get_wtime();

                // ExFF & TBL set difference
                tmpVec.resize(ExFF.size() + TBL.size());
                __gnu_parallel::sort(TBL.begin(), TBL.end());
                __gnu_parallel::sort(ExFF.begin(), ExFF.end());
                it = std::set_difference( ExFF.begin(), ExFF.end(), TBL.begin(), TBL.end(), tmpVec.begin() );
                tmpVec.resize(it - tmpVec.begin()); 
                tmpVec.swap(ExFF);
                tmpVec.clear();

                // Find unique elements
                __gnu_parallel::sort(ExFF.begin(),ExFF.end());
                it = std::unique (ExFF.begin(), ExFF.end()); 
                ExFF.resize(std::distance(ExFF.begin(),it));

                t_1_end = omp_get_wtime();
                t_1_elapsed = t_1_end - t_1_begin;
                t_overhead += t_1_elapsed;
                if (!QUIET && TIMING) log->log("Elapsed time (omp-b-2: ExFF) = %lf sec\n", t_1_elapsed);   

                // .....................................................................

                // Find the direction of Outer to Edge points
                t_1_begin = omp_get_wtime();

                Check.clear();
                ExTBL.clear();

                for (int i = 0; i < ExFF.size(); i ++)  {
                    Check.push_back(1);
                    ExTBL.push_back(0.0);
                }
                
                #pragma omp parallel for private(g1,g2,sum,count,isEmpty,val,val_min_abs,val_min,min_dir) schedule(runtime)
                for (int i = 0; i < ExFF.size(); i ++)
                {
                    g1 = (int)(ExFF[i] / M1);
                    g2 = (int)(ExFF[i] % M1);
                    sum = 0.0;
                    count = 0;
                    isEmpty = true;
                    val_min_abs = BIG_NUMBER;
                    val_min = BIG_NUMBER;
                    min_dir = -1;

                    if ( F[(g1-1)*W1+g2] != 0.0 )  {

                        if ( std::abs(F[(g1-1)*W1+g2]) < val_min_abs && F[(g1-2)*W1+g2] != 0.0 )  {
                            val_min_abs = std::abs(F[(g1-1)*W1+g2]);
                            val_min = F[(g1-1)*W1+g2];
                            min_dir = 0;
                        }
                        if ( F[(g1-2)*W1+g2] != 0.0 )  {

                            val = exp( 2.0 * std::log(F[(g1-1)*W1+g2]) - std::log(F[(g1-2)*W1+g2]) );

                            if ( !(isnan(val) || isnan(-val)) && !(isinf(val) || isinf(-val)) )  
                            {
                                sum += val;
                                count += 1;
                                isEmpty = false;
                            }
                        }
                    }

                    if ( F[(g1+1)*W1+g2] != 0.0 )  {

                        if ( std::abs(F[(g1+1)*W1+g2]) < val_min_abs && F[(g1+2)*W1+g2] != 0.0  )  {
                            val_min_abs = std::abs(F[(g1+1)*W1+g2]);
                            val_min = F[(g1+1)*W1+g2];
                            min_dir = 0;
                        }
                        if ( F[(g1+2)*W1+g2] != 0.0 )  {

                            val = exp( 2.0 * std::log(F[(g1+1)*W1+g2]) - std::log(F[(g1+2)*W1+g2]) );

                            if ( !(isnan(val) || isnan(-val)) && !(isinf(val) || isinf(-val)) ) 
                            {
                                sum += val;
                                count += 1;
                                isEmpty = false;
                            }
                        }
                    }

                    if ( F[g1*W1+(g2-1)] != 0.0 )  {

                        if ( std::abs(F[g1*W1+(g2-1)]) < val_min_abs && F[g1*W1+(g2-2)] != 0.0  )  {
                            val_min_abs = std::abs(F[g1*W1+(g2-1)]);
                            val_min = F[g1*W1+(g2-1)];
                            min_dir = 1;
                        }
                        if ( F[g1*W1+(g2-2)] != 0.0 )  {

                            val = exp( 2.0 * std::log(F[g1*W1+(g2-1)]) - std::log(F[g1*W1+(g2-2)]) );

                            if ( !(isnan(val) || isnan(-val)) && !(isinf(val) || isinf(-val)) ) 
                            {
                                sum += val;
                                count += 1;
                                isEmpty = false;
                            }
                        }
                    }

                    if ( F[g1*W1+(g2+1)] != 0.0 )  {

                        if ( std::abs(F[g1*W1+(g2+1)]) < val_min_abs && F[g1*W1+(g2+2)] != 0.0 )  {
                            val_min_abs = std::abs(F[g1*W1+(g2+1)]);
                            val_min = F[g1*W1+(g2+1)];
                            min_dir = 1;
                        }
                        if ( F[g1*W1+(g2+2)] != 0.0 )  {

                            val = exp( 2.0 * std::log(F[g1*W1+(g2+1)]) - std::log(F[g1*W1+(g2+2)]) );

                            if ( !(isnan(val) || isnan(-val)) && !(isinf(val) || isinf(-val)) ) 
                            {
                                sum += val;
                                count += 1;
                                isEmpty = false;
                            }
                        }
                    }

                    if ( isEmpty )
                    {
                        Check[i] = 0; 

                    }  else  {
                        // Assume the probability of outer points always smaller than edge point.
                        // if larger, then set the edge point with smallest P to outer
                        // point instead of using the extrapolation result.
                        if ( std::abs(sum/count) > val_min_abs )  {
                            ExTBL[i] = val_min * exp(-ExReduce * H[min_dir]);
                        }
                        else  {
                            ExTBL[i] = sum / count;
                        }
                    }
                }

                count = 0;

                #pragma omp parallel for reduction (+:count) schedule(runtime)
                for ( int i = 0; i < ExFF.size(); i++ )  {
                    if (Check[i] == 1)  {
                        F[ExFF[i]] = ExTBL[i];
                        count += 1;
                    }
                }

                if (count == 0)  {
                    ExFF.clear();
                    ExTBL.clear();
                    Check.clear();
                }
                t_1_end = omp_get_wtime();
                t_1_elapsed = t_1_end - t_1_begin;
                t_overhead += t_1_elapsed;
                if (!QUIET && TIMING) log->log("Elapsed time (omp-b-3: ExFF) = %.4e sec\n", t_1_elapsed);  
            }

            // ............................................................................................. Extrapolation

            if ( isFirstExtrp )  {

                // Check Extending nonzero Area

                t_1_begin = omp_get_wtime();

                tmpVec.clear();

                #pragma omp parallel for reduction(merge: tmpVec) private(g1, g2) schedule(runtime)
                for (int i = 0; i < ExFF.size(); i++)  {
                    if (Check[i] == 1)  {
                        g1 = (int)(ExFF[i] / M1);
                        g2 = (int)(ExFF[i] % M1);

                        if ( !TAMask[g1*W1+g2] )
                            tmpVec.push_back(g1*W1+g2);
                        if ( g1+1 < BoxShape[0]-EDGE-1 && !TAMask[(g1+1)*W1+g2] )
                            tmpVec.push_back((g1+1)*W1+g2);
                        if ( g1-1 > EDGE && !TAMask[(g1-1)*W1+g2] )
                            tmpVec.push_back((g1-1)*W1+g2);
                        if ( g2+1 < BoxShape[1]-EDGE-1 && !TAMask[g1*W1+(g2+1)] )
                            tmpVec.push_back(g1*W1+(g2+1));
                        if ( g2-1 > EDGE && !TAMask[g1*W1+(g2-1)] )
                            tmpVec.push_back(g1*W1+(g2-1));
                    }
                }

                for (int i = 0; i < tmpVec.size(); i ++)  {
                    g1 = (int)(tmpVec[i] / M1);
                    g2 = (int)(tmpVec[i] % M1);
                    TAMask[tmpVec[i]] = 1;
                
                    x1_min = (g1 < x1_min) ? g1 : x1_min;
                    x1_max = (g1 > x1_max) ? g1 : x1_max;
                    x2_min = (g2 < x2_min) ? g2 : x2_min;
                    x2_max = (g2 > x2_max) ? g2 : x2_max;
                }
                tmpVec.clear();

                t_1_end = omp_get_wtime();
                t_1_elapsed = t_1_end - t_1_begin;
                t_overhead += t_1_elapsed;
                if (!QUIET && TIMING) log->log("Elapsed time (omp-c-1: CASE 1 TA) = %lf sec\n", t_1_elapsed); 
        
                // Runge–Kutta 4
                #pragma omp parallel 
                {
                    // RK4-1
                    #pragma omp single nowait
                    {
                        t_1_begin = omp_get_wtime();
                    }
                    #pragma omp for private(xx1,xx2,f0,f1p,f1m,f2p,f2m) schedule(runtime)
                    for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                        for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                            if ( TAMask[i1*W1+i2] )  {
                                xx1 = Box[0] + i1 * H[0];
                                xx2 = Box[2] + i2 * H[1];
                                f0 = F[i1*W1+i2];
                                f1p = F[(i1+1)*W1+i2];
                                f1m = F[(i1-1)*W1+i2];
                                f2p = F[i1*W1+(i2+1)];
                                f2m = F[i1*W1+(i2-1)];

                                KK1[i1*W1+i2] = -k2h0m * xx2 * (f1p - f1m) + 
                                                k2h1 * POTENTIAL_X(xx1, xx2) * (f2p - f2m) +
                                                kgamma * (f0 + i2h1 * xx2 * (f2p - f2m) + mkT2h1sq * (f2p + f2m - 2 * f0));

                                FF[i1*W1+i2] = F[i1*W1+i2] + KK1[i1*W1+i2] / 6.0;
                            }
                        }
                    }
                    #pragma omp single nowait
                    {
                        t_1_end = omp_get_wtime();
                        t_1_elapsed = t_1_end - t_1_begin;
                        t_truncate += t_1_elapsed;
                        if (!QUIET && TIMING) log->log("Elapsed time (omp-kk-11: CASE 1 KK1) = %lf sec\n", t_1_elapsed);
                        t_1_begin = omp_get_wtime();
                    }

                    // RK4-2
                    #pragma omp for private(xx1,xx2,f0,f1p,f1m,f2p,f2m,kk0,kk1p,kk1m,kk2p,kk2m) schedule(runtime)
                    for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                        for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                            if ( TAMask[i1*W1+i2] )  {
                                xx1 = Box[0] + i1 * H[0];
                                xx2 = Box[2] + i2 * H[1];
                                f0 = F[i1*W1+i2];
                                f1p = F[(i1+1)*W1+i2];
                                f1m = F[(i1-1)*W1+i2];
                                f2p = F[i1*W1+(i2+1)];
                                f2m = F[i1*W1+(i2-1)];
                                kk0 = KK1[i1*W1+i2];
                                kk1p = KK1[(i1+1)*W1+i2];
                                kk1m = KK1[(i1-1)*W1+i2];
                                kk2p = KK1[i1*W1+(i2+1)];
                                kk2m = KK1[i1*W1+(i2-1)];

                                KK2[i1*W1+i2] = -k2h0m * xx2 * (f1p + 0.5 * kk1p - f1m - 0.5 * kk1m) + 
                                                k2h1 * POTENTIAL_X(xx1, xx2) * (f2p + 0.5 * kk2p - f2m - 0.5 * kk2m) +
                                                kgamma * (f0 + 0.5 * kk0 + i2h1 * xx2 * (f2p + 0.5 * kk2p - f2m - 0.5 * kk2m) + mkT2h1sq * (f2p + 0.5 * kk2p + f2m + 0.5 * kk2m - 2 * f0 - kk0));

                                FF[i1*W1+i2] += KK2[i1*W1+i2] / 3.0;
                            }
                        }
                    }
                    #pragma omp single nowait
                    {
                        t_1_end = omp_get_wtime();
                        t_1_elapsed = t_1_end - t_1_begin;
                        t_truncate += t_1_elapsed;
                        if (!QUIET && TIMING) log->log("Elapsed time (omp-kk-12: CASE 1 KK2) = %lf sec\n", t_1_elapsed);
                        t_1_begin = omp_get_wtime();
                    }

                    // RK4-3
                    #pragma omp for private(xx1,xx2,f0,f1p,f1m,f2p,f2m,kk0,kk1p,kk1m,kk2p,kk2m) schedule(runtime)
                    for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                        for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                            if ( TAMask[i1*W1+i2] )  {
                                xx1 = Box[0] + i1 * H[0];
                                xx2 = Box[2] + i2 * H[1];
                                f0 = F[i1*W1+i2];
                                f1p = F[(i1+1)*W1+i2];
                                f1m = F[(i1-1)*W1+i2];
                                f2p = F[i1*W1+(i2+1)];
                                f2m = F[i1*W1+(i2-1)];
                                kk0 = KK2[i1*W1+i2];
                                kk1p = KK2[(i1+1)*W1+i2];
                                kk1m = KK2[(i1-1)*W1+i2];
                                kk2p = KK2[i1*W1+(i2+1)];
                                kk2m = KK2[i1*W1+(i2-1)];

                                KK3[i1*W1+i2] = -k2h0m * xx2 * (f1p + 0.5 * kk1p - f1m - 0.5 * kk1m) + k2h1 * POTENTIAL_X(xx1, xx2) * (f2p + 0.5 * kk2p -  f2m - 0.5 * kk2m) +
                                            kgamma * (f0 + 0.5 * kk0 + i2h1 * xx2 * (f2p + 0.5 * kk2p - f2m - 0.5 * kk2m) + mkT2h1sq * (f2p + 0.5 * kk2p + f2m + 0.5 * kk2m - 2 * f0 - kk0));   

                                FF[i1*W1+i2] += KK3[i1*W1+i2] / 3.0;
                            }
                        }            
                    }
                    #pragma omp single nowait
                    {
                        t_1_end = omp_get_wtime();
                        t_1_elapsed = t_1_end - t_1_begin;
                        t_truncate += t_1_elapsed;
                        if (!QUIET && TIMING) log->log("Elapsed time (omp-kk-13: CASE 1 KK3) = %lf sec\n", t_1_elapsed);
                        t_1_begin = omp_get_wtime();
                    }

                    // RK4-4
                    #pragma omp for private(xx1,xx2,f0,f1p,f1m,f2p,f2m,kk0,kk1p,kk1m,kk2p,kk2m) schedule(runtime)
                    for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                        for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                            if ( TAMask[i1*W1+i2] == 1 )  {
                                xx1 = Box[0] + i1 * H[0];
                                xx2 = Box[2] + i2 * H[1];
                                f0 = F[i1*W1+i2];
                                f1p = F[(i1+1)*W1+i2];
                                f1m = F[(i1-1)*W1+i2];
                                f2p = F[i1*W1+(i2+1)];
                                f2m = F[i1*W1+(i2-1)];
                                kk0 = KK3[i1*W1+i2];
                                kk1p = KK3[(i1+1)*W1+i2];
                                kk1m = KK3[(i1-1)*W1+i2];
                                kk2p = KK3[i1*W1+(i2+1)];
                                kk2m = KK3[i1*W1+(i2-1)];

                                KK4[i1*W1+i2] = -k2h0m * xx2 * (f1p + kk1p - f1m - kk1m) + 
                                                k2h1 * POTENTIAL_X(xx1, xx2) * (f2p + kk2p -  f2m - kk2m) +
                                                kgamma * (f0 + kk0 + i2h1 * xx2 * (f2p + kk2p - f2m - kk2m) + mkT2h1sq * (f2p + kk2p + f2m + kk2m - 2 * f0 - 2 * kk0));

                                FF[i1*W1+i2] += KK4[i1*W1+i2] / 6.0;
                            }
                        }
                    }
                    #pragma omp single nowait
                    {
                        t_1_end = omp_get_wtime();
                        t_1_elapsed = t_1_end - t_1_begin;
                        t_truncate += t_1_elapsed;
                        if (!QUIET && TIMING) log->log("Elapsed time (omp-kk-14: CASE 1 KK4) = %lf sec\n", t_1_elapsed);
                        t_1_begin = omp_get_wtime();
                    }
                } // OMP PARALLEL
                isFirstExtrp = false;

            } // if ( isFirstExtrp )
            else if (ExFF.size() == 0)  {

                // In the case no valid ExFF found, reset TBL to break the while loop
                TBL.clear();
            }
            else  
            {
                // Extrapolation loop when multiple expanding occured

                if ( ExFF.size() > 0 )  {

                    t_1_begin = omp_get_wtime();

                    tmpVec.clear();

                    #pragma omp parallel for reduction(merge: tmpVec) private(g1, g2) schedule(runtime)
                    for (int i = 0; i < ExFF.size(); i++)  {
                        if (Check[i] == 1)  {
                            g1 = (int)(ExFF[i] / M1);
                            g2 = (int)(ExFF[i] % M1);

                            if ( !TAMask[g1*W1+g2] )
                                tmpVec.push_back(g1*W1+g2);
                            if ( g1+1 < BoxShape[0]-EDGE-1 && !TAMask[(g1+1)*W1+g2] )
                                tmpVec.push_back((g1+1)*W1+g2);
                            if ( g1-1 > EDGE && !TAMask[(g1-1)*W1+g2] )
                                tmpVec.push_back((g1-1)*W1+g2);
                            if ( g2+1 < BoxShape[1]-EDGE-1 && !TAMask[g1*W1+(g2+1)] )
                                tmpVec.push_back(g1*W1+(g2+1));
                            if ( g2-1 > EDGE && !TAMask[g1*W1+(g2-1)] )
                                tmpVec.push_back(g1*W1+(g2-1));
                        }
                    }

                    for (int i = 0; i < tmpVec.size(); i ++)  {

                        g1 = (int)(tmpVec[i] / M1); 
                        g2 = (int)(tmpVec[i] % M1);
                        TAMask[tmpVec[i]] = 1;

                        // Update TA box
                        x1_min = (g1 < x1_min) ? g1 : x1_min;
                        x2_min = (g2 < x2_min) ? g2 : x2_min;
                        x1_max = (g1 > x1_max) ? g1 : x1_max;
                        x2_max = (g2 > x2_max) ? g2 : x2_max;
                    }
                    tmpVec.clear();
                    ExBD.clear();

                    #pragma omp parallel for reduction(merge: ExBD) private(g1, g2, n1, n2) schedule(runtime)
                    for (int i = 0; i < ExFF.size(); i++) {
                        if (Check[i] == 1)  {
                            g1 = (int)(ExFF[i] / M1);
                            g2 = (int)(ExFF[i] % M1);
                            ExBD.push_back(ExFF[i]);

                            for (int j = 0; j < nneigh; j ++)  {

                                n1 = neighlist[j][0];
                                n2 = neighlist[j][1];

                                if (TAMask[(g1+n1)*W1+(g2+n2)])
                                    ExBD.push_back(GridToIdx(g1+n1,g2+n2));
                            }
                        }
                    }

                    // Find unique elements (ExBD)
                    __gnu_parallel::sort(ExBD.begin(),ExBD.end());
                    it = std::unique (ExBD.begin(), ExBD.end()); 
                    ExBD.resize(std::distance(ExBD.begin(),it));

                    t_1_end = omp_get_wtime();
                    t_1_elapsed = t_1_end - t_1_begin;
                    t_overhead += t_1_elapsed;
                    if (!QUIET && TIMING) log->log("Elapsed time (omp-cx-1: CASE 1 ExBD) = %lf sec\n", t_1_elapsed); 
                } // if ExFF.size() > 0

                // Runge–Kutta 4

                t_1_begin = omp_get_wtime();

                // RK4-1
                #pragma omp parallel for private(g1,g2,xx1,xx2,f0,f1p,f1m,f2p,f2m) schedule(runtime)
                for (int i = 0; i < ExBD.size(); i++)  {

                    g1 = (int)(ExBD[i] / M1);
                    g2 = (int)(ExBD[i] % M1);
                    xx1 = Box[0] + g1 * H[0];
                    xx2 = Box[2] + g2 * H[1];
                    f0 = F[g1*W1+g2];
                    f1p = F[(g1+1)*W1+g2];
                    f1m = F[(g1-1)*W1+g2];
                    f2p = F[g1*W1+(g2+1)];
                    f2m = F[g1*W1+(g2-1)];

                    KK1[g1*W1+g2] = -k2h0m * xx2 * (f1p - f1m) + 
                                    k2h1 * POTENTIAL_X(xx1, xx2) * (f2p - f2m) +
                                    kgamma * (f0 + i2h1 * xx2 * (f2p - f2m) + mkT2h1sq * (f2p + f2m - 2 * f0));

                    FF[g1*W1+g2] = F[g1*W1+g2] + KK1[g1*W1+g2] / 6.0;
                }
                t_1_end = omp_get_wtime();
                t_1_elapsed = t_1_end - t_1_begin;
                t_overhead += t_1_elapsed;
                if (!QUIET && TIMING) log->log("Elapsed time (omp-kkx-11: CASE 1 KK1) = %lf sec\n", t_1_elapsed);
                t_1_begin = omp_get_wtime();

                // RK4-2
                #pragma omp parallel for private(g1,g2,xx1,xx2,f0,f1p,f1m,f2p,f2m,kk0,kk1p,kk1m,kk2p,kk2m) schedule(runtime)
                for (int i = 0; i < ExBD.size(); i++)  {

                    g1 = (int)(ExBD[i] / M1);
                    g2 = (int)(ExBD[i] % M1);
                    xx1 = Box[0] + g1 * H[0];
                    xx2 = Box[2] + g2 * H[1];
                    f0 = F[g1*W1+g2];
                    f1p = F[(g1+1)*W1+g2];
                    f1m = F[(g1-1)*W1+g2];
                    f2p = F[g1*W1+(g2+1)];
                    f2m = F[g1*W1+(g2-1)];
                    kk0 = KK1[g1*W1+g2];
                    kk1p = KK1[(g1+1)*W1+g2];
                    kk1m = KK1[(g1-1)*W1+g2];
                    kk2p = KK1[g1*W1+(g2+1)];
                    kk2m = KK1[g1*W1+(g2-1)];

                    KK2[g1*W1+g2] = -k2h0m * xx2 * (f1p + 0.5 * kk1p - f1m - 0.5 * kk1m) + 
                                    k2h1 * POTENTIAL_X(xx1, xx2) * (f2p + 0.5 * kk2p - f2m - 0.5 * kk2m) +
                                    kgamma * (f0 + 0.5 * kk0 + i2h1 * xx2 * (f2p + 0.5 * kk2p - f2m - 0.5 * kk2m) + mkT2h1sq * (f2p + 0.5 * kk2p + f2m + 0.5 * kk2m - 2 * f0 - kk0));

                    FF[g1*W1+g2] += KK2[g1*W1+g2] / 3.0;
                }
                t_1_end = omp_get_wtime();
                t_1_elapsed = t_1_end - t_1_begin;
                t_overhead += t_1_elapsed;
                if (!QUIET && TIMING) log->log("Elapsed time (omp-kkx-12: CASE 1 KK2) = %lf sec\n", t_1_elapsed);
                t_1_begin = omp_get_wtime();

                // RK4-3
                #pragma omp parallel for private(g1,g2,xx1,xx2,f0,f1p,f1m,f2p,f2m,kk0,kk1p,kk1m,kk2p,kk2m) schedule(runtime)
                for (int i = 0; i < ExBD.size(); i++)  {

                    g1 = (int)(ExBD[i] / M1);
                    g2 = (int)(ExBD[i] % M1);
                    xx1 = Box[0] + g1 * H[0];
                    xx2 = Box[2] + g2 * H[1];
                    f0 = F[g1*W1+g2];
                    f1p = F[(g1+1)*W1+g2];
                    f1m = F[(g1-1)*W1+g2];
                    f2p = F[g1*W1+(g2+1)];
                    f2m = F[g1*W1+(g2-1)];
                    kk0 = KK2[g1*W1+g2];
                    kk1p = KK2[(g1+1)*W1+g2];
                    kk1m = KK2[(g1-1)*W1+g2];
                    kk2p = KK2[g1*W1+(g2+1)];
                    kk2m = KK2[g1*W1+(g2-1)];

                    KK3[g1*W1+g2] = -k2h0m * xx2 * (f1p + 0.5 * kk1p - f1m - 0.5 * kk1m) + 
                                    k2h1 * POTENTIAL_X(xx1, xx2) * (f2p + 0.5 * kk2p -  f2m - 0.5 * kk2m) +
                                    kgamma * (f0 + 0.5 * kk0 + i2h1 * xx2 * (f2p + 0.5 * kk2p - f2m - 0.5 * kk2m) + mkT2h1sq * (f2p + 0.5 * kk2p + f2m + 0.5 * kk2m - 2 * f0 - kk0));             

                    FF[g1*W1+g2] += KK3[g1*W1+g2] / 3.0;
                }
                t_1_end = omp_get_wtime();
                t_1_elapsed = t_1_end - t_1_begin;
                t_overhead += t_1_elapsed;
                if (!QUIET && TIMING) log->log("Elapsed time (omp-kkx-13: CASE 1 KK3) = %lf sec\n", t_1_elapsed);
                t_1_begin = omp_get_wtime();

                // RK4-4
                #pragma omp parallel for private(g1,g2,xx1,xx2,f0,f1p,f1m,f2p,f2m,kk0,kk1p,kk1m,kk2p,kk2m) schedule(runtime)
                for (int i = 0; i < ExBD.size(); i++)  {

                    g1 = (int)(ExBD[i] / M1);
                    g2 = (int)(ExBD[i] % M1);
                    xx1 = Box[0] + g1 * H[0];
                    xx2 = Box[2] + g2 * H[1];
                    f0 = F[g1*W1+g2];
                    f1p = F[(g1+1)*W1+g2];
                    f1m = F[(g1-1)*W1+g2];
                    f2p = F[g1*W1+(g2+1)];
                    f2m = F[g1*W1+(g2-1)];
                    kk0 = KK3[g1*W1+g2];
                    kk1p = KK3[(g1+1)*W1+g2];
                    kk1m = KK3[(g1-1)*W1+g2];
                    kk2p = KK3[g1*W1+(g2+1)];
                    kk2m = KK3[g1*W1+(g2-1)];

                    KK4[g1*W1+g2] = -k2h0m * xx2 * (f1p + kk1p - f1m - kk1m) + 
                                    k2h1 * POTENTIAL_X(xx1, xx2) * (f2p + kk2p -  f2m - kk2m) +
                                    kgamma * (f0 + kk0 + i2h1 * xx2 * (f2p + kk2p - f2m - kk2m) + mkT2h1sq * (f2p + kk2p + f2m + kk2m - 2 * f0 - 2 * kk0));

                    FF[g1*W1+g2] += KK4[g1*W1+g2] / 6.0;
                }
                t_1_end = omp_get_wtime();
                t_1_elapsed = t_1_end - t_1_begin;
                t_overhead += t_1_elapsed;
                if (!QUIET && TIMING) log->log("Elapsed time (omp-kkx-14: CASE 1 KK4) = %lf sec\n", t_1_elapsed);
                t_1_begin = omp_get_wtime();

                ExBD.clear();
            } // isFirstExtrp == false and ExFF.size() > 0

            // Check Multiple Expanding 
            // TBL = index of FF that FF(TBL) is higher than TolL

            if ( ExFF.size() > 0 )  {
                t_1_begin = omp_get_wtime();
                TBL.clear();
                tmpVec.clear();

                #pragma omp parallel for reduction(merge: tmpVec) private(g1, g2, b1, b2, b3, b4, b5)
                for (int i = 0; i < ExFF.size(); i++)
                {
                    g1 = (int)(ExFF[i] / M1);
                    g2 = (int)(ExFF[i] % M1);
                    b1 = FF[g1*W1+g2] >= TolH;
                    b2 = std::abs(FF[(g1+1)*W1+g2] - FF[(g1-1)*W1+g2]) >= TolHdX1;
                    b3 = std::abs(FF[g1*W1+(g2+1)] - FF[g1*W1+(g2-1)]) >= TolHdX2;
                    b4 = g1 > EDGE && g2 > EDGE;
                    b5 = g1 < BoxShape[0] - EDGE - 1 && g2 < BoxShape[1] - EDGE - 1;

                    if (  ( b1 || b2 || b3 ) && b4 && b5 )  {
                        tmpVec.push_back(ExFF[i]);
                    }
                }        
                tmpVec.swap(TBL);
                tmpVec.clear(); 

                // TBL & TBL_P set difference
                tmpVec.resize(TBL_P.size() + TBL.size());
                __gnu_parallel::sort (TBL.begin(), TBL.end());
                __gnu_parallel::sort (TBL_P.begin(), TBL_P.end());
                it=std::set_difference( TBL.begin(), TBL.end(), TBL_P.begin(), TBL_P.end(), tmpVec.begin() );
                tmpVec.resize(it - tmpVec.begin()); 
                tmpVec.swap(TBL);
                tmpVec.clear();

                // Combine TBL and TBL_P
                TBL_P.reserve(TBL_P.size() + TBL.size());
                TBL_P.insert(TBL_P.end(), TBL.begin(), TBL.end());

                // Find unique elements
                __gnu_parallel::sort(TBL_P.begin(),TBL_P.end());
                it = std::unique (TBL_P.begin(), TBL_P.end()); 
                TBL_P.resize(std::distance(TBL_P.begin(),it));

                // Update isFirstExtrp
                isFirstExtrp = (TBL.size() == 0) ? 1 : 0;
                Excount += 1;
                if (Excount == ExLimit) TBL.clear();

                t_1_end = omp_get_wtime();
                t_1_elapsed = t_1_end - t_1_begin;
                t_overhead += t_1_elapsed;
                //if (!QUIET) log->log("TBL size = %d TBL_P size = %d\n", TBL.size(), TBL_P.size()); 
                if (!QUIET && TIMING) log->log("Elapsed time (omp-c-3 CASE 1 TBL) = %lf sec\n", t_1_elapsed); 
            }
        }// TBL.size() != 0 && !isFullGrid && Excount < ExLimit

        // .........................................................................................

        // CASE 2: Truncating without extrapolation

        if ( !isExtrapolate && !isFullGrid )
        {
            // RK4-1
            #pragma omp parallel
            {
                #pragma omp single nowait
                {
                    t_1_begin = omp_get_wtime();
                }
                #pragma omp for private(xx1,xx2,f0,f1p,f1m,f2p,f2m) schedule(runtime)
                for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                    for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                        if (TAMask[i1*W1+i2])  {
                            xx1 = Box[0] + i1 * H[0];
                            xx2 = Box[2] + i2 * H[1];
                            f0 = F[i1*W1+i2];
                            f1p = F[(i1+1)*W1+i2];
                            f1m = F[(i1-1)*W1+i2];
                            f2p = F[i1*W1+(i2+1)];
                            f2m = F[i1*W1+(i2-1)];

                            KK1[i1*W1+i2] = -k2h0m * xx2 * (f1p - f1m) + 
                                            k2h1 * POTENTIAL_X(xx1, xx2) * (f2p - f2m) +
                                            kgamma * (f0 + i2h1 * xx2 * (f2p - f2m) + mkT2h1sq * (f2p + f2m - 2 * f0));

                            FF[i1*W1+i2] = F[i1*W1+i2] + KK1[i1*W1+i2] / 6.0;
                        }
                    }
                }
                #pragma omp single nowait
                {
                    t_1_end = omp_get_wtime();
                    t_1_elapsed = t_1_end - t_1_begin;
                    t_overhead += t_1_elapsed;
                    if (!QUIET && TIMING) log->log("Elapsed time (omp-kk-21: CASE 2 KK1) = %lf sec\n", t_1_elapsed);
                    t_1_begin = omp_get_wtime();
                }

                // RK4-2
                #pragma omp for private(xx1,xx2,f0,f1p,f1m,f2p,f2m,kk0,kk1p,kk1m,kk2p,kk2m) schedule(runtime)
                for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                    for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                        if (TAMask[i1*W1+i2])  {
                            xx1 = Box[0] + i1 * H[0];
                            xx2 = Box[2] + i2 * H[1];
                            f0 = F[i1*W1+i2];
                            f1p = F[(i1+1)*W1+i2];
                            f1m = F[(i1-1)*W1+i2];
                            f2p = F[i1*W1+(i2+1)];
                            f2m = F[i1*W1+(i2-1)];
                            kk0 = KK1[i1*W1+i2];
                            kk1p = KK1[(i1+1)*W1+i2];
                            kk1m = KK1[(i1-1)*W1+i2];
                            kk2p = KK1[i1*W1+(i2+1)];
                            kk2m = KK1[i1*W1+(i2-1)];

                            KK2[i1*W1+i2] = -k2h0m * xx2 * (f1p + 0.5 * kk1p - f1m - 0.5 * kk1m) + 
                                            k2h1 * POTENTIAL_X(xx1, xx2) * (f2p + 0.5 * kk2p - f2m - 0.5 * kk2m) +
                                            kgamma * (f0 + 0.5 * kk0 + i2h1 * xx2 * (f2p + 0.5 * kk2p - f2m - 0.5 * kk2m) + mkT2h1sq * (f2p + 0.5 * kk2p + f2m + 0.5 * kk2m - 2 * f0 - kk0));

                            FF[i1*W1+i2] += KK2[i1*W1+i2] / 3.0;
                        }
                    }
                }
                #pragma omp single nowait
                {
                    t_1_end = omp_get_wtime();
                    t_1_elapsed = t_1_end - t_1_begin;
                    t_overhead += t_1_elapsed;
                    if (!QUIET && TIMING) log->log("Elapsed time (omp-kk-22: CASE 2 KK2) = %lf sec\n", t_1_elapsed);
                    t_1_begin = omp_get_wtime();
                }

                // RK4-3
                #pragma omp for private(xx1,xx2,f0,f1p,f1m,f2p,f2m,kk0,kk1p,kk1m,kk2p,kk2m) schedule(runtime)
                for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                    for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                        if (TAMask[i1*W1+i2])  {
                            xx1 = Box[0] + i1 * H[0];
                            xx2 = Box[2] + i2 * H[1];
                            f0 = F[i1*W1+i2];
                            f1p = F[(i1+1)*W1+i2];
                            f1m = F[(i1-1)*W1+i2];
                            f2p = F[i1*W1+(i2+1)];
                            f2m = F[i1*W1+(i2-1)];
                            kk0 = KK2[i1*W1+i2];
                            kk1p = KK2[(i1+1)*W1+i2];
                            kk1m = KK2[(i1-1)*W1+i2];
                            kk2p = KK2[i1*W1+(i2+1)];
                            kk2m = KK2[i1*W1+(i2-1)];

                            KK3[i1*W1+i2] = -k2h0m * xx2 * (f1p + 0.5 * kk1p - f1m - 0.5 * kk1m) + 
                                            k2h1 * POTENTIAL_X(xx1, xx2) * (f2p + 0.5 * kk2p -  f2m - 0.5 * kk2m) +
                                            kgamma * (f0 + 0.5 * kk0 + i2h1 * xx2 * (f2p + 0.5 * kk2p - f2m - 0.5 * kk2m) + mkT2h1sq * (f2p + 0.5 * kk2p + f2m + 0.5 * kk2m - 2 * f0 - kk0));  

                            FF[i1*W1+i2] += KK3[i1*W1+i2] / 3.0;
                        }
                    }
                }
                #pragma omp single nowait
                {
                    t_1_end = omp_get_wtime();
                    t_1_elapsed = t_1_end - t_1_begin;
                    t_overhead += t_1_elapsed;
                    if (!QUIET && TIMING) log->log("Elapsed time (omp-kk-23: CASE 2 KK3) = %lf sec\n", t_1_elapsed);
                    t_1_begin = omp_get_wtime();
                }

                // RK4-4
                #pragma omp for private(xx1,xx2,f0,f1p,f1m,f2p,f2m,kk0,kk1p,kk1m,kk2p,kk2m) schedule(runtime)
                for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                    for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                        if (TAMask[i1*W1+i2])  {
                            xx1 = Box[0] + i1 * H[0];
                            xx2 = Box[2] + i2 * H[1];
                            f0 = F[i1*W1+i2];
                            f1p = F[(i1+1)*W1+i2];
                            f1m = F[(i1-1)*W1+i2];
                            f2p = F[i1*W1+(i2+1)];
                            f2m = F[i1*W1+(i2-1)];
                            kk0 = KK3[i1*W1+i2];
                            kk1p = KK3[(i1+1)*W1+i2];
                            kk1m = KK3[(i1-1)*W1+i2];
                            kk2p = KK3[i1*W1+(i2+1)];
                            kk2m = KK3[i1*W1+(i2-1)];

                            KK4[i1*W1+i2] = -k2h0m * xx2 * (f1p + 1.0 * kk1p - f1m - 1.0 * kk1m) + 
                                            k2h1 * POTENTIAL_X(xx1, xx2) * (f2p + 1.0 * kk2p -  f2m - 1.0 * kk2m) +
                                            kgamma * (f0 + 1.0 * kk0 + i2h1 * xx2 * (f2p + 1.0 * kk2p - f2m - 1.0 * kk2m) + mkT2h1sq * (f2p + 1.0 * kk2p + f2m + 1.0 * kk2m - 2 * f0 - 2*  kk0));

                            FF[i1*W1+i2] += KK4[i1*W1+i2] / 6.0;
                        }
                    }                
                }
                #pragma omp single nowait
                {
                    t_1_end = omp_get_wtime();
                    t_1_elapsed = t_1_end - t_1_begin;
                    t_overhead += t_1_elapsed;
                    if (!QUIET && TIMING) log->log("Elapsed time (omp-kk-24: CASE 2 KK4) = %lf sec\n", t_1_elapsed);
                    t_1_begin = omp_get_wtime();
                }
            } // OMP PARALLEL
        } 
        else if ( !isExtrapolate && isFullGrid )
        {
            // .........................................................................................

            // CASE 3: Full grid

            // RK4-1
            #pragma omp parallel
            {
                #pragma omp single nowait
                {
                    t_1_begin = omp_get_wtime();
                }
                #pragma omp for private(xx1,xx2,f0,f1p,f1m,f2p,f2m) schedule(runtime)
                for (int i1 = EDGE; i1 < BoxShape[0] - EDGE; i1 ++)  {
                    for (int i2 = EDGE; i2 < BoxShape[1] - EDGE; i2 ++)  {
                        xx1 = Box[0] + i1 * H[0];
                        xx2 = Box[2] + i2 * H[1];
                        f0 = F[i1*W1+i2];
                        f1p = F[(i1+1)*W1+i2];
                        f1m = F[(i1-1)*W1+i2];
                        f2p = F[i1*W1+(i2+1)];
                        f2m = F[i1*W1+(i2-1)];

                        KK1[i1*W1+i2] = -k2h0m * xx2 * (f1p - f1m) + 
                                        k2h1 * POTENTIAL_X(xx1, xx2) * (f2p - f2m) +
                                        kgamma * (f0 + i2h1 * xx2 * (f2p - f2m) + mkT2h1sq * (f2p + f2m - 2 * f0));

                        FF[i1*W1+i2] = F[i1*W1+i2] + KK1[i1*W1+i2] / 6.0;
                    }
                }
                #pragma omp single nowait
                {
                    t_1_end = omp_get_wtime();
                    t_1_elapsed = t_1_end - t_1_begin;
                    t_full += t_1_elapsed;
                    if (!QUIET && TIMING) log->log("Elapsed time (omp-kk-31: CASE 3 KK1) = %lf sec\n", t_1_elapsed);
                    t_1_begin = omp_get_wtime();
                }
                // RK4-2
                #pragma omp for private(xx1,xx2,f0,f1p,f1m,f2p,f2m,kk0,kk1p,kk1m,kk2p,kk2m) schedule(runtime)
                for (int i1 = EDGE; i1 < BoxShape[0] - EDGE; i1 ++)  {
                    for (int i2 = EDGE; i2 < BoxShape[1] - EDGE; i2 ++)  {
                        xx1 = Box[0] + i1 * H[0];
                        xx2 = Box[2] + i2 * H[1];
                        f0 = F[i1*W1+i2];
                        f1p = F[(i1+1)*W1+i2];
                        f1m = F[(i1-1)*W1+i2];
                        f2p = F[i1*W1+(i2+1)];
                        f2m = F[i1*W1+(i2-1)];
                        kk0 = KK1[i1*W1+i2];
                        kk1p = KK1[(i1+1)*W1+i2];
                        kk1m = KK1[(i1-1)*W1+i2];
                        kk2p = KK1[i1*W1+(i2+1)];
                        kk2m = KK1[i1*W1+(i2-1)];

                        KK2[i1*W1+i2] = -k2h0m * xx2 * (f1p + 0.5 * kk1p - f1m - 0.5 * kk1m) + 
                                        k2h1 * POTENTIAL_X(xx1, xx2) * (f2p + 0.5 * kk2p - f2m - 0.5 * kk2m) +
                                        kgamma * (f0 + 0.5 * kk0 + i2h1 * xx2 * (f2p + 0.5 * kk2p - f2m - 0.5 * kk2m) + mkT2h1sq * (f2p + 0.5 * kk2p + f2m + 0.5 * kk2m - 2 * f0 - kk0));

                        FF[i1*W1+i2] += KK2[i1*W1+i2] / 3.0;
                    }
                }
                #pragma omp single nowait
                {
                    t_1_end = omp_get_wtime();
                    t_1_elapsed = t_1_end - t_1_begin;
                    t_full += t_1_elapsed;
                    if (!QUIET && TIMING) log->log("Elapsed time (omp-kk-32: CASE 3 KK2) = %lf sec\n", t_1_elapsed);
                    t_1_begin = omp_get_wtime();
                }

                // RK4-3
                #pragma omp for private(xx1,xx2,f0,f1p,f1m,f2p,f2m,kk0,kk1p,kk1m,kk2p,kk2m) schedule(runtime)
                for (int i1 = EDGE; i1 < BoxShape[0] - EDGE; i1 ++)  {
                    for (int i2 = EDGE; i2 < BoxShape[1] - EDGE; i2 ++)  {
                        xx1 = Box[0] + i1 * H[0];
                        xx2 = Box[2] + i2 * H[1];
                        f0 = F[i1*W1+i2];
                        f1p = F[(i1+1)*W1+i2];
                        f1m = F[(i1-1)*W1+i2];
                        f2p = F[i1*W1+(i2+1)];
                        f2m = F[i1*W1+(i2-1)];
                        kk0 = KK2[i1*W1+i2];
                        kk1p = KK2[(i1+1)*W1+i2];
                        kk1m = KK2[(i1-1)*W1+i2];
                        kk2p = KK2[i1*W1+(i2+1)];
                        kk2m = KK2[i1*W1+(i2-1)];

                        KK3[i1*W1+i2] = -k2h0m * xx2 * (f1p + 0.5 * kk1p - f1m - 0.5 * kk1m) + 
                                        k2h1 * POTENTIAL_X(xx1, xx2) * (f2p + 0.5 * kk2p -  f2m - 0.5 * kk2m) +
                                        kgamma * (f0 + 0.5 * kk0 + i2h1 * xx2 * (f2p + 0.5 * kk2p - f2m - 0.5 * kk2m) + mkT2h1sq * (f2p + 0.5 * kk2p + f2m + 0.5 * kk2m - 2 * f0 - kk0)); 

                        FF[i1*W1+i2] += KK3[i1*W1+i2] / 3.0;              
                    }
                }
                #pragma omp single nowait
                {
                    t_1_end = omp_get_wtime();
                    t_1_elapsed = t_1_end - t_1_begin;
                    t_full += t_1_elapsed;
                    if (!QUIET && TIMING) log->log("Elapsed time (omp-kk-33: CASE 3 KK3) = %lf sec\n", t_1_elapsed);
                    t_1_begin = omp_get_wtime();
                }

                // RK4-4
                #pragma omp for private(xx1,xx2,f0,f1p,f1m,f2p,f2m,kk0,kk1p,kk1m,kk2p,kk2m) schedule(runtime)
                for (int i1 = EDGE; i1 < BoxShape[0] - EDGE; i1 ++)  {
                    for (int i2 = EDGE; i2 < BoxShape[1] - EDGE; i2 ++)  {
                        xx1 = Box[0] + i1 * H[0];
                        xx2 = Box[2] + i2 * H[1];
                        f0 = F[i1*W1+i2];
                        f1p = F[(i1+1)*W1+i2];
                        f1m = F[(i1-1)*W1+i2];
                        f2p = F[i1*W1+(i2+1)];
                        f2m = F[i1*W1+(i2-1)];
                        kk0 = KK3[i1*W1+i2];
                        kk1p = KK3[(i1+1)*W1+i2];
                        kk1m = KK3[(i1-1)*W1+i2];
                        kk2p = KK3[i1*W1+(i2+1)];
                        kk2m = KK3[i1*W1+(i2-1)];

                        KK4[i1*W1+i2] = -k2h0m * xx2 * (f1p + kk1p - f1m - kk1m) + 
                                        k2h1 * POTENTIAL_X(xx1, xx2) * (f2p + kk2p -  f2m - kk2m) +
                                        kgamma * (f0 + kk0 + i2h1 * xx2 * (f2p + kk2p - f2m - kk2m) + mkT2h1sq * (f2p + kk2p + f2m + kk2m - 2 * f0 - 2 * kk0));     

                        FF[i1*W1+i2] += KK4[i1*W1+i2] / 6.0;                             
                    }
                }
                #pragma omp single nowait
                {
                    t_1_end = omp_get_wtime();
                    t_1_elapsed = t_1_end - t_1_begin;
                    t_full += t_1_elapsed;
                    if (!QUIET && TIMING) log->log("Elapsed time (omp-kk-34: CASE 3 KK4) = %lf sec\n", t_1_elapsed);
                    t_1_begin = omp_get_wtime();
                }
            }
        }
        // .........................................................................................

        // NORMALIZATION AND TRUNCATION

        t_1_begin = omp_get_wtime();
        norm = 0.0;

        if (!isFullGrid)  {

            #pragma omp parallel for reduction (+:norm) schedule(runtime)
            for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                    if (TAMask[i1*W1+i2])
                        norm += FF[i1*W1+i2];
                }
            }
        }  
        else  {

            #pragma omp parallel for reduction (+:norm) schedule(runtime)
            for (int i1 = EDGE; i1 < BoxShape[0]-EDGE; i1 ++)  {
                for (int i2 = EDGE; i2 < BoxShape[1]-EDGE; i2 ++)  {
                    norm += FF[i1*W1+i2];
                }
            }
        }
        norm *= H[0] * H[1];

        if ( (tt + 1) % PERIOD == 0 )
            log->log("[KleinKramers2d] Normalization factor = %.16e\n",norm);

        norm = 1.0 / norm; 

        t_1_end = omp_get_wtime();
        t_1_elapsed = t_1_end - t_1_begin;
        t_full += t_1_elapsed;
        t_truncate += t_1_elapsed;
        if (!QUIET && TIMING) log->log("Elapsed time (omp-e-1-1 Norm) = %lf sec\n", t_1_elapsed);
        t_1_begin = omp_get_wtime();

        if (!isFullGrid)  {
            #pragma omp parallel for private(val) schedule(runtime)
            for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                    if (TAMask[i1*W1+i2])  {
                        val = norm * FF[i1*W1+i2];
                        FF[i1*W1+i2] = val;
                        F[i1*W1+i2] = val;
                        PF[i1*W1+i2] = val;
                    }
                }
            }
        }  
        else  {
            #pragma omp parallel for private(val) schedule(runtime)
            for (int i1 = EDGE; i1 < BoxShape[0]-EDGE; i1 ++)  {
                for (int i2 = EDGE; i2 < BoxShape[1]-EDGE; i2 ++)  {
                    val = norm * FF[i1*W1+i2];
                    FF[i1*W1+i2] = val;
                    F[i1*W1+i2] = val;
                    PF[i1*W1+i2] = val;
                }
            }
        }
        t_1_end = omp_get_wtime();
        t_1_elapsed = t_1_end - t_1_begin;
        t_full += t_1_elapsed;
        t_truncate += t_1_elapsed;
        if (!QUIET && TIMING) log->log("Elapsed time (omp-e-1-2 FF) = %lf sec\n", t_1_elapsed); 

        if ( (tt + 1) % PERIOD == 0 )
        {
            // REPORT MEASUREMENTS
            // ----------------------------------------------------------------------------
            // Compute Transmittance
            // ----------------------------------------------------------------------------

            if (isTrans)  {
            
                t_1_begin = omp_get_wtime();
                pftrans = 0.0;

                if (!isFullGrid)  {
                    #pragma omp parallel for reduction (+:pftrans)
                    for (int i1 = idx_x0; i1 <= x1_max; i1 ++)  {
                        for (int i2 = x2_min; i2 <= x2_max; i2 ++)
                            pftrans+=PF[i1*W1+i2];
                    }
                }
                else  {
                    #pragma omp parallel for reduction (+:pftrans)
                    for (int i1 = idx_x0; i1 < BoxShape[0]-EDGE; i1 ++)  {
                        for (int i2 = EDGE; i2 < BoxShape[1]-EDGE; i2 ++)
                            pftrans+=PF[i1*W1+i2];
                    }
                }
                pftrans *= H[0] * H[1];
                PF_trans.push_back(pftrans);
                log->log("[KleinKramers2d] idx_x0 = %d\n", idx_x0);
                log->log("[KleinKramers2d] Time %lf, Trans = %.16e\n", ( tt + 1 ) * kk, pftrans);
                t_1_end = omp_get_wtime();
                t_1_elapsed = t_1_end - t_1_begin; 
                if (!QUIET && TIMING) log->log("Elapsed time (omp-x-2 trans) = %lf sec\n", t_1_elapsed); 
            }

            if (isCorr)  {

                // Density
                #pragma omp parallel for private(density)
                for (int i1 = EDGE; i1 < BoxShape[0] - EDGE; i1 ++)  {
                    density = 0.0;
                    for (int i2 = EDGE; i2 < BoxShape[1] - EDGE; i2 ++)  {
                        density += PF[i1*W1+i2];
                    }
                    Ft[i1] = density * H[1];
                }

                corr = 0.0;
                for (int i1 = EDGE; i1 < BoxShape[0] - EDGE; i1 ++)  {
                    corr += Ft[i1] * F0[i1];
                }
                corr *= H[0];
                log->log("[KleinKramers2d] Time %lf, Corr = %.16e\n", ( tt + 1 ) * kk, corr/corr_0);
            }
        }

        // Truncation and TA

        tmpVec.clear();
        
        if ( !isFullGrid )
        {
            t_1_begin = omp_get_wtime();
            ta_size = 0;
            #pragma omp parallel for private(b1, b2, b3)
            for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                    if (TAMask[i1*W1+i2])  {
                        b1 = std::abs(PF[i1*W1+i2]) < TolH;
                        b2 = std::abs(PF[(i1+1)*W1+i2]*TAMask[(i1+1)*W1+i2] - PF[(i1-1)*W1+i2]*TAMask[(i1-1)*W1+i2]) < TolHdX1;
                        b3 = std::abs(PF[i1*W1+(i2+1)]*TAMask[i1*W1+(i2+1)] - PF[i1*W1+(i2-1)]*TAMask[i1*W1+(i2-1)]) < TolHdX2;
                        //TAMask[i1*W1+i2] = 0;
            
                        if (b1 && b2 && b3)
                            F[i1*W1+i2] = 0.0;
                        //else 
                            //TAMask[i1*W1+i2] = 1;
                    }
                }
            }   

            t_1_end = omp_get_wtime();
            t_1_elapsed = t_1_end - t_1_begin;
            t_overhead += t_1_elapsed;
            if (!QUIET && TIMING) log->log("Elapsed time (omp-e-3-1 TA) = %.4e sec\n", t_1_elapsed);
            t_1_begin = omp_get_wtime();

            #pragma omp parallel for
            for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                    if (F[i1*W1+i2] == 0.0)  {
                        TAMask[i1*W1+i2] = 0;
                        PF[i1*W1+i2] = 0.0;
                    }  else  {
                        if (!TAMask[i1*W1+i2]) {
                            TAMask[i1*W1+i2] = 1;
                        }
                    }
                }
            }
            t_1_end = omp_get_wtime();
            t_1_elapsed = t_1_end - t_1_begin;
            t_overhead += t_1_elapsed;
            if (!QUIET && TIMING) log->log("Elapsed time (omp-e-3-2 TA) = %.4e sec\n", t_1_elapsed);
            t_1_begin = omp_get_wtime();

            // Rebuild TA box
            if ( (tt + 1) %  SORT_PERIOD == 0 )  {

                x1_min = BIG_NUMBER;
                x1_max = -BIG_NUMBER;
                x2_min = BIG_NUMBER;
                x2_max = -BIG_NUMBER;

                #pragma omp parallel for reduction(min: x1_min, x2_min) reduction(max: x1_max, x2_max) schedule(runtime)
                for (int i1 = EDGE; i1 < BoxShape[0] - EDGE; i1 ++)  {
                    for (int i2 = EDGE; i2 < BoxShape[1] - EDGE; i2 ++)  {
                        if (TAMask[i1*W1+i2])  {
                            if (i1 < x1_min)  x1_min = i1;
                            if (i1 > x1_max)  x1_max = i1;
                            if (i2 < x2_min)  x2_min = i2;
                            if (i2 > x2_max)  x2_max = i2;
                        }
                    }
                }
            }
            t_1_end = omp_get_wtime();
            t_1_elapsed = t_1_end - t_1_begin;
            t_overhead += t_1_elapsed;
            if (!QUIET && TIMING) log->log("Elapsed time (omp-e-4 TA) = %lf sec\n", t_1_elapsed);

            // TB
            t_1_begin = omp_get_wtime();
            tmpVec.clear();

            #pragma omp parallel for reduction(merge: tmpVec) schedule(runtime)
            for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                    if (TAMask[i1*W1+i2])  {
                        if (!TAMask[(i1-1)*W1+i2] || !TAMask[(i1+1)*W1+i2] || \
                            !TAMask[i1*W1+(i2-1)] || !TAMask[i1*W1+(i2+1)])
                            tmpVec.push_back(i1*M1+i2);
                    }
                }
            }
            tmpVec.swap(TB);
            tmpVec.clear();

            t_1_end = omp_get_wtime();
            t_1_elapsed = t_1_end - t_1_begin;
            t_overhead += t_1_elapsed;
            if (!QUIET && TIMING) log->log("Elapsed time (omp-e-5 TB) = %lf sec\n", t_1_elapsed);
            // `````````````````````````````````````````````````````````````````

            // TA expansion
            t_1_begin = omp_get_wtime();

            #pragma omp parallel for reduction(merge: tmpVec) private(g1, g2) schedule(runtime)
            for (int i = 0; i < TB.size(); i++)
            {
                g1 = (int)(TB[i] / M1);
                g2 = (int)(TB[i] % M1);

                if ( g1+1 < BoxShape[0]-EDGE-1 && !TAMask[(g1+1)*W1+g2])
                    tmpVec.push_back(GridToIdx(g1+1,g2));
                if ( g1-1 > EDGE && !TAMask[(g1-1)*W1+g2])
                    tmpVec.push_back(GridToIdx(g1-1,g2));
                if ( g2+1 < BoxShape[1]-EDGE-1 && !TAMask[g1*W1+(g2+1)])
                    tmpVec.push_back(GridToIdx(g1,g2+1));
                if ( g2-1 > EDGE && !TAMask[g1*W1+(g2-1)])
                    tmpVec.push_back(GridToIdx(g1,g2-1));
            }
            t_1_end = omp_get_wtime();
            t_1_elapsed = t_1_end - t_1_begin;
            t_overhead += t_1_elapsed;
            if (!QUIET && TIMING) log->log("Elapsed time (omp-e-6 TAEX) = %lf sec\n", t_1_elapsed);
            t_1_begin = omp_get_wtime();

            #pragma omp parallel for reduction(min: x1_min, x2_min) \
                                     reduction(max: x1_max, x2_max) \
                                     private(g1, g2) schedule(runtime)
            for (int i = 0; i < tmpVec.size(); i ++)  {
                g1 = (int)(tmpVec[i] / M1);
                g2 = (int)(tmpVec[i] % M1);
                TAMask[tmpVec[i]] = 1;

                // Update TA box
                x1_min = (g1 < x1_min) ? g1 : x1_min;
                x2_min = (g2 < x2_min) ? g2 : x2_min;
                x1_max = (g1 > x1_max) ? g1 : x1_max;
                x2_max = (g2 > x2_max) ? g2 : x2_max;
            }
            tmpVec.clear();

            ta_size = 0;
            #pragma omp parallel for reduction(+: ta_size) schedule(runtime)
            for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                    if (TAMask[i1*W1+i2])
                        ta_size += 1;    
                }
            }

            t_1_end = omp_get_wtime();
            t_1_elapsed = t_1_end - t_1_begin;
            t_overhead += t_1_elapsed;
            if (!QUIET && TIMING) log->log("Elapsed time (omp-e-7 TARB) = %lf sec\n", t_1_elapsed);
        }

        // Reset

        t_1_begin = omp_get_wtime();

        if (!isFullGrid)  {
            #pragma omp parallel for
            for (int i1 = std::max(x1_min-ExLimit-DIMENSIONS, EDGE); i1 <= std::min(x1_max+ExLimit+DIMENSIONS, BoxShape[0]-EDGE); i1 ++)  {
                for (int i2 = std::max(x2_min-ExLimit-DIMENSIONS, EDGE); i2 <= std::min(x2_max+ExLimit+DIMENSIONS, BoxShape[1]-EDGE); i2 ++)  {

                    FF[i1*W1+i2] = 0.0;
                    KK1[i1*W1+i2] = 0.0;
                    KK2[i1*W1+i2] = 0.0;
                    KK3[i1*W1+i2] = 0.0;
                    KK4[i1*W1+i2] = 0.0;
                }
            }
        }
        else  {
            #pragma omp parallel for
            for (int i1 = EDGE; i1 < BoxShape[0] - EDGE ; i1 ++)  {
                for (int i2 = EDGE; i2 < BoxShape[1] - EDGE ; i2 ++)  {
                    FF[i1*W1+i2] = 0.0;
                    KK1[i1*W1+i2] = 0.0;
                    KK2[i1*W1+i2] = 0.0;
                    KK3[i1*W1+i2] = 0.0;
                    KK4[i1*W1+i2] = 0.0;
                }
            }
        }
        t_1_end = omp_get_wtime();
        t_1_elapsed = t_1_end - t_1_begin;
        t_full += t_1_elapsed;
        t_truncate += t_1_elapsed;
        if ( !QUIET && TIMING ) log->log("Elapsed time (omp-e-8: reset) = %lf sec\n", t_1_elapsed);  

        if ( (tt + 1) % PERIOD == 0 )
        {   
            t_0_end = omp_get_wtime();
            t_0_elapsed = t_0_end - t_0_begin;

            if ( !QUIET ) log->log("[KleinKramers2d] Step: %d, Elapsed time: %lf sec\n", tt + 1, t_0_elapsed);

            if ( !isFullGrid && !QUIET )  {

                ta_size = 0;
                #pragma omp parallel for reduction(+: ta_size) schedule(runtime)
                for (int i1 = x1_min; i1 <= x1_max; i1 ++)  {
                    for (int i2 = x2_min; i2 <= x2_max; i2 ++)  {
                        if (TAMask[i1*W1+i2])
                            ta_size += 1;    
                    }
                }
                tb_size = TB.size();
                if ( ( ta_size * 1.0 ) / GRIDS_TOT > SwitchLevel )  {
                    isFullGrid = true;
                    log->log("[KleinKramers2d] Switch to FG mode\n");
                }

                log->log("[KleinKramers2d] Core computation time = %lf\n", t_truncate);
                log->log("[KleinKramers2d] Overhead time = %lf\n", t_overhead);
                log->log("[KleinKramers2d] TA size = %d, TB size = %d\n", ta_size, tb_size);
                log->log("[KleinKramers2d] TA Range [%d, %d][%d, %d]\n", x1_min, x1_max, x2_min, x2_max);
                log->log("[KleinKramers2d] TA / total grids = %lf\n", ( ta_size * 1.0 ) / GRIDS_TOT);
                log->log("[KleinKramers2d] ExCount = %d ExLimit = %d\n", Excount, ExLimit);
            }
            else if ( isFullGrid && !QUIET )  {

                log->log("[KleinKramers2d] Core computation time = %lf\n", t_full);
            }
            if ( !QUIET ) log->log("\n........................................................\n\n");
        }         
    } // Time iteration 

    delete [] F;
    delete [] FF;
    delete [] PF;
    delete [] KK1;
    delete [] KK2;
    delete [] KK3;
    delete [] KK4;

    if ( !isFullGrid )
        delete [] TAMask;

    log->log("[KleinKramers2d] Evolve done.\n");
}
/* =============================================================================== */

/* KK2DPOT_DW2 */

inline double KleinKramers2d::Wavefunction_DW2(double x1, double x2)
{
    // b0
    //return PIHB_INV * exp(-2 * A[0] * pow(x1 - Wave0[0],2)) * exp(-0.5 * HBSQ_INV * (pow(x2 - Wave0[1],2)/A[1]));

    // b3
    return  x1 > 0 ? 0.0 : 1.0 / 2.1291477 * exp(-0.05 * x2 * x2 - 200.0 * 0.025 * pow(1-exp(-SMALL_B), -4) * pow(1-exp(-(SMALL_B)*(1+x1)), 2) * pow(1-exp(-(SMALL_B)*(1-x1)), 2));

    // b6
    // return  x1 > 0 ? 0.0 : 1.0 / 1.2295339 * exp(-0.05 * x2 * x2 - 200.0 * 0.025 * pow(1-exp(-SMALL_B), -4) * pow(1-exp(-(SMALL_B)*(1+x1)), 2) * pow(1-exp(-(SMALL_B)*(1-x1)), 2));

    // b9
    // return  x1 > 0 ? 0.0 : 1.0 / 0.84346893 * exp(-0.05 * x2 * x2 - 200.0 * 0.025 * pow(1-exp(-SMALL_B), -4) * pow(1-exp(-(SMALL_B)*(1+x1)), 2) * pow(1-exp(-(SMALL_B)*(1-x1)), 2));
}
/* ------------------------------------------------------------------------------- */

inline double KleinKramers2d::Potential_DW2(double x1, double x2)
{
    //return 0.025 * (x1 * x1 - 1.0) * (x1 * x1 - 1.0);
    return 0.025 * pow(1-exp(-SMALL_B), -4) * pow(1-exp(-(SMALL_B)*(1+x1)), 2) * pow(1-exp(-(SMALL_B)*(1-x1)), 2);
}
/* ------------------------------------------------------------------------------- */

inline double KleinKramers2d::Vx_DW2(double x1, double x2)
{
    //return 0.1 * x1 * (x1 * x1 - 1.0);
    return 2 * (SMALL_B) * 0.025 * pow(1-exp(-SMALL_B), -4) * (\
           (1-exp(-(SMALL_B)*(1+x1))) * exp(-(SMALL_B)*(1+x1)) * pow(1-exp(-(SMALL_B)*(1-x1)), 2) - \
           (1-exp(-(SMALL_B)*(1-x1))) * exp(-(SMALL_B)*(1-x1)) * pow(1-exp(-(SMALL_B)*(1+x1)), 2) \
           );
}

/* ------------------------------------------------------------------------------- */

VectorXi KleinKramers2d::IdxToGrid(int idx)
{
    int x1 = (int)(idx / M1);
    int x2 = (int)(idx % M1);

    VectorXi grid;
    grid.resize(DIMENSIONS);
    grid << x1, x2;

    return grid;
}
/* ------------------------------------------------------------------------------- */

inline int KleinKramers2d::GridToIdx(int x1, int x2)
{
    return (int)(x1 * M1 + x2);
}
/* ------------------------------------------------------------------------------- */
