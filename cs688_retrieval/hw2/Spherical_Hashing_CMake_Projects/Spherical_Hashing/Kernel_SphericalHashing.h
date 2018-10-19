#pragma once

#include "Common.h"
#include "Points.h"
#include "Kernel.h"


class Kernel_Sphere
{
public :
    Kernel_Point c;
    REAL_TYPE r, rSq;

    Points *ps;
    int inputDim;

    void Initialize(Points *_ps);
};

template<int BCODE_LEN>
class Kernel_SphericalHashing
{
public :
	Points *ps;

	Kernel *kernel;

	Kernel_Sphere *s;

	Points tps;

	// for fast distance calculation
	REAL_TYPE *distSQ_Residuals;
	REAL_TYPE *sumW;

	Points M;	int nM;	
	REAL_TYPE **kM;

	REAL_TYPE **kM_DPS;
	REAL_TYPE  *kM_DPS_Self;
	REAL_TYPE **kM_QPS;
	REAL_TYPE  *kM_QPS_Self;

	bitset<NUM_TRAIN_SAMPLES> *table;
	Index_Distance **ids;	

	REAL_TYPE **kM_TPS;
	REAL_TYPE *kM_Self;

	void Train();
	void Set_Spheres();

	void SetRadius_Ratio(int sIndex, REAL_TYPE ratio);

	REAL_TYPE Compute_K_Center_TPS(int sIndex, int tpIndex);
	REAL_TYPE Compute_Distance_SQ_From_Center_TPS(int sIndex, int tpIndex);
	void Compute_KernelMatrix_TPS();

	void AllocateMem_KMs(Points *dps, Points *qps);
	void ReleaseMem_KMs();
	void Construct_KMs(Points *dps, Points *qps);

	void Set_Centers_RandomWeights();
	void Compute_Table();
	void Compute_Overlaps(int **overlaps);

	void Compute_DistSQ_Residuals();
	REAL_TYPE Compute_Distance_SQ_From_Center(int sIndex, REAL_TYPE *x);

	void Initialize(Points *_ps, Kernel *_kernel, int _nM);
	void Set_Milestones();

	REAL_TYPE Compute_Distance_SQ_From_Center_KM_DPS(int sIndex, int pIndex);
	void Compute_BCode_KM_DPS(int pIndex, bitset<BCODE_LEN> &y);

	REAL_TYPE Compute_Distance_SQ_From_Center_KM_QPS(int sIndex, int pIndex);
	void Compute_BCode_KM_QPS(int pIndex, bitset<BCODE_LEN> &y);	
};

