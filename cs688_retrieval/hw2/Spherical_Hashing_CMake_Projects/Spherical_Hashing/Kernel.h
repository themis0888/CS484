#pragma once

#include "Common.h"
#include "Points.h"

class Kernel_Point
{
public :
	int n;
	REAL_TYPE *w;
	Points *ps;
	void Initialize(int _n, Points *_ps)
	{
		ps = _ps;
		n = _n;
		w = new REAL_TYPE [n];
		Set_Zero();
	}

	void Set_Zero()
	{
		for(int i=0;i<n;i++)
		{
			w[i] = 0.0;
		}
	}

	void Print_W()
	{
		for(int i=0;i<n;i++)
		{
			printf("%f\t",w[i]);
		}
	}
};

class Kernel
{
public :
	REAL_TYPE (*kernelFunc)(REAL_TYPE*, REAL_TYPE*, int, REAL_TYPE*);

	int inputDim, nParams;
	REAL_TYPE *params;

	bool kMatComputed;
	REAL_TYPE **kMat;

	Points *ps;

	void Initialize(REAL_TYPE (*_kernelFunc)(REAL_TYPE*, REAL_TYPE*, int, REAL_TYPE*) , int _inputDim , int _nParams , REAL_TYPE *_params);
	void Construct_KMat(Points *_ps);

	REAL_TYPE KF(REAL_TYPE *x, REAL_TYPE *y);
	
	REAL_TYPE Compute_Distance(REAL_TYPE *x, REAL_TYPE *y);
	REAL_TYPE Compute_Distance_SQ(REAL_TYPE *x, REAL_TYPE *y);

	REAL_TYPE Compute_Distance_P_KP(REAL_TYPE *x, Kernel_Point *kp);
	REAL_TYPE Compute_Distance_P_KP_SQ(REAL_TYPE *x, Kernel_Point *kp);

	REAL_TYPE K_kMat(int xIndex, int yIndex);
	REAL_TYPE Compute_Distance_kMat(int xIndex, int yIndex);
	REAL_TYPE Compute_Distance_SQ_kMat(int xIndex, int yIndex);

	REAL_TYPE Compute_Distance_P_KP_kMat(int xIndex, Kernel_Point *kp);
	REAL_TYPE Compute_Distance_P_KP_SQ_kMat(int xIndex, Kernel_Point *kp);
};

REAL_TYPE KernelFunc_RBF(REAL_TYPE *x, REAL_TYPE *y, int inputDim, REAL_TYPE *params);
REAL_TYPE KernelFunc_Linear(REAL_TYPE *x, REAL_TYPE *y, int inputDim, REAL_TYPE *params);
REAL_TYPE KernelFunc_L2(REAL_TYPE *x, REAL_TYPE *y, int inputDim, REAL_TYPE *params);
REAL_TYPE KernelFunc_ChiSquare(REAL_TYPE *x, REAL_TYPE *y, int inputDim, REAL_TYPE *params);
REAL_TYPE KernelFunc_Histogram_Intersection(REAL_TYPE *x, REAL_TYPE *y, int inputDim, REAL_TYPE *params);
