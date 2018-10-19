#include "Kernel.h"

#include <omp.h>

void Kernel::Initialize(REAL_TYPE (*_kernelFunc)(REAL_TYPE*, REAL_TYPE*, int, REAL_TYPE*) , int _inputDim , int _nParams , REAL_TYPE *_params )
{
	kernelFunc = _kernelFunc;
	inputDim = _inputDim;
	nParams = _nParams;
	if( nParams != 0 )
	{
		params = new REAL_TYPE [ nParams ];
		for(int i=0;i<nParams;i++)
		{
			params[i] = _params[i];
		}
	}
	kMatComputed = false;
}

REAL_TYPE Kernel::KF(REAL_TYPE *x, REAL_TYPE *y)
{
	return kernelFunc( x , y , inputDim , params );
}

REAL_TYPE Kernel::Compute_Distance_SQ(REAL_TYPE *x, REAL_TYPE *y)
{
	return ( KF(x,x) - 2.0*KF(x,y) + KF(y,y) );
}

REAL_TYPE Kernel::Compute_Distance(REAL_TYPE *x, REAL_TYPE *y)
{
	return sqrt( Compute_Distance_SQ(x,y) );
}

REAL_TYPE Kernel::Compute_Distance_P_KP_SQ(REAL_TYPE *x, Kernel_Point *kp)
{
	REAL_TYPE sumW, tmp, ret;
	sumW = 0.0;
	for(int i=0;i<kp->n;i++)
	{
		sumW += kp->w[i];
	}

	ret = KF( x , x );	
	
	tmp = 0.0;
	for(int i=0;i<kp->n;i++)
	{
		if( kp->w[i] != 0.0 )
		{
			tmp += 2.0 * kp->w[i] * KF( x , kp->ps->d[i] );
		}
	}
	tmp = tmp / sumW;
	ret = ret - tmp;

	tmp = 0.0;
	for(int i=0;i<kp->n;i++)
	{
		for(int j=0;j<kp->n;j++)
		{
			tmp += kp->w[i] * kp->w[j] * KF( kp->ps->d[i] , kp->ps->d[j] );
		}
	}
	tmp = tmp / ( sumW * sumW );
	ret = ret + tmp;

	return ret;
}


REAL_TYPE Kernel::Compute_Distance_P_KP(REAL_TYPE *x, Kernel_Point *kp)
{
	return sqrt( Compute_Distance_P_KP_SQ( x , kp ) );
}

void Kernel::Construct_KMat(Points *_ps)
{
	ps = _ps;
	kMat = new REAL_TYPE * [ ps->nP ];
	for(int i=0;i<ps->nP;i++)
	{
		kMat[i] = new REAL_TYPE [ ps->nP ];
	}

	for(int i=0;i<ps->nP;i++)
	{
		//kMat[i][i] = K( ps->d[i] , ps->d[i] );
		#pragma omp parallel for
		for(int j=i;j<ps->nP;j++)
		{
			kMat[i][j] = KF( ps->d[i] , ps->d[j] );
			kMat[j][i] = kMat[i][j];
		}
	}
	kMatComputed = true;
}

REAL_TYPE Kernel::K_kMat(int xIndex, int yIndex)
{
	return kMat[xIndex][yIndex];
}

REAL_TYPE Kernel::Compute_Distance_SQ_kMat(int xIndex, int yIndex)
{
	return ( K_kMat(xIndex,xIndex) - 2.0*K_kMat(xIndex,yIndex) + K_kMat(yIndex,yIndex) );
}

REAL_TYPE Kernel::Compute_Distance_kMat(int xIndex, int yIndex)
{
	return sqrt( Compute_Distance_SQ_kMat( xIndex , yIndex ) );
}

REAL_TYPE Kernel::Compute_Distance_P_KP_SQ_kMat(int xIndex, Kernel_Point *kp)
{
	REAL_TYPE sumW, tmp, ret;
	sumW = 0.0;
	for(int i=0;i<kp->n;i++)
	{
		sumW += kp->w[i];
	}

	ret = K_kMat( xIndex , xIndex );	
	
	tmp = 0.0;
	for(int i=0;i<kp->n;i++)
	{
		if( kp->w[i] != 0.0 )
		{
			tmp += 2.0 * kp->w[i] * K_kMat( xIndex , i );
		}
	}
	tmp = tmp / sumW;
	ret = ret - tmp;

	
	tmp = 0.0;
	for(int i=0;i<kp->n;i++)
	{
		for(int j=0;j<kp->n;j++)
		{
			tmp += kp->w[i] * kp->w[j] * K_kMat( i , j );
		}
	}
	tmp = tmp / ( sumW * sumW );
	ret = ret + tmp;
	

	return ret;
}


REAL_TYPE Kernel::Compute_Distance_P_KP_kMat(int xIndex, Kernel_Point *kp)
{
	return sqrt( Compute_Distance_P_KP_SQ_kMat( xIndex , kp ) );
}

REAL_TYPE KernelFunc_RBF(REAL_TYPE *x, REAL_TYPE *y, int inputDim, REAL_TYPE *params)
{
    REAL_TYPE dist = Compute_Distance_L2Sq<REAL_TYPE>( x , y , inputDim );
    return exp( -1.0 * dist / ( params[0] * params[0] * 2.0 ) );
}

REAL_TYPE KernelFunc_Linear(REAL_TYPE *x, REAL_TYPE *y, int inputDim, REAL_TYPE *params)
{
    REAL_TYPE ret = 0.0;
    for(int i=0;i<inputDim;i++)
    {
        ret += x[i] * y[i];
    }
    return ret;
}

REAL_TYPE KernelFunc_L2(REAL_TYPE *x, REAL_TYPE *y, int inputDim, REAL_TYPE *params)
{
    return ( -1.0 * sqrt( Compute_Distance_L2Sq<REAL_TYPE>( x , y , inputDim ) ) );
}

REAL_TYPE KernelFunc_ChiSquare(REAL_TYPE *x, REAL_TYPE *y, int inputDim, REAL_TYPE *params)
{
    REAL_TYPE ret = 0.0;
    for(int i=0;i<inputDim;i++)
    {
        ret += ( x[i] * y[i] ) / ( x[i] + y[i] + 0.001 );
    }
    ret *= 2.0;
    return ret;
}

REAL_TYPE KernelFunc_Histogram_Intersection(REAL_TYPE *x, REAL_TYPE *y, int inputDim, REAL_TYPE *params)
{
    REAL_TYPE ret = 0.0;
    for(int i=0;i<inputDim;i++)
    {
        ret += min( x[i] , y[i] );
    }
    return ret;
}
