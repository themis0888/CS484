#pragma once

#include "Common.h"
#include "Points.h"

template<int BCODE_LEN>
class LSH
{
public :
	int dim;
	REAL_TYPE **pM;

	void Initialize(int _dim)
	{
		dim = _dim;
		pM = new REAL_TYPE * [ dim ];
		for(int k=0;k<dim;k++)
		{
			pM[k] = new REAL_TYPE [ BCODE_LEN ];
			for(int i=0;i<BCODE_LEN;i++)
			{
				pM[k][i] = Rand_Gaussian<REAL_TYPE>();
			}
		}
	}

    void Compute_BCode(REAL_TYPE *x, bitset<BCODE_LEN> &y)
	{
		REAL_TYPE tmp;
		for(int i=0;i<BCODE_LEN;i++)
		{
			tmp = 0.0;
			for(int k=0;k<dim;k++)
			{
				tmp += x[k] * pM[k][i];
			}
			if( tmp > 0.0 )
			{
				y[i] = 1;
			}
			else
			{
				y[i] = 0;
			}
		}
	}
};
