#pragma once

#include "Common.h"
#include "Points.h"


class Sphere
{
public :
	REAL_TYPE *c, r, rSq;
	
	void Initialize(int _dim)
	{
		c = new REAL_TYPE [ _dim ];
		r = 0.0;		rSq = 0.0;
	}

	// function to set radius to include desired portion of training set
	void Set_Radius(Points *ps, Index_Distance *ids);
};

template<int BCODE_LEN>
class SphericalHashing
{
public :
	Points *ps;

	// training set
	Points tps;

	Sphere *s;

	Index_Distance **ids;
	bitset<NUM_TRAIN_SAMPLES> *table;

	void Initialize(Points *_ps);
	void Compute_Table();
	void Compute_Num_Overlaps(int **overlaps);
	void Set_Spheres();

	void ReleaseMem();

    void Compute_BCode(REAL_TYPE *x, bitset<BCODE_LEN> &y)
	{
		for(int i=0;i<BCODE_LEN;i++)
		{
			if( Compute_Distance_L2Sq<REAL_TYPE>( s[i].c , x , ps->dim ) > s[i].rSq )
			{
				y[i] = 0;
			}
			else
			{
				y[i] = 1;
			}
		}
	}
};
