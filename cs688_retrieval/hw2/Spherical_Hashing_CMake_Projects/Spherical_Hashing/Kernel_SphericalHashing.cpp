#include "Kernel_SphericalHashing.h"

#include <omp.h>

void Kernel_Sphere::Initialize(Points *_ps)
{
    ps = _ps;
    inputDim = ps->dim;

    c.Initialize( ps->nP , ps );
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::Initialize(Points *_ps, Kernel *_kernel, int _nM)
{
	this->ps = _ps;
	this->kernel = _kernel;
	this->nM = _nM;

	distSQ_Residuals = new REAL_TYPE [ BCODE_LEN ];
	sumW = new REAL_TYPE [ BCODE_LEN ];
}

template<int BCODE_LEN>
REAL_TYPE Kernel_SphericalHashing<BCODE_LEN>::Compute_Distance_SQ_From_Center_KM_DPS(int sIndex, int pIndex)
{
	REAL_TYPE ret;
	ret = 0.0;
	for(int i=0;i<nM;i++)
	{
		ret += s[sIndex].c.w[i] * kM_DPS[i][pIndex];
	}
	return ret;
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::Compute_BCode_KM_DPS(int pIndex, bitset<BCODE_LEN> &y)
{
	for(int i=0;i<BCODE_LEN;i++)
	{
		if( Compute_Distance_SQ_From_Center_KM_DPS( i , pIndex ) > s[i].rSq )
		{
			y[i] = 1;
		}
		else
		{
			y[i] = 0;
		}
	}
}

template<int BCODE_LEN>
REAL_TYPE Kernel_SphericalHashing<BCODE_LEN>::Compute_Distance_SQ_From_Center_KM_QPS(int sIndex, int pIndex)
{
	REAL_TYPE ret;
	ret = 0.0;
	for(int i=0;i<nM;i++)
	{
		ret += s[sIndex].c.w[i] * kM_QPS[i][pIndex];
	}
	return ret;
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::Compute_BCode_KM_QPS(int pIndex, bitset<BCODE_LEN> &y)
{
	for(int i=0;i<BCODE_LEN;i++)
	{
		if( Compute_Distance_SQ_From_Center_KM_QPS( i , pIndex ) > s[i].rSq )
		{
			y[i] = 1;
		}
		else
		{
			y[i] = 0;
		}
	}
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::AllocateMem_KMs(Points *dps, Points *qps)
{
	kM_DPS = new REAL_TYPE * [ nM ];
	for(int i=0;i<nM;i++)
	{
		kM_DPS[i] = new REAL_TYPE [ dps->nP ];
	}
	kM_DPS_Self = new REAL_TYPE [ dps->nP ];
	
	kM_QPS = new REAL_TYPE * [ nM ];
	for(int i=0;i<nM;i++)
	{
		kM_QPS[i] = new REAL_TYPE [ qps->nP ];
	}
	kM_QPS_Self = new REAL_TYPE [ qps->nP ];
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::Construct_KMs(Points *dps, Points *qps)
{
	#pragma omp parallel for
	for(int i=0;i<dps->nP;i++)
	{
		kM_DPS_Self[i] = kernel->KF( dps->d[i] , dps->d[i] );
	}
	#pragma omp parallel for
	for(int i=0;i<qps->nP;i++)
	{
		kM_QPS_Self[i] = kernel->KF( qps->d[i] , qps->d[i] );
	}
	for(int i=0;i<nM;i++)
	{
		#pragma omp parallel for
		for(int j=0;j<dps->nP;j++)
		{
			kM_DPS[i][j] = kernel->KF( M.d[i] , dps->d[j] );
		}
	}
	for(int i=0;i<nM;i++)
	{
		#pragma omp parallel for
		for(int j=0;j<qps->nP;j++)
		{
			kM_QPS[i][j] = kernel->KF( M.d[i] , qps->d[j] );
		}
	}
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::ReleaseMem_KMs()
{
	for(int i=0;i<nM;i++)
	{
		delete [] kM_DPS[i];
		delete [] kM_QPS[i];
	}
	delete [] kM_DPS;
	delete [] kM_QPS;
	delete [] kM_DPS_Self;
	delete [] kM_QPS_Self;
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::Set_Milestones()
{
	M.Initialize( nM , tps.dim );
	bool *check = new bool [ tps.nP ];
	SetVector_Val<bool>( check , tps.nP , false );
	for(int i=0;i<nM;i++)
	{
		while(true)
		{
			int rIdx = Rand_Uniform_Int( 0 , tps.nP - 1 );
			if( !check[rIdx] )
			{
				SetVector_Vec<REAL_TYPE>( M.d[i] , tps.d[rIdx] , tps.dim );
				check[rIdx] = true;
                break;
			}
		}
	}
	delete [] check;

	kM = new REAL_TYPE * [ nM ];
	for(int i=0;i<nM;i++)
	{
		kM[i] = new REAL_TYPE [ nM ];
	}	
	for(int i=0;i<nM;i++)
	{
		#pragma omp parallel for
		for(int j=i;j<nM;j++)
		{
			kM[i][j] = kernel->KF( M.d[i] , M.d[j] );
			kM[j][i] = kM[i][j];
		}
	}
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::Compute_DistSQ_Residuals()
{
	#pragma omp parallel for
	for(int k=0;k<BCODE_LEN;k++)
	{
		sumW[k] = 0.0;
		for(int i=0;i<nM;i++)
		{
			sumW[k] += s[k].c.w[i];
		}
		distSQ_Residuals[k] = 0.0;		
		for(int i=0;i<nM;i++)
		{
			for(int j=0;j<nM;j++)
			{
				distSQ_Residuals[k] += ( s[k].c.w[i] * s[k].c.w[j] * kM[i][j] );
			}
		}
	}
}

template<int BCODE_LEN>
REAL_TYPE Kernel_SphericalHashing<BCODE_LEN>::Compute_Distance_SQ_From_Center(int sIndex, REAL_TYPE *x)
{
	REAL_TYPE ret, tmp;
	ret = kernel->KF( x , x );	
	
	tmp = 0.0;
	for(int i=0;i<nM;i++)
	{
		if( s[sIndex].c.w[i] != 0 )
		{
			tmp += 2.0 * s[sIndex].c.w[i] * kernel->KF( x , M.d[i] );
		}
	}
	ret = ret - tmp + distSQ_Residuals[sIndex];
	return ret;
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::Train()
{
	tps.Initialize( NUM_TRAIN_SAMPLES , ps->dim );	
	bool *checkList = new bool [ ps->nP ];
	SetVector_Val<bool>( checkList , ps->nP , true );
	int index;

	// sampling training set
	for(int i=0;i<NUM_TRAIN_SAMPLES;i++)
	{
		while(true)
		{
			index = Rand_Uniform_Int( 0 , ps->nP - 1 );
			if( checkList[index] )
			{
				checkList[index] = false;
				break;
			}
		}
		SetVector_Vec<REAL_TYPE>( tps.d[i] , ps->d[index] , tps.dim );		
	}
	delete [] checkList;

	table = new bitset<NUM_TRAIN_SAMPLES> [ BCODE_LEN ];
	for(int i=0;i<BCODE_LEN;i++)
	{
		for(int j=0;j<NUM_TRAIN_SAMPLES;j++)
		{
			table[i][j] = 0;
		}
	}
	
	ids = new Index_Distance * [BCODE_LEN];
	for(int i=0;i<BCODE_LEN;i++)
	{
		ids[i] = new Index_Distance [ tps.nP ];
	}

	s = new Kernel_Sphere [BCODE_LEN];
	for(int i=0;i<BCODE_LEN;i++)
	{
		s[i].Initialize( &tps );
	}

	Set_Milestones();

	Compute_KernelMatrix_TPS();

	Set_Spheres();
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::Set_Centers_RandomWeights()
{
	//#pragma omp parallel for
	for(int i=0;i<BCODE_LEN;i++)
	{
		for(int k=0;k<nM;k++)
		{
			s[i].c.w[k] = Rand_Uniform<REAL_TYPE>( -1.0 , 1.0 );
		}
		Normalize_Vector<REAL_TYPE>( s[i].c.w , nM );
	}
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::Set_Spheres()
{
	int **overlaps = new int * [ BCODE_LEN ];
	for(int i=0;i<BCODE_LEN;i++)
	{
		overlaps[i] = new int [ BCODE_LEN ];
	}

	REAL_TYPE **dW_Mat = new REAL_TYPE * [BCODE_LEN ];
	for(int i=0;i<BCODE_LEN;i++)
	{
		dW_Mat[i] = new REAL_TYPE [ nM ];
	}
	REAL_TYPE *dW = new REAL_TYPE [ nM ];

	REAL_TYPE *distSD;
	distSD = new REAL_TYPE[ BCODE_LEN ];
	REAL_TYPE meanDistSD, varDistSD;
	REAL_TYPE maxSD, minSD;
	REAL_TYPE minR, maxR;
	
	int p0, p1;
	REAL_TYPE nowDist;
	REAL_TYPE maxDist = -1.0;
	double avgDist = 0.0;
	for(int i=0;i<tps.nP;i++)
	{
		p0 = Rand_Uniform_Int( 0 , tps.nP - 1 );
		while( true )
		{
			p1 = Rand_Uniform_Int( 0 , tps.nP - 1 );
			if( p0 != p1 )
			{
				break;
			}
		}
		nowDist = kernel->Compute_Distance_SQ( tps.d[p0] , tps.d[p1] );
		nowDist = sqrt(nowDist);
		avgDist += (double)nowDist;
		maxDist = max( maxDist , nowDist );
	}
	avgDist /= (double)( tps.nP );
    //printf("** avg Dist: %f\n",avgDist);
    //printf("** max Dist: %f\n",maxDist);

    //printf("Iterative Process for Kernelized Spherical Hashing Begin\n");
    REAL_TYPE includingRatio = INCLUDING_RATIO;
    REAL_TYPE overlapRatio = OVERLAP_RATIO;

	Set_Centers_RandomWeights();
	
	#pragma omp parallel for
	for(int i=0;i<BCODE_LEN;i++)
	{
		SetRadius_Ratio(i,includingRatio);
	}
	

	int interNum;
	int targetInterNum = (REAL_TYPE)(tps.nP) * overlapRatio;
	REAL_TYPE tmpOverlap, alpha, wConst;

    int maxIterations = 50;


	for(int iter=0;iter<maxIterations;iter++)
	{
		Compute_Table();
		Compute_Overlaps(overlaps);

		REAL_TYPE mean, variance, cnt;
		mean = 0.0;		cnt = 0.0;		variance = 0.0;
		for(int i=0;i<BCODE_LEN-1;i++)
		{
			for(int j=i+1;j<BCODE_LEN;j++)
			{
				mean += (REAL_TYPE)( overlaps[i][j] );
				cnt += 1.0;
			}
		}
		mean /= cnt;
        for(int i=0;i<BCODE_LEN-1;i++)
		{
            for(int j=i+1;j<BCODE_LEN;j++)
			{
				variance += ( (REAL_TYPE)( overlaps[i][j] ) - mean ) * ( (REAL_TYPE)( overlaps[i][j] ) - mean );
			}
		}
		variance /= cnt;
        //printf("Iteartion #%d\n",iter);
        //printf("Mean:%f ( SDev:%f )\t",mean,sqrt(variance));

		double rMean, rVar;
		rMean = 0.0;
		rVar = 0.0;
		cnt = (REAL_TYPE)BCODE_LEN;
		minR = s[0].r;		maxR = s[0].r;
        for(int i=0;i<BCODE_LEN;i++)
		{
			rMean += (double)s[i].r;
			minR = min( minR , s[i].r );
			maxR = max( maxR , s[i].r );
		}
		rMean /= (double)cnt;
        for(int i=0;i<BCODE_LEN;i++)
		{
			rVar += ( (double)s[i].r - rMean ) * ( (double)s[i].r - rMean );
		}
		rVar /= (double)cnt;
        //printf("Radius Mean: %f ( SDev: %f  min: %f  max: %f)\n", rMean , sqrt(rVar) ,minR,maxR);

		REAL_TYPE allowedErrorMean, allowedErrorVar;
				
		allowedErrorMean = (REAL_TYPE)tps.nP * overlapRatio * 0.10;
		allowedErrorVar = (REAL_TYPE)tps.nP * overlapRatio * 0.15;

		/*
		printf("Overlaps\n");
		for(int i=0;i<BCODE_LEN;i++)
		{
			for(int j=0;j<BCODE_LEN;j++)
			{
				printf("%d\t",overlaps[i][j]);
			}
			printf("\n");
		}
		printf("\n");
		*/


		if( fabs( mean - ( (REAL_TYPE)tps.nP * overlapRatio ) ) < allowedErrorMean && sqrt(variance) < allowedErrorVar )
		{
            //printf("----- Converged (iteration count %d) \n",iter);
            //printf("Radius Mean: %f ( SDev: %f  min: %f  max: %f)\n", rMean , sqrt(rVar) ,minR,maxR);
            //printf("Error ratio: mean %f%%\tstd-dev %f%%\n\n",100.0*fabs( mean - ( (REAL_TYPE)tps.nP * overlapRatio ) ) / ( (REAL_TYPE)tps.nP * overlapRatio ) , 100.0*sqrt(variance) /( (REAL_TYPE)tps.nP * overlapRatio )  );
			break;
		}

		SetMatrix_Val<REAL_TYPE>( dW_Mat , BCODE_LEN , nM , 0.0 );
		for(int i=0;i<BCODE_LEN;i++)
		{
			for(int j=i+1;j<BCODE_LEN;j++)
			{
				tmpOverlap = (REAL_TYPE)(overlaps[i][j]) / (REAL_TYPE)(tps.nP);
				alpha = ( tmpOverlap - overlapRatio ) / overlapRatio;
				alpha /= 2.0;

				Sub_Vector<REAL_TYPE>( s[j].c.w , s[i].c.w , dW , nM );
				Scalar_Vector<REAL_TYPE>( dW , alpha , nM );
				Add_Vector<REAL_TYPE>( dW_Mat[j] , dW , dW_Mat[j] , nM );
				Scalar_Vector<REAL_TYPE>( dW , -1.0 , nM );
				Add_Vector<REAL_TYPE>( dW_Mat[i] , dW , dW_Mat[i] , nM );
			}
		}
		
		#pragma omp parallel for
		for(int i=0;i<BCODE_LEN;i++)
		{
			wConst = 1.0 / (REAL_TYPE)(BCODE_LEN);
			Scalar_Vector<REAL_TYPE>( dW_Mat[i] , wConst , nM );
			Add_Vector<REAL_TYPE>( s[i].c.w , dW_Mat[i] , s[i].c.w , nM );
		}
		#pragma omp parallel for
		for(int i=0;i<BCODE_LEN;i++)
		{
			SetRadius_Ratio( i , includingRatio );
		}
	}

	for(int i=0;i<BCODE_LEN;i++)
	{
		delete [] overlaps[i];
		delete [] dW_Mat[i];
	}
	delete [] overlaps;
	delete [] dW_Mat;
	delete [] dW;
	
	/*
	printf("\n ** Sphere Information ** \n" );
	for(int i=0;i<BCODE_LEN;i++)
	{
		printf("%3d\t%f\t%f\t%f\n",i,s[i].r,s[i].rSq,sqrt( ksh->distSQ_Residuals[i] ) );
	}
	*/
	
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::Compute_KernelMatrix_TPS()
{
	kM_TPS = new REAL_TYPE * [ nM ];
	for(int i=0;i<nM;i++)
	{
		kM_TPS[i] = new REAL_TYPE [ tps.nP ];
	}
	kM_Self = new REAL_TYPE [ tps.nP ];

	for(int i=0;i<nM;i++)
	{
		#pragma omp parallel for
		for(int k=0;k<tps.nP;k++)
		{
			kM_TPS[i][k] = kernel->KF( M.d[i] , tps.d[k] );
		}
	}
	#pragma omp parallel for
	for(int i=0;i<tps.nP;i++)
	{
		kM_Self[i] = kernel->KF( tps.d[i] , tps.d[i] );
	}
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::SetRadius_Ratio(int sIndex, REAL_TYPE ratio)
{
	for(int i=0;i<tps.nP;i++)
	{
		ids[sIndex][i].index = i;
		ids[sIndex][i].distSq = Compute_K_Center_TPS( sIndex , i );
		ids[sIndex][i].dist = ids[sIndex][i].distSq;
	}
	sort( &ids[sIndex][0] , &ids[sIndex][tps.nP] );
	int tIndex = (int)( (REAL_TYPE)( tps.nP ) * ratio );
	s[sIndex].rSq = ids[sIndex][tIndex-1].distSq;
	s[sIndex].r = s[sIndex].rSq;
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::Compute_Table()
{
	for(int i=0;i<BCODE_LEN;i++)
	{
		#pragma omp parallel for
		for(int j=0;j<tps.nP;j++)
		{
			table[i][j] = 0;
			if( Compute_K_Center_TPS( i , j ) > s[i].rSq )
			{
				table[i][j] = 1;
			}
		}
	}
}

template<int BCODE_LEN>
void Kernel_SphericalHashing<BCODE_LEN>::Compute_Overlaps(int **overlaps)
{
    for(int i=0;i<BCODE_LEN;i++)
	{
		overlaps[i][i] = table[i].count();
		#pragma omp parallel for
        for(int j=i+1;j<BCODE_LEN;j++)
		{
			overlaps[i][j] = ( table[i] & table[j] ).count();
			overlaps[j][i] = overlaps[i][j];
		}
	}
}


template<int BCODE_LEN>
REAL_TYPE Kernel_SphericalHashing<BCODE_LEN>::Compute_K_Center_TPS(int sIndex, int tpIndex)
{
	REAL_TYPE ret;
	ret = 0.0;
    for(int i=0;i<nM;i++)
	{
		ret += s[sIndex].c.w[i] * kM_TPS[i][tpIndex];
	}
	return ret;
}

template<int BCODE_LEN>
REAL_TYPE Kernel_SphericalHashing<BCODE_LEN>::Compute_Distance_SQ_From_Center_TPS(int sIndex, int tpIndex)
{
	REAL_TYPE ret, tmp;
	ret = kM_Self[tpIndex];
	tmp = 0.0;
    for(int i=0;i<nM;i++)
	{
		tmp += s[sIndex].c.w[i] * kM_TPS[i][tpIndex];
	}
    ret = ret - ( 2.0 * tmp ) + distSQ_Residuals[sIndex];
	return ret;
}
