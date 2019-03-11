#include "SphericalHashing.h"
#include "SphericalHashing.cpp"
#include "Evaluation.h"

#include <string>

#ifdef USE_PARALLELIZATION
#include <omp.h>
#endif

// dps: data points set
// qps: query points set
Points dps, qps;

// nP: number of data points
// nQ: number of query points

int nP, nQ;



REAL_TYPE *dataCenter;

// initialize data and query points
void Initialize_Data(char *_dps_path, char *_qps_path)
{
	dps.Initialize_From_File( _dps_path );
	qps.Initialize_From_File( _qps_path );
	nP = dps.nP;		nQ = qps.nP;

	// you can control the number of queries here
	nQ = 100;
	qps.nP = nQ;

	dataCenter = new REAL_TYPE[ dps.dim ];
	// compute mean position of data points
	dps.Compute_Center( dataCenter );
}



template<int BCODE_LEN>
void Process()
{

	Stopwatch T0("");
	SphericalHashing<BCODE_LEN> sh;
	sh.Initialize( &dps );
	T0.Reset();		T0.Start();
	sh.Set_Spheres();
	T0.Stop();
	printf("- Learning Spherical Hashing Finished (%f seconds)\n",T0.GetTime());

	bitset<BCODE_LEN> *bCodeData_SH = new bitset<BCODE_LEN> [ nP ];
	bitset<BCODE_LEN> *bCodeQuery_SH = new bitset<BCODE_LEN> [ nQ ];

	Result_Element<int> *resSH_HD = new Result_Element<int> [ nP ];
	Result_Element<double> *resSH_SHD = new Result_Element<double> [ nP ];

	T0.Reset();		T0.Start();
	// compute binary codes of Spherical Hashing
#ifdef USE_PARALLELIZATION
	#pragma omp parallel for
#endif
	for(int i=0;i<nP;i++)
	{
		sh.Compute_BCode( dps.d[i] , bCodeData_SH[i] );
	}

#ifdef USE_PARALLELIZATION
	#pragma omp parallel for
#endif
	for(int i=0;i<nQ;i++)
	{
		sh.Compute_BCode( qps.d[i] , bCodeQuery_SH[i] );
	}
	T0.Stop();
	printf("- Spherical Hashing: Computing Binary Codes Finished (%f seconds)\n",T0.GetTime() );

	FILE *sh_query_output = fopen("./SphericalHashing_query.out", "w");
	fprintf(sh_query_output, "%d %d\n", nQ, BCODE_LEN);
	for(int i=0; i<nQ; i++)
	{
		std::string bcodequery = bCodeQuery_SH[i].to_string();
		fprintf(sh_query_output, "%s\n", bcodequery.data());
	}
	fclose(sh_query_output);

	FILE *sh_data_output = fopen("./SphericalHashing_data.out", "w");
	fprintf(sh_data_output, "%d %d\n", nP, BCODE_LEN);
	for(int i=0; i<nP; i++)
	{
		std::string bcodedata = bCodeData_SH[i].to_string();
		fprintf(sh_data_output, "%s\n", bcodedata.data());
	}
	fclose(sh_data_output);


}

void Process_Wrapper(int _bcode_len)
{
	if( _bcode_len == 8 ) { Process<8>(); }
	if( _bcode_len == 16 ) { Process<16>(); }
	if( _bcode_len == 32 ) { Process<32>(); }
	if( _bcode_len == 64 ) { Process<64>(); }
	if( _bcode_len == 128 ) { Process<128>(); }
	if( _bcode_len == 256 ) { Process<256>(); }
	if( _bcode_len == 512 ) { Process<512>(); }
}

int main(int argc, char **argv)
{
	if( argc != 4 )
	{
		fprintf( stderr , "USAGE: EXE  DATA_FILE_NAME  QUERY_FILE_NAME  CODE_LEN\n" );
		exit(1);
	}

    printf("------------------------------------------------------------\n");
    printf("Tester for Spherical Hashing with Euclidean Distance\n" );
    printf("\tData  : %s\n" , argv[1] );
    printf("\tQuery : %s\n" , argv[2] );
    //printf("\tK     : %d\n" , atoi( argv[3] ) );
    printf("\tC-LEN : %d\n" , atoi( argv[3] ) );
    printf("\n");

	srand( (unsigned int)( time(NULL) ) );

	Stopwatch T0("");
	T0.Reset();		T0.Start();
	Initialize_Data( argv[1] , argv[2] );
	T0.Stop();
	printf("- Reading Data Finished (%f seconds)\n",T0.GetTime() );

	
	
	int bcode_len = atoi( argv[3] );

	Process_Wrapper( bcode_len );

	return 1;
}
