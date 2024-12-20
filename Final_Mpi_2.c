#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <mpi.h>
#include <mpi-ext.h>
#include <signal.h>
using namespace std;

#define Max(a, b) ((a) > (b) ? (a) : (b))

#define N (2 * 2 * 2 * 2 * 2 * 2 + 2)
#define N2 (N * N)
#define N3 (N * N2)

double maxeps = 0.1e-7;
int itmax = 100;
double eps;
double s = 0., e;
int deadlock = 50;
int proc_to_kill = -1;
double A[N][N][N];
MPI_Comm communicator;

void relax();
void init();
void verify();
void save_state();
void load_state();
void error_handler(MPI_Comm *pcomm, int *error_code, ...);
void pr_m(double a[N][N][N]);
int ranksize, myrank;
int startrow, nrow, lastrow; 
int has_died = 0;
int it;

MPI_Request req_buf[6];
MPI_Status stat_buf[6];

MPI_Comm main_comm;
MPI_Errhandler errh;

int main(int argc, char **argv) {
    /* initialisation of MPI */
    MPI_Init(&argc, &argv);
	main_comm = MPI_COMM_WORLD;
    MPI_Comm_size(main_comm, &ranksize);
	MPI_Comm_rank(main_comm, &myrank);
	
	MPI_Comm_create_errhandler(error_handler, &errh);
    MPI_Comm_set_errhandler(main_comm, errh);
	
	if (argc == 3){
		char * iteration_num = argv[1];
		char * proc_num = argv[2];
		deadlock = strtol(iteration_num, &iteration_num, 10);
		proc_to_kill = strtol(proc_num, &proc_num, 10);
	}
	
	if (ranksize != 1) {
		ranksize -= 1;
	}
	
	if (myrank == 0){
		init();
	}
	
	MPI_Bcast(A, N3, MPI_DOUBLE, 0, main_comm);
    save_state();
    for (it = 0; it < itmax; ++it) {
		if (it != -1 && it && it % 10 == 0 && it != deadlock){
			if (!has_died){
				save_state();
			}
		}
        eps = 0.;
        relax();
		
		if (proc_to_kill != -1 && it == deadlock && myrank == proc_to_kill && !has_died) {
			raise(SIGKILL);
		}

        if (!myrank)
            printf("it=%4i   eps=%f\n", it, eps);
        if (eps < maxeps)
            break;
		MPI_Barrier(main_comm);
    }
	
    verify();

    MPI_Finalize();

    return 0;
}

    void init()
    {

        for(int k=0; k<=N-1; k++)
        for(int j=0; j<=N-1; j++)
        for(int i=0; i<=N-1; i++)
        {
            if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
            A[i][j][k]= 0.;
            else A[i][j][k]= ( 4. + i + j + k) ;
        }
    }

void relax() {
    double localEps = 0.;
	if (myrank != ranksize){
	for (int i = 1; i < N - 1; ++i) {
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			if (j >= N - 1) break;
			for (int k = 1; k < N - 1; ++k) {
				
				if (myrank > 0) {
                    MPI_Recv(&A[i][j - 1][k], 1, MPI_DOUBLE, myrank - 1, 0, main_comm, &stat_buf[0]);
                }
				
				if (!myrank && j != 1 && ranksize != 1) {
					MPI_Recv(&A[i][j - 1][k], 1, MPI_DOUBLE, ranksize - 1, 0, main_comm, &stat_buf[1]);
					
				}
				
				double oldVal = A[i][j][k];
                A[i][j][k] = (A[i - 1][j][k] + A[i + 1][j][k] + 
							  A[i][j - 1][k] + A[i][j + 1][k] +
                              A[i][j][k - 1] + A[i][j][k + 1]) / 6.0;
                localEps = Max(localEps, fabs(oldVal - A[i][j][k]));
				
				if (myrank < ranksize - 1 && (((N - 2) % ranksize == 0) || j != N - 2)) {
                    MPI_Send(&A[i][j][k], 1, MPI_DOUBLE, myrank + 1, 0, main_comm);
                }
				if (myrank == ranksize - 1 && myrank && j != N - 2) {
                    MPI_Send(&A[i][j][k], 1, MPI_DOUBLE, 0, 0, main_comm);
                }

			}
		}
	for (int j = myrank + 1; j < N - 1; j += ranksize) {
		if (j >= N - 1) { 
			break;
		}
		if (!myrank && ranksize > 1 && (((N - 2) % ranksize == 0) || j != N - 2)){
			MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 1, 0, main_comm);
		}
		if (!myrank && j != 1 && ranksize > 2) {
			MPI_Send(&A[i][j][0], N, MPI_DOUBLE, ranksize - 1, 0, main_comm);
		}
	}
	
	if ((myrank == ranksize - 1 || myrank == 1) && ranksize > 1) {
		
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			if (j >= N - 1) { 
				break;
			}
			
			if (myrank == 1) {
				MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, 0, 0, main_comm, &stat_buf[1]);
				
			} else if (j != N - 2 && ranksize > 2) {
				MPI_Recv(&A[i][j + 1][0], N, MPI_DOUBLE, 0, 0, main_comm, &stat_buf[0]);
			}
			
		}
		
	}
	if (myrank > 0) {
		
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			
			if (j >= N - 1) { 
				break;
			}
			if (myrank == 1 && ranksize > 2) {
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 2, 0, main_comm);
				
			} else if (myrank != ranksize - 1 && ranksize > 2) {
				if ((N - 2) % ranksize == 0 || j != N - 2){
				MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm, &stat_buf[1]);
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, myrank + 1, 0, main_comm);
				} else if (j == N - 2) {
					MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm, &stat_buf[1]);
				}
			} else if (ranksize > 2) {
				MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm, &stat_buf[0]);
			}
			
		}
	}
	
		
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			if (j >= N - 1) { 
				break;
			}
			if (myrank == ranksize - 1 && ranksize > 2) {
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm);
			} else if (myrank != 1 && myrank != 0 && ranksize > 2) {	
			if ((N - 2) % ranksize == 0 || j != N - 2){
				MPI_Recv(&A[i][j + 1][0], N, MPI_DOUBLE, myrank + 1, 0, main_comm, &stat_buf[1]);
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm);
			} else if (j == N - 2) {
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm);
			}
			} else if (myrank == 1 && ranksize > 2) {
				MPI_Recv(&A[i][j + 1][0], N, MPI_DOUBLE, myrank + 1, 0, main_comm, &stat_buf[0]);
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 0, 0, main_comm);		
			} else if (myrank == 1) {
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 0, 0, main_comm);
				
			} else if (!myrank && ranksize > 1 && j != N - 2) {
				MPI_Recv(&A[i][j + 1][0], N, MPI_DOUBLE, 1, 0, main_comm, &stat_buf[0]);
				
			}
		}
		
		if (myrank == ranksize - 1 && ranksize > 2) {
			for (int j = myrank + 1; j < N - 1; j += ranksize) {
				if (j >= N - 1) { 
					break;
				}
				if (j != N - 2) {
					MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 0, 0, main_comm);
					
				}
			}
		}
		
		if (myrank == 0 && ranksize > 2) {
			for (int j = myrank + 1; j < N - 1; j += ranksize) {
				if (j >= N - 1) { 
					break;
				}
				if (j != 1) {
					MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, ranksize - 1, 0, main_comm, &stat_buf[0]);
				}
			}
		}
	}
	}
    MPI_Allreduce(&localEps, &eps, 1 , MPI_DOUBLE, MPI_MAX, main_comm);
}

void pr_m(double a[N][N][N]){
	for (int i = 0; i < N; i++){
		for (int j = 0; j < N; j++){
			for (int k = 0; k < N; k++){
				std::cout.precision(2);
				std::cout << a[i][j][k] << " ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl << std::endl;
	}
}

void verify() {

    double sTmp = 0.;
	if (myrank != ranksize){
	for(int i = 0; i < N - 1; i++) 
		for (int j = myrank + 1; j < N - 1; j += ranksize){
			if (j >= N - 1)
				break;
			for(int k = 0; k < N - 1; k++){
                sTmp = sTmp + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N3);
            }
		}
	}
    MPI_Allreduce(&sTmp, &s, 1 , MPI_DOUBLE, MPI_SUM, main_comm);
	if (!myrank) {
        printf("  S = %f\n", s);
    }
}

void save_state() {
	MPI_File file; 
	MPI_File_open(main_comm, "respawn.txt", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &file); 
	int count = 0;

	if (myrank != ranksize){
	for (int i = 1; i < N - 1; ++i) {
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			if (j >= N - 1) break;
			MPI_Offset offset = ((i - 1) * (N - 2) + (j - 1)) * N * sizeof(double) + myrank * (N - 2) * (N - 2) * N * sizeof(double);
			MPI_File_write_at(file, offset, &A[i][j][0], N, MPI_DOUBLE, MPI_STATUS_IGNORE);
			count++;
		}
	}
	}
	//cout << "saved: " << myrank << " " << A[1][myrank + 1][0] << endl;
	 
	 MPI_File_close(&file);
 }

void error_handler(MPI_Comm *pcomm, int *error_code, ...) {    
	
	MPI_Comm comm = *pcomm;

    MPI_Comm_rank(comm, &myrank);
    MPI_Comm_size(comm, &ranksize);
	
	std::cout << "Proccess " << myrank << " is alive and went to error handler" << std::endl;
	
	MPIX_Comm_revoke(*pcomm);
	MPIX_Comm_shrink(*pcomm, &main_comm);
	MPI_Comm_set_errhandler(main_comm, errh);
	
	MPI_Comm_rank(main_comm,&myrank); 
	MPI_Comm_size(main_comm,&ranksize);
	
	has_died = 1;

	load_state();
	
	if (it != 0) {
		int minus = it % 10 == 0 ? 10 : it % 10;
		it -= minus + 1;
	} else {
		it = -1;
	}
 }

void load_state() {
	 MPI_File file; 
	 MPI_File_open(main_comm, "respawn.txt", MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
	int count = 0;
	for (int i = 1; i < N - 1; ++i) {
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			if (j >= N - 1) break;
			MPI_Offset offset = ((i - 1) * (N - 2) + (j - 1)) * N * sizeof(double) + myrank * (N - 2) * (N - 2) * N * sizeof(double);
			MPI_File_read_at(file, offset, &A[i][j][0], N, MPI_DOUBLE, MPI_STATUS_IGNORE);
			count++;
		}
	}

	for (int i = 1; i < N - 1; ++i) {
	for (int j = myrank + 1; j < N - 1; j += ranksize) {
		if (j >= N - 1) { 
			break;
		}
		if (!myrank && ranksize > 1 && (((N - 2) % ranksize == 0) || j != N - 2)){
			MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 1, 0, main_comm);
		}
		if (!myrank && j != 1 && ranksize > 2) {
			MPI_Send(&A[i][j][0], N, MPI_DOUBLE, ranksize - 1, 0, main_comm);
		}
	}
	
	if ((myrank == ranksize - 1 || myrank == 1) && ranksize > 1) {
		
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			if (j >= N - 1) { 
				break;
			}
			
			if (myrank == 1) {
				MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, 0, 0, main_comm, &stat_buf[1]);
			} else if (j != N - 2 && ranksize > 2) {
				MPI_Recv(&A[i][j + 1][0], N, MPI_DOUBLE, 0, 0, main_comm, &stat_buf[0]);
			}
			
		}
		
	}
	if (myrank > 0) {
		
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			
			if (j >= N - 1) { 
				break;
			}
			if (myrank == 1 && ranksize > 2) {
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 2, 0, main_comm);
				
			} else if (myrank != ranksize - 1 && ranksize > 2) {
				if ((N - 2) % ranksize == 0 || j != N - 2){
				MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm, &stat_buf[1]);
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, myrank + 1, 0, main_comm);
				} else if (j == N - 2) {
					MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm, &stat_buf[1]);
				}
			} else if (ranksize > 2) {
				MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm, &stat_buf[0]);
			}
			
		}
	}
	
		
		for (int j = myrank + 1; j < N - 1; j += ranksize) {
			if (j >= N - 1) { 
				break;
			}
			if (myrank == ranksize - 1 && ranksize > 2) {
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm);
			} else if (myrank != 1 && myrank != 0 && ranksize > 2) {	
			if ((N - 2) % ranksize == 0 || j != N - 2){
				MPI_Recv(&A[i][j + 1][0], N, MPI_DOUBLE, myrank + 1, 0, main_comm, &stat_buf[1]);
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm);
			} else if (j == N - 2) {
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, myrank - 1, 0, main_comm);
			}
			} else if (myrank == 1 && ranksize > 2) {
				MPI_Recv(&A[i][j + 1][0], N, MPI_DOUBLE, myrank + 1, 0, main_comm, &stat_buf[0]);
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 0, 0, main_comm);		
			} else if (myrank == 1) {
				MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 0, 0, main_comm);
				
			} else if (!myrank && ranksize > 1 && j != N - 2) {
				MPI_Recv(&A[i][j + 1][0], N, MPI_DOUBLE, 1, 0, main_comm, &stat_buf[0]);
				
			}
		}
		
		if (myrank == ranksize - 1 && ranksize > 2) {
			for (int j = myrank + 1; j < N - 1; j += ranksize) {
				if (j >= N - 1) { 
					break;
				}
				if (j != N - 2) {
					MPI_Send(&A[i][j][0], N, MPI_DOUBLE, 0, 0, main_comm);
					
				}
			}
		}
		
		if (myrank == 0 && ranksize > 2) {
			for (int j = myrank + 1; j < N - 1; j += ranksize) {
				if (j >= N - 1) { 
					break;
				}
				if (j != 1) {
					MPI_Recv(&A[i][j - 1][0], N, MPI_DOUBLE, ranksize - 1, 0, main_comm, &stat_buf[0]);
				}
			}
		}	
	}
	 MPI_File_close(&file);
 }
