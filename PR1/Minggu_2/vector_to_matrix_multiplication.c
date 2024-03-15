#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <time.h>

#define N 2000  /* number of rows and columns in matrix */

int main(int argc, char **argv) {
    int numtasks, taskid, numworkers, source, dest, rows, offset, i, j;
    double *a, *b, *c;
    struct timeval start_total, stop_total, start_comm, stop_comm, start_comp, stop_comp;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    numworkers = numtasks - 1;

    if (taskid == 0) {
        a = (double *)malloc(N * N * sizeof(double));
        b = (double *)malloc(N * sizeof(double));
        c = (double *)malloc(N * sizeof(double));

        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                a[i * N + j] = 1.0;
            }
            b[i] = 2.0;
        }

        gettimeofday(&start_total, NULL);
        gettimeofday(&start_comm, NULL);

        /* send matrix data to the worker tasks */
        rows = N / numworkers;
        offset = 0;

        for (dest = 1; dest <= numworkers; dest++) {
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&rows, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&a[offset * N], rows * N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            MPI_Send(b, N, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);
            offset += rows;
        }

        gettimeofday(&stop_comm, NULL);

        /* wait for results from all worker tasks */
        for (i = 1; i <= numworkers; i++) {
            source = i;
            MPI_Recv(&offset, 1, MPI_INT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&rows, 1, MPI_INT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&c[offset], rows, MPI_DOUBLE, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        gettimeofday(&stop_total, NULL);

        double comm_time = (stop_comm.tv_sec + stop_comm.tv_usec * 1e-6) - (start_comm.tv_sec + start_comm.tv_usec * 1e-6);
        double total_time = (stop_total.tv_sec + stop_total.tv_usec * 1e-6) - (start_total.tv_sec + start_total.tv_usec * 1e-6);
        double comp_time = total_time - comm_time;

        printf("Communication Time: %.6f seconds\n", comm_time);
        printf("Computation Time: %.6f seconds\n", comp_time);
        printf("Total Time: %.6f seconds\n", total_time);

        free(a);
        free(b);
        free(c);
    }

    if (taskid > 0) {
        MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&rows, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        a = (double *)malloc(rows * N * sizeof(double));
        b = (double *)malloc(N * sizeof(double));
        c = (double *)malloc(rows * sizeof(double));

        MPI_Recv(a, rows * N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(b, N, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        gettimeofday(&start_comp, NULL);

        /* Matrix-vector multiplication */
        for (i = 0; i < rows; i++) {
            c[i] = 0.0;
            for (j = 0; j < N; j++) {
                c[i] += a[i * N + j] * b[j];
            }
        }

        gettimeofday(&stop_comp, NULL);

        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&rows, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(c, rows, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);

        free(a);
        free(b);
        free(c);
    }

    MPI_Finalize();
    return 0;
}
