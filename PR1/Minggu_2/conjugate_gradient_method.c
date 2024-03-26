#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define TOLERANCE_THRESHOLD 1e-6

typedef struct {
    int size;
    double **data;
} Matrix;

typedef struct {
    int size;
    double *data;
} Vector;

void readData(FILE *file, int *ARR_NUM, Matrix **matrices, Vector **b, Vector **x_0) {
    char line[1000000];
    // assuming the format is always "M=number"
    fgets(line, sizeof(line), file);
    sscanf(line, "M=%d\n", ARR_NUM);
    *matrices = (Matrix *)malloc(*ARR_NUM * sizeof(Matrix));
    *b = (Vector *)malloc(*ARR_NUM * sizeof(Vector));
    *x_0 = (Vector *)malloc(*ARR_NUM * sizeof(Vector));

    for (int i = 0; i < *ARR_NUM; i++) {
        int size;
        fgets(line, sizeof(line), file);
        sscanf(line, "N=%d\n", &size);

        // Read matrix
        (*matrices)[i].size = size;
        (*matrices)[i].data = (double **)malloc(size * sizeof(double *));
        for (int j = 0; j < size; j++) {
            (*matrices)[i].data[j] = (double *)malloc(size * sizeof(double));
            fgets(line, sizeof(line), file);
            char *token = strtok(line, " ");
            for (int k = 0; k < size; k++) {
                if (token != NULL) {
                    sscanf(token, "%lf", &(*matrices)[i].data[j][k]);
                    token = strtok(NULL, " \t\n");
                } else {
                    printf("Error: Incomplete row in the matrix.\n");
                    exit(1);
                }
            }
        }

        // Read vectors
        (*b)[i].size = size;
        (*b)[i].data = (double *)malloc(size * sizeof(double));
        fgets(line, sizeof(line), file);
        char *token = strtok(line, " \t\n");
        for (int j = 0; j < size; j++) {
            if (token != NULL) {
                sscanf(token, "%lf", &(*b)[i].data[j]);
                token = strtok(NULL, " \t\n");
            } else {
                printf("Error: Incomplete vector.\n");
                exit(1);
            }
        }

        (*x_0)[i].size = size;
        (*x_0)[i].data = (double *)malloc(size * sizeof(double));
        fgets(line, sizeof(line), file);
        token = strtok(line, " \t\n");
        for (int j = 0; j < size; j++) {
            if (token != NULL) {
                sscanf(token, "%lf", &(*x_0)[i].data[j]);
                token = strtok(NULL, " \t\n");
            } else {
                printf("Error: Incomplete vector.\n");
                exit(1);
            }
        }
    }
}

void printData(int ARR_NUM, Matrix *matrices, Vector *b, Vector *x_0) {
    for (int i = 0; i < ARR_NUM; i++) {
        printf("Matrix %d:\n", i + 1);
        for (int row = 0; row < matrices[i].size; row++) {
            for (int col = 0; col < matrices[i].size; col++) {
                printf("%lf ", matrices[i].data[row][col]);
            }
            printf("\n");
        }
        printf("\n");

        printf("Vector b (target):\n");
        for (int j = 0; j < b[i].size; j++) {
            printf("%lf ", b[i].data[j]);
        }
        printf("\n");

        printf("Vector x_0 (initial guess):\n");
        for (int j = 0; j < x_0[i].size; j++) {
            printf("%lf ", x_0[i].data[j]);
        }
        printf("\n\n");
    }
}

int main(int argc, char* argv[]) {
    int ARR_NUM, rank, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    FILE *file = fopen("input.txt", "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        return 1;
    }

    Matrix *matrices;
    Vector *b;
    Vector *x_0;

    readData(file, &ARR_NUM, &matrices, &b, &x_0);
    //printData(ARR_NUM, matrices, b, x_0);

    // implement conjugate gradient method
    int n, local_n;
    double *x, *x_s,*r, *p, *Ap, *temp, *Ap_local, *p_local;
    double alpha, beta, r_dot_r, r_dot_r_new, p_dot_Ap, p_dot_Ap_local, r_dot_r_local, r_dot_r_new_local;
    int max_iter =1000;

    // Parallel implementation
    // calculate time
    double computation_time,communication_time;
    double total_computation_time = 0;
    double total_communication_time = 0;
    for(int i=0; i < ARR_NUM; i++){

        

        // Initialize variables
        n = matrices[i].size;
        local_n = n / comm_size;
        x = (double *)malloc(n * sizeof(double));
        r = (double *)malloc(n * sizeof(double));
        p = (double *)malloc(n * sizeof(double));
        Ap = (double *)malloc(n * sizeof(double));
        temp = (double *)malloc(n * sizeof(double));
        Ap_local = (double *)malloc(local_n * sizeof(double));
        p_local = (double *)malloc(local_n * sizeof(double));
        int start = rank * local_n;

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            computation_time = MPI_Wtime();
        }
        
        // initialize x, r, p
        for (int j = 0; j < n; j++) {
            x[j] = x_0[i].data[j];
        }

        // A * x
        for (int j = start; j < start+local_n; j++) {
            temp[j] = 0;
            for (int k = 0; k < n; k++) {
                temp[j] += matrices[i].data[j][k] * x[k];
            }
        }

        // r = b - A * x and p = r
        for (int j = start; j < start+local_n; j++) {
            r[j] = b[i].data[j] - temp[j];
            p[j] = r[j];  
        }
        if(rank == 0){
            total_computation_time += MPI_Wtime() - computation_time;
        }
        

        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0){
            communication_time = MPI_Wtime();
        }
        MPI_Allgather(p+start, local_n, MPI_DOUBLE, p, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
        if(rank == 0){
            total_communication_time += MPI_Wtime() - communication_time;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            computation_time = MPI_Wtime();
        }
        r_dot_r = 0;
        for (int j = start; j < start+local_n; j++) {
            r_dot_r += r[j] * r[j];
        }

        if(rank == 0){
            total_computation_time += MPI_Wtime() - computation_time;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        if(rank == 0){
            communication_time = MPI_Wtime();
        }
        MPI_Allreduce(&r_dot_r, &r_dot_r, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if(rank == 0){
            total_communication_time += MPI_Wtime() - communication_time;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        

        int iter = 0;
        while(r_dot_r > TOLERANCE_THRESHOLD && iter < max_iter){
            if(rank == 0){
                computation_time = MPI_Wtime();
            }
            // Ap
            for (int j = start; j < start+local_n; j++) {
                Ap[j] = 0;
                for (int k = 0; k < n; k++) {
                    Ap[j] += matrices[i].data[j][k] * p[k];
                }
            }
            if(rank == 0){
                total_computation_time += MPI_Wtime() - computation_time;
            }

            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0){
                communication_time = MPI_Wtime();
            }
            MPI_Allgather(Ap+start, local_n, MPI_DOUBLE, Ap, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
            if(rank == 0){
                total_communication_time += MPI_Wtime() - communication_time;
            }
            MPI_Barrier(MPI_COMM_WORLD);

            if(rank == 0){
                computation_time = MPI_Wtime();
            }
            // p_dot_Ap
            p_dot_Ap = 0;
            for (int j = start; j < start+local_n; j++) {
                p_dot_Ap += p[j] * Ap[j];
            }
            if(rank == 0){
                total_computation_time += MPI_Wtime() - computation_time;
            }

            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0){
                communication_time = MPI_Wtime();
            }
            MPI_Allreduce(&p_dot_Ap, &p_dot_Ap, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            if(rank == 0){
                total_communication_time += MPI_Wtime() - communication_time;
            }
            MPI_Barrier(MPI_COMM_WORLD);

            if(rank == 0){
                computation_time = MPI_Wtime();
            }
            // alpha
            alpha = r_dot_r / p_dot_Ap;
            
            // update x, r
            for (int j = start; j < start+local_n; j++) {
                x[j] += alpha * p[j];
                r[j] -= alpha * Ap[j];
            }

            // r_dot_r_new
            r_dot_r_new = 0;
            for (int j = start; j < start+local_n; j++) {
                r_dot_r_new += r[j] * r[j];
            }

            if(rank == 0){
                total_computation_time += MPI_Wtime() - computation_time;
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0){
                communication_time = MPI_Wtime();
            }
            MPI_Allreduce(&r_dot_r_new, &r_dot_r_new, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            if(rank == 0){
                total_communication_time += MPI_Wtime() - communication_time;
            }
            MPI_Barrier(MPI_COMM_WORLD);

            if(rank == 0){
                computation_time = MPI_Wtime();
            }
            // calculate beta
            beta = r_dot_r_new / r_dot_r;

            // update p
            for (int j = start; j < start+local_n; j++) {
                p[j] = r[j] + beta * p[j];
            }
            if(rank == 0){
                total_computation_time += MPI_Wtime() - computation_time;
            }

            MPI_Barrier(MPI_COMM_WORLD);
            if(rank == 0){
                communication_time = MPI_Wtime();
            }
            MPI_Allgather(p+start, local_n, MPI_DOUBLE, p, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
            if(rank == 0){
                total_communication_time += MPI_Wtime() - communication_time;
            }
            MPI_Barrier(MPI_COMM_WORLD);

            // update r_dot_r
            r_dot_r = r_dot_r_new;
            iter++;
             
        }
        

        
        // End of computation
        MPI_Allgather(x+start, local_n, MPI_DOUBLE, x, local_n, MPI_DOUBLE, MPI_COMM_WORLD);
        if(rank == 0){
            // compute the x minimum but serially
            x_s = (double *)malloc(n * sizeof(double));
            
            for (int j = 0; j < n; j++) {
                x_s[j] =x_0[i].data[j];
            }

            for (int j = 0; j < n; j++) {
                temp[j] = 0;
                for (int k = 0; k < n; k++) {
                    temp[j] += matrices[i].data[j][k] * x_s[k];
                }
            }

            for (int j = 0; j < n; j++) {
                r[j] = b[i].data[j] - temp[j];
                p[j] = r[j];
            }

            r_dot_r = 0;
            for (int j = 0; j < n; j++) {
                r_dot_r += r[j] * r[j];
            }

            iter = 0;
            while (r_dot_r > TOLERANCE_THRESHOLD && iter < max_iter) {
                // calculate Ap
                for (int j = 0; j < n; j++) {
                    Ap[j] = 0;
                    for (int k = 0; k < n; k++) {
                        Ap[j] += matrices[i].data[j][k] * p[k];
                    }
                }

                // calculate alpha
                p_dot_Ap = 0;
                for (int j = 0; j < n; j++) {
                    p_dot_Ap += p[j] * Ap[j];
                }
                alpha = r_dot_r / p_dot_Ap;

                // update x, r
                for (int j = 0; j < n; j++) {
                    x_s[j] += alpha * p[j];
                    r[j] -= alpha * Ap[j];
                }

                // calculate r_dot_r_new
                r_dot_r_new = 0;
                for (int j = 0; j < n; j++) {
                    r_dot_r_new += r[j] * r[j];
                }

                // calculate beta
                beta = r_dot_r_new / r_dot_r;

                // update p
                for (int j = 0; j < n; j++) {
                    p[j] = r[j] + beta * p[j];
                }

                // update r_dot_r
                r_dot_r = r_dot_r_new;

                iter++;
            }
            // compare x and x_s
            int check= 0;
            for (int j = 0; j < n; j++) {
                if (x[j]-x_s[j]>0.00000001) {
                    check = 1;
                    break;
                }
                
            }
            printf("Matrix N=%d\n", matrices[i].size);
            printf("Total computation time: %lf\n", total_computation_time);
            printf("Total communication time: %lf\n", total_communication_time);
            printf("error: %lf\n", r_dot_r);
            printf("Result is correct?: %s\n", check==0 ? "true" : "false");
            printf("Number of iteration: %d", iter);
            printf("\n\n");
        }

        free(x);
        free(r);
        free(p);

    }
    // Free allocated memory
    for (int i = 0; i < ARR_NUM; i++) {
        for (int j = 0; j < matrices[i].size; j++) {
            free(matrices[i].data[j]);
        }
        free(matrices[i].data);
        free(b[i].data);
        free(x_0[i].data);
    }
    free(matrices);
    free(b);
    free(x_0);

    fclose(file);
    MPI_Finalize();
    return 0;
}
