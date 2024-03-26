#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>

#define TOLERANCE_THRESHOLD 1e-6

void readData(FILE *file, int *ARR_NUM, double ****arr_matrices, int **arr_matrices_size, double ***arr_b, double ***arr_x_0) {
    char line[1000000];
    char* token;
    // assuming the format is always "M=number"
    fgets(line, sizeof(line), file);
    sscanf(line, "M=%d\n", ARR_NUM);

    // allocate memory for arrays
    *arr_matrices = (double ***)malloc(*ARR_NUM * sizeof(double **));
    *arr_matrices_size = (int *)malloc(*ARR_NUM * sizeof(int));
    *arr_b = (double **)malloc(*ARR_NUM * sizeof(double *));
    *arr_x_0 = (double **)malloc(*ARR_NUM * sizeof(double *));


    // read data
    for (int i = 0; i < *ARR_NUM; i++) {
        int size;
        fgets(line, sizeof(line), file);
        sscanf(line, "N=%d\n", &size);

        //read matrices
        (*arr_matrices)[i] = (double **)malloc(size * sizeof(double *));
        (*arr_matrices_size)[i] = size;
        for (int j = 0; j < size; j++) {
            (*arr_matrices)[i][j] = (double *)malloc(size * sizeof(double));
            fgets(line, sizeof(line), file);
            token = strtok(line, " \t\n");
            for (int k = 0; k < size; k++) {
                if (token != NULL) {
                    sscanf(token, "%lf", &(*arr_matrices)[i][j][k]);
                    token = strtok(NULL, " \t\n");
                } else {
                    printf("Error: Incomplete matrix.\n");
                    exit(1);
                }
            }
        }

        // Read b
        (*arr_b)[i] = (double *)malloc(size * sizeof(double));
        fgets(line, sizeof(line), file);
        token = strtok(line, " \t\n");
        for (int j = 0; j < size; j++) {
            if (token != NULL) {
                sscanf(token, "%lf", &(*arr_b)[i][j]);
                token = strtok(NULL, " \t\n");
            } else {
                printf("Error: Incomplete vector.\n");
                exit(1);
            }
        }
        
        // Read x_0
        (*arr_x_0)[i] = (double *)malloc(size * sizeof(double));
        fgets(line, sizeof(line), file);
        token = strtok(line, " \t\n");
        for (int j = 0; j < size; j++) {
            if (token != NULL) {
                sscanf(token, "%lf", &(*arr_x_0)[i][j]);
                token = strtok(NULL, " \t\n");
            } else {
                printf("Error: Incomplete vector.\n");
                exit(1);
            }
        }
    }
}


int main(int argc, char* argv[]) {
    int ARR_NUM,rank, comm_size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    
    int* arr_matrices_size;
    double*** arr_matrices;
    double** arr_b;
    double** arr_x_0;

    if (rank == 0) {
        FILE *file = fopen("input.txt", "r");
        if (file == NULL) {
            printf("Error: File not found.\n");
            exit(1);
        }
        readData(file, &ARR_NUM, &arr_matrices, &arr_matrices_size, &arr_b, &arr_x_0);
        fclose(file);
    }
    //broadcast ARR_NUM
    MPI_Bcast(&ARR_NUM, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank != 0){
        arr_matrices_size = (int *)malloc(ARR_NUM * sizeof(int));
        arr_matrices = (double ***)malloc(ARR_NUM * sizeof(double **));
        arr_b = (double **)malloc(ARR_NUM * sizeof(double *));
        arr_x_0 = (double **)malloc(ARR_NUM * sizeof(double *));
    }

    //broadcast arr_matrices_size
    MPI_Bcast(arr_matrices_size, ARR_NUM, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank != 0){
        for(int i = 0; i < ARR_NUM; i++){
            arr_matrices[i] = (double **)malloc(arr_matrices_size[i] * sizeof(double *));
            for(int j = 0; j < arr_matrices_size[i]; j++){
                arr_matrices[i][j] = (double *)malloc(arr_matrices_size[i] * sizeof(double));
            }
            arr_b[i] = (double *)malloc(arr_matrices_size[i] * sizeof(double));
            arr_x_0[i] = (double *)malloc(arr_matrices_size[i] * sizeof(double));
        }
    }

    //broadcast data
    for(int i = 0; i < ARR_NUM; i++){
        for(int j=0; j < arr_matrices_size[i]; j++){
            MPI_Bcast(arr_matrices[i][j], arr_matrices_size[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        MPI_Bcast(arr_b[i], arr_matrices_size[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(arr_x_0[i], arr_matrices_size[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

  
    
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
    for(int i=0; i < 5; i++){

        

        // Initialize variables
        n = arr_matrices_size[i] ;
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
            x[j] = arr_x_0[i][j];
        }

        // A * x
        for (int j = start; j < start+local_n; j++) {
            temp[j] = 0;
            for (int k = 0; k < n; k++) {
                temp[j] += arr_matrices[i][j][k] * x[k];
            }
        }

        // r = b - A * x and p = r
        for (int j = start; j < start+local_n; j++) {
            r[j] = arr_b[i][j] - temp[j];
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
                    Ap[j] += arr_matrices[i][j][k] * p[k];
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
                x_s[j] =arr_x_0[i][j];
            }

            for (int j = 0; j < n; j++) {
                temp[j] = 0;
                for (int k = 0; k < n; k++) {
                    temp[j] += arr_matrices[i][j][k] * x_s[k];
                }
            }

            for (int j = 0; j < n; j++) {
                r[j] = arr_b[i][j] - temp[j];
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
                        Ap[j] += arr_matrices[i][j][k] * p[k];
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
            printf("Matrix N=%d\n", arr_matrices_size[i]);
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
    MPI_Finalize();
    return 0;
}
