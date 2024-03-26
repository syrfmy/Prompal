#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char* argv[]) {
    int rank, num_of_process;
    char prompt[100];
    MPI_Init(&argc, &argv);
    MPI_Comm parentcomm, intercomm;
    MPI_Comm_get_parent(&parentcomm);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (parentcomm == MPI_COMM_NULL) {

        if(rank == 0) {
        sprintf(prompt, "How many processes do you want to start?");
        puts(prompt);
        scanf("%d", &num_of_process);
        } 
        MPI_Comm_spawn(argv[0], MPI_ARGV_NULL, num_of_process, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &intercomm, MPI_ERRCODES_IGNORE);
        // send message to all child
        for(int i = 0; i < num_of_process; i++) {
            MPI_Send( &i, 1, MPI_INT, i, 0, intercomm);
        }
        MPI_Barrier(intercomm);
        int temp;
        
        //recieve message from all child
        for(int i = 0; i < num_of_process; i++) {
            MPI_Recv(&temp, 1, MPI_INT, i, 0, intercomm, MPI_STATUS_IGNORE);
            printf("Recieve random number %d from the %d child process\n", temp, i); 
        }
        MPI_Barrier(intercomm);

    
        printf("Rank %d from staticly initialized process\n", rank);     
    }else {
        int random_number;
        srand((unsigned)time(NULL)+rank*1000);
        
        MPI_Recv(&num_of_process, 1, MPI_INT, 0, 0, parentcomm, MPI_STATUS_IGNORE);
        printf("You are the %d child the parent process\n", num_of_process);
        MPI_Barrier(parentcomm);
        //generate random number different for each child process

        random_number =  rand() % 100;  
        MPI_Send(&random_number, 1, MPI_INT, 0, 0, parentcomm);
        MPI_Barrier(parentcomm);
       
    }
    
    MPI_Finalize();

    return 0;
}