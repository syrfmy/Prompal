#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

void merge(int arr[], int l, int m, int r) {
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r - m;

    int L[n1], R[n2];

    for (i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    i = 0;
    j = 0;
    k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void insertion_sort(int arr[], int start, int end) {
    int i, key, j;
    for (i = start + 1; i <= end; i++) {
        key = arr[i];
        j = i - 1;

        while (j >= start && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

int main(int argc, char *argv[]) {
    int rank, size, temp, n, parent_rank;
    int *arr;
    int left, right, mid;
    int perm_tree_height = atoi(argv[1]);
    int tree_height = atoi(argv[1]);
    char tree_height_str[10];
    MPI_Comm arr_of_child_comm[atoi(argv[1])];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm parent_comm;
    MPI_Comm_get_parent(&parent_comm);
    
    if (parent_comm == MPI_COMM_NULL) {
        parent_rank=-1;
        n = atoi(argv[2]);
        arr = (int *)malloc(n * sizeof(int));
        //initialize the array with random number
        srand((unsigned)time(NULL) + getpid());
        printf("Original array: ");
        for (int i = 0; i < n; i++) {
            arr[i] = rand() % 100;
            printf("%d ", arr[i]);
        }
        printf("\n");
        printf("Process id is define as (x,y) where x is the height of the parent node and y is the child node\n\n");
        printf("Spawn     (-1,%d) - Master Process\n", perm_tree_height);
    }else{
        // get parent parent tree height
        MPI_Recv(&parent_rank, 1, MPI_INT, 0, 0, parent_comm, MPI_STATUS_IGNORE);
        printf("Spawn     (%d,%d)\n", parent_rank, perm_tree_height);
        //recieve the size of the array from the parent
        MPI_Recv(&n, 1, MPI_INT, 0, 0, parent_comm, MPI_STATUS_IGNORE);
        arr = (int *)malloc(n * sizeof(int));
        //recieve the portion of the array from the parent
        MPI_Recv(arr, n, MPI_INT, 0, 0, parent_comm, MPI_STATUS_IGNORE);
    }

    while(tree_height > 0) {
        tree_height -=1;
        sprintf(tree_height_str, "%d", tree_height);
        char * child_argv[2] = {tree_height_str, NULL};
        MPI_Comm child_comm;
        MPI_Comm_spawn(argv[0], child_argv, 1, MPI_INFO_NULL, 0, MPI_COMM_WORLD, &child_comm, MPI_ERRCODES_IGNORE);
        arr_of_child_comm[tree_height] = child_comm;
        
        // calculate the portion of the array to be sent to the child
        left = (n / 2);
        right = n-1;
        temp = right - left + 1;

        // recalculate the portion of the array of the parent
        n = (n / 2);

        // send the parent tree height to the child
        MPI_Send(&perm_tree_height, 1, MPI_INT, 0, 0, child_comm);
        // send the size of the array to the child
        MPI_Send(&temp, 1, MPI_INT, 0, 0, child_comm);
        // send the portion of the array to the child
        MPI_Send(&arr[left], temp, MPI_INT, 0, 0, child_comm);
    }

    //sort the portion of the array
    insertion_sort(arr, 0, n-1);

    //Wait for all children first
    for (int i =0; i<perm_tree_height; i++) {
        MPI_Barrier(arr_of_child_comm[i]);
    }
    
    //print the sorted portion of the array for each of the process
    printf("Sort in   (%d,%d): ",parent_rank ,perm_tree_height);
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    
    //unblock the parent process
    if(parent_comm != MPI_COMM_NULL) {
        MPI_Barrier(parent_comm);
    }

    for(int i=0; i<perm_tree_height; i++) {
        //recieve the size of the array from the child
        MPI_Recv(&temp, 1, MPI_INT, 0, 0, arr_of_child_comm[i], MPI_STATUS_IGNORE);
        //recieve the sorted portion of the array from the child
        MPI_Recv(&arr[n], temp, MPI_INT, 0, 0, arr_of_child_comm[i], MPI_STATUS_IGNORE);
        //merge the sorted portion of the array
        merge(arr, 0, n-1, n+temp-1);
        //recalculate the size of the array
        n += temp;
        //print the merged portion of the array for each of the process
        printf("Merged in (%d,%d): ", parent_rank, perm_tree_height);
        for (int i = 0; i < n; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");

        //wait for children first
        MPI_Barrier(arr_of_child_comm[i]);
        //free the child comm
        MPI_Comm_free(&arr_of_child_comm[i]);
    }
    if(parent_comm != MPI_COMM_NULL) {
        //send the size of the array to the parent
        MPI_Send(&n, 1, MPI_INT, 0, 0, parent_comm);
        //send the sorted portion of the array to the parent
        MPI_Send(arr, n, MPI_INT, 0, 0, parent_comm);
        //block the parent process
        MPI_Barrier(parent_comm);
    }

    //print the sorted array
    if(parent_comm == MPI_COMM_NULL) {
        printf("\n");
        printf("Sorted array: ");
        for (int i = 0; i < n; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }
    MPI_Finalize();
    return 0;
}
