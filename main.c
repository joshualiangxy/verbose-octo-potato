#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include "tasks.h"
#include "utils.h"

void master()
{

}

void map_worker(int rank)
{

}

void reduce_worker(int rank, MapTaskOutput* map(char*))
{
    map("poggers");
}

int main(int argc, char** argv)
{
    // Get command-line params
    char* input_files_dir = argv[1];
    int num_files = atoi(argv[2]);
    int num_map_workers = atoi(argv[3]);
    int num_reduce_workers = atoi(argv[4]);
    char* output_file_name = argv[5];
    int map_reduce_task_num = atoi(argv[6]);

    MPI_Init(&argc, &argv);

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Identify the specific map function to use
    MapTaskOutput* (*map) (char*);
    switch (map_reduce_task_num) {
        case 1:
            map = &map1;
            break;
        case 2:
            map = &map2;
            break;
        case 3:
            map = &map3;
            break;
    }

    // Distinguish between master, map workers and reduce workers
    if (rank == 0) {
        printf("Rank (%d): This is the master process\n", rank);
        master();
    } else if ((rank >= 1) && (rank <= num_map_workers)) {
        printf(
            "Rank (%d => %d): This is a map worker process\n",
            rank,
            rank - 1
        );
        reduce_worker(rank - 1, map);
    } else {
        printf(
            "Rank (%d => %d): This is a reduce worker process\n",
            rank,
            rank - num_map_workers - 1
        );
        map_worker(rank - num_map_workers - 1);
    }

    // Clean up
    MPI_Finalize();
    return 0;
}

