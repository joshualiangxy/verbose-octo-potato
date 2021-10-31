#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "tasks.h"
#include "utils.h"

enum TAG { MAP_SEND, MAP_RECEIVE, REDUCE_SEND, REDUCE_RECEIVE, EXIT };

int NUM_MASTER = 1;
int MAX_MAP_KEYS = 4 * 48;
int MAX_REDUCE_KEYS = 26;

typedef struct {
    char key[8];
    int val;
    int partition;
} KeyValueMessage;

void send_to_reducer(
    KeyValueMessage key_values[],
    int length,
    int num_map_workers,
    int num_reduce_workers,
    MPI_Datatype mpi_key_value_type
) {
  KeyValueMessage partition_values[num_reduce_workers][MAX_MAP_KEYS];
  MPI_Request send_requests[num_reduce_workers];
  int i, counts[num_reduce_workers];

  memset(counts, 0, num_reduce_workers * sizeof(int));

  for (i = 0; i < length; i++) {
    KeyValueMessage key_value = key_values[i];
    int partition = key_value.partition;
    partition_values[partition][counts[partition]] = key_value;

    counts[partition]++;
    }

    for (i = 0; i < num_reduce_workers; i++) {
        int reduce_worker_rank = NUM_MASTER + num_map_workers + i;
        MPI_Isend(
            &partition_values[i],
            counts[i],
            mpi_key_value_type,
            reduce_worker_rank,
            REDUCE_SEND,
            MPI_COMM_WORLD,
            &send_requests[i]
        );
    }
}

void write_to_file(KeyValueMessage results[], int length, FILE* output_file)
{
    int i;

    for (i = 0; i < length; i++)
        fprintf(output_file, "%s %d", results[i].key, results[i].val);
}

void master(
    int num_files,
    int num_map_workers,
    int num_reduce_workers,
    FILE* output_file,
    MPI_Datatype mpi_key_value_type
) {
    int i, completed = 0;
    MPI_Request send_requests[num_files], receive_requests[num_files];
    KeyValueMessage map_results[num_files][MAX_MAP_KEYS],
        reduce_results[num_reduce_workers][MAX_REDUCE_KEYS];
    int buffer[num_files];
    int map_worker_rank, reduce_worker_rank;
    int index;
    MPI_Status status;

    for (i = 0; i < num_files; i++) {
        if (i < num_map_workers) {
            map_worker_rank = NUM_MASTER + i;
        } else {
            MPI_Waitany(i - 1, receive_requests, &index, &status);
            map_worker_rank = status.MPI_SOURCE;
            send_to_reducer(
                map_results[index],
                status.count,
                num_map_workers,
                num_reduce_workers,
                mpi_key_value_type
            );
            completed++;
        }

        buffer[i] = i;
        MPI_Isend(
            &buffer[i],
            1,
            MPI_INT,
            map_worker_rank,
            MAP_SEND,
            MPI_COMM_WORLD,
            &send_requests[i]
        );
        MPI_Irecv(
            map_results[i],
            MAX_MAP_KEYS,
            mpi_key_value_type,
            map_worker_rank,
            MAP_RECEIVE,
            MPI_COMM_WORLD,
            &receive_requests[i]
        );
    }

    for (i = 0; i < num_map_workers; i++) {
        map_worker_rank = NUM_MASTER + i;
        MPI_Isend(
            NULL,
            0,
            MPI_INT,
            map_worker_rank,
            EXIT,
            MPI_COMM_WORLD,
            &send_requests[i]
        );
    }

    while (completed++ < num_files) {
        MPI_Waitany(num_files, receive_requests, &index, &status);
        map_worker_rank = status.MPI_SOURCE;
        send_to_reducer(
            map_results[index],
            status.count,
            num_map_workers,
            num_reduce_workers,
            mpi_key_value_type
        );
    }

    for (i = 0; i < num_reduce_workers; i++) {
        reduce_worker_rank = NUM_MASTER + num_map_workers + i;
        MPI_Isend(
            NULL,
            0,
            mpi_key_value_type,
            reduce_worker_rank,
            EXIT,
            MPI_COMM_WORLD,
            &send_requests[i]
        );
        MPI_Irecv(
            &reduce_results[i],
            MAX_REDUCE_KEYS,
            mpi_key_value_type,
            reduce_worker_rank,
            REDUCE_RECEIVE,
            MPI_COMM_WORLD,
            &receive_requests[i]
        );
    }

    for (i = 0; i < num_reduce_workers; i++) {
        MPI_Waitany(num_files, receive_requests, &index, &status);
        write_to_file(reduce_results[i], status.count, output_file);
    }
}

void map_worker(
    int rank,
    MapTaskOutput* map(char*),
    MPI_Datatype mpi_key_value_type
) {
    FILE* input_file;
    while (true) {
        MPI_Status status;
    }
}

void reduce_worker(int rank, MPI_Datatype mpi_key_value_type)
{
    while (true) {
        MPI_Status status;
    }
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

    int world_size, rank;

    int structlen = 3;
    MPI_Datatype mpi_key_value_type;
    MPI_Datatype types[2] = { MPI_CHAR, MPI_INT, MPI_INT };
    int blocklengths[2] = { 8, 1, 1 };
    int displacements[2] = {
        offsetof(KeyValueMessage, key),
        offsetof(KeyValueMessage, val),
        offsetof(KeyValueMessage, partition)
    };

    FILE* output_file = fopen(output_file_name, "w");
    if (output_file == NULL) {
        fprintf(
            stderr,
            "Failed to open %s for writing. Aborting...\n",
            output_file_name
        );
        exit(EXIT_FAILURE);
    }

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Type_create_struct(
        structlen,
        blocklengths,
        displacements,
        types,
        &mpi_key_value_type
    );
    MPI_Type_commit(&mpi_key_value_type);

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
        master(
            num_files,
            num_map_workers,
            num_reduce_workers,
            output_file,
            mpi_key_value_type
        );
    } else if ((rank >= 1) && (rank <= num_map_workers)) {
        printf(
            "Rank (%d => %d): This is a map worker process\n",
            rank,
            rank - NUM_MASTER
        );
        map_worker(rank - 1, map, mpi_key_value_type);
    } else {
        printf(
            "Rank (%d => %d): This is a reduce worker process\n",
            rank,
            rank - num_map_workers - NUM_MASTER
        );
        reduce_worker(rank - num_map_workers - 1, mpi_key_value_type);
    }

    MPI_Type_free(&mpi_key_value_type);

    // Clean up
    MPI_Finalize();
    return 0;
}

