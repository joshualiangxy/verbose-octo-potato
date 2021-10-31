#include <mpi.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "tasks.h"
#include "utils.h"

enum TAG { MAP_SEND, MAP_RECEIVE, REDUCE_SEND, REDUCE_RECEIVE, EXIT };

int MASTER_RANK = 0;
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
        MPI_Isend(partition_values[i], counts[i], mpi_key_value_type, reduce_worker_rank,
                REDUCE_SEND, MPI_COMM_WORLD, &send_requests[i]);
    }
}

void write_to_file(
    KeyValueMessage results[],
    int length,
    FILE* output_file,
    bool is_last
) {
    int i;

    for (i = 0; i < length; i++) {
        fprintf(output_file, "%s %d", results[i].key, results[i].val);

        if (!is_last || i < length - 1)
            fprintf(output_file, "\n");
    }
}

void master(
    int num_files,
    int num_map_workers,
    int num_reduce_workers,
    FILE* output_file,
    MPI_Datatype mpi_key_value_type
) {
    int i, completed = 0, count;
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
            MPI_Waitany(i, receive_requests, &index, &status);
            MPI_Get_count(&status, mpi_key_value_type, &count);
            map_worker_rank = status.MPI_SOURCE;

            send_to_reducer(
                map_results[index],
                count,
                num_map_workers,
                num_reduce_workers,
                mpi_key_value_type
            );
            completed++;
        }

        buffer[i] = i;
        MPI_Isend(&buffer[i], 1, MPI_INT, map_worker_rank,
                MAP_SEND, MPI_COMM_WORLD, &send_requests[i]);
        MPI_Irecv(map_results[i], MAX_MAP_KEYS, mpi_key_value_type, map_worker_rank,
                MAP_RECEIVE, MPI_COMM_WORLD, &receive_requests[i]);
    }

    for (i = 0; i < num_map_workers; i++) {
        map_worker_rank = NUM_MASTER + i;
        MPI_Isend(NULL, 0, MPI_INT, map_worker_rank,
                EXIT, MPI_COMM_WORLD, &send_requests[i]);
    }

    while (completed++ < num_files) {
        MPI_Waitany(num_files, receive_requests, &index, &status);
        MPI_Get_count(&status, mpi_key_value_type, &count);
        map_worker_rank = status.MPI_SOURCE;

        send_to_reducer(
            map_results[index],
            count,
            num_map_workers,
            num_reduce_workers,
            mpi_key_value_type
        );
    }

    for (i = 0; i < num_reduce_workers; i++) {
        reduce_worker_rank = NUM_MASTER + num_map_workers + i;
        MPI_Isend(NULL, 0, mpi_key_value_type, reduce_worker_rank,
            EXIT, MPI_COMM_WORLD, &send_requests[i]);
        MPI_Irecv(&reduce_results[i], MAX_REDUCE_KEYS, mpi_key_value_type, reduce_worker_rank,
            REDUCE_RECEIVE, MPI_COMM_WORLD, &receive_requests[i]);
    }

    for (i = 0; i < num_reduce_workers; i++) {
        MPI_Waitany(num_files, receive_requests, &index, &status);
        MPI_Get_count(&status, mpi_key_value_type, &count);
        write_to_file(
            reduce_results[i],
            count,
            output_file,
            i == num_reduce_workers - 1
        );
    }

    fclose(output_file);
}

void partition_results(
    KeyValue* key_values,
    int length,
    int num_reduce_workers,
    KeyValueMessage* send_buffer
) {
    int i;

    for (i = 0; i < length; i++) {
        KeyValue key_value = key_values[i];

        send_buffer[i].val = key_value.val;
        send_buffer[i].partition = partition(key_value.key, num_reduce_workers);
        strcpy(send_buffer[i].key, key_value.key);
    }
}

void map_worker(
    char* input_files_dir,
    int num_reduce_workers,
    MapTaskOutput* (*map)(char*),
    MPI_Datatype mpi_key_value_type
) {
    FILE* input_file;
    int file_num;
    long file_size;
    char* file_contents;
    MPI_Status status;
    MPI_Request send_request = NULL;
    KeyValueMessage* send_buffer;
    MapTaskOutput* results = NULL;

    int file_name_size = strlen(input_files_dir) + 16;

    while (true) {
        char file_name[file_name_size];

        MPI_Recv(&file_num, 1, MPI_INT, MASTER_RANK,
                MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == EXIT) break;
        if (status.MPI_TAG != MAP_SEND) {
            printf("Wrong tag sent to map worker: %d\n", status.MPI_TAG);
            continue;
        }

        snprintf(file_name, sizeof(char) * file_name_size,
                "%s%d.txt", input_files_dir, file_num);

        input_file = fopen(file_name, "rb");

        fseek(input_file, 0, SEEK_END);
        file_size = ftell(input_file);
        fseek(input_file, 0, SEEK_SET);

        file_contents = (char*) malloc(file_size + 1);
        fread(file_contents, sizeof(char), file_size, input_file);

        fclose(input_file);

        file_contents[file_size] = 0;

        if (send_request != NULL)
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
        if (results != NULL) {
            free_map_task_output(results);
            results = NULL;
        }

        results = map(file_contents);

        send_buffer = (KeyValueMessage*) malloc(
            sizeof(KeyValueMessage) * results->len
        );

        partition_results(results->kvs, results->len,
                num_reduce_workers, send_buffer);

        MPI_Isend(send_buffer, results->len, mpi_key_value_type, MASTER_RANK,
                MAP_RECEIVE, MPI_COMM_WORLD, &send_request);
    }

    MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    if (results != NULL) {
        free_map_task_output(results);
        results = NULL;
    }
}

void reduce_worker(MPI_Datatype mpi_key_value_type)
{
    MPI_Status status;
    MPI_Request send_request;
    KeyValueMessage receive_buffer[MAX_MAP_KEYS], send_buffer[MAX_REDUCE_KEYS];
    char keys[MAX_REDUCE_KEYS][8];
    int values[MAX_REDUCE_KEYS][MAX_MAP_KEYS];
    int lengths[MAX_REDUCE_KEYS];
    int i, j, count, key_count = 0;

    while (true) {
        MPI_Recv(&receive_buffer, MAX_MAP_KEYS, mpi_key_value_type, MASTER_RANK,
                MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == EXIT) break;
        if (status.MPI_TAG != REDUCE_SEND) {
            printf("Wrong tag sent to reduce worker: %d\n", status.MPI_TAG);
            continue;
        }

        MPI_Get_count(&status, mpi_key_value_type, &count);

        for (i = 0; i < count; i++) {
            KeyValueMessage key_value = receive_buffer[i];
            bool found = false;

            for (j = 0; j < key_count; j++) {
                if (strcmp(key_value.key, keys[j]) != 0) continue;

                found = true;
                values[j][lengths[j]] = key_value.val;
                lengths[j]++;
                break;
            }

            if (!found) {
                lengths[key_count] = 1;
                values[key_count][0] = key_value.val;
                strcpy(keys[key_count], key_value.key);
                key_count++;
            }
        }

        for (i = 0; i < key_count; i++) {
            KeyValue key_value = reduce(keys[i], values[i], lengths[i]);
            values[i][0] = key_value.val;
            lengths[i] = 1;
        }
    }

    for (i = 0; i < key_count; i++) {
        send_buffer[i].val = values[i][0];
        send_buffer[i].partition = 0;
        strcpy(send_buffer[i].key, keys[i]);
    }

    MPI_Isend(send_buffer, key_count, mpi_key_value_type, MASTER_RANK,
            REDUCE_RECEIVE, MPI_COMM_WORLD, &send_request);
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
    MPI_Datatype types[3] = { MPI_CHAR, MPI_INT, MPI_INT };
    int blocklengths[3] = { 8, 1, 1 };
    MPI_Aint displacements[3] = {
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
    MapTaskOutput* (*map)(char*);
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
        printf("Rank (%d): This is a map worker process\n", rank);
        map_worker(input_files_dir, num_reduce_workers, map, mpi_key_value_type);
    } else {
        printf("Rank (%d): This is a reduce worker process\n", rank);
        reduce_worker(mpi_key_value_type);
    }

    MPI_Type_free(&mpi_key_value_type);

    // Clean up
    MPI_Finalize();
    return 0;
}

