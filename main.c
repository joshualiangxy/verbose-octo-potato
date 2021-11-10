#include <mpi.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <unordered_map>
#include <iostream>
#include "tasks.h"
#include "utils.h"

enum TAG {
    MAP_SEND,
    MAP_RECEIVE,
    MAP_RECEIVE_LENGTH,
    REDUCE_SEND,
    REDUCE_SEND_LENGTH,
    REDUCE_RECEIVE,
    REDUCE_RECEIVE_LENGTH,
    EXIT
};

int MASTER_RANK = 0;
int NUM_MASTER = 1;

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
    KeyValueMessage partition_values[num_reduce_workers][length];
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
        MPI_Send(&counts[i], 1, MPI_INT, reduce_worker_rank,
                REDUCE_SEND_LENGTH, MPI_COMM_WORLD);
        MPI_Isend(partition_values[i], counts[i], mpi_key_value_type, reduce_worker_rank,
                REDUCE_SEND, MPI_COMM_WORLD, &send_requests[i]);
    }

    MPI_Waitall(num_reduce_workers, send_requests, MPI_STATUS_IGNORE);
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
    int i, completed = 0, count, length;
    MPI_Request map_send_requests[num_files],
        map_receive_requests[num_files],
        map_exit_requests[num_map_workers],
        reduce_receive_requests[num_reduce_workers],
        reduce_exit_requests[num_reduce_workers];
    KeyValueMessage* map_results[num_files];
    KeyValueMessage* reduce_results[num_reduce_workers];
    int buffer[num_files];
    int map_worker_rank, reduce_worker_rank;
    int index;
    MPI_Status status;

    for (i = 0; i < num_files; i++) {
        if (i < num_map_workers) {
            map_worker_rank = NUM_MASTER + i;
        } else {
            MPI_Waitany(i, map_receive_requests, &index, &status);
            MPI_Get_count(&status, mpi_key_value_type, &count);
            map_worker_rank = status.MPI_SOURCE;

            send_to_reducer(
                map_results[index],
                count,
                num_map_workers,
                num_reduce_workers,
                mpi_key_value_type
            );

            free(map_results[index]);
            map_results[index] = NULL;

            completed++;
        }

        buffer[i] = i;
        MPI_Isend(&buffer[i], 1, MPI_INT, map_worker_rank,
                MAP_SEND, MPI_COMM_WORLD, &map_send_requests[i]);
        MPI_Recv(&length, 1, MPI_INT, map_worker_rank,
                MAP_RECEIVE_LENGTH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        map_results[i] = (KeyValueMessage*) malloc(
            sizeof(KeyValueMessage) * length
        );

        MPI_Irecv(map_results[i], length, mpi_key_value_type, map_worker_rank,
                MAP_RECEIVE, MPI_COMM_WORLD, &map_receive_requests[i]);
    }

    for (i = 0; i < num_map_workers; i++) {
        map_worker_rank = NUM_MASTER + i;
        MPI_Isend(NULL, 0, MPI_INT, map_worker_rank,
                EXIT, MPI_COMM_WORLD, &map_exit_requests[i]);
    }

    while (completed++ < num_files) {
        MPI_Waitany(num_files, map_receive_requests, &index, &status);
        MPI_Get_count(&status, mpi_key_value_type, &count);

        send_to_reducer(
            map_results[index],
            count,
            num_map_workers,
            num_reduce_workers,
            mpi_key_value_type
        );

        free(map_results[index]);
        map_results[index] = NULL;
    }

    for (i = 0; i < num_reduce_workers; i++) {
        reduce_worker_rank = NUM_MASTER + num_map_workers + i;
        MPI_Isend(NULL, 0, MPI_INT, reduce_worker_rank,
                EXIT, MPI_COMM_WORLD, &reduce_exit_requests[i]);

        MPI_Recv(&length, 1, MPI_INT, reduce_worker_rank, REDUCE_RECEIVE_LENGTH,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        reduce_results[i] = (KeyValueMessage*) malloc(
            sizeof(KeyValueMessage) * length
        );

        MPI_Irecv(reduce_results[i], length, mpi_key_value_type, reduce_worker_rank,
                REDUCE_RECEIVE, MPI_COMM_WORLD, &reduce_receive_requests[i]);
    }

    for (i = 0; i < num_reduce_workers; i++) {
        MPI_Waitany(num_reduce_workers, reduce_receive_requests, &index, &status);
        MPI_Get_count(&status, mpi_key_value_type, &count);
        write_to_file(
            reduce_results[index],
            count,
            output_file,
            i == num_reduce_workers - 1
        );

        free(reduce_results[index]);
        reduce_results[index] = NULL;
    }

    MPI_Waitall(num_reduce_workers, reduce_exit_requests, MPI_STATUS_IGNORE);

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

        free(file_contents);
        file_contents = NULL;

        send_buffer = (KeyValueMessage*) malloc(
            sizeof(KeyValueMessage) * results->len
        );

        partition_results(results->kvs, results->len,
                num_reduce_workers, send_buffer);

        MPI_Send(&(results->len), 1, MPI_INT, MASTER_RANK,
                MAP_RECEIVE_LENGTH, MPI_COMM_WORLD);
        MPI_Isend(send_buffer, results->len, mpi_key_value_type, MASTER_RANK,
                MAP_RECEIVE, MPI_COMM_WORLD, &send_request);
    }

    if (send_request != NULL)
        MPI_Wait(&send_request, MPI_STATUS_IGNORE);
    if (results != NULL) {
        free_map_task_output(results);
        results = NULL;
    }
}

void reduce_worker(MPI_Datatype mpi_key_value_type)
{
    using std::vector;
    using std::unordered_map;
    using std::string;

    MPI_Status status;
    KeyValueMessage* receive_buffer;
    KeyValueMessage* send_buffer;
    vector<string> keys;
    vector<vector<int>> values;
    vector<int> lengths;
    unordered_map<string, int> key_map;
    int i, j, length, key_count = 0;
    int counter = 0;

    while (true) {
        MPI_Recv(&length, 1, MPI_INT, MASTER_RANK,
                MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == EXIT) break;
        if (status.MPI_TAG != REDUCE_SEND_LENGTH) {
            printf("Wrong tag sent to reduce worker: %d\n", status.MPI_TAG);
            continue;
        }

        receive_buffer = (KeyValueMessage*) malloc(
            sizeof(KeyValueMessage) * length
        );

        MPI_Recv(receive_buffer, length, mpi_key_value_type, MASTER_RANK,
                REDUCE_SEND, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (i = 0; i < length; i++) {
            KeyValueMessage key_value = receive_buffer[i];
            bool found = false;

            auto iter = key_map.find(key_value.key);

            if (iter != key_map.end()) {
                int index = iter->second;

                if (lengths[index] >= values[index].size())
                    values[index].push_back(key_value.val);
                else
                    values[index][lengths[index]] = key_value.val;

                lengths[index]++;
            } else {
                key_map.insert({ key_value.key, key_count });
                keys.push_back(key_value.key);
                values.push_back({ key_value.val });
                lengths.push_back(1);
                key_count++;
            }
        }

        for (i = 0; i < key_count; i++) {
            KeyValue key_value = reduce(&keys[i][0], &values[i][0], lengths[i]);
            values[i][0] = key_value.val;
            lengths[i] = 1;
        }

        free(receive_buffer);
        receive_buffer = NULL;
    }

    send_buffer = (KeyValueMessage*) malloc(
        sizeof(KeyValueMessage) * key_count
    );

    for (i = 0; i < key_count; i++) {
        send_buffer[i].val = values[i][0];
        send_buffer[i].partition = 0;
        strcpy(send_buffer[i].key, &keys[i][0]);
    }

    MPI_Send(&key_count, 1, MPI_INT, MASTER_RANK,
            REDUCE_RECEIVE_LENGTH, MPI_COMM_WORLD);
    MPI_Send(send_buffer, key_count, mpi_key_value_type, MASTER_RANK,
            REDUCE_RECEIVE, MPI_COMM_WORLD);

    free(send_buffer);
    send_buffer = NULL;
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

    int structlen = 3;
    MPI_Datatype tmp_type, mpi_key_value_type;
    MPI_Datatype types[3] = { MPI_CHAR, MPI_INT, MPI_INT };
    int blocklengths[3] = { 8, 1, 1 };
    MPI_Aint displacements[3] = {
        offsetof(KeyValueMessage, key),
        offsetof(KeyValueMessage, val),
        offsetof(KeyValueMessage, partition)
    };
    MPI_Aint lb, extent;

    int world_size, rank;

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
        &tmp_type
    );
    MPI_Type_get_extent(tmp_type, &lb, &extent);
    MPI_Type_create_resized(tmp_type, lb, extent, &mpi_key_value_type);
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

