#include <iostream>
#include <mpi.h>
#include <string>
#include <vector>

const int N = 5; // Matrix size
const int P = 8; // Parts number

const int length = 64;
const int part_len = (length / 2) / P;
const bool even = length % 2 == 0;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) { // Root process
        if (size != N * N) {
            std::cerr << "Number of processes must be equal to " << N * N << ", currently is " << size << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        if ((length / 2) % P != 0) {
            std::cerr << "Number of parts is not correct for this message" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        std::string data = "1234567812345678123456781234567812345678123456781234567812345678";
        std::cout << "Initial data: " << data << std::endl;
        MPI_Request requests[2 * P];
        int request_count = 0;
        for (int part_num = 0; part_num < P; part_num++) {
            std::string part1 = data.substr(part_num * part_len, part_len);
            std::string part2 = data.substr(length / 2 + part_num * part_len, part_len);
            MPI_Issend(part1.c_str(), part1.size(), MPI_CHAR, 1, 0, MPI_COMM_WORLD, &requests[request_count++]);
            MPI_Issend(part2.c_str(), part2.size(), MPI_CHAR, N, 0, MPI_COMM_WORLD, &requests[request_count++]);
        }
        if (!even) {
            std::string tmp(1, data[length - 1]);
            MPI_Ssend(tmp.c_str(), 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        }
    }
    else if (rank == N * N - 1) { // End process
        std::string data_end_first(length / 2, '\0');
        std::string data_end_second(length / 2, '\0');
        std::string data_end(length, '\0');
        std::vector<std::string> parts1(P, std::string(part_len, '\0'));
        std::vector<std::string> parts2(P, std::string(part_len, '\0'));
        int bias = 0;
        MPI_Request requests[2 * P];
        int request_count = 0;
        for (int part_num = 0; part_num < P; part_num++) {
            MPI_Irecv(&parts1[part_num][0], part_len, MPI_CHAR, N * N - N - 1, N * N - N - 1, MPI_COMM_WORLD, &requests[request_count++]);

            MPI_Irecv(&parts2[part_num][0], part_len, MPI_CHAR, N * N - 2, N * N - 2, MPI_COMM_WORLD, &requests[request_count++]);
        }

        std::string tmp(1, '\0');
        if (!even) {
            MPI_Recv(&tmp[0], 1, MPI_CHAR, N * N - N - 1, N * N - N - 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        MPI_Waitall(request_count, requests, MPI_STATUSES_IGNORE);

        for (int part_num = 0; part_num < P; part_num++) {
            data_end_first.replace(bias, part_len, parts1[part_num]);
            data_end_second.replace(bias, part_len, parts2[part_num]);
            bias += part_len;
        }

        data_end = data_end_first + data_end_second + tmp;
        std::cout << "Final result: " << data_end << std::endl;
    }
    else { // In between
        int send, receive;
        if ((rank < N - 1) || (rank >= N * N - N)) {
            send = rank + 1;
            if (rank == N * N - N) {
                receive = rank - N;
            }
            else {
                receive = rank - 1;
            }
        }
        else if (((rank + 1) % N == 0) || (rank % N == 0)) {
            send = rank + N;
            if (rank == N - 1) {
                receive = rank - 1;
            }
            else {
                receive = rank - N;
            }
        }
        else {
            send = -1;
            receive = -1;
        }

        if (send != -1 && receive != -1) { // only send on the sides of the matrix
            for (int part_num = 0; part_num < P; part_num++) {
                std::string part(part_len, '\0');
                MPI_Recv(&part[0], part_len, MPI_CHAR, receive, receive, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(part.c_str(), part.size(), MPI_CHAR, send, rank, MPI_COMM_WORLD);
            }

            if (!even && (((rank < N - 1)) || ((rank + 1) % N == 0))) {
                std::string tmp(1, '\0');
                MPI_Recv(&tmp[0], 1, MPI_CHAR, receive, receive, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Ssend(tmp.c_str(), 1, MPI_CHAR, send, rank, MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return 0;
}
