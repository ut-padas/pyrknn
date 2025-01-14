from mpi4py import MPI

def main():
    # Initialize the MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the rank of the process
    size = comm.Get_size()  # Get the total number of processes

    # Print a message from each process
    print(f"Hello from process {rank} out of {size}")

    # Test communication: gather ranks at process 0
    ranks = comm.gather(rank, root=0)

    if rank == 0:
        print(f"Gathered ranks at root process: {ranks}")

if __name__ == "__main__":
    main()
