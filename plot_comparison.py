import matplotlib.pyplot as plt

# Generate images based on columns
with open("torques_time_optimal.csv", "r") as f:
    lines = f.readlines()
    time_optimal_data = [list(map(float, line.strip().split(","))) for line in lines]

with open("torques_hybrid.csv", "r") as f:
    lines = f.readlines()
    hybrid_data = [list(map(float, line.strip().split(","))) for line in lines]

with open("torques_with_object.csv", "r") as f:
    lines = f.readlines()
    with_object_data = [list(map(float, line.strip().split(","))) for line in lines]

# Transpose data to get columns
time_optimal_data = list(zip(*time_optimal_data))
hybrid_data = list(zip(*hybrid_data))
with_object_data = list(zip(*with_object_data))

# Combine the data per column into plots
for i in range(len(time_optimal_data)):
    plt.figure()
    plt.plot(time_optimal_data[i], label="Time Optimal")
    plt.plot(hybrid_data[i], label="Hybrid")
    plt.plot(with_object_data[i], label="With Object")
    plt.title(f"Torque Comparison for Joint {i+1}")
    plt.xlabel("Time Step")
    plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.grid()
    plt.savefig(f"torque_comparison_joint_{i+1}.png")
    plt.close()