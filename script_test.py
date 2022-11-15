import numpy as np

arr = np.array([1.0, 1.0, 0.5, 0.5, 0.0, 0.2, 0.5, 1.0, 1.0])

def sectional_average(data, height_cutoff=0.1):
    prob_groups = np.where(data > height_cutoff)[0]
    prob_groups = np.split(prob_groups, np.where(np.diff(prob_groups) != 1)[0]+1)

    return prob_groups

print(sectional_average(arr))

avg_test = sectional_average(arr)

for num in avg_test[0]:
    print(arr[num])

temp_array = np.array(arr[avg_test[0]])

print(temp_array.mean())
