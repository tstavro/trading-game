
'''
    metric
'''

import numpy as np
import matplotlib.pyplot as plt

seed = 20
process_size = 1000
np.random.seed(seed)
returns = np.random.normal(loc=0, scale=1, size=process_size)/100
process = np.cumsum(returns)+1 

#---------------------------------------

# a1 = [i for i in range(20)]
# for i in range(1, len(a1)+1):
#     # print(i, np.mean(a1[:i][-4:]))
#     print(a1[:i][-4:])

#---------------------------------------

win_len = 50
mov_av = np.zeros(len(process))
mov_av = []
for i in range(1, len(process)+1):
    mov_av.append( np.mean(process[:i][-win_len:]))
# mov_av = np.asarray(mov_av)
mov_av = np.array(mov_av)

plt.figure(figsize=(10,8))
plt.plot(process)
plt.plot(mov_av)
plt.grid(True)
plt.show()

plt.figure(figsize=(10,8))
plt.axhline(y=0, color = 'gray', linestyle = '--')
plt.plot(process-mov_av)
plt.title('Random Process - movavg')
plt.grid(True)
plt.show()

win_len = 100
max_values = []
differences = [np.nan for i in range(win_len)]
for i in range(win_len, len(process) + 1):
    window = process[i - win_len:i]
    max_value = np.max(window)
    max_values.append(max_value)
    differences.append(max_value - process[i - 1])

plt.figure(figsize=(10,8))
plt.plot(process, label='Process')
plt.plot(range(win_len - 1, len(process)), max_values, label='Max Moving Window')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Random Process with Max Moving Window')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,8))
plt.plot(differences, label='Max Value - Process')
plt.xlabel('Time')
plt.ylabel('Difference')
plt.title('Difference between Max Value and Random Process')
plt.legend()
plt.grid(True)
plt.show()

win_len = 100
max_value = 0
for i in range(len(process)):
    if i - win_len >= 0:
        window = process[i - win_len:i]
        max_value = np.max(window)
        if max_value - process[i] > 0.2:
            print(i, 'done')
            break

    

