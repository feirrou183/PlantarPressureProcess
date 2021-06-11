import matplotlib.pyplot as plt


'''
velocity_list = [1.27,0.95,1.05]
std_list = [0.16,0.25,0.23]
name_list = ['ST','STEP','SPIN']
'''
velocity_list = [1.27,1.16,1.02,0.90]
std_list = [0.16,0.20,0.24,0.18]
name_list = ['ST','30°','60°','90°']


if __name__ == '__main__':
    plt.bar(range(len(velocity_list)),velocity_list,yerr = std_list, error_kw = {'ecolor' : '0.2', 'capsize' :6},tick_label = name_list)
    plt.ylabel('VELOCITY(mm/s)')
    plt.xlabel('STRATEGY')
    plt.show()
