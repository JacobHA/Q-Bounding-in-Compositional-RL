from glob import glob
import json
import numpy as np
import matplotlib.pyplot as plt

kl_divs = []
kl_stds = []


size = 10
if size < 10:
    size_ = f'0{size}'
else:
    size_ = size
mode='kld'

filename = f'single_out_b05_r1000_s{size_}_m{mode}/single_out_b05_r1000_s{size_}_m{mode}_n???.json'
file_list = glob(filename)
# file_list = glob('chimera_code_006/single_out_b05_r10000_s06_fmin_n???.json')
# file_list = glob('output_data_n???.json')
# file_list = glob('single_out_b05_r1000_s10_mkld/single_out_b05_r1000_s10_mkld_n???.json')

# file_list = glob('single_out_b05_r1000_s10_mkld/single_out_b05_r1000_s10_mkld_n???.json')
# file_list = glob('single_out_b05_r1000_s10_mnorm/single_out_b05_r1000_s10_mnorm_n???.json')
# file_list = glob('single_out_b05_r1000_s06_mkld/single_out_b05_r1000_s06_mkld_n???.json')
# file_list = glob('single_out_b05_r1000_s06_mnorm/single_out_b05_r1000_s06_mnorm_n???.json')

file_list.sort()

num_states = []
for file_path in file_list:
    with open(file_path, 'r') as file:
        content = json.load(file)
    
    num_states.append(content['META']['num_rewards'])
    kld_run = content['kld_run']
    kl_divs.append(np.mean(kld_run))
    kl_stds.append(np.std(kld_run))

kl_divs = np.array(kl_divs)
kl_stds = np.array(kl_stds)

plt.figure()
# increase font size
# plt.errorbar(num_states, kl_divs, kl_stds)
plt.rc('font', family='Times New Roman')
plt.plot(num_states, kl_divs, 'ko-')
plt.fill_between(num_states, kl_divs - kl_stds, kl_divs + kl_stds, alpha=0.2, color='k')
plt.xlabel('Reward Density (Number of states with rewards)', fontsize=16)


if mode == 'kld':
    plt.ylabel('KL Divergence', fontsize=16)
    plt.title('KL divergence as a function of density', fontsize=20)

# plt.ylabel('KL divergence', fontsize=22)
if mode == 'norm':
    plt.ylabel(r'Unweighted average of  $f(Q) - \widetilde{Q}^*$ ', fontsize=16)
    plt.title(r'Average of $f(Q) - \widetilde{Q}^*$ as a function of reward density', fontsize=20)


plt.savefig(f'{size}x{size}_{mode}.pdf')
# plt.show()