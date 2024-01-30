import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 加载 .mat 文件
# Load the provided .mat file and extract the data for heatmap visualization
file_path = 'Data/A_0.3_L_2.0_wave_data.mat'
loaded_data = io.loadmat(file_path)

# Extracting data for each region: deepsea, normal, slope
deepsea_data = loaded_data['wave_data']['deepsea'][0,0]
normal_data = loaded_data['wave_data']['normal'][0,0]
slope_data = loaded_data['wave_data']['slope'][0,0]
fig,ax=plt.subplots(1,3,figsize=(10,8))
# Function to create a heatmap from the data
def plot_heatmap(data, title,i):
    # Reshape data for heatmap plotting
    x = np.unique(data[:, 0])
    t = np.unique(data[:, 1])
    u_xt = data[:, 2].reshape(640, 100)
    print(title,np.max(u_xt))

    cax=ax[i].pcolormesh(x, t, u_xt, cmap='jet', vmin=0, vmax=0.5)
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('t')
    if i ==0:
      fig.colorbar(cax, ax=ax[0], label='u(x,t)')
    ax[i].set_xlabel('x')
    ax[i].set_ylabel('t')
    ax[i].set_title(title)

for i in range (3):
   title=["deepsea","slope","normal"]
   data=loaded_data['wave_data'][title[i]][0,0]
   print(data.shape)

   plot_heatmap(data, title=title[i],i=i)
plt.tight_layout()
plt.show()