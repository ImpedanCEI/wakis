import numpy as np
import h5py
import matplotlib.pyplot as plt

ninj = 1046
Nt = 1716


hf = h5py.File('../hdf5/Ez_abc.h5', 'r')
hf2 = h5py.File('../hdf5/Jz_abc.h5', 'r')
z = np.array(hf['z'])

fig, ax = plt.subplots(1, 1, figsize=[9,5], dpi=140)

axx = ax.twinx()
keystoplot = [ninj+(Nt-ninj)*0.1, ninj+(Nt-ninj)*0.3, ninj+(Nt-ninj)*0.6, ninj+(Nt-ninj)*0.8, Nt-1]
colors = plt.cm.rainbow([0.2, 0.4, 0.6, 0.8, 0.9])
colorsJ = [0.2, 0.4, 0.6, 0.8, 1.]

for i, key in enumerate(keystoplot):
    Etoplot = np.array(hf[list(hf.keys())[int(key)]])
    Jtoplot = np.array(hf2[list(hf2.keys())[int(key)]])
    axx.plot(z, Etoplot, color=colors[i], ls='none', marker='d', markevery=2, ms=3, label=f'ABC timestep #{int(key)}')
    ax.plot(z, Jtoplot, color='k', label=f'Jz timestep #{int(key)}', alpha=colorsJ[i])
    ax.fill_between(z, Jtoplot, color='k', alpha=0.1)

hf.close()
hf2.close()

hf = h5py.File('../hdf5/Ez_pec.h5', 'r')
hf2 = h5py.File('../hdf5/Jz_pec.h5', 'r')

for i, key in enumerate(keystoplot):
    Etoplot = np.array(hf[list(hf.keys())[int(key)]])
    Jtoplot = np.array(hf2[list(hf2.keys())[int(key)]])
    axx.plot(z, Etoplot, color=colors[i], label=f'PEC timestep #{int(key)}', alpha=0.8, lw=2)

ax.set_xlabel('$z$ [m]')
axx.set_ylabel('$E_z$ [V/m]', color='darkgreen', fontweight='bold')
lims = max(np.abs(plt.ylim()[0]), np.abs(plt.ylim()[0]))
axx.set_ylim(-1e5, 1e5)
ax.set_ylabel('Charge distribution [C/m]', color='r', fontweight='bold')
ax.set_ylim(ymin=-np.max(Jtoplot)*1.5, ymax=np.max(Jtoplot)*1.5)
axx.legend(loc='lower center', ncol=2)

hf.close()
hf2.close()

fig.suptitle('$E_z(x_s,y_s,z)$ field and $J_z(x_s,y_s,z)$ for different timesteps')
fig.tight_layout()
fig.savefig('PECvsABC.png')
plt.show()