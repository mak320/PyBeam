import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['science', 'notebook', 'grid'])


l_d = 4.7
rho_b = 1.65
ang_b = np.pi / 8

# ====================== Components ====================== #
def DriftZone(L):
    """Tranfer Matrix of an L long drift"""
    return [{'TransMat': np.array([[1, L], [0, 1]]),
             'len': L
             }]

def ThinQuad(f):
    """Transfer Matrix of a Quadrupole with a focal length f"""
    return [{'TransMat': np.array([[1, 0], [-1/f, 1]]),
             'len': 0
             }]

def Dip(rho, L):
    """Transfer Matrix for a Sector Dipole, with bending radius rho and length L"""
    return [{'TransMat': np.array([[np.cos(L/rho), rho*np.sin(L/rho)],
                                    [-np.sin(L/rho), np.cos(L/rho)]]),
             'len' : L
            }]

def Quad(k, L):
    """Quadrupole magnet with quad gradient k and lenght L
        k >= 0 --> Focusing
        k < 0 --> Defocusing
    """
    omega = np.sqrt(np.abs(k))
    if k >= 0:
        return [{'TransMat': np.array([[np.cos(omega * L), 1 / omega * np.sin(omega * L)],
                                       [-omega * np.sin(omega * L), np.cos(omega * L)]]),
                 'len': L
                 }]

    else:
        return [{'TransMat': np.array([[np.cosh(omega * L), 1 / omega * np.sinh(omega * L)],
                                       [-omega * np.sinh(omega * L), np.cosh(omega * L)]]),
                 'len': L
                 }]

def BendingMag(k, rho, ang, PoB):
    """Combined function magnet consisting of Dipole and quadrupole modes
    Bending radius rho, quad gradient k, which bends the reference trajectory by an angle ang
    The dipole component only acts in the plane of bending
    """
    L = rho * (ang / np.pi)

    if PoB:
        k = k + 1/rho**2
    else:
        k *= -1

    omega = np.sqrt(np.abs(k))
    if k >= 0:  # Focus

        return [{'TransMat': np.array([[np.cos(omega * L), 1 / omega * np.sin(omega * L)],
                                       [-omega * np.sin(omega * L), np.cos(omega * L)]]),
                 'len': L
                 }]
    else:  # De-focus
        return [{'TransMat': np.array([[np.cosh(omega * L), 1 / omega * np.sinh(omega * L)],
                                       [omega * np.sinh(omega * L), np.cosh(omega * L)]]),
                 'len': L
                 }]

# ====================== Functionalities ====================== #


def Beamline(k1, k2):
    # ====================== Global Parameters ====================== #
    l_d = 4.7
    rho_b = 1.65
    ang_b = np.pi / 8

    # ====================== Beamline Design ====================== #
    Beamline_horiz = DriftZone(l_d)  + \
                     BendingMag(k1, rho_b, ang_b, PoB= True) + \
                     BendingMag(k2, rho_b, ang_b, PoB= True) + \
                     BendingMag(k2, rho_b, ang_b, PoB= True) + \
                     BendingMag(k1, rho_b, ang_b, PoB= True)

    Beamline_vert =DriftZone(l_d)  + \
                     BendingMag(k1, rho_b, ang_b, PoB= False)  + \
                     BendingMag(k2, rho_b, ang_b, PoB= False)  + \
                     BendingMag(k2, rho_b, ang_b, PoB= False)  + \
                     BendingMag(k1, rho_b, ang_b, PoB= False)

    return {'horiz': Beamline_horiz, 'vert': Beamline_vert}

def EffTransMat(beamline):
    """Multiplies component transfer matricies in order to get Effective Transfer Matrix"""
    EffT = np.eye(beamline[0]['TransMat'].shape[0])
    Len = 0
    for component in beamline[-1::-1]: #revese beamline since matrix multiplication acts for the right
        EffT = EffT @ component['TransMat']
        Len = Len + component['len']
    return {'TransMat': EffT, 'len': Len}

def transportParticles(x0, beamline):
    trans_coords = [x0]
    s = [0]
    for component in beamline:
        trans_coords.append(component['TransMat'] @ trans_coords[-1])
        s.append(s[-1] + component['len'])
    trans_coords = np.array(trans_coords).transpose()
    return {'x': trans_coords[:, 0, :].T,
            'p_x': trans_coords[:, 1, :].T,
            's': np.array(s)}

def ScanStability(lim_k1, lim_k2, N = 400):
    k1s = np.linspace(-lim_k1, lim_k1, N)
    k2s = np.linspace(-lim_k2, lim_k2, N )[-1::-1]

    k1s = k1s[k1s != 0]
    k2s = k2s[k2s != 0]

    K1, K2 = np.meshgrid(k1s, k2s)

    Trace_horiz = np.zeros((N,N))
    Trace_vert = np.zeros((N, N))

    for i2 in range(N):
        for i1 in range(N):

            k1 = k1s[i1]
            k2 = k2s[i2]

            Beamline_horiz = Beamline(k1, k2)['horiz']

            Beamline_horiz = Beamline_horiz * 4

            OTM_horiz = EffTransMat(Beamline_horiz)['TransMat']

            Beamline_vert = Beamline(k1, k2)['vert']

            Beamline_vert = Beamline_vert * 4

            OTM_vert = EffTransMat(Beamline_vert)['TransMat']

            Trace_horiz[i2, i1] = np.trace(OTM_horiz)
            Trace_vert[i2, i1] = np.trace(OTM_vert)

    fig = plt.figure('Stability')
    ax = fig.add_subplot(111)

    cs_horiz = ax.contour(K1, K2, Trace_horiz, levels=[-2, 2], linestyles=['solid', 'solid'], colors='red')
    ax.clabel(cs_horiz)

    cs_vert = ax.contour(K1, K2, Trace_vert, levels=[-2, 2], linestyles=['solid', 'solid'], colors='blue')
    ax.clabel(cs_vert)

    stale_horiz = np.abs(Trace_horiz) < 2
    stale_vert = np.abs(Trace_vert) < 2

    stable = np.logical_and(stale_horiz, stale_vert)
    stable = stable.astype(int)

    ax.contourf(K1, K2, stable, levels=[0, 0.5, 1.5], colors=[(0, 0, 0, 0), (0, 128, 0, 0.4)])

    ax.set_xlabel(r'$k_1$')
    ax.set_ylabel(r'$k_2$')

    ax.hlines(y=[0, -1/rho_b**2], xmin=-lim_k1, xmax=lim_k1, colors='gray', linestyles=['solid', 'dashed'])
    ax.vlines(x=[0, -1 / rho_b ** 2], ymin=-lim_k2, ymax=lim_k2, colors='gray', linestyles=['solid', 'dashed'])
    ax.scatter(x = -4.99, y = 4.42, marker='x', s = 30)

    plt.show()

    return Trace_horiz, Trace_vert

def twiss_inv(beamline, ret_tune = False):
    OTM = EffTransMat(beamline)
    R = OTM['TransMat']
    psi = np.arccos((R[0, 0]+R[1, 1])/2)
    if (R[0, 1]<0):
        psi = 2*np.pi-psi
    Q = psi/(2*np.pi)
    beta = (R[0, 1]) / np.sin(psi)
    alpha = (R[0, 0] - R[1, 1])/(2*np.sin(psi))
    gamma = (1 + alpha**2)/beta
    if ret_tune:
        return np.array([[beta, -alpha], [-alpha, gamma]]), Q
    else:
        return np.array([[beta, -alpha], [-alpha, gamma]])

def update_twiss (A0 , beamline):
    """ Transport the sigma matrix along the beamline """
    Ai = [A0]
    s = [0]
    for component in beamline :
        Ai.append(component['TransMat'] @ Ai[-1] @ component['TransMat']. transpose())
        s. append(s[ -1] + component ['len'])
    Ai =np.array(Ai).transpose()
    return {'beta': Ai[0][0],
            'alpha01': Ai[0][1],
            'alpha10': Ai[1][0],  # equal to alpha-new01
            'gamma': Ai[1][1],
            's': np. array(s),
            'Ai': Ai}

def ScanBetaMax(lim_k1, lim_k2, N = 400):

    k1s = np.linspace(-lim_k1, lim_k1, N)
    k2s = np.linspace(-lim_k2, lim_k2, N)[-1::-1]

    k1s = k1s[k1s != 0]
    k2s = k2s[k2s != 0]

    K1, K2 = np.meshgrid(k1s, k2s)

    Trace_horiz = np.zeros((N, N))
    Trace_vert = np.zeros((N, N))

    BetaMax_horiz = np.zeros((N, N))
    BetaMax_vert = np.zeros((N, N))


    for i2 in range(N):
        for i1 in range(N):
            k1 = k1s[i1]
            k2 = k2s[i2]

            Beamline_horiz =Beamline(k1, k2)['horiz']

            Beamline_horiz = Beamline_horiz * 4

            OTM_horiz = EffTransMat(Beamline_horiz)['TransMat']

            Beamline_vert = Beamline(k1, k2)['vert']

            Beamline_vert = Beamline_vert * 4

            OTM_vert = EffTransMat(Beamline_vert)['TransMat']

            Trace_horiz[i2, i1] = np.trace(OTM_horiz)
            Trace_vert[i2, i1] = np.trace(OTM_vert)

            stable_horiz = (np.abs(np.trace(OTM_horiz)) < 2)
            stable_vert = (np.abs(np.trace(OTM_vert)) < 2)

            stable = np.logical_and(stable_horiz, stable_vert)

            if stable:
                twissMat0X, Q_X = twiss_inv(Beamline_horiz*4)
                twissMat0Y, Q_Y = twiss_inv(Beamline_vert*4)



                BetaMax_horiz[i2, i1] = np.max(np.abs(update_twiss(twissMat0X, Beamline_horiz)['beta']))
                BetaMax_vert[i2, i1] = np.max(np.abs(update_twiss(twissMat0Y, Beamline_vert)['beta']))

            else:
                BetaMax_horiz[i2, i1] = np.inf
                BetaMax_vert[i2, i1] = np.inf



    BetaMax = np.fmax(BetaMax_horiz, BetaMax_vert)

    fig, axes = plt.subplot_mosaic([['left', 'right'], ['bottom', 'bottom']], constrained_layout=True)

    axes['left'].contourf(K1, K2, np.log(BetaMax_horiz), levels=150, vmax=4.5, cmap='turbo')
    axes['right'].contourf(K1, K2, np.log(BetaMax_vert), levels=150, vmax=4.5, cmap='turbo')

    c = axes['bottom'].contourf(K1, K2, np.log(BetaMax), levels=150, vmax=4.5, cmap='turbo')
    cb = fig.colorbar(c, label=r'$\ln(Max[|\beta(s)|])$')

    lev1x = axes['left'].contour(K1, K2, Trace_horiz, levels=[-2, 2], linestyles=['solid', 'solid'], colors='red', linewidths = 0.75)
    axes['left'].clabel(lev1x)
    lev1y = axes['left'].contour(K1, K2, Trace_vert, levels=[-2, 2], linestyles=['solid', 'solid'], colors='blue', linewidths = 0.75)
    axes['left'].clabel(lev1y)

    lev2x = axes['right'].contour(K1, K2, Trace_horiz, levels=[-2, 2], linestyles=['solid', 'solid'], colors='red', linewidths= 0.75)
    axes['right'].clabel(lev2x)
    lev1y = axes['right'].contour(K1, K2, Trace_vert, levels=[-2, 2], linestyles=['solid', 'solid'], colors='blue', linewidths= 0.75)
    axes['right'].clabel(lev1y)

    lev3x = axes['bottom'].contour(K1, K2, Trace_horiz, levels=[-2, 2], linestyles=['solid', 'solid'], colors='red', linewidths= 0.75)
    axes['bottom'].clabel(lev3x)
    lev1y = axes['bottom'].contour(K1, K2, Trace_vert, levels=[-2, 2], linestyles=['solid', 'solid'], colors='blue', linewidths= 0.75)
    axes['bottom'].clabel(lev1y)

    for ax in ['left', 'right', 'bottom']:
        # guide lines
        axes[ax].hlines(y=0, xmin=np.min(k1s), xmax=np.max(k1s), colors='gray', linewidths=0.5)
        axes[ax].hlines(y=-1 / rho_b ** 2, xmin=np.min(k1s), xmax=np.max(k1s), colors='gray', linewidths=0.5)
        axes[ax].vlines(x=0, ymin=np.min(k2s), ymax=np.max(k2s), colors='gray', linewidths=0.5)
        axes[ax].vlines(x=-1 / rho_b ** 2, ymin=np.min(k2s), ymax=np.max(k2s), colors='gray', linewidths=0.5)
        # axis labels
        axes[ax].set_xlabel(r'$k_1$')
        axes[ax].set_ylabel(r'$k_2$')

    # title
    axes['left'].set_title('Horizontal Plane')
    axes['right'].set_title('Vertical Plane')
    axes['bottom'].set_xlabel('Max from both planes')

    # finding and plotting global minima
    def find_min_idx(x):
        k = x.argmin()
        ncol = x.shape[1]
        return int(k / ncol), k % ncol

    GMinBetaX = np.array([
        k1s[find_min_idx(BetaMax_horiz)[1]],
        k2s[find_min_idx(BetaMax_horiz)[0]]
    ])

    GMinBetaY = np.array([
        k1s[find_min_idx(BetaMax_vert)[1]],
        k2s[find_min_idx(BetaMax_vert)[0]]
    ])
    GMinBeta = np.array([
        k1s[find_min_idx(BetaMax)[1]],
        k2s[find_min_idx(BetaMax)[0]]
    ])

    axes['left'].scatter(x=GMinBetaX[0], y=GMinBetaX[1], marker='*', s= 50, c='orange')
    axes['right'].scatter(x=GMinBetaY[0], y=GMinBetaY[1], marker='*', s=50, c='orange')
    axes['bottom'].scatter(x=GMinBeta[0], y=GMinBeta[1], marker='*', s=50, c='orange')

    plt.show()

    A0X_horiz = twiss_inv(Beamline(GMinBetaX[0], GMinBetaX[1])['horiz'])
    A0X_vert = twiss_inv(Beamline(GMinBetaX[0], GMinBetaX[1])['vert'])
    GMinBetaFuncX_horiz = update_twiss(A0X_horiz, Beamline(GMinBetaX[0], GMinBetaX[1])['horiz']*4)
    GMinBetaFuncX_vert = update_twiss(A0X_vert, Beamline(GMinBetaX[0], GMinBetaX[1])['vert']*4)

    A0Y_horiz = twiss_inv(Beamline(GMinBetaY[0], GMinBetaY[1])['horiz'])
    A0Y_vert = twiss_inv(Beamline(GMinBetaY[0], GMinBetaY[1])['vert'])
    GMinBetaFuncY_horiz = update_twiss(A0Y_horiz, Beamline(GMinBetaY[0], GMinBetaY[1])['horiz']*4)
    GMinBetaFuncY_vert = update_twiss(A0Y_vert, Beamline(GMinBetaY[0], GMinBetaY[1])['vert']*4)

    A0both_horiz = twiss_inv(Beamline(GMinBeta[0], GMinBeta[1])['horiz'])
    A0both_vert = twiss_inv(Beamline(GMinBeta[0], GMinBeta[1])['vert'])
    GMinBetaFunc_horiz = update_twiss(A0both_horiz, Beamline(GMinBeta[0], GMinBeta[1])['horiz']*4)
    GMinBetaFunc_vert = update_twiss(A0both_vert, Beamline(GMinBeta[0], GMinBeta[1])['vert']*4)

    figu = plt.figure('BetaFunc_GlobMin')
    axi1 = figu.add_subplot(311)
    axi2 = figu.add_subplot(312)
    axi3 = figu.add_subplot(313)

    axi1.plot(GMinBetaFuncX_horiz['s'], np.abs(GMinBetaFuncX_horiz['beta']), 'o-')
    axi1.plot(GMinBetaFuncX_vert['s'], np.abs(GMinBetaFuncX_vert['beta']), 'o-')

    axi2.plot(GMinBetaFuncY_horiz['s'], np.abs(GMinBetaFuncY_horiz['beta']), 'o-')
    axi2.plot(GMinBetaFuncY_vert['s'], np.abs(GMinBetaFuncY_vert['beta']), 'o-')

    axi3.plot(GMinBetaFunc_horiz['s'], np.abs(GMinBetaFunc_horiz['beta']), 'o-')
    axi3.plot(GMinBetaFunc_vert['s'], np.abs(GMinBetaFunc_vert['beta']), 'o-')

    plt.show()
    return BetaMax_horiz, BetaMax_vert, BetaMax

def BetaFunc(k1, k2):
    Beamline_horiz = Beamline(k1, k2)['horiz']

    Beamline_vert = Beamline(k1, k2)['vert']

    twissMat0X = twiss_inv(Beamline_horiz*4)
    twissMat0Y = twiss_inv(Beamline_vert*4)

    Twiss_horiz = update_twiss(twissMat0X, Beamline_horiz*4)
    Twiss_vert = update_twiss(twissMat0Y, Beamline_vert*4)

    fig = plt.figure('BetaFunction_(k1,k2)=(%.3f, %.3f)'%(k1, k2))
    ax = fig.add_subplot(111)

    ax.plot(Twiss_horiz['s'], np.abs(Twiss_horiz['beta']), 'o-r', label=r'$|\beta_x(s)|$')
    ax.plot(Twiss_vert['s'], np.abs(Twiss_vert['beta']), 'o-b', label=r'$|\beta_y(s)|$')

    plt.show()


def ScanTune(lim_k1, lim_k2, N=200):
    k1s = np.linspace(-lim_k1, lim_k1, N)
    k2s = np.linspace(-lim_k2, lim_k2, N)[-1::-1]

    k1s = k1s[k1s != 0]
    k2s = k2s[k2s != 0]

    K1, K2 = np.meshgrid(k1s, k2s)

    Trace_horiz = np.zeros((N, N))
    Trace_vert = np.zeros((N, N))

    Qarr_horiz = np.zeros((N, N))
    Qarr_vert = np.zeros((N, N))

    for i2 in range(N):
        for i1 in range(N):
            k1 = k1s[i1]
            k2 = k2s[i2]

            Beamline_horiz = Beamline(k1, k2)['horiz']

            Beamline_horiz = Beamline_horiz

            OTM_horiz = EffTransMat(Beamline_horiz)['TransMat']

            Beamline_vert = Beamline(k1, k2)['vert']

            Beamline_vert = Beamline_vert

            OTM_vert = EffTransMat(Beamline_vert)['TransMat']

            Trace_horiz[i2, i1] = np.trace(OTM_horiz)
            Trace_vert[i2, i1] = np.trace(OTM_vert)

            stable_horiz = (np.abs(np.trace(OTM_horiz)) < 2)
            stable_vert = (np.abs(np.trace(OTM_vert)) < 2)

            stable = np.logical_and(stable_horiz, stable_vert)

            if stable:
                Q_X = twiss_inv(Beamline_horiz*4, ret_tune=True)[1]
                Q_Y = twiss_inv(Beamline_vert*4, ret_tune=True)[1]

                Qarr_horiz[i2, i1] = Q_X
                Qarr_vert[i2, i1] = Q_Y

            else:
                Qarr_horiz[i2, i1] = 0
                Qarr_vert[i2, i1] = 0

    fig, axes = plt.subplot_mosaic([['left', 'right'], ['left', 'right']], constrained_layout=True)

    axes['left'].contourf(K1, K2, Qarr_horiz, levels=150, vmax=1, cmap='binary')
    c = axes['right'].contourf(K1, K2, Qarr_vert, levels=150, vmax=1, cmap='binary')
    cb = fig.colorbar(c, label=r'$Q$')

    lev1x = axes['left'].contour(K1, K2, Trace_horiz, levels=[-2, 2], linestyles=['solid', 'solid'], colors='red',
                                 linewidths=0.75)
    axes['left'].clabel(lev1x)
    lev1y = axes['left'].contour(K1, K2, Trace_vert, levels=[-2, 2], linestyles=['solid', 'solid'], colors='blue',
                                 linewidths=0.75)
    axes['left'].clabel(lev1y)

    lev2x = axes['right'].contour(K1, K2, Trace_horiz, levels=[-2, 2], linestyles=['solid', 'solid'], colors='red',
                                  linewidths=0.75)
    axes['right'].clabel(lev2x)
    lev1y = axes['right'].contour(K1, K2, Trace_vert, levels=[-2, 2], linestyles=['solid', 'solid'], colors='blue',
                                  linewidths=0.75)
    axes['right'].clabel(lev1y)

    for ax in ['left', 'right']:
        # guide lines
        axes[ax].hlines(y=0, xmin=np.min(k1s), xmax=np.max(k1s), colors='black', linewidths=0.75)
        axes[ax].hlines(y=-1 / rho_b ** 2, xmin=np.min(k1s), xmax=np.max(k1s), colors='black', linewidths=0.75)
        axes[ax].vlines(x=0, ymin=np.min(k2s), ymax=np.max(k2s), colors='black', linewidths=0.75)
        axes[ax].vlines(x=-1 / rho_b ** 2, ymin=np.min(k2s), ymax=np.max(k2s), colors='black', linewidths=0.75)
        # axis labels
        axes[ax].set_xlabel(r'$k_1$')
        axes[ax].set_ylabel(r'$k_2$')

    plt.show()


ScanTune(30, 30, N = 100)









