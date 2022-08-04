import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

plt.style.use(['science', 'notebook', 'grid'])


# ====================== Components ====================== #
def OF_correction(rigidity, L, B):
    ang_OF = np.arctan(L / (rigidity/B))
    return ang_OF


def DriftZone(L):
    """Transfer Matrix of an L long drift"""
    return [{'TransMat': np.array([[1, L, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]]),
             'len': L
             }]


def ThinQuad(f):
    """Transfer Matrix of a Quadrupole with a focal length f"""
    return [{'TransMat': np.array([[1, 0, 0],
                                   [-1 / f, 1, 0],
                                   [0, 0, 1]]),
             'len': 0
             }]


def Dip(rho, L):
    """Transfer Matrix for a Sector Dipole, with bending radius rho and length L"""
    return [{'TransMat': np.array([[np.cos(L / rho), rho * np.sin(L / rho), rho * (1 - np.cos(L / rho))],
                                   [-np.sin(L / rho), np.cos(L / rho), np.sin(L / rho)],
                                   [0, 0, 1]]),
             'len': L
             }]


def Quad(k, L):
    """Quadrupole magnet with quad gradient k and length L
        k >= 0 --> Focusing
        k < 0 --> De-focusing
    """
    omega = np.sqrt(np.abs(k))
    if k >= 0:
        return [{'TransMat': np.array([[np.cos(omega * L), 1 / omega * np.sin(omega * L), 0],
                                       [-omega * np.sin(omega * L), np.cos(omega * L), 0],
                                       [0, 0, 1]]),
                 'len': L
                 }]

    else:
        return [{'TransMat': np.array([[np.cosh(omega * L), 1 / omega * np.sinh(omega * L), 0],
                                       [-omega * np.sinh(omega * L), np.cosh(omega * L), 0],
                                       [0, 0, 1]]),
                 'len': L
                 }]


def BendingMag(k, rho, ang, PoB):
    """Combined function magnet consisting of Dipole and quadrupole modes
    Bending radius rho, quad gradient k, which bends the reference trajectory by an angle ang
    The dipole component only acts in the plane of bending
    """
    L = rho * ang

    if PoB:
        k = k + 1 / rho ** 2
        omega = np.sqrt(np.abs(k))
        if k > 0:
            return [{'TransMat': np.array(
                [[np.cos(omega * L), 1 / omega * np.sin(omega * L), (1 - np.cos(omega * L)) / (rho * omega ** 2)],
                 [-omega * np.sin(omega * L), np.cos(omega * L), np.sin(omega * L) / (rho * omega)],
                 [0, 0, 1]]),
                'len': L
            }]
        else:
            return [{'TransMat': np.array(
                [[np.cosh(omega * L), 1 / omega * np.sinh(omega * L), (1 - np.cos(omega * L)) / (rho * omega ** 2)],
                 [omega * np.sinh(omega * L), np.cosh(omega * L), np.sin(omega * L) / (rho * omega)],
                 [0, 0, 1]]),
                'len': L
            }]

    else:
        k = -k
        omega = np.sqrt(np.abs(k))
        if k > 0:
            return [{'TransMat': np.array([[np.cos(omega * L), 1 / omega * np.sin(omega * L), 0],
                                           [-omega * np.sin(omega * L), np.cos(omega * L), 0],
                                           [0, 0, 1]]),
                     'len': L
                     }]
        else:
            return [{'TransMat': np.array([[np.cosh(omega * L), 1 / omega * np.sinh(omega * L), 0],
                                           [omega * np.sinh(omega * L), np.cosh(omega * L), 0],
                                           [0, 0, 1]]),
                     'len': L
                     }]


# ====================== Beamline ====================== #
def Beamline(k1, k2, horiz, NStep=1):
    global l_d, rho_b, ang_b
    l_d = 4.7  # meter
    rho_b = 1.65  # meter
    ang_b = np.pi / 12  # radian

    global emittance, momentum_err, Nsec
    emittance = 4e-6 # meter radian
    momentum_err = 1e-4 # dimensionless
    Nsec = 3
    Nmags = 4

    # Septum parameters
    B_OF = 0.7 # Tesla
    rigidity_OF = 6.6 # Tesle meter
    L_OF = 1 # m



    ang_OF = OF_correction(rigidity_OF, L_OF, B_OF)

    ang_cor = ang_OF/(Nsec * Nmags)
    
    if horiz:
        beamline = DriftZone(l_d / NStep) * NStep + \
                   BendingMag(k1, rho_b, (ang_b - ang_cor) / NStep, PoB=True) * NStep + \
                   BendingMag(k2, rho_b, (ang_b - ang_cor) / NStep, PoB=True) * NStep + \
                   BendingMag(k2, rho_b, (ang_b - ang_cor) / NStep, PoB=True) * NStep + \
                   BendingMag(k1, rho_b, (ang_b - ang_cor) / NStep, PoB=True) * NStep +
                   # DriftZone(((l_d - L_OF) / 2) /NStep) * NStep + \
                   #

 \
                else:
        beamline = DriftZone(l_d / NStep) * NStep + \
                   BendingMag(k1, rho_b, (ang_b - ang_cor) / NStep, PoB=False) * NStep + \
                   BendingMag(k2, rho_b, (ang_b - ang_cor) / NStep, PoB=False) * NStep + \
                   BendingMag(k2, rho_b, (ang_b - ang_cor) / NStep, PoB=False) * NStep + \
                   BendingMag(k1, rho_b, (ang_b - ang_cor) / NStep, PoB=False) * NStep

    return beamline


# ====================== Functionalities ====================== #
def EffTransMat(beamline):
    """Multiplies component transfer matrices in order to get Effective Transfer Matrix"""
    EffT = np.eye(beamline[0]['TransMat'].shape[0])
    Tot_len = 0
    for component in beamline[-1::-1]:  # reverse beamline since matrix multiplication acts for the right
        EffT = EffT @ component['TransMat']
        Tot_len = Tot_len + component['len']
    return {'TransMat': EffT, 'len': Tot_len}


def Twiss(beamline):
    """Finds the Twiss function for a given beamline configuration"""
    if not isinstance(beamline, list):
        raise TypeError('beamline must be of type: list')

    OTM = EffTransMat(beamline)
    R = OTM['TransMat']
    psi = np.arccos((R[0, 0] + R[1, 1]) / 2)
    if R[0, 1] < 0:
        psi = 2 * np.pi - psi
    Q = psi / (2 * np.pi)
    beta = (R[0, 1]) / np.sin(psi)
    alpha = (R[0, 0] - R[1, 1]) / (2 * np.sin(psi))
    gamma = (1 + alpha ** 2) / beta

    A0 = np.array([[gamma, alpha], [alpha, beta]])

    Ai = [A0]
    s = [0]

    for component in beamline:
        T_comp = np.array([[component['TransMat'][0, 0], component['TransMat'][0, 1]],
                           [component['TransMat'][1, 0], component['TransMat'][1, 1]]])
        T_inv = np.linalg.inv(T_comp)
        Ai.append(T_inv.T @ Ai[-1] @ T_inv)
        s.append(s[-1] + component['len'])
    Ai = np.array(Ai).transpose()
    return {'gamma': Ai[0][0],
            'alpha01': Ai[0][1],
            'alpha10': Ai[1][0],  # equal to alpha-new01
            'beta': Ai[1][1],
            's': np.array(s),
            'Ai': Ai,
            'Q': Q}


def Dispersion(beamline):
    """Takes in a beamline and calculates the dispersion as a function of path length travelled along the beamline"""
    if not (isinstance(beamline, list)):
        raise TypeError("beamline must be of type list, make sure to use 'horiz'/'vert' key!")

    OTM = EffTransMat(beamline)
    R = OTM['TransMat']

    M = np.array([[1 - R[1, 1], R[0, 1]],
                  [R[1, 0], 1 - R[0, 0]]])

    DVec = np.array([R[0, 2], R[1, 2]])

    disp0 = 1 / np.linalg.det(M) * M.dot(DVec)
    disp0 = np.append(disp0, [1])

    disp_arr = [disp0]
    s_arr = [0]
    for component in beamline:
        s_arr.append(s_arr[-1] + component['len'])
        disp_arr.append(component['TransMat'] @ disp_arr[-1])

    disp_arr = np.array(disp_arr)

    eta = disp_arr[:, 0]
    etaPr = disp_arr[:, 1]

    return {'eta': eta, 'etaPr': etaPr, 's': s_arr}


def PlotTwiss(k1, k2, interpolate, PlotDispersion=True, PlotBeamSize=True):

    if interpolate:
        interpSteps = 10
        linemark = '-'
    else:
        interpSteps = 1
        linemark = 'o-'

    if PlotBeamSize:
        Nfig1 = 211
        Nfig2 = 212
    else:
        Nfig1 = 111

    fig = plt.figure('TwissPlot_at_(k1,k2)=(%.3f, %.3f)' % (k1, k2))
    ax = fig.add_subplot(Nfig1)

    TwissFunc_horiz = Twiss(Beamline(k1, k2, horiz=True, NStep=interpSteps) * Nsec)
    TwissFunc_vert = Twiss(Beamline(k1, k2, horiz=False, NStep=interpSteps) * Nsec)

    ax.plot(TwissFunc_horiz['s'], TwissFunc_horiz['beta'], linemark, label=r'$|\beta_x(s)|$', c='red')
    ax.plot(TwissFunc_vert['s'], TwissFunc_vert['beta'], linemark, label=r'$|\beta_y(s)|$', c='blue')

    if PlotDispersion:
        DispersionFunc_horiz = Dispersion(Beamline(k1, k2, horiz=True, NStep=interpSteps) * Nsec)
        DispersionFunc_vert = Dispersion(Beamline(k1, k2, horiz=False, NStep=interpSteps) * Nsec)

        ax.plot(DispersionFunc_horiz['s'], DispersionFunc_horiz['eta'], linemark, label=r'$\eta_x(s)$', c='green')
        ax.plot(DispersionFunc_vert['s'], DispersionFunc_vert['eta'], linemark, label=r'$\eta_y(s)$',
                c='black', lw=0.75)

    ax.legend(loc='upper right', fontsize = 12)
    ax.set_xlabel(r's [m]')
    ax.set_ylabel(r'Beta Function [m]')
    ax.text(0.03, 0.93, r'$(k_1, k_2) = (%.3f, %.3f) [Tm]$' % (k1, k2), transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'), size=12)

    if PlotBeamSize:
        DispersionFunc_horiz = Dispersion(Beamline(k1, k2, horiz=True, NStep=interpSteps) * Nsec)
        DispersionFunc_vert = Dispersion(Beamline(k1, k2, horiz=False, NStep=interpSteps) * Nsec)

        BeamSize_horiz = np.sqrt(np.abs(emittance * TwissFunc_horiz['beta']) +
                                 (momentum_err * DispersionFunc_horiz['eta']))

        BeamSize_vert = np.sqrt(np.abs(emittance * TwissFunc_vert['beta']) +
                                (momentum_err * DispersionFunc_vert['eta']))


        axi = fig.add_subplot(Nfig2)

        axi.plot(DispersionFunc_horiz['s'], BeamSize_horiz, linemark, label=r'$\sigma_x(s)$', c='red')
        axi.plot(DispersionFunc_vert['s'], BeamSize_vert, linemark, label=r'$\sigma_y(s)$', c='blue')

        axi.legend(loc='upper left', fontsize=12)
        axi.set_xlabel(r'$s [m]$')
        axi.set_ylabel('Beamsize [m]')

    plt.show()


#  PlotTwiss(4.573, -3.065, True)


def ScanStability(lim_k1, lim_k2, N=400):
    k1s = np.linspace(-lim_k1, lim_k1, N)
    k2s = np.linspace(-lim_k2, lim_k2, N)[-1::-1]

    k1s = k1s[k1s != 0]
    k2s = k2s[k2s != 0]

    K1, K2 = np.meshgrid(k1s, k2s)

    Trace_horiz = np.zeros((N, N))
    Trace_vert = np.zeros((N, N))

    for i2 in range(N):
        for i1 in range(N):

            k1 = k1s[i1]
            k2 = k2s[i2]

            Beamline_horiz = Beamline(k1, k2, horiz=True)

            Beamline_horiz = Beamline_horiz * Nsec

            OTM_horiz = EffTransMat(Beamline_horiz)['TransMat']

            Beamline_vert = Beamline(k1, k2, horiz=False)

            Beamline_vert = Beamline_vert * Nsec

            OTM_vert = EffTransMat(Beamline_vert)['TransMat']

            Trace_horiz[i2, i1] = OTM_horiz[0, 0] + OTM_horiz[1, 1]
            Trace_vert[i2, i1] = OTM_vert[0, 0] + OTM_vert[1, 1]

    fig = plt.figure('Stability')
    ax = fig.add_subplot(111)

    cs_horiz = ax.contour(K1, K2, Trace_horiz, levels=[-2, 2], linestyles=['solid', 'solid'],
                          colors='red', linewidths=0.75)
    ax.clabel(cs_horiz)

    cs_vert = ax.contour(K1, K2, Trace_vert, levels=[-2, 2], linestyles=['solid', 'solid'],
                         colors='blue', linewidths=0.75)
    ax.clabel(cs_vert)

    stale_horiz = np.abs(Trace_horiz) < 2
    stale_vert = np.abs(Trace_vert) < 2

    stable = np.logical_and(stale_horiz, stale_vert)
    stable = stable.astype(int)

    ax.contourf(K1, K2, stable, levels=[0, 0.5, 1.5], colors=[(0, 0, 0, 0), (0, 1, 0, 0.5)])

    ax.set_xlabel(r'$k_1$')
    ax.set_ylabel(r'$k_2$')

    ax.hlines(y=[0, -1/rho_b**2], xmin=-lim_k1, xmax=lim_k1, colors='gray',
              linestyles=['solid', 'dashed'], linewidths=1)
    ax.vlines(x=[0, -1 / rho_b ** 2], ymin=-lim_k2, ymax=lim_k2, colors='gray',
              linestyles=['solid', 'dashed'], linewidths=1)

    plt.show()

    return Trace_horiz, Trace_vert


#  ScanStability(10, 10, 200)


def ScanBetaMax(lim_k1, lim_k2, N=400):

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

            Beamline_horiz = Beamline(k1, k2, horiz=True)

            Beamline_horiz = Beamline_horiz * Nsec

            OTM_horiz = EffTransMat(Beamline_horiz)['TransMat']

            Beamline_vert = Beamline(k1, k2, horiz=False)

            Beamline_vert = Beamline_vert * Nsec

            OTM_vert = EffTransMat(Beamline_vert)['TransMat']

            Trace_horiz[i2, i1] = OTM_horiz[0, 0] + OTM_horiz[1, 1]
            Trace_vert[i2, i1] = OTM_vert[0, 0] + OTM_vert[1, 1]

            stable_horiz = (np.abs(OTM_horiz[0, 0] + OTM_horiz[1, 1]) < 2)
            stable_vert = (np.abs(OTM_vert[0, 0] + OTM_vert[1, 1]) < 2)

            stable = np.logical_and(stable_horiz, stable_vert)

            if stable:
                TwissFunc_horiz = Twiss(Beamline_horiz)
                TwissFunc_vert = Twiss(Beamline_vert)

                BetaMax_horiz[i2, i1] = np.max(TwissFunc_horiz['beta'])
                BetaMax_vert[i2, i1] = np.max(TwissFunc_vert['beta'])

            else:
                BetaMax_horiz[i2, i1] = np.inf
                BetaMax_vert[i2, i1] = np.inf

    BetaMax = np.fmax(BetaMax_horiz, BetaMax_vert)

    fig, axes = plt.subplot_mosaic([['left', 'right'], ['bottom', 'bottom']], constrained_layout=True)

    axes['left'].contourf(K1, K2, np.log(BetaMax_horiz), levels=150, vmax=4.5, cmap='turbo')
    axes['right'].contourf(K1, K2, np.log(BetaMax_vert), levels=150, vmax=4.5, cmap='turbo')

    c = axes['bottom'].contourf(K1, K2, np.log(BetaMax), levels=150, vmax=4.5, cmap='turbo')
    cb = fig.colorbar(c, label=r'$\ln(Max[|\beta(s)|])$')

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

    lev3x = axes['bottom'].contour(K1, K2, Trace_horiz, levels=[-2, 2], linestyles=['solid', 'solid'], colors='red',
                                   linewidths=0.75)
    axes['bottom'].clabel(lev3x)
    lev1y = axes['bottom'].contour(K1, K2, Trace_vert, levels=[-2, 2], linestyles=['solid', 'solid'], colors='blue',
                                   linewidths=0.75)
    axes['bottom'].clabel(lev1y)

    for ax in ['left', 'right', 'bottom']:
        # guide lines
        axes[ax].hlines(y=[0, -1/rho_b**2], xmin=np.min(k1s), xmax=np.max(k1s), linestyles=['solid', 'dashed'],
                        colors='gray', linewidths=1)

        axes[ax].vlines(x=[0, -1/rho_b**2], ymin=np.min(k2s), ymax=np.max(k2s),  linestyles=['solid', 'dashed'],
                        colors='gray', linewidths=0.5)

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

    print(r'Global Min of Max[|beta_x|]: (k1, k2) = (%.3f, %.3f)' % (GMinBetaX[0], GMinBetaX[1]))
    print(r'Global Min of Max[|beta_y|]: (k1, k2) = (%.3f, %.3f)' % (GMinBetaY[0], GMinBetaY[1]))
    print(r'Global Min of Max[|beta|]: (k1, k2) = (%.3f, %.3f)' % (GMinBeta[0], GMinBeta[1]))

    axes['left'].scatter(x=GMinBetaX[0], y=GMinBetaX[1], marker='*', s=50, c='orange')
    axes['right'].scatter(x=GMinBetaY[0], y=GMinBetaY[1], marker='*', s=50, c='orange')
    axes['bottom'].scatter(x=GMinBeta[0], y=GMinBeta[1], marker='*', s=50, c='orange')

    plt.show()

    PlotTwiss(GMinBetaX[0], GMinBetaX[1], interpolate=True, PlotDispersion=False, PlotBeamSize=False)
    PlotTwiss(GMinBetaY[0], GMinBetaY[1], interpolate=True, PlotDispersion=False, PlotBeamSize=False)
    PlotTwiss(GMinBeta[0], GMinBeta[1], interpolate=True, PlotDispersion=False, PlotBeamSize=False)

    return BetaMax_horiz, BetaMax_vert, BetaMax


def ScanBeamSizeMax(lim_k1, lim_k2, N=400):

    k1s = np.linspace(-lim_k1, lim_k1, N)
    k2s = np.linspace(-lim_k2, lim_k2, N)[-1::-1]

    k1s = k1s[k1s != 0]
    k2s = k2s[k2s != 0]

    K1, K2 = np.meshgrid(k1s, k2s)

    Trace_horiz = np.zeros((N, N))
    Trace_vert = np.zeros((N, N))

    BeamSizeMax_horiz = np.zeros((N, N))
    BeamSizeMax_vert = np.zeros((N, N))

    for i2 in range(N):
        for i1 in range(N):
            k1 = k1s[i1]
            k2 = k2s[i2]

            Beamline_horiz = Beamline(k1, k2, horiz=True)

            Beamline_horiz = Beamline_horiz * Nsec

            OTM_horiz = EffTransMat(Beamline_horiz)['TransMat']

            Beamline_vert = Beamline(k1, k2, horiz=False)

            Beamline_vert = Beamline_vert * Nsec

            OTM_vert = EffTransMat(Beamline_vert)['TransMat']

            Trace_horiz[i2, i1] = OTM_horiz[0, 0] + OTM_horiz[1, 1]
            Trace_vert[i2, i1] = OTM_vert[0, 0] + OTM_vert[1, 1]

            stable_horiz = (np.abs(OTM_horiz[0, 0] + OTM_horiz[1, 1]) < 2)
            stable_vert = (np.abs(OTM_vert[0, 0] + OTM_vert[1, 1]) < 2)

            stable = np.logical_and(stable_horiz, stable_vert)

            if stable:
                TwissFunc_horiz = Twiss(Beamline_horiz)
                TwissFunc_vert = Twiss(Beamline_vert)

                DispFunc_horiz = Dispersion(Beamline_horiz)
                DispFunc_vert = Dispersion(Beamline_vert)

                BeamSizeMax_horiz[i2, i1] = np.max(np.sqrt(np.abs(emittance*TwissFunc_horiz['beta']) +
                                                           (momentum_err*DispFunc_horiz['eta'])**2))
                BeamSizeMax_vert[i2, i1] = np.max(np.sqrt(np.abs(emittance*TwissFunc_vert['beta']) +
                                                           (momentum_err*DispFunc_vert['eta'])**2))

            else:
                BeamSizeMax_horiz[i2, i1] = np.inf
                BeamSizeMax_vert[i2, i1] = np.inf

    BeamSizeMax = np.fmax(BeamSizeMax_horiz, BeamSizeMax_vert)

    fig, axes = plt.subplot_mosaic([['left', 'right'], ['bottom', 'bottom']], constrained_layout=True)

    axes['left'].pcolor(K1, K2, BeamSizeMax_horiz, cmap='turbo', shading='auto',
                        norm=colors.LogNorm(vmin=np.min(BeamSizeMax), vmax=0.01*np.max([BeamSizeMax != np.inf])))
    axes['right'].pcolor(K1, K2, BeamSizeMax_vert, cmap='turbo', shading='auto',
                        norm=colors.LogNorm(vmin=np.min(BeamSizeMax), vmax=0.01*np.max([BeamSizeMax != np.inf])))
    c = axes['bottom'].pcolor(K1, K2, BeamSizeMax, cmap='turbo', shading='auto',
                        norm=colors.LogNorm(vmin=np.min(BeamSizeMax), vmax=0.01*np.max([BeamSizeMax != np.inf])))
    cb = fig.colorbar(c, label=r'$Max[\sigma(s)]$')

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

    lev3x = axes['bottom'].contour(K1, K2, Trace_horiz, levels=[-2, 2], linestyles=['solid', 'solid'], colors='red',
                                   linewidths=0.75)
    axes['bottom'].clabel(lev3x)
    lev1y = axes['bottom'].contour(K1, K2, Trace_vert, levels=[-2, 2], linestyles=['solid', 'solid'], colors='blue',
                                   linewidths=0.75)
    axes['bottom'].clabel(lev1y)

    for ax in ['left', 'right', 'bottom']:
        # guide lines
        axes[ax].hlines(y=[0, -1/rho_b**2], xmin=np.min(k1s), xmax=np.max(k1s), linestyles=['solid', 'dashed'],
                        colors='gray', linewidths=1)

        axes[ax].vlines(x=[0, -1/rho_b**2], ymin=np.min(k2s), ymax=np.max(k2s),  linestyles=['solid', 'dashed'],
                        colors='gray', linewidths=0.5)

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

    GMinBeamSizeX = np.array([
        k1s[find_min_idx(BeamSizeMax_horiz)[1]],
        k2s[find_min_idx(BeamSizeMax_horiz)[0]]
    ])

    GMinBeamSizeY = np.array([
        k1s[find_min_idx(BeamSizeMax_vert)[1]],
        k2s[find_min_idx(BeamSizeMax_vert)[0]]
    ])
    GMinBeamSize = np.array([
        k1s[find_min_idx(BeamSizeMax)[1]],
        k2s[find_min_idx(BeamSizeMax)[0]]
    ])

    print(r'Global Min of $Max[\sigma_x(s)]$: (k1, k2) = (%.3f, %.3f)' % (GMinBeamSizeX[0], GMinBeamSizeX[1]))
    print(r'Global Min of $Max[\sigma_y(s)]$: (k1, k2) = (%.3f, %.3f)' % (GMinBeamSizeY[0], GMinBeamSizeY[1]))
    print(r'Global Min of $Max[\sigma(s)]$: (k1, k2) = (%.3f, %.3f)' % (GMinBeamSize[0], GMinBeamSize[1]))

    axes['left'].scatter(x=GMinBeamSizeX[0], y=GMinBeamSizeX[1], marker='*', s=50, c='orange')
    axes['right'].scatter(x=GMinBeamSizeY[0], y=GMinBeamSizeY[1], marker='*', s=50, c='orange')
    axes['bottom'].scatter(x=GMinBeamSize[0], y=GMinBeamSize[1], marker='*', s=50, c='orange')

    plt.show()

    PlotTwiss(GMinBeamSizeX[0], GMinBeamSizeX[1], interpolate=True, PlotDispersion=True, PlotBeamSize=True)
    PlotTwiss(GMinBeamSizeY[0], GMinBeamSizeY[1], interpolate=True, PlotDispersion=True, PlotBeamSize=True)
    PlotTwiss(GMinBeamSize[0], GMinBeamSize[1], interpolate=True, PlotDispersion=True, PlotBeamSize=True)

    return BeamSizeMax_horiz, BeamSizeMax_vert, BeamSizeMax

ScanBeamSizeMax(10, 10, 400)

#ScanBetaMax(10, 10, 100)


