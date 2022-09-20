import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.integrate import cumulative_trapezoid

plt.style.use(['science', 'notebook'])


# ====================== Components ====================== #
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
                                       [omega * np.sinh(omega * L), np.cosh(omega * L), 0],
                                       [0, 0, 1]]),
                 'len': L
                 }]


def BendingMag(k, rho, ang, PoB):
    """Combined function magnet consisting of dipole and quadrupole modes
    :param k: quadrupole strength [1/m^2]
    :param rho: bending radius [m]
    :param ang: angle bent by the magent [rad]
    :param PoB: Plane of Bending, parameter must be either True of False. When considering the 'x' transverse plane use
    PoB = True to have dipole field components active. For 'y' transverse use PoB = False, this disables dipole filed
    effects.
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


# ====================== Ring  ====================== #
Nsec = 3
Nmags = 4

l_d = 4.7  # meter
rho_b = 1.65  # meter
ang_b = 2 * np.pi / (Nsec * Nmags)  # radian
l_bend = 2 * np.pi / 3 * rho_b  # meter

emittance = 3e-6  # meter radian
momentum_err = 1e-4  # dimensionless


def Beamline(k1, k2, horiz, NStep=1):
    """
    :param k1 outer quadrupole gradient
    :param k2 inner quadrupole gradient
    :param horiz specify True if interested in (x, x') and False if interested in (y, y')
    :param NStep the number of interpolation steps one component in the beam line is split into
    """

    if horiz:
        beamline = DriftZone(l_d / NStep) * NStep + \
                   BendingMag(k1, rho_b, ang_b / NStep, PoB=True) * NStep + \
                   BendingMag(k2, rho_b, ang_b / NStep, PoB=True) * NStep + \
                   BendingMag(k2, rho_b, ang_b / NStep, PoB=True) * NStep + \
                   BendingMag(k1, rho_b, ang_b / NStep, PoB=True) * NStep

        beamline = beamline * Nsec

    else:
        beamline = DriftZone(l_d / NStep) * NStep + \
                   BendingMag(k1, rho_b, ang_b / NStep, PoB=False) * NStep + \
                   BendingMag(k2, rho_b, ang_b / NStep, PoB=False) * NStep + \
                   BendingMag(k2, rho_b, ang_b / NStep, PoB=False) * NStep + \
                   BendingMag(k1, rho_b, ang_b / NStep, PoB=False) * NStep

        beamline = beamline * Nsec

    return beamline


# ====================== Functionalities ====================== #
def DistAlongBeamLine(beamline):
    s_arr = [0]
    for component in beamline:
        s_arr.append(s_arr[-1] + component['Len'])
    return np.array(s_arr)



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
    psi_arr = [psi]

    for component in beamline:
        T_comp = np.array([[component['TransMat'][0, 0], component['TransMat'][0, 1]],
                           [component['TransMat'][1, 0], component['TransMat'][1, 1]]])
        psi_arr.append((T_comp[0, 0] + T_comp[1, 1])/2)
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
            'Q': Q,
            'psi': psi_arr,
            'twiss_inv': A0}


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

    fig = plt.figure('TwissPlot_at_(k1,k2)=(%.3f,%.3f)' % (k1, k2))
    ax = fig.add_subplot(Nfig1)

    TwissFunc_horiz = Twiss(Beamline(k1, k2, horiz=True, NStep=interpSteps))
    TwissFunc_vert = Twiss(Beamline(k1, k2, horiz=False, NStep=interpSteps))

    ax.plot(TwissFunc_horiz['s'], TwissFunc_horiz['beta'], linemark, label=r'$|\beta_x(s)|$', c='red')
    ax.plot(TwissFunc_vert['s'], TwissFunc_vert['beta'], linemark, label=r'$|\beta_y(s)|$', c='blue')
    ax.set_xlabel('s')

    # ax.plot(TwissFunc_horiz['s'], TwissFunc_horiz['gamma'], linemark, label=r'$|\gamma_x(s)|$', c='yellow')
    # ax.plot(TwissFunc_vert['s'], TwissFunc_vert['gamma'], linemark, label=r'$|\gamma_y(s)|$', c='orange')

    # l_section = l_d + (2 * np.pi / 3) * rho_b
    #
    # ax.vlines(x=[l_section, 2 * l_section, 3 * l_section], ymin=min(TwissFunc_horiz['beta']) - 1,
    #           ymax=min(TwissFunc_horiz['beta']) + 1)

    if PlotDispersion:
        DispersionFunc_horiz = Dispersion(Beamline(k1, k2, horiz=True, NStep=interpSteps))

        ax.plot(DispersionFunc_horiz['s'], DispersionFunc_horiz['eta'], linemark, label=r'$\eta_x(s)$', c='green')
        # ax.plot(DispersionFunc_vert['s'], DispersionFunc_vert['eta'], linemark, label=r'$\eta_y(s)$',
        #         c='black', lw=0.75)

    ax.legend(loc='upper right', fontsize=12)
    # ax.set_xlabel(r's [m]')
    ax.set_ylabel(r'Beta Function [m]')
    ax.text(0.03, 0.93, r'$(k_1, k_2) = (%.3f, %.3f) [Tm]$' % (k1, k2), transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='black'), size=12)

    if PlotBeamSize:
        DispersionFunc_horiz = Dispersion(Beamline(k1, k2, horiz=True, NStep=interpSteps))
        DispersionFunc_vert = Dispersion(Beamline(k1, k2, horiz=False, NStep=interpSteps))

        BeamSize_horiz = np.sqrt(np.abs(emittance * TwissFunc_horiz['beta']) +
                                 (momentum_err * DispersionFunc_horiz['eta'])**2)

        BeamSize_vert = np.sqrt(np.abs(emittance * TwissFunc_vert['beta']) +
                                (momentum_err * DispersionFunc_vert['eta'])**2)

        axi = fig.add_subplot(Nfig2)

        axi.plot(DispersionFunc_horiz['s'], BeamSize_horiz, linemark, label=r'$\sigma_x(s)$', c='red')
        axi.plot(DispersionFunc_vert['s'], BeamSize_vert, linemark, label=r'$\sigma_y(s)$', c='blue')

        axi.legend(loc='upper left', fontsize=12)
        axi.set_xlabel(r'$s [m]$')
        axi.set_ylabel('Beamsize [m]')

    plt.show()



def Transition_Gamma(k1, k2):
    BL_x = Beamline(k1=k1, k2=k2, horiz=True, NStep=1)
    GlobArcLen = DistAlongBeamLine(BL_x)
    bend1 = np.logical_and(l_d <= GlobArcLen, GlobArcLen >= l_d + l_bend)
    bend2 = np.logical_and(2*l_d+l_bend <= GlobArcLen, GlobArcLen >= 2*l_d + 2*l_bend)
    bend3 = np.logical_and(3*l_d+2*l_bend <= GlobArcLen, GlobArcLen >= 3*l_d + 3*l_bend)

    eta_x = Dispersion(BL_x)['eta']





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

            Beamline_horiz = Beamline_horiz

            OTM_horiz = EffTransMat(Beamline_horiz)['TransMat']

            Beamline_vert = Beamline(k1, k2, horiz=False)

            Beamline_vert = Beamline_vert

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

    ax.set_xlabel(r'$k_1\ \left[\frac{1}{m^2}\right]$')
    ax.set_ylabel(r'$k_2\ \left[\frac{1}{m^2}\right]$')

    ax.hlines(y=[0, -1 / rho_b ** 2], xmin=-lim_k1, xmax=lim_k1, colors='gray',
              linestyles=['solid', 'dashed'], linewidths=1)
    ax.vlines(x=[0, -1 / rho_b ** 2], ymin=-lim_k2, ymax=lim_k2, colors='gray',
              linestyles=['solid', 'dashed'], linewidths=1)

    return Trace_horiz, Trace_vert


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

            Beamline_horiz = Beamline_horiz

            OTM_horiz = EffTransMat(Beamline_horiz)['TransMat']

            Beamline_vert = Beamline(k1, k2, horiz=False)

            Beamline_vert = Beamline_vert

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
        axes[ax].hlines(y=[0, -1 / rho_b ** 2], xmin=np.min(k1s), xmax=np.max(k1s), linestyles=['solid', 'dashed'],
                        colors='gray', linewidths=1)

        axes[ax].vlines(x=[0, -1 / rho_b ** 2], ymin=np.min(k2s), ymax=np.max(k2s), linestyles=['solid', 'dashed'],
                        colors='gray', linewidths=0.5)

        # axis labels
        axes[ax].set_xlabel(r'$k_1\ \left[\frac{1}{m^2}\right]$')
        axes[ax].set_ylabel(r'$k_2\ \left[\frac{1}{m^2}\right]$')

    # title
    axes['left'].set_title('Horizontal Plane')
    axes['right'].set_title('Vertical Plane')
    axes['bottom'].set_title('Max from both planes')

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

    axes['left'].scatter(x=GMinBetaX[0], y=GMinBetaX[1], marker='*', s=50, c='crimson')
    axes['right'].scatter(x=GMinBetaY[0], y=GMinBetaY[1], marker='*', s=50, c='crimson')
    axes['bottom'].scatter(x=GMinBeta[0], y=GMinBeta[1], marker='*', s=50, c='crimson')

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

            Beamline_horiz = Beamline_horiz

            OTM_horiz = EffTransMat(Beamline_horiz)['TransMat']

            Beamline_vert = Beamline(k1, k2, horiz=False)

            Beamline_vert = Beamline_vert

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

                BeamSizeMax_horiz[i2, i1] = np.max(np.sqrt(np.abs(emittance * TwissFunc_horiz['beta']) +
                                                           (momentum_err * DispFunc_horiz['eta']) ** 2))
                BeamSizeMax_vert[i2, i1] = np.max(np.sqrt(np.abs(emittance * TwissFunc_vert['beta']) +
                                                          (momentum_err * DispFunc_vert['eta']) ** 2))

            else:
                BeamSizeMax_horiz[i2, i1] = np.inf
                BeamSizeMax_vert[i2, i1] = np.inf

    BeamSizeMax = np.fmax(BeamSizeMax_horiz, BeamSizeMax_vert)

    fig, axes = plt.subplot_mosaic([['left', 'right'], ['bottom', 'bottom']], constrained_layout=True)

    axes['left'].pcolor(K1, K2, BeamSizeMax_horiz, cmap='turbo', shading='auto',
                        norm=colors.LogNorm(vmin=np.min(BeamSizeMax), vmax=0.01 * np.max([BeamSizeMax != np.inf])))
    axes['right'].pcolor(K1, K2, BeamSizeMax_vert, cmap='turbo', shading='auto',
                         norm=colors.LogNorm(vmin=np.min(BeamSizeMax), vmax=0.01 * np.max([BeamSizeMax != np.inf])))
    c = axes['bottom'].pcolor(K1, K2, BeamSizeMax, cmap='turbo', shading='auto',
                              norm=colors.LogNorm(vmin=np.min(BeamSizeMax),
                                                  vmax=0.01 * np.max([BeamSizeMax != np.inf])))
    fig.colorbar(c, label=r'$Max[\sigma(s)]$')

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
        axes[ax].hlines(y=[0, -1 / rho_b ** 2], xmin=np.min(k1s), xmax=np.max(k1s), linestyles=['solid', 'dashed'],
                        colors='gray', linewidths=1)

        axes[ax].vlines(x=[0, -1 / rho_b ** 2], ymin=np.min(k2s), ymax=np.max(k2s), linestyles=['solid', 'dashed'],
                        colors='gray', linewidths=0.5)

        # axis labels
        axes[ax].set_xlabel(r'$k_1\ \left[\frac{1}{m^2}\right]$')
        axes[ax].set_ylabel(r'$k_2\ \left[\frac{1}{m^2}\right]$')

    # title
    axes['left'].set_title('Horizontal Plane')
    axes['right'].set_title('Vertical Plane')
    axes['bottom'].set_title('Max from both planes')

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

    axes['left'].scatter(x=GMinBeamSizeX[0], y=GMinBeamSizeX[1], marker='*', s=50, c='crimson')
    axes['right'].scatter(x=GMinBeamSizeY[0], y=GMinBeamSizeY[1], marker='*', s=50, c='crimson')
    axes['bottom'].scatter(x=GMinBeamSize[0], y=GMinBeamSize[1], marker='*', s=50, c='crimson')

    plt.show()

    PlotTwiss(GMinBeamSizeX[0], GMinBeamSizeX[1], interpolate=True, PlotDispersion=True, PlotBeamSize=True)
    PlotTwiss(GMinBeamSizeY[0], GMinBeamSizeY[1], interpolate=True, PlotDispersion=True, PlotBeamSize=True)
    PlotTwiss(GMinBeamSize[0], GMinBeamSize[1], interpolate=True, PlotDispersion=True, PlotBeamSize=True)

    return BeamSizeMax_horiz, BeamSizeMax_vert, BeamSizeMax


def ResonanceAnalysis(k1, k2):
    Beamline_horiz = Beamline(k1, k2, horiz=True, NStep=1)
    Beamline_vert = Beamline(k1, k2, horiz=False, NStep=1)

    Q_x = Twiss(Beamline_horiz)['Q']
    Q_y = Twiss(Beamline_vert)['Q']

    def ProperFractions(n):
        Fracs = []
        Labels = []
        for i in range(1, n):
            for j in range(i + 1, n + 1):

                if __gcd(i, j) == 1:
                    Fracs.append(i / j)
                    Labels.append(str(i) + '/' + str(j))

        Fracs = np.array(Fracs)
        Fracs = np.append(Fracs, 1 / Fracs)
        return Fracs, Labels

    def __gcd(a, b):
        if b == 0:
            return a
        else:
            return __gcd(b, a % b)

    x = np.linspace(0, 1, 1000)

    fig = plt.figure('ResonancePlot', figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    ax1.set_xlabel(r"$Q_x$")
    ax1.set_ylabel(r"$Q_y$")
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ResOrder = 2
    slopes = ProperFractions(ResOrder)[0]

    for slope in slopes:
        ax1.plot(x, slope * x, '--k')
        ax1.plot(x, -slope * x + 1, '--k')
        ax1.plot(x, slope * (x - 1) + 1, '--k')
        ax1.plot(x, -slope * (x - 1), '--k')
        ax1.plot(x, x, '-k')
        ax1.plot(x, -x + 1, '-k')

        ax1.hlines(y=ProperFractions(ResOrder + 1)[0], xmin=0, xmax=1, colors='k', linestyles='solid')
        ax1.vlines(x=ProperFractions(ResOrder + 1)[0], ymin=0, ymax=1, colors='k', linestyles='solid')

    '''Abs. Phase Advance obtained from the Hill Eq. beta'(s) = 1 / psi(s)  ---> psi(s) = integral 1 / beta(s)'''

    Beta_horiz = Twiss(Beamline_horiz)['beta']
    Beta_vert = Twiss(Beamline_vert)['beta']
    s_horiz = Twiss(Beamline_horiz)['s']
    s_vert = Twiss(Beamline_vert)['s']

    running_psi_horiz = cumulative_trapezoid(y=1 / Beta_horiz, x=s_horiz, initial=0)
    running_psi_vert = cumulative_trapezoid(y=1 / Beta_vert, x=s_vert, initial=0)

    psi_horiz = running_psi_horiz[-1]
    psi_vert = running_psi_vert[-1]
    Q_horiz = psi_horiz / (2 * np.pi)
    Q_vert = psi_vert / (2 * np.pi)

    ax1.scatter(x=Q_horiz, y=Q_vert, c='red', s=50, marker="x")

    ax2.plot(s_horiz, running_psi_horiz, label=r'$\psi_x(s)$')
    ax2.plot(s_vert, running_psi_vert, label=r'$\psi_y(s)$')
    ax2.set_xlabel(r'$s$ [m]')
    ax2.set_ylabel(r'$\psi$ [rad]')
    ax2.legend()
    ax2.text(0.03, 0.73, r'$(Q_x, Q_y) = (%.3f, %.3f)$' % (Q_horiz, Q_vert), transform=ax2.transAxes,
             bbox=dict(facecolor='white', edgecolor='black'), size=12)
    ax2.text(0.03, 0.63, r'$(k_1, k_2) = (%.3f, %.3f)$' % (k1, k2), transform=ax2.transAxes,
             bbox=dict(facecolor='white', edgecolor='black'), size=12)

    ax2.grid()

    plt.show()

    return Q_horiz, Q_vert


def TesterFunc(k1, k2):
    lim_k = 0.5
    ScanStability(-k1 + lim_k, -k2 + lim_k, 200)
    plt.scatter(x=k1, y=k2, c='k', marker='o', s=10)
    plt.show()
    PlotTwiss(k1, k2, interpolate=True, PlotDispersion=True, PlotBeamSize=True)
    ResonanceAnalysis(k1, k2)





TesterFunc(-0.3, -0.1)








#
#
# ResonanceAnalysis(-0.186, -0.186)
#
# PlotTwiss(-0.186, -0.186, interpolate=True, PlotDispersion=True, PlotBeamSize=True)

# ScanBetaMax(1/1.65**2, 1/1.65**2, 200)

# ScanBetaMax(1.7, 1.7, 200)
#
# ScanBeamSizeMax(1.7, 1.7, 200)
#
# ResonanceAnalysis(-0.214, -0.162)
# #
# ResonanceAnalysis(-0.311, -0.100)
#
# PlotTwiss(-0.311, -0.100, interpolate=True, PlotDispersion=True, PlotBeamSize=True)
#
# ResonanceAnalysis(-1.273, 0.299)
# ScanStability(7, 7, 600)

# ScanBetaMax(7, 7, 600)


#ScanBeamSizeMax(7, 7, 400)



# PAx = PhaseAdv(Beamline(0.129, -0.222, horiz=True, NStep=30))
# PAy = PhaseAdv(Beamline(0.129, -0.222, horiz=False, NStep=30))
#
#
#
#
# len_bend = rho_b* 2* np.pi / 3
# len_drift = l_d
#
# plt.plot(PAx['s'], PAx['UpperRight'], label = r'$\mathbf{T^x}_{12}(s)$')
# plt.plot(PAy['s'], PAy['UpperRight'], label = r'$\mathbf{T^y}_{12}(s)$')
# plt.axvspan(0, len_drift, color='red', alpha=0.3)
# plt.axvspan(len_drift+len_bend, 2*len_drift+len_bend, color='red', alpha=0.3)
# plt.axvspan(2*len_drift+2*len_bend, 3*len_drift+2*len_bend, color='red', alpha=0.3, label='Drift Zone')
# plt.xlabel('$s$ [m]')
# plt.legend()
#
# plt.show()

# Twiss_x = Twiss(Beamline(-0.129, -0.222, horiz=True, NStep=30))
# Twiss_y = Twiss(Beamline(-0.129, -0.222, horiz=False, NStep=30))
#
# len_bend = rho_b* 2* np.pi / 3
# len_drift = l_d
#
#
# plt.plot(Twiss_x['s'], Twiss_x['beta']*np.sin(Twiss_x['psi']), legend=r"$\beta(s) \sin(\phi)$")
# plt.plot(Twiss_y['s'], Twiss_y['beta']*np.sin(Twiss_y['psi']))
# plt.axvspan(0, len_drift, color='red', alpha=0.3)
# plt.axvspan(len_drift+len_bend, 2*len_drift+len_bend, color='red', alpha=0.3)
# plt.axvspan(2*len_drift+2*len_bend, 3*len_drift+2*len_bend, color='red', alpha=0.3, label='Drift Zone')
# plt.legend()

