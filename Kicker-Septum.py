import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation
plt.style.use(['science', 'notebook', 'grid'])

'''This code does not yet use the dispersion module'''

# ====================== Globals ====================== #
K1, K2 = -0.129, -0.222  # 1/(meter^2)

l_d = 4.7  # meter
l_kicker = 0.3  # meter
l_septum = 1  # meter
rho_b = 1.65  # meter
Nsec = 3
Nmag = 4
ang_b = 2 * np.pi / (Nsec * Nmag)  # radian

emittance = 4e-6  # meter radian
momentum_err = 1e-4

MinDistKS = 0.45  # Kicker-Septum minimal distance
MinDistBS = 0.3  # Bending Section-Septum minimal distance




# ====================== Components ====================== #


def DriftZone(L):
    """Transfer Matrix of an L long drift"""
    return [{'TransMat': np.array([[1, L],
                                   [0, 1]]),
             'len': L
             }]


def ThinQuad(f):
    """Transfer Matrix of a Quadrupole with a focal length f"""
    return [{'TransMat': np.array([[1, 0],
                                   [-1 / f, 1]]),
             'len': 0
             }]


def Dip(rho, L, PoB):
    """Transfer Matrix for a Sector Dipole, with bending radius rho and length L"""

    if PoB:
        return [{'TransMat': np.array([[np.cos(L / rho), rho * np.sin(L / rho)],
                                       [-np.sin(L / rho), np.cos(L / rho)]]),
                'len': L
                }]
    else:
        return [{'TransMat': np.eye(2),
                 'len': L
                 }]


def Quad(k, L):
    """Quadrupole magnet with quad gradient k and length L
        k >= 0 --> Focusing
        k < 0 --> De-focusing
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
    L = rho * ang

    if PoB:
        k = k + 1 / rho ** 2
        omega = np.sqrt(np.abs(k))
        if k > 0:
            return [{'TransMat': np.array([[np.cos(omega * L), 1 / omega * np.sin(omega * L)],
                                           [-omega * np.sin(omega * L), np.cos(omega * L)]]),
                'len': L
            }]
        else:
            return [{'TransMat': np.array([[np.cosh(omega * L), 1 / omega * np.sinh(omega * L)],
                                           [omega * np.sinh(omega * L), np.cosh(omega * L)]]),
                'len': L
            }]

    else:
        k = -k
        omega = np.sqrt(np.abs(k))
        if k > 0:
            return [{'TransMat': np.array([[np.cos(omega * L), 1 / omega * np.sin(omega * L)],
                                           [-omega * np.sin(omega * L), np.cos(omega * L)]]),
                     'len': L
                     }]
        else:
            return [{'TransMat': np.array([[np.cosh(omega * L), 1 / omega * np.sinh(omega * L)],
                                           [omega * np.sinh(omega * L), np.cosh(omega * L)]]),
                     'len': L
                     }]


# ====================== Base Capability ====================== #


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
            'alpha': Ai[0][1],
            'beta': Ai[1][1],
            's': np.array(s),
            'TwissMat': Ai,
            'TwissInv': A0}


def TransportParticles(x0, beamline):
    """Transports Particle, specified in one transverse plane by [x, x'] along the beamline made up of a series of
    components with defined transfer matricies. The function returns arrays containting x(s), x'(s) and s"""

    state = [x0]
    s = [0]

    for component in beamline:
        state.append(component['TransMat'] @ state[-1])
        s.append(s[-1] + component['len'])
    state = np.array(state).transpose()
    return {'x': state[:, 0, :],
            'px': state[:, 1, :],
            's': s}


def GaussianBeam(mean_x, mean_px, sig_x, sig_px, Npar=1000, plot=False):
    beam = np.random.randn(2, Npar)
    beam[0, :] = sig_x * beam[0, :] + mean_x
    beam[1, :] = sig_px * beam[1, :] + mean_px
    return beam


# ====================== Ring Characterization ====================== #

def StoredTwissFunc(H, Step):

    def Beamline(horiz=H, NStep=Step):


        if horiz:
            beamline = DriftZone(l_d / NStep) * NStep + \
                       BendingMag(K1, rho_b, ang_b / NStep, PoB=True) * NStep + \
                       BendingMag(K2, rho_b, ang_b / NStep, PoB=True) * NStep + \
                       BendingMag(K2, rho_b, ang_b / NStep, PoB=True) * NStep + \
                       BendingMag(K1, rho_b, ang_b / NStep, PoB=True) * NStep

            beamline = beamline * 1

        else:
            beamline = DriftZone(l_d / NStep) * NStep + \
                       BendingMag(K1, rho_b, ang_b / NStep, PoB=False) * NStep + \
                       BendingMag(K2, rho_b, ang_b / NStep, PoB=False) * NStep + \
                       BendingMag(K2, rho_b, ang_b / NStep, PoB=False) * NStep + \
                       BendingMag(K1, rho_b, ang_b / NStep, PoB=False) * NStep

            beamline = beamline * 1

        return beamline

    BL = Beamline(horiz=H, NStep=Step)
    twissfunc = Twiss(BL)

    return twissfunc


NInterpSteps = 20

Twiss_x = StoredTwissFunc(H=True, Step=NInterpSteps)
Twiss_y = StoredTwissFunc(H=False, Step=NInterpSteps)



def OneBendBetween(s1, s2, horiz=True, Nstep=1):
    if horiz:
        beamline = DriftZone(L=(l_d - s1 - l_kicker) / Nstep) * Nstep + \
                   BendingMag(K1, rho_b, ang_b / Nstep, PoB=True) * Nstep + \
                   BendingMag(K2, rho_b, ang_b / Nstep, PoB=True) * Nstep + \
                   BendingMag(K2, rho_b, ang_b / Nstep, PoB=True) * Nstep + \
                   BendingMag(K1, rho_b, ang_b / Nstep, PoB=True) * Nstep + \
                   DriftZone(L=s2 / Nstep) * Nstep

    else:
        beamline = DriftZone(L=(l_d - s1 - l_kicker) / Nstep) + \
                   BendingMag(K1, rho_b, ang_b / Nstep, PoB=False) * Nstep + \
                   BendingMag(K2, rho_b, ang_b / Nstep, PoB=False) * Nstep + \
                   BendingMag(K2, rho_b, ang_b / Nstep, PoB=False) * Nstep + \
                   BendingMag(K1, rho_b, ang_b / Nstep, PoB=False) * Nstep + \
                   DriftZone(L=s2 / Nstep) * Nstep

    return beamline


def TwoBendBetween(s1, s2, horiz=True, Nstep=1):
    if horiz:
        beamline = DriftZone(L=(l_d - s1 - l_kicker) / Nstep) * Nstep + \
                   BendingMag(K1, rho_b, ang_b / Nstep, PoB=True) * Nstep + \
                   BendingMag(K2, rho_b, ang_b / Nstep, PoB=True) * Nstep + \
                   BendingMag(K2, rho_b, ang_b / Nstep, PoB=True) * Nstep + \
                   BendingMag(K1, rho_b, ang_b / Nstep, PoB=True) * Nstep + \
                   DriftZone(l_d / Nstep) * Nstep + \
                   BendingMag(K1, rho_b, ang_b / Nstep, PoB=True) * Nstep + \
                   BendingMag(K2, rho_b, ang_b / Nstep, PoB=True) * Nstep + \
                   BendingMag(K2, rho_b, ang_b / Nstep, PoB=True) * Nstep + \
                   BendingMag(K1, rho_b, ang_b / Nstep, PoB=True) * Nstep + \
                   DriftZone(L=s2 / Nstep) * Nstep

    else:
        beamline = DriftZone(L=(l_d - s1 - l_kicker) / Nstep) * Nstep + \
                   BendingMag(K1, rho_b, ang_b / Nstep, PoB=False) * Nstep + \
                   BendingMag(K2, rho_b, ang_b / Nstep, PoB=False) * Nstep + \
                   BendingMag(K2, rho_b, ang_b / Nstep, PoB=False) * Nstep + \
                   BendingMag(K1, rho_b, ang_b / Nstep, PoB=False) * Nstep + \
                   DriftZone(l_d / Nstep) * Nstep + \
                   BendingMag(K1, rho_b, ang_b / Nstep, PoB=False) * Nstep + \
                   BendingMag(K2, rho_b, ang_b / Nstep, PoB=False) * Nstep + \
                   BendingMag(K2, rho_b, ang_b / Nstep, PoB=False) * Nstep + \
                   BendingMag(K1, rho_b, ang_b / Nstep, PoB=False) * Nstep + \
                   DriftZone(L=s2/ Nstep) *Nstep

    return beamline


# ====================== Optimization ====================== #

def EfficientKick():
    N = 100
    s1_arr = np.linspace(0, l_d - l_kicker, N)
    s2_arr = np.linspace(0, l_d - l_septum, N)[-1::-1]

    # s1_arr = np.linspace(0, 10, N)
    # s2_arr = np.linspace(0, 10, N)[-1::-1]
    OneBendKickEff = np.zeros((N, N))
    TwoBendKickEff = np.zeros((N, N))

    S1, S2 = np.meshgrid(s1_arr, s2_arr)

    for i2 in range(N):
        for i1 in range(N):
            s1 = s1_arr[i1]
            s2 = s2_arr[i2]

            OneBendBL = OneBendBetween(s1=s1, s2=s2, horiz=True)
            OneBendOTM = EffTransMat(OneBendBL)['TransMat']

            TwoBendBL = TwoBendBetween(s1=s1, s2=s2, horiz=True)
            TwoBendOTM = EffTransMat(TwoBendBL)['TransMat']

            OneBendKickEff[i2, i1] = OneBendOTM[0, 1]
            TwoBendKickEff[i2, i1] = TwoBendOTM[0, 1]

    def find_min_idx(x):
        k = x.argmin()
        ncol = x.shape[1]
        return int(k / ncol), k % ncol

    MaxKick1_i1, MaxKick1_i2 = find_min_idx(OneBendKickEff)
    MaxKick2_i1, MaxKick2_i2 = find_min_idx(TwoBendKickEff)

    MaxKick1_s1 = s1_arr[MaxKick1_i1]
    MaxKick1_s2 = s2_arr[MaxKick1_i2]
    MaxKick2_s1 = s1_arr[MaxKick2_i1]
    MaxKick2_s2 = s2_arr[MaxKick2_i2]

    fig = plt.figure('Placement', figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    cont1 = ax1.contour(S1, S2, OneBendKickEff, levels = list(np.linspace(1.5, 3, 20)))

    ax1.clabel(cont1)
    ax1.set_xlabel('$s_1$ [m]')
    ax1.set_ylabel('$s_2$ [m]')
    ax1.set_title('One Bending Section \n between the Kicker and the Septum')

    cont2 = ax2.contour(S1, S2, TwoBendKickEff, levels=30)
    ax2.clabel(cont2)
    ax2.set_xlabel('$s_1$ [m]')
    ax2.set_title('Two Bending Sections \n between the Kicker and the Septum')
    #
    # ax1.scatter(MaxKick1_s1, MaxKick1_s2, c='red')
    # ax2.scatter(MaxKick2_s1, MaxKick2_s2, c='red')
    #
    print(MaxKick1_s1, MaxKick1_s2)
    print(np.max(OneBendKickEff))
    print(MaxKick2_s1, MaxKick2_s2)
    print(np.max(TwoBendKickEff))
    plt.show()

    return OneBendKickEff, TwoBendKickEff


def BeamPass(NBendBetween, s1, s2, Ninterp=10):
    Beta_at_Kick_x = np.interp(s1, Twiss_x['s'], Twiss_x['beta'])
    # Beta_at_Kick_y = np.interp(s1, Twiss_y['s'], Twiss_y['beta'])
    Gamma_at_Kick_x = np.interp(s1, Twiss_x['s'], Twiss_x['gamma'])
    # Gamma_at_Kick_y = np.interp(s1, Twiss_y['s'], Twiss_y['gamma'])

    Nsigma = 3

    XBeam_at_Kick = GaussianBeam(mean_x=0, mean_px=0,
                                 sig_x=Nsigma * np.sqrt(emittance * Beta_at_Kick_x),
                                 sig_px=Nsigma * np.sqrt(emittance * Gamma_at_Kick_x))

    # YBeam_at_Kick = GaussianBeam(mean_x=0, mean_px=0,
    #                              sig_x=2 / Nsigma * (emittance * Beta_at_Kick_y),
    #                              sig_px=2 / Nsigma * (emittance * Gamma_at_Kick_y))

    def Kicker(par_beam, kickSt):
        beam_x = par_beam[0, :]
        beam_px = par_beam[1, :]
        beam_px = beam_px + kickSt
        KickedBeam = np.vstack((beam_x, beam_px))
        return KickedBeam

    KickStrength = -0.01

    XBeam_at_Kick = Kicker(XBeam_at_Kick, KickStrength)

    SBendBetween = {0, 1, 2}
    if not(NBendBetween in SBendBetween):
        raise TypeError('NBendBetween can only take on value in {0, 1, 2}')
    elif NBendBetween == 0:
        BL = DriftZone(L=(s2 - s1) / Ninterp) * Ninterp
    elif NBendBetween == 1:
        BL = OneBendBetween(s1, s2, horiz=True, Nstep=Ninterp)
    elif NBendBetween == 2:
        BL = TwoBendBetween(s1, s2, horiz=True, Nstep=Ninterp)

    def get_data(i=0):
        XBeamProp = TransportParticles(XBeam_at_Kick, BL)
        Beam_x = XBeamProp['x'][:, i]
        Beam_px = XBeamProp['px'][:, i]
        s = XBeamProp['s'][i]
        return Beam_x, Beam_px, s

    beam_x, beam_px, s_text = get_data()
    g = sns.JointGrid(x=beam_x, y=beam_px, height=6)
    lim = (-0.07, 0.07)

    def prep_axis(g, xlim, ylim):
        g.ax_joint.clear()
        g.ax_joint.set_xlim(xlim)
        g.ax_joint.set_ylim(ylim)
        g.ax_marg_x.clear()
        g.ax_marg_x.set_xlim(xlim)
        g.ax_marg_y.clear()
        g.ax_marg_y.set_ylim(ylim)
        plt.setp(g.ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(g.ax_marg_y.get_yticklabels(), visible=False)
        plt.setp(g.ax_marg_x.yaxis.get_majorticklines(), visible=False)
        plt.setp(g.ax_marg_x.yaxis.get_minorticklines(), visible=False)
        plt.setp(g.ax_marg_y.xaxis.get_majorticklines(), visible=False)
        plt.setp(g.ax_marg_y.xaxis.get_minorticklines(), visible=False)
        plt.setp(g.ax_marg_x.get_yticklabels(), visible=False)
        plt.setp(g.ax_marg_y.get_xticklabels(), visible=False)
        for txt in g.fig.texts:
            txt.set_visible(False)
        g.set_axis_labels("x [m]", "x' [rad]")
        plt.tight_layout()

    def animate(i):
        prep_axis(g, lim, lim)
        g.x, g.y, s_text = get_data(i)
        g.fig.text(0.25, 0.75, 's = %.3f m ' % s_text, fontsize=16, bbox=dict(facecolor='white', edgecolor='black'))
        g.plot_joint(sns.kdeplot, cmap='Blues')

        g.plot_marginals(sns.kdeplot, color="blue", shade=True)

    FrameNum = (len(BL) * Ninterp) + 2

    ani = matplotlib.animation.FuncAnimation(g.fig, animate, frames= FrameNum)

    plt.show()
    # ani.save('output.gif', writer='pillow')


def OptimizePlacement(NBendBetween):

    N = 80
    s1_arr = np.linspace(0, l_d - l_kicker, N)
    s2_arr = np.linspace(0, l_d - l_septum, N)

    S1, S2 = np.meshgrid(s1_arr, s2_arr)

    Nsigma = 3
    SeptumWall = 5e-3
    SafetyDist = 4e-3
    VacPipe = 2e-3

    DispArr = np.zeros((N, N))
    T12 = np.zeros((N, N))

    BeamPipeSize = np.zeros((N, N))
    Midpoint = np.zeros((N, N))

    for i2 in range(N):
        for i1 in range(N):
            s1 = s1_arr[i1]
            s2 = s2_arr[i2]

            SBendBetween = {0, 1, 2}
            if not (NBendBetween in SBendBetween):
                raise TypeError('NBendBetween can only take on value in {0, 1, 2}')
            elif NBendBetween == 0:
                BL = DriftZone(L=(s2 - s1) / NInterpSteps) * NInterpSteps
            elif NBendBetween == 1:
                BL = OneBendBetween(s1, s2, horiz=True, Nstep=NInterpSteps)
                title_text = 'One bending section between the Kicker and the Septum'
            elif NBendBetween == 2:
                BL = TwoBendBetween(s1, s2, horiz=True, Nstep=NInterpSteps)
                title_text = 'Two Bending sections between the Kicker and the Septum'

            T12[i2, i1] = EffTransMat(BL)['TransMat'][0, 1]

            Beta_at_Sep = np.interp(s2, Twiss_x['s'], Twiss_x['beta'])
            DispArr[i2, i1] = SeptumWall + 2 * SafetyDist + 2 * VacPipe + 2 * Nsigma * np.sqrt(emittance * Beta_at_Sep)

            NominalTraj = [[0], [0.005]]

            TranspNominalTraj = TransportParticles(NominalTraj, BL)
            BeamCent = TranspNominalTraj['x'].flatten()
            BetaFunc = np.interp(TranspNominalTraj['s'], Twiss_x['s'], Twiss_x['beta'])

            BeamPipeSize[i2, i1] = np.max(BeamCent + np.sqrt(emittance * BetaFunc))
            Midpoint[i2, i1] = np.max(BeamCent)

    fig = plt.figure('KickerSeptumOptim', figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    c1 = ax1.contour(S1, S2, DispArr/T12, levels=np.linspace(-1, 1, 30), cmap='turbo')
    cb = fig.colorbar(c1, location="left")

    ax1.set_xlabel(r'$s_1$ [m]')
    ax1.set_ylabel(r'$s_2$ [m]')
    

    ax1.set_title("$\kappa= \int B ds$")

    c2 = ax2.contour(S1, S2, BeamPipeSize, levels=15, cmap='turbo')
    ax2.clabel(c2)
    ax2.set_xlabel(r'$s_1$ [m]')
    ax2.set_ylabel(r'$s_2$ [m]')
    ax2.set_title('Min. Beam Pipe size')
    fig.suptitle(title_text)

    fig.tight_layout()

    plt.show()


OptimizePlacement(1)



def BeamPipe(NBendBetween):
    N = 100
    s1_arr = np.linspace(0, l_d - l_kicker, N)
    s2_arr = np.linspace(0, l_d - l_septum, N)

    S1, S2 = np.meshgrid(s1_arr, s2_arr)

    BeamPipeSize = np.zeros((N, N))
    Midpoint = np.zeros((N, N))
    for i2 in range(N):
        for i1 in range(N):
            s1 = s1_arr[i1]
            s2 = s2_arr[i2]

            SBendBetween = {0, 1, 2}
            if not (NBendBetween in SBendBetween):
                raise TypeError('NBendBetween can only take on value in {0, 1, 2}')
            elif NBendBetween == 0:
                BL = DriftZone(L=(s2 - s1) / NInterpSteps) * NInterpSteps
            elif NBendBetween == 1:
                BL = OneBendBetween(s1, s2, horiz=True, Nstep=NInterpSteps)
                title_text = 'One bending section between the Kicker and the Septum'
            elif NBendBetween == 2:
                BL = TwoBendBetween(s1, s2, horiz=True, Nstep=NInterpSteps)
                title_text = 'Two Bending sections between the Kicker and the Septum'

            NominalTraj = [[0], [0.005]]

            TranspNominalTraj = TransportParticles(NominalTraj, BL)
            BeamCent = TranspNominalTraj['x'].flatten()
            BetaFunc = np.interp(TranspNominalTraj['s'], Twiss_x['s'], Twiss_x['beta'])

            BeamPipeSize[i2, i1] = np.max(BeamCent + np.sqrt(emittance * BetaFunc))
            Midpoint[i2, i1] = np.max(BeamCent)

    fig = plt.figure('Beampipe')
    ax = fig.add_subplot(111)
    c1 = ax.contour(S1, S2, Midpoint, levels=15, cmap='turbo')
    ax.clabel(c1)
    ax.set_xlabel(r'$s_1$ [m]')
    ax.set_ylabel(r'$s_2$ [m]')
    ax.set_title(title_text)




            # plt.plot(TranspNominalTraj['s'], BeamCent)
            # plt.show()

            # print('Beam Cent')
            # print(BeamCent.shape)
            #
            # print('beta func')
            # print(BetaFunc)






    #         BeamWidth = 3 * np.sqrt(emittance * )
    #
    #         BeamPipeSize[i2, i1] = np.max(BeamCent + BeamWidth)
    #
    # plt.contourf(S1, S2, BeamPipeSize, cmap = 'turbo')
    #
    # plt.colorbar(label='Min Beam Pipe Radius [m]' )
    # plt.show()
    #
    #
    #






















    #         BeamPipeSize[i2, i1] = np.max(BeamCenter + 2 * nsig * np.sqrt(emittance * Beta_at_Sep))
    #
    #
    #
    # plt.contour(S1, S2, BeamPipeSize)
    #

    # fig = plt.figure('IntegratedMagneticField')
    # ax = fig.add_subplot(111)
    #
    #
    # c = ax.contour(S1, S2, DispArr/T12, levels=list(np.linspace(-1, 1, 300)), cmap='turbo')
    # cb = fig.colorbar(c, label=r'$\kappa = \int B ds$')
    #
    # ax.set_xlabel(r'$s_1$')
    # ax.set_ylabel(r'$s_2$')
    # ax.grid()







plt.show()



    # BeamPipeSize = np.zeros((N, N))
    # for i2 in range(N):
    #     for i1 in range(N):
    #         s1 = s1_arr[i1]
    #         s2 = s2_arr[i2]
    #
    #         SBendBetween = {0, 1, 2}
    #         if not (NBendBetween in SBendBetween):
    #             raise TypeError('NBendBetween can only take on value in {0, 1, 2}')
    #         elif NBendBetween == 0:
    #             BL = DriftZone(L=(s2 - s1))
    #         elif NBendBetween == 1:
    #             BL = OneBendBetween(s1, s2, horiz=True, Nstep=1)
    #         elif NBendBetween == 2:
    #             BL = TwoBendBetween(s1, s2, horiz=True, Nstep=1)
    #
    #         Twiss_x = Twiss(Section(k1, k2, horiz=True))
    #
    #         Beta_at_septum = np.interp(s2, Twiss_x['s'], Twiss_x['beta'])
    #
    #         Displacement = SeptumWall + 2 * SafetyDist + 2 * VacPipe + 2 * nsig * np.sqrt(emittance * Beta_at_septum)
    #
    #         NominalTraj = [[0], [0]]
    #         BeamCenter = TransportParticles(NominalTraj, BL)
    #
    #         BeamPipeSize[i2, i1] = Displacement + BeamCenter['x'][-1]
    #
    # plt.contour(S1, S2, BeamPipeSize)
    # plt.show()
