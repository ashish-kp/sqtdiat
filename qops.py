import numpy as np
import matplotlib.pyplot as plt
import qiskit as qt
from qiskit.visualization import plot_bloch_vector
import qutip as qt
from scipy.optimize import minimize

basis_h = np.array([1, 0])
basis_v = np.array([0, 1])
basis_d = np.array([1, 1]) / np.sqrt(2)
basis_a = np.array([1, -1]) / np.sqrt(2)
basis_l = np.array([1, 1j]) / np.sqrt(2)
basis_r = np.array([1, -1j]) / np.sqrt(2)

bell_1 = np.array([1, 0, 0, 1]) / np.sqrt(2)
bell_2 = np.array([1, 0, 0, -1]) / np.sqrt(2)
bell_3 = np.array([0, 1, 1, 0]) / np.sqrt(2)
bell_4 = np.array([0, 1, -1, 0]) / np.sqrt(2)

I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]]) 
Z = np.array([[1, 0], [0, -1]])

def norm_state_vec(state_vec):
    """Normalizes a given list of complex numbers and returns a numpy array."""
    if type(state_vec) != np.array:
        state_vec = np.array(state_vec)
    norm_fact = 0
    for x in state_vec:
        norm_fact += x * np.conj(x)
    return 1 / np.sqrt(np.real(norm_fact)) * state_vec

def proj(v, u):
    if type(u) or type(v) != np.array:
        u, v = np.array(u), np.array(v)
    return np.dot(u, v) / np.dot(u, u) * u

def gram_schmidt(basis_set):
    if type(basis_set) != np.array:
        basis_set = np.array(basis_set)
    u = np.ones(basis_set.shape)
    u[0] = basis_set[0]
    for i in range(0, basis_set.shape[0]):
        subt = np.zeros(basis_set.shape[1])
        for j in range(0, i):
            subt = subt + proj(basis_set[i], u[j])
        u[i] = basis_set[i] - subt
        u[i] = u[i] / np.sqrt(np.dot(u[i], np.conj(u[i])))
    return u

def trace(sqr_mat):
    """Returns the trace of a given square matrix"""
    if type(sqr_mat) != np.array:
        sqr_mat = np.array(sqr_mat)
    if sqr_mat.shape[0] == sqr_mat.shape[1]:
        return np.sum([sqr_mat[i][i] for i in range(sqr_mat.shape[0])])
    else:
        return "Entered matrix is not a square matrix."

def density_mat_from_state_vec(state_vec):
    """Creates a density matrix from a state vector."""
    if type(state_vec) != np.array:
        state_vec = np.array(state_vec)
    return np.dot(state_vec.reshape(state_vec.shape[0], 1), np.conj(state_vec.reshape(1, state_vec.shape[0])))

def randm_dens():
    """Generates a random 1 qubit density matrix"""
    sig_x = np.array([[0, 1], [1, 0]])
    sig_y = np.array([[0, -1j], [1j, 0]])
    sig_z = np.array([[1, 0], [0, -1]])
    x, z = np.random.random() - 0.5, np.random.random() - 0.5
    y = (-1)**(np.random.randint(2)) * np.sqrt(1 - x**2 - z**2)
    return (np.eye(2) + x * sig_x + y * sig_y + z * sig_z) / 2


def complex_conjugate(dens_mat):
    """Returns the conjugate of a given matrix"""
    if type(dens_mat) != np.array:
        dens_mat = np.array(dens_mat)
        return np.conj(dens_mat).T

def nxn_valid_quantum(sqr_mat):
    """Checks if a given square matrix is a valid density matrix by checking unit trace, Hermiticity and positivity"""
    if type(sqr_mat) != np.array:
        sqr_mat = np.array(sqr_mat)
    flag = True
    if np.sum(sqr_mat == np.conj(sqr_mat).T) != (sqr_mat.shape[0] * sqr_mat.shape[1]): 
        raise ValueError("The given matrix is not Hermitian")
    for x in np.linalg.eigvals(sqr_mat):
        if x < (-1 * 10**-10): flag = False; raise ValueError("Negative eigen values")
    if trace(sqr_mat) < 0.999 or trace(sqr_mat) > 1.0001: flag = False; raise ValueError("Trace is not equal to 1")
    if trace(np.dot(sqr_mat, sqr_mat)) > 1.0001: flag = False; raise ValueError("Trace of rho squared is greater than 1")
    return flag


def arr2num(arr):
    num = 0
    arr = arr[::-1]
    for i in range(len(arr)):
        num += arr[i] * 2**i
        # print(num, arr[i], i)
    return num

def dec2bin_num(num, n):
    bin = []
    while num > 0:
        bin.append(num % 2)
        num //= 2
    while len(bin) < n:
        bin.append(0)
    return bin[::-1]

def generate_bin(n):
    arr = []
    for x in range(2**n):
        arr.append(dec2bin_num(x, n))
    return arr

def linear_entropy(dens_mat):
    """Returns the linear entropy for a given valid density matrix"""
    if type(dens_mat) != np.array:
        dens_mat = np.array(dens_mat)
    if nxn_valid_quantum(dens_mat):
        lin_en = 1 - trace(dens_mat @ dens_mat)
        if abs(lin_en) < 1 * 10**-10: 
            return 0
        else: 
            return lin_en

def partial_transpose(dens_mat, qubit, n):
    """Returns a partially transposed density matrix with the partial transpose performed in the nth qubit.
        Requires : dens_mat = a valid density matrix
                   qubit = integer, the position of the qubit to be flipped.
                   n = dimension of the density matrix either row or column

    """
    if nxn_valid_quantum(dens_mat):
        n = int(np.log2(dens_mat.shape[0]))
        all_binary = generate_bin(n)
        all_binary_2 = generate_bin(n)
        part_qubit = n - qubit - 1
        cnt = 0
        orig = dens_mat
        ppt_dens = np.zeros((2**n, 2**n))
        for i in range(2**n):
            for j in range(2**n):
                if all_binary[i][part_qubit] != all_binary[j][part_qubit]:
                    dens_mat = orig
                    # old_i, old_j = arr2num(all_binary[i]), arr2num(all_binary[j])
                    all_binary_2[i][part_qubit], all_binary_2[j][part_qubit] = all_binary[j][part_qubit], all_binary[i][part_qubit]
                    # print(arr2num(all_binary[i]), arr2num(all_binary[j]), "New",arr2num(all_binary_2[i]), arr2num(all_binary_2[j]))
                    ppt_dens[arr2num(all_binary[i])][arr2num(all_binary[j])] = dens_mat[arr2num(all_binary_2[i])][arr2num(all_binary_2[j])]
                    # print(dens_mat, "\n", ppt_dens)
                else:
                    ppt_dens[i][j] = dens_mat[i][j]
        return ppt_dens

def concurrence(dens_mat):
    """Returns the concurrence for a 2 qubit quantum system.
    Requires : dens_mat = a valid 4 x 4 density matrix."""
    if type(dens_mat) != np.array:
        dens_mat = np.array(dens_mat)
    if nxn_valid_quantum(dens_mat) and dens_mat.shape[0] == 4 and dens_mat.shape[1] == 4:
        eta = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]])
        fin = dens_mat @ eta @ dens_mat.T @ eta 
        eig_val_arr = np.linalg.eigvals(fin)
        r_v = sorted(eig_val_arr, reverse = True)
        val = np.sqrt(r_v[0])
        for x in range(1, 4):
            val -= np.sqrt(r_v[x])
        return max(0. , np.real(np.round(val, 4)))

def bin_ent(x):
    """Returns the binary entropy -xlogx - (1-x)log(1-x)"""
    if x == 0:
        return 0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def entanglement(dens_mat):
    """Returns the entanglement of a given 2 qubit density matrix.
    Requires dens_mat = A valid 4 x 4 density matrix."""
    return bin_ent((1 + np.sqrt(1 - concurrence(dens_mat)**2)) / 2)

def ideal_two_q_tomography(arr):
    """Returns a 4x4 density matrix corresponding to a 2 qubit system which is obtained through performing 16 measurements.
    Requires an array in which the measurements (or the counts) are given in the order : HH, HV, HD, HL, VH, VV, VD, VL, DH, DV, DD, DL, LH, LV, LD, LL"""
    s = [np.eye(2, dtype = 'complex128'), np.array([[0, 1], [1, 0]], dtype = 'complex128'), np.array([[0, -1j], [1j, 0]], dtype = 'complex128'), np.array([[1, 0], [0, -1]], dtype = 'complex128')]
    n_tot = arr[0] + arr[1] + arr[4] + arr[5]
    arr = np.array(arr) / n_tot
    PHH, PHV, PHD, PHL, PVH, PVV, PVD, PVL, PDH, PDV, PDD, PDL, PLH, PLV, PLD, PLL = arr

    arr_16_sq = np.array([[PDD, PDL, PDH, PDV], [PLD, PLL, PLH, PLV], [PHD, PHL, PHH, PHV], [PVD, PVL, PVH, PVV]])
    S_66 = np.zeros((6, 6))
    for i in range(len(S_66)):
        for j in range(len(S_66)):
            if i < 4 and j < 4:
                # print(i, j)
                S_66[i + 2][j + 2] = arr_16_sq[i][j]
    # print(S_66)
    for j in range(5, -1, -1):
        S_66[1][j] = S_66[j][5] + S_66[4][j] - S_66[3][j]
        S_66[0][j] = S_66[j][5] + S_66[4][j] - S_66[2][j]
    for i in range(5, -1, -1):
        S_66[i][1] = S_66[i][5] + S_66[i][4] - S_66[i][3]
        S_66[i][0] = S_66[i][5] + S_66[i][4] - S_66[i][2]
    # print(S_66)

    S_66[:, [2, 0]] = S_66[:, [0, 2]]
    S_66[:, [2, 1]] = S_66[:, [1, 2]]
    S_66[:, [2, 3]] = S_66[:, [3, 2]]
    S_66[[2, 0], :] = S_66[[0, 2], :]
    S_66[[2, 1], :] = S_66[[1, 2], :]
    S_66[[2, 3], :] = S_66[[3, 2], :]

    # print(S_66)
    Stokes = np.zeros((16, 1))
    Stokes[0][0] = 1
    for i in range(4):
        for j in range(4):
            # if i != 
            if i != 0 and j != 0:
                # print(bb[(i - 1) * 2 : (i - 1) * 2 + 2, (j - 1) * 2: (j - 1) * 2 + 2], i, j, "next\n")
                temp_arr = np.array(S_66[(i - 1) * 2 : (i - 1) * 2 + 2, (j - 1) * 2: (j - 1) * 2 + 2]).reshape(1, 4)
                Stokes[i * 4 + j] = temp_arr @ np.array([1, -1, -1, 1]).reshape(4, 1)
            if i == 0 and j != 0:
                # print(bb[(j - 1) * 2 : j * 2, (j - 1) * 2: 2 * j], i, j, "next1\n")
                temp_arr = np.array(S_66[(j - 1) * 2 : j * 2, (j - 1) * 2: 2 * j]).reshape(1, 4)
                Stokes[i * 4 + j] = temp_arr @ np.array([1, -1, 1, -1]).reshape(4, 1)
            if i != 0 and j == 0:
                # print(bb[(i - 1) * 2 : i * 2, (i - 1) * 2: 2 * i], i, j, "next2\n")
                temp_arr = np.array(S_66[(i - 1) * 2 : i * 2, (i - 1) * 2: 2 * i]).reshape(1, 4)
                Stokes[i * 4 + j] = temp_arr @ np.array([1, 1, -1, -1]).reshape(4, 1)
    # Stokes = Stokes.reshape(16, 1) 
    b = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            b = b + Stokes[4 * i + j] * np.kron(s[i], s[j])
            # print(Stokes[4 * i + j], i, j)
    b = b / 4
    return np.round(np.real(b), 3)

def ppt(dens_mat, qubit):
    if type(dens_mat) != np.array:
        dens_mat = np.array(dens_mat) 
    n = dens_mat.shape[0]
    new_mat = partial_transpose(dens_mat, qubit, n)
    eig_vals_dm = np.linalg.eigvals(new_mat)
    for x in np.round(eig_vals_dm, 4):
        if x < 1e-4:
            return False
    return True

def ideal_one_q_tomography(N_H, N_V, N_D, N_R):
    N_t = N_H + N_V
    P_H = N_H / N_t
    P_V = N_V / N_t
    P_D = N_D / N_t
    P_R = N_R / N_t
    return np.array([[2 * P_H, (2 * P_D - 1) - 1j * (1 - 2 * P_R)], [(2 * P_D - 1) + 1j * (1 - 2 * P_R), 2 * P_V]]) * 0.5

def sqrt_dens_mat(dens_mat):
    # if nxn_valid_quantum(dens_mat):
    eig_vals, eig_vecs = np.linalg.eig(dens_mat)
    eig_vals = np.real(eig_vals)
    iden = np.eye(len(eig_vals))
    for i in range(len(eig_vals)):
        if np.abs(eig_vals[i]) < 10e-4:
            val = 0
        else:
            val = np.sqrt(eig_vals[i])
        iden[i] = val * iden[i]
    print(iden)
    return eig_vecs.T @ iden @ np.linalg.inv(eig_vecs.T)

def fidelity(rho1, rho2):
    if type(rho1) != np.array:
        rho1 = np.array(rho1)
    if type(rho2) != np.array:
        rho2 = np.array(rho1)
    if rho1.shape == rho2.shape:
        if nxn_valid_quantum(rho1) and nxn_valid_quantum(rho2):
            sqrt_rho1 = sqrt_dens_mat(rho1)
            val = sqrt_rho1 @ rho2 @ sqrt_rho1
            if np.imag(trace(val)) < 10e-4:
                return np.round(np.real((trace(sqrt_dens_mat(val)))**2), 4)
    else:
        raise ValueError(f"Given density matrices are not of same dimension {rho1.shape}, {rho2.shape}")

def stokes_two_qubit(dens_mat):
    if nxn_valid_quantum(dens_mat):
        if type(dens_mat) != np.array:
            dens_mat = np.array(dens_mat)
        pauli = [np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
        stokes_vec = []
        for x in pauli:
            for y in pauli:
                stokes_vec.append(np.round(np.real(trace(np.kron(x, y) @ dens_mat)), 5))
                # print(np.kron(x, y))
        return stokes_vec

def plot_bloch_vector_from_dm(dens_mat):
    if stokes_vec_from_dens_mat(dens_mat):
        _, x, y, z = stokes_vec_from_dens_mat(dens_mat)
        return plot_bloch_vector([x, y, z])

def stokes_vec_from_dens_mat(dens_mat):
    if type(dens_mat) != np.array:
        dens_mat = np.array(dens_mat)
    s0 = np.real(trace(dens_mat))
    s1 = np.real((trace(np.array([[0, 1], [1, 0]]) @ dens_mat)))
    s2 = np.real((trace(np.array([[0, -1j], [1j, 0]]) @ dens_mat)))
    s3 = np.real(trace(np.array([[1, 0], [0, -1]]) @ dens_mat))
    return [s0, s1, s2, s3]

def rho_from_t(t_arr):
    t1, t2, t3, t4 = t_arr
    T_d = np.array([[t1, 0], [t2 + 1j * t3, t4]])
    return np.conj(T_d).T @ T_d

def project(state_vec):
    return density_mat_from_state_vec(norm_state_vec(state_vec))



def cons2(t_arr):
#     return x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 - 1
    t1, t2, t3, t4 = t_arr
    return t1**2 + t2**2 + t3**2 + t4**2 - 1

def single_qubit_tomography(N_all, x0 = [1, 1, 1, 1], C = [0, 0]):
    """Returns a density matrix for the given four population values measured in the Horizontal, Vertical, Diagonal and Left Circular Basis.

    Requires the following:

    a) N_all = a four element array, containing the populations in the following order [N_H, N_V, N_D, N_L]
    b) x0 (Optional) = four element array, containing the four initial values for t1, t2, t3 and t4. 
    c) C = two element crosstalk array, containing the crosstalk between horizontal to vertical and vertical to horizontal crosstalk percentages.

    Returns: 

    The density matrix for the given measured values.
    """
    N_all[0] = N_all[0] + (C[0]) * N_all[0] - (C[1] * N_all[1])
    N_all[1] = N_all[1] + (C[1]) * N_all[1] - (C[0] * N_all[0])
    N_all = np.array(N_all) / (N_all[0] + N_all[1])
    cons = [{'type': 'eq', 'fun': cons2}]
    
    def objective_func(t_arr):
        rho = rho_from_t(t_arr)
        M_vu = np.array([project([1, 0]), project([0, 1]), project([1, 1]), project([1, 1j])])
        L = 0
        for i in range(4):
            L += (np.trace(M_vu[i] @ rho) - N_all[i])**2 / (2 * np.trace(M_vu[i] @ rho))
        return np.real(L)
    
    res = minimize(objective_func, x0, method='SLSQP', constraints=cons)

    t1, t2, t3, t4 = res.x
    T_d = np.array([[t1, 0], [t2 + (1j * t3), t4]])
    return np.round(np.conj(T_d).T @ T_d, 4)

def state_rand(n):
    """Generates a random complex state vector.
   
    Parameters:
        n (int): The length of the state vector.
   
    Returns:
        state_vector (numpy array): The complex state vector.
    """
    vec1 = np.random.rand(2**n)
    vec2 = np.random.rand(2**n)
    state_vector = vec1 + vec2 * 1j
    return state_vector
 
def density_rand(n):
    """Generates a random density matrix.
   
    Parameters:
        n (int): The length of the state vector.
   
    Returns:
        density_random (numpy array): The random density matrix.
        trace (float): The trace of the density matrix.
    """
    state_vector = state_rand(n)
    density_random = density_mat_from_state_vec(state_vector)
    return density_random

def partial_trace(tot, pttx, rho = 'ran'):
    """
    Returns the partial trace of a density matrix
    """
    kk = [2**i for i in pttx]
    kk = kk[::-1]
    jj = [2**i for i in range(tot) if i not in pttx]
    jj = jj[::-1]
    if rho == 'ran':
        rho = density_mat_from_state_vec(normalize_state_vec([i for i in range(1, 2**tot + 1)]))

    def binfromdec(number, rng):
        arr = []
        while number > 0:
            arr.append(str(number % 2))
            number //= 2 
        while len(arr) < rng:
            arr.append(str(0))
        return "".join(arr[::-1])


    gg = []

    for i in range(2**(tot - len(pttx))):
        str1 = binfromdec(i, tot - len(pttx))
        sum = 0
        for j in range(len(str1)):
            sum += int(str1[j]) * jj[j]
        gg.append(sum)

    hh = []
    for i in range(2**(len(pttx))):
        str1 = binfromdec(i, len(pttx))
        sum = 0
        for j in range(len(str1)):
            sum += int(str1[j]) * kk[j]
        hh.append(sum)
    hh = np.array(hh)

    ans = np.zeros((2**(tot - len(pttx)), 2**(tot - len(pttx))))
    for i in range(len(hh)):
        
        ll = hh[i] + gg
        for j in range(ans.shape[0]):
            for k in range(ans.shape[0]):
                ans[j][k] += rho[ll[j]][ll[k]]
    return ans
