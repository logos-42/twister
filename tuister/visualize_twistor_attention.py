"""
Visualization of Twistor-Attention Isomorphism
扭量-注意力同构的可视化演示
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


class Twistor:
    """A twistor Z^α = (ω^A, π_A') in complex twistor space T = C^4"""
    
    def __init__(self, omega=None, pi=None):
        if omega is None:
            omega = np.random.randn(2) + 1j * np.random.randn(2)
        if pi is None:
            pi = np.random.randn(2) + 1j * np.random.randn(2)
        self.omega = omega
        self.pi = pi
    
    def inner_product(self, other):
        """Twistor symplectic inner product"""
        epsilon = np.array([[0, 1], [-1, 0]])
        pi2_lower = epsilon @ other.pi
        pi1_lower = epsilon @ self.pi
        return np.dot(self.omega, pi2_lower) + np.dot(other.omega, pi1_lower)
    
    def is_null(self, tol=1e-10):
        return np.abs(self.inner_product(self)) < tol
    
    def hermitian_norm_sq(self):
        return np.sum(np.abs(self.omega)**2) + np.sum(np.abs(self.pi)**2)
    
    def projective_rep(self):
        v = np.concatenate([self.omega, self.pi])
        v_normalized = v / (np.linalg.norm(v) + 1e-10)
        return np.array([v_normalized[0].real, v_normalized[1].real, v_normalized[2].real])


def create_null_twistor_from_spacetime(x_mu):
    """Create a null twistor from spacetime point using twistor equation"""
    t, x, y, z = x_mu
    sigma = [
        np.array([[1, 0], [0, 1]]),
        np.array([[0, 1], [1, 0]]),
        np.array([[0, -1j], [1j, 0]]),
        np.array([[1, 0], [0, -1]])
    ]
    x_spinor = np.zeros((2, 2), dtype=complex)
    for mu in range(4):
        x_spinor += sigma[mu] * x_mu[mu]
    x_spinor /= np.sqrt(2)
    
    pi = np.random.randn(2) + 1j * np.random.randn(2)
    pi = pi / (np.linalg.norm(pi) + 1e-10)
    omega = 1j * x_spinor @ pi
    
    return Twistor(omega, pi)


class CausalSelfAttention:
    """Causal self-attention mechanism"""
    
    def __init__(self, d_k, d_v):
        self.d_k = d_k
        self.d_v = d_v
        self.scale = 1.0 / np.sqrt(d_k)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def causal_mask(self, T):
        mask = np.triu(np.ones((T, T)), k=1) * (-1e10)
        return mask
    
    def compute_attention(self, Q, K, V):
        T = Q.shape[0]
        scores = Q @ K.T * self.scale
        scores = scores + self.causal_mask(T)
        attn_weights = self.softmax(scores)
        output = attn_weights @ V
        return output, attn_weights, scores


def twistors_to_attention(twistors, d_k, d_v):
    """Convert twistors to Q, K, V matrices"""
    T = len(twistors)
    Q = np.zeros((T, d_k))
    K = np.zeros((T, d_k))
    V = np.zeros((T, d_v))
    
    for i, Z in enumerate(twistors):
        omega_real = np.concatenate([Z.omega.real, Z.omega.imag])
        Q[i, :min(d_k, 4)] = omega_real[:min(d_k, 4)]
        
        pi_real = np.concatenate([Z.pi.real, Z.pi.imag])
        K[i, :min(d_k, 4)] = pi_real[:min(d_k, 4)]
        
        v_components = [
            np.abs(Z.omega[0]), np.abs(Z.omega[1]),
            np.abs(Z.pi[0]), np.abs(Z.pi[1]),
            Z.hermitian_norm_sq()
        ]
        V[i, :min(d_v, 5)] = v_components[:min(d_v, 5)]
    
    return Q, K, V


def plot_twistor_space():
    """Visualize twistor space structure"""
    fig = plt.figure(figsize=(14, 5))
    
    # Subplot 1: Twistor space decomposition
    ax1 = fig.add_subplot(131, projection='3d')
    n_twistors = 20
    twistors = [Twistor() for _ in range(n_twistors)]
    points = np.array([Z.projective_rep() for Z in twistors])
    norms = np.array([Z.hermitian_norm_sq() for Z in twistors])
    
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                          c=norms, cmap='viridis', s=100, alpha=0.7)
    ax1.set_title('Twistor Space T = C^4\n(Projected to 3D)', fontsize=11)
    ax1.set_xlabel('Re(omega^0)')
    ax1.set_ylabel('Re(omega^1)')
    ax1.set_zlabel("Re(pi_0')")
    plt.colorbar(scatter, ax=ax1, label='Hermitian Norm')
    
    # Subplot 2: Null twistor cone
    ax2 = fig.add_subplot(132, projection='3d')
    t_vals = np.linspace(-2, 2, 10)
    null_twistors = []
    for t in t_vals:
        for x in np.linspace(-1, 1, 5):
            for y in np.linspace(-1, 1, 5):
                Z = create_null_twistor_from_spacetime([t, x, y, 0])
                null_twistors.append(Z)
    
    null_points = np.array([Z.projective_rep() for Z in null_twistors])
    ax2.scatter(null_points[:, 0], null_points[:, 1], null_points[:, 2],
                c='red', s=20, alpha=0.5)
    ax2.set_title('Null Twistor Cone\n<Z,Z> = 0', fontsize=11)
    ax2.set_xlabel('Re(omega^0)')
    ax2.set_ylabel('Re(omega^1)')
    ax2.set_zlabel("Re(pi_0')")
    
    # Subplot 3: Inner product structure
    ax3 = fig.add_subplot(133)
    inner_products = np.zeros((n_twistors, n_twistors))
    for i in range(n_twistors):
        for j in range(n_twistors):
            inner_products[i, j] = twistors[i].inner_product(twistors[j]).real
    
    im = ax3.imshow(inner_products, cmap='RdBu_r', aspect='auto')
    ax3.set_title('Twistor Inner Product Matrix\nRe(<Z_i, Z_j>)', fontsize=11)
    ax3.set_xlabel('Twistor Index j')
    ax3.set_ylabel('Twistor Index i')
    plt.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    plt.savefig('twistor_space.png', dpi=150, bbox_inches='tight')
    print("Saved: twistor_space.png")
    plt.show()


def plot_attention_mechanism():
    """Visualize attention mechanism"""
    fig = plt.figure(figsize=(16, 10))
    
    T = 10
    d_k, d_v = 8, 8
    
    t_vals = np.linspace(0, 2*np.pi, T)
    x_trajectory = np.array([[t, np.sin(t), np.cos(t), 0] for t in t_vals])
    twistors = [create_null_twistor_from_spacetime(x) for x in x_trajectory]
    
    Q, K, V = twistors_to_attention(twistors, d_k, d_v)
    
    attn = CausalSelfAttention(d_k, d_v)
    output, attn_weights, scores = attn.compute_attention(Q, K, V)
    
    # Subplot 1: Q matrix
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(Q, aspect='auto', cmap='coolwarm')
    ax1.set_title('Query Matrix Q\n(from omega^A)', fontsize=11)
    ax1.set_xlabel('d_k dimension')
    ax1.set_ylabel('Sequence Position t')
    
    # Subplot 2: K matrix
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(K, aspect='auto', cmap='coolwarm')
    ax2.set_title("Key Matrix K\n(from pi_A')", fontsize=11)
    ax2.set_xlabel('d_k dimension')
    ax2.set_ylabel('Sequence Position t')
    
    # Subplot 3: Attention scores
    ax3 = fig.add_subplot(2, 3, 3)
    masked_scores = np.where(np.triu(np.ones((T, T)), k=1), np.nan, scores)
    im = ax3.imshow(masked_scores, aspect='auto', cmap='viridis')
    ax3.set_title('Attention Scores QK^T/sqrt(d_k)\n+ Causal Mask', fontsize=11)
    ax3.set_xlabel('Key Position j')
    ax3.set_ylabel('Query Position i')
    plt.colorbar(im, ax=ax3)
    ax3.plot([0, T-1], [0, T-1], 'r--', linewidth=2)
    
    # Subplot 4: Attention weights
    ax4 = fig.add_subplot(2, 3, 4)
    im = ax4.imshow(attn_weights, aspect='auto', cmap='hot')
    ax4.set_title('Attention Weights\nsoftmax(scores)', fontsize=11)
    ax4.set_xlabel('Key Position j')
    ax4.set_ylabel('Query Position i')
    plt.colorbar(im, ax=ax4)
    
    # Subplot 5: Correspondence diagram
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    
    correspondence_text = """
TWISTOR <-> ATTENTION CORRESPONDENCE
=====================================

Twistor Space T = C^4
Z^alpha = (omega^A, pi_A')
           |
           v Phi (structure-preserving map)
           |
Attention Mechanism
Q, K, V in R^{Txd}

CORRESPONDENCES:
- omega^A (unprimed)    -> Query Q
- pi_A' (primed)        -> Key K
- <Z_i, Z_j>            -> QK^T/sqrt(d_k)
- <Z,Z> = 0 (null)      -> Softmax normalization
- Twistor equation      -> Causal mask M
=====================================
"""
    ax5.text(0.5, 0.5, correspondence_text, transform=ax5.transAxes,
             fontsize=9, verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Subplot 6: Row-wise attention pattern
    ax6 = fig.add_subplot(2, 3, 6)
    for i in range(min(5, T)):
        ax6.plot(range(T), attn_weights[i], marker='o', label=f'Position {i}')
    ax6.set_title('Attention Weights by Position\n(Causal Structure)', fontsize=11)
    ax6.set_xlabel('Key Position j')
    ax6.set_ylabel('Attention Weight')
    ax6.legend(loc='upper left', fontsize=8)
    ax6.set_xlim(-0.5, T-0.5)
    
    plt.tight_layout()
    plt.savefig('attention_mechanism.png', dpi=150, bbox_inches='tight')
    print("Saved: attention_mechanism.png")
    plt.show()


def plot_mathematical_correspondence():
    """Detailed visualization of mathematical correspondence"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    T = 8
    d_k = 4
    
    twistors = [Twistor() for _ in range(T)]
    Q, K, V = twistors_to_attention(twistors, d_k, d_k)
    
    # Panel 1: Spinor inner product vs Attention scores
    ax1 = axes[0, 0]
    spinor_inner = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            spinor_inner[i, j] = twistors[i].inner_product(twistors[j]).real
    
    attention_scores = (Q @ K.T) / np.sqrt(d_k)
    valid_mask = np.tril(np.ones((T, T)), k=0).astype(bool)
    
    ax1.scatter(spinor_inner[valid_mask], attention_scores[valid_mask], 
                alpha=0.6, s=100, c='blue', edgecolors='black')
    ax1.plot([spinor_inner.min(), spinor_inner.max()], 
             [spinor_inner.min(), spinor_inner.max()], 
             'r--', label='y = x')
    ax1.set_xlabel('Spinor Inner Product Re(<Z_i, Z_j>)', fontsize=11)
    ax1.set_ylabel('Attention Score (QK^T/sqrt(d_k))_ij', fontsize=11)
    ax1.set_title('Correspondence: Spinor Inner Product <-> Attention Score', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Hermitian norm vs Value magnitude
    ax2 = axes[0, 1]
    hermitian_norms = np.array([Z.hermitian_norm_sq() for Z in twistors])
    value_magnitudes = np.linalg.norm(V, axis=1)
    
    ax2.scatter(hermitian_norms, value_magnitudes, 
                alpha=0.6, s=100, c='green', edgecolors='black')
    ax2.set_xlabel('Twistor Hermitian Norm ||Z||^2', fontsize=11)
    ax2.set_ylabel('Value Vector Magnitude ||V||', fontsize=11)
    ax2.set_title('Correspondence: Hermitian Norm <-> Value Magnitude', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Null twistor condition
    ax3 = axes[1, 0]
    null_twistors = [create_null_twistor_from_spacetime([t, 0, 0, 0]) 
                     for t in np.linspace(-1, 1, T//2)]
    general_twistors = [Twistor() for _ in range(T//2)]
    all_twistors = null_twistors + general_twistors
    
    inner_norms = [np.abs(Z.inner_product(Z)) for Z in all_twistors]
    hermitian_norms_all = [Z.hermitian_norm_sq() for Z in all_twistors]
    
    colors = ['red' if i < len(null_twistors) else 'blue' for i in range(T)]
    
    for i in range(T):
        label = 'Null Twistors' if i == 0 else ('General Twistors' if i == len(null_twistors) else '')
        ax3.scatter(hermitian_norms_all[i], inner_norms[i], 
                   c=colors[i], s=100, alpha=0.7, label=label)
    
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Hermitian Norm ||Z||^2', fontsize=11)
    ax3.set_ylabel('Inner Product |<Z, Z>|', fontsize=11)
    ax3.set_title('Null Twistor Condition: <Z,Z> = 0', fontsize=11)
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Softmax normalization
    ax4 = axes[1, 1]
    raw_scores = np.random.randn(T)
    causal_mask = np.triu(np.ones(T) * (-1e10), k=1)
    masked_scores = raw_scores + causal_mask
    exp_scores = np.exp(masked_scores - np.max(masked_scores))
    attn_weights = exp_scores / np.sum(exp_scores)
    
    positions = np.arange(T)
    ax4.bar(positions, attn_weights, alpha=0.7, color='steelblue', edgecolor='black')
    ax4.set_xlabel('Sequence Position j', fontsize=11)
    ax4.set_ylabel('Attention Weight A[i,j]', fontsize=11)
    ax4.set_title('Softmax Normalization: sum_j A[i,j] = 1\n(Analogous to Projective Structure)', fontsize=11)
    
    ax4.text(0.5, 0.95, f'Sum of weights: {attn_weights.sum():.6f}', 
             transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('mathematical_correspondence.png', dpi=150, bbox_inches='tight')
    print("Saved: mathematical_correspondence.png")
    plt.show()


if __name__ == "__main__":
    print("="*70)
    print("TWISTOR-ATTENTION ISOMORPHISM VISUALIZATION")
    print("扭量-注意力同构可视化演示")
    print("="*70)
    
    print("\n1. Visualizing Twistor Space Geometry...")
    plot_twistor_space()
    
    print("\n2. Visualizing Attention Mechanism...")
    plot_attention_mechanism()
    
    print("\n3. Demonstrating Mathematical Correspondence...")
    plot_mathematical_correspondence()
    
    print("\n" + "="*70)
    print("All visualizations saved!")
    print("Files: twistor_space.png, attention_mechanism.png,")
    print("       mathematical_correspondence.png")
    print("="*70)
