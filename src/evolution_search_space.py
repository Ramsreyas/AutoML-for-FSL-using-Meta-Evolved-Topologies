# src/evolution_search_space.py
# ==============================================================================
# MODULE: SEARCH SPACE DEFINITIONS
# Research Standard: Define a discrete search space to make optimization feasible.
# ==============================================================================

# --- 1. Architectural Constraints ---
# These limits ensure models fit in GPU memory and train within a reasonable time.
# For a 100-unit budget, we keep models relatively shallow (Macro-NAS style).
MIN_LAYERS = 3
MAX_LAYERS = 9

# --- 2. Gene Definitions (The "LEGO Bricks") ---
# We use a tuple representation: (Layer_Type, Params...)

# Convolutional Genes: ('conv', out_channels, kernel_size, padding)
# We limit choices to standard efficient values to speed up convergence.
CONV_GENES = [
    ('conv', 32, 3, 1),
    ('conv', 64, 3, 1),
    ('conv', 128, 3, 1),
    ('conv', 64, 5, 2),
]

# Pooling Genes: ('pool', kernel_size, stride)
# Pooling is aggressive (stride 2) to reduce feature map size quickly.
POOL_GENES = [
    ('pool', 2, 2),
]

# Activation Genes: ('activation', type)
# BatchNorm is treated as a structural gene often paired with convs.
STRUCTURAL_GENES = [
    ('relu',),
    ('batchnorm',),
]

# The Master Gene Pool
# The evolutionary engine will randomly select from this list.
GENE_POOL = CONV_GENES + POOL_GENES + STRUCTURAL_GENES

# --- 3. Sanity Check Functions ---
# In research, you must ensure 'validity'. A model with 5 pools on a 28x28 image
# will crash because the image size becomes 0x0.

def get_output_shape(genotype, input_size=28):
    """
    Calculates the spatial output size of a genotype.
    Returns 0 if the architecture creates invalid (negative/zero) dimensions.
    """
    current_size = input_size
    
    for gene in genotype:
        layer_type = gene[0]
        
        if layer_type == 'conv':
            # Conv output = (W - K + 2P) / S + 1
            # Here S=1, so: W - K + 2P + 1
            kernel, padding = gene[2], gene[3]
            current_size = (current_size - kernel + 2*padding) + 1
            
        elif layer_type == 'pool':
            # Pool output = (W - K) / S + 1
            kernel, stride = gene[1], gene[2]
            current_size = (current_size - kernel) // stride + 1
            
        if current_size <= 0:
            return 0
            
    return current_size

def is_valid_architecture(genotype):
    """
    Checks if a random genotype constitutes a valid, trainable neural network.
    """
    # Constraint 1: Length
    if not (MIN_LAYERS <= len(genotype) <= MAX_LAYERS):
        return False
    
    # Constraint 2: Must have at least one convolution (to extract features)
    has_conv = any(g[0] == 'conv' for g in genotype)
    if not has_conv:
        return False
        
    # Constraint 3: Must not reduce image size to <= 0
    if get_output_shape(genotype) <= 0:
        return False
        
    return True