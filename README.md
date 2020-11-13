# Wreath Product Net
Model-related code for Equivariant Maps for Hierarchical Structures

## Vanilla

Vanilla.py contains the basic equivariant layer, with translation and permutation symmetries. wreathProdNetVanilla.py describes the entire network (residual-style architecture).

## Attn

Attn.py contains the augmented equivariant layer, with class-level and geometric attention mechanisms. Note the core layer remains unchanged (and these additions do not affect equivariant properties). wreathProdNetAttn.py describes the entire network (also a residual architecture).
