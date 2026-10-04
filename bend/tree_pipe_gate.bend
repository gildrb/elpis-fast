# Proof gate of ext 9601 (bend/tree_pipe_laws.bend): the tree pipe proofs plus
# the proof modules that discharge the laws they build on (spec_inv_tree's T0
# and its dependencies: tree acceptance, derive and path, chain acceptance,
# round advance, chain speculation invariance). No main, no foreign proof code
# and no axioms; for `bend-verdict --verdict` without the whole PROOF.bend.
import Base
import ./tree_pipe_laws.bend as Laws
import ./tree_pipe_proof.bend as TreePipeProof
import ./spec_inv_tree_proof.bend as SpecInvTreeProof
import ./spec_inv_proof.bend as SpecInvProof
import ./exl3_tree_accept_proof.bend as TreeAcceptProof
import ./exl3_tree_derive_proof.bend as TreeDeriveProof
import ./exl3_tree_path_proof.bend as TreePathProof
import ./exl3_accept_proof.bend as AcceptProof
import ./exl3_round_proof.bend as RoundProof
