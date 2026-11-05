"""Names of the distance modes, as they appear in artifact filenames (C_<mode>_<side>.npy).

    euclidean               L2 distance in latent space
    ensemble_eol_samedec    length of the straight segment under the expected pullback metric
                            of the decoder ensemble: per segment step, the root mean over
                            decoders of the squared output difference, summed over steps
"""

CHEAP_MODES = ["euclidean", "ensemble_eol_samedec"]
ALL_MODES = list(CHEAP_MODES)


def resolve(mode: str) -> str:
    """Validate a mode name."""
    if mode in ALL_MODES:
        return mode
    raise ValueError(f"unknown distance mode {mode!r}; valid modes: {ALL_MODES}")
