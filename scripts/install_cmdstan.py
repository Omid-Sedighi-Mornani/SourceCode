import cmdstanpy

# Installiere CmdStan vollständig (dauert ~5 Minuten)
cmdstanpy.install_cmdstan(
    version='2.37.0',
    overwrite=True,
    verbose=True
)

# Prüfe Installation
print(cmdstanpy.cmdstan_path())