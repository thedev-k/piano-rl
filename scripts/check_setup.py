import sys

libraries = [
    ("numpy", "numpy"),
    ("gymnasium", "gymnasium"),
    ("stable-baselines3", "stable_baselines3"),
    ("torch (CPU)", "torch"),
    ("pretty_midi", "pretty_midi"),
    ("mido", "mido"),
    ("music21", "music21"),
    ("matplotlib", "matplotlib"),
    ("pytest", "pytest"),
    ("tensorboard", "tensorboard"),
]

all_passed = True
print(f"Python {sys.version.split()[0]} on {sys.platform}\n")
print(f"{'Library':<22} | {'Status':<10} | {'Version'}")
print("-" * 50)

for display_name, module_name in libraries:
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "installed (no __version__)")
        if module_name == "torch":
            cuda_available = module.cuda.is_available()
            print(f"{display_name:<22} | SUCCESS    | {version} (CUDA available: {cuda_available})")
        else:
            print(f"{display_name:<22} | SUCCESS    | {version}")
    except ImportError as e:
        all_passed = False
        print(f"{display_name:<22} | FAILED     | Error: {e}")

print("-" * 50)
if all_passed:
    print("All required libraries were imported successfully!")
else:
    print("Some libraries failed to import. Please check errors above.")
    sys.exit(1)
