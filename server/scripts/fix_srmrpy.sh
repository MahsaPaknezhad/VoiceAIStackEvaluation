#!/bin/bash
# Fix numpy compatibility issue in multiple packages
PACKAGES=("srmrpy" "gammatone" "pyloudnorm" "resampy")

for package in "${PACKAGES[@]}"; do
    PACKAGE_DIR=$(python -c "import $package; import os; print(os.path.dirname($package.__file__))" 2>/dev/null)
    if [ -d "$PACKAGE_DIR" ]; then
        find "$PACKAGE_DIR" -name "*.py" -exec sed -i 's/\bnp\.float\b/np.float64/g' {} \;
        echo "Fixed $package numpy compatibility"
    else
        echo "Package $package not found or not installed"
    fi
done
