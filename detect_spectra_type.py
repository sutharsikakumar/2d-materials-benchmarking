# Claude 4.5 generated code 

"""
2D Material Spectroscopy Technique Identifier
Analyzes raw spectroscopy data to identify the measurement technique
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from pathlib import Path
import sys


class SpectroscopyIdentifier:
    """Identifies spectroscopy technique from raw data characteristics"""
    
    def __init__(self, x_data, y_data):
        self.x = np.array(x_data)
        self.y = np.array(y_data)
        self.technique = None
        self.confidence = 0.0
        self.details = {}
        
    def analyze(self):
        """Main analysis pipeline"""
        # Calculate data statistics
        self._calculate_statistics()
        
        # Run classification heuristics
        scores = {
            'Raman': self._check_raman(),
            'PL (Photoluminescence)': self._check_pl(),
            'UV-Vis Absorption': self._check_uvvis(),
            'FTIR': self._check_ftir(),
            'XPS': self._check_xps(),
            'XRD': self._check_xrd(),
            'Reflectance Contrast': self._check_reflectance(),
            'ARPES': self._check_arpes(),
            'THz Spectroscopy': self._check_thz()
        }
        
        # Find best match
        self.technique = max(scores, key=scores.get)
        self.confidence = scores[self.technique]
        
        return self.technique, self.confidence, scores
    
    def _calculate_statistics(self):
        """Calculate statistical features of the data"""
        self.details['x_range'] = (self.x.min(), self.x.max())
        self.details['y_range'] = (self.y.min(), self.y.max())
        self.details['x_span'] = self.x.max() - self.x.min()
        self.details['n_points'] = len(self.x)
        
        # Normalize y for peak detection
        y_norm = (self.y - self.y.min()) / (self.y.max() - self.y.min() + 1e-10)
        
        # Find peaks
        peaks, properties = signal.find_peaks(y_norm, prominence=0.1, width=1)
        self.details['n_peaks'] = len(peaks)
        
        if len(peaks) > 0:
            self.details['peak_sharpness'] = np.mean(properties['prominences'])
            self.details['avg_peak_width'] = np.mean(properties['widths'])
        else:
            self.details['peak_sharpness'] = 0
            self.details['avg_peak_width'] = 0
    
    def _check_raman(self):
        """Check if data matches Raman spectroscopy"""
        score = 0.0
        x_min, x_max = self.details['x_range']
        
        # Raman: wavenumber typically 100-3500 cm⁻¹
        if 50 < x_min < 500 and 1000 < x_max < 4000:
            score += 40
        
        # Sharp peaks are characteristic
        if self.details['n_peaks'] >= 2 and self.details['peak_sharpness'] > 0.3:
            score += 30
        
        # Typical x-span for Raman
        if 500 < self.details['x_span'] < 3500:
            score += 20
        
        # Y-values typically positive, wide range
        if self.y.min() >= 0:
            score += 10
        
        return score
    
    def _check_pl(self):
        """Check if data matches Photoluminescence"""
        score = 0.0
        x_min, x_max = self.details['x_range']
        
        # PL: wavelength typically 400-1000 nm or energy 1-4 eV
        wavelength_range = 300 < x_min < 500 and 600 < x_max < 1200
        energy_range = 1 < x_min < 2 and 2.5 < x_max < 5
        
        if wavelength_range or energy_range:
            score += 40
        
        # Usually fewer, broader peaks than Raman
        if 1 <= self.details['n_peaks'] <= 5:
            score += 25
        
        # Broader peaks than Raman
        if self.details['avg_peak_width'] > 5:
            score += 20
        
        # Smooth baseline
        if self.y.min() >= 0:
            score += 15
        
        return score
    
    def _check_uvvis(self):
        """Check if data matches UV-Vis absorption/reflectance"""
        score = 0.0
        x_min, x_max = self.details['x_range']
        y_min, y_max = self.details['y_range']
        
        # UV-Vis: wavelength typically 200-800 nm
        if 180 < x_min < 300 and 600 < x_max < 900:
            score += 40
        
        # Absorption: 0-3 range, or % values 0-100
        if (0 <= y_min and y_max <= 4) or (0 <= y_min and 80 <= y_max <= 100):
            score += 30
        
        # Broad features, not many sharp peaks
        if self.details['n_peaks'] <= 3:
            score += 20
        
        # Smooth curve
        if self.details['avg_peak_width'] > 10:
            score += 10
        
        return score
    
    def _check_ftir(self):
        """Check if data matches FTIR"""
        score = 0.0
        x_min, x_max = self.details['x_range']
        y_min, y_max = self.details['y_range']
        
        # FTIR: wavenumber typically 400-4000 cm⁻¹
        if 300 < x_min < 600 and 3000 < x_max < 4500:
            score += 40
        
        # % Transmittance (0-100%) is common
        if 0 <= y_min and 50 <= y_max <= 120:
            score += 25
        
        # Multiple absorption bands
        if 3 <= self.details['n_peaks'] <= 20:
            score += 20
        
        # Different range than Raman
        if self.details['x_span'] > 2000:
            score += 15
        
        return score
    
    def _check_xps(self):
        """Check if data matches XPS"""
        score = 0.0
        x_min, x_max = self.details['x_range']
        
        # XPS: binding energy typically 0-1200 eV
        if -10 < x_min < 100 and 500 < x_max < 1500:
            score += 40
        
        # Sharp peaks at specific binding energies
        if self.details['n_peaks'] >= 2 and self.details['peak_sharpness'] > 0.25:
            score += 30
        
        # Wide scan range
        if self.details['x_span'] > 500:
            score += 20
        
        # Positive intensity (CPS)
        if self.y.min() >= 0:
            score += 10
        
        return score
    
    def _check_xrd(self):
        """Check if data matches XRD"""
        score = 0.0
        x_min, x_max = self.details['x_range']
        
        # XRD: 2θ typically 10-80 degrees
        if 5 < x_min < 15 and 40 < x_max < 90:
            score += 45
        
        # Sharp diffraction peaks
        if self.details['n_peaks'] >= 3 and self.details['peak_sharpness'] > 0.3:
            score += 30
        
        # Relatively narrow x-range
        if 30 < self.details['x_span'] < 80:
            score += 15
        
        # Positive intensity
        if self.y.min() >= 0:
            score += 10
        
        return score
    
    def _check_reflectance(self):
        """Check if data matches Reflectance contrast spectroscopy"""
        score = 0.0
        x_min, x_max = self.details['x_range']
        y_min, y_max = self.details['y_range']
        
        # Wavelength range similar to UV-Vis
        if 350 < x_min < 500 and 600 < x_max < 900:
            score += 35
        
        # % Reflectance (0-100) or small contrast values
        if (0 <= y_min and y_max <= 1.5) or (-0.2 <= y_min and y_max <= 0.5):
            score += 30
        
        # Usually smooth, few features
        if self.details['n_peaks'] <= 2:
            score += 25
        
        # Relatively flat or gentle curves
        if self.details['avg_peak_width'] > 15:
            score += 10
        
        return score
    
    def _check_arpes(self):
        """Check if data matches ARPES (1D cut)"""
        score = 0.0
        x_min, x_max = self.details['x_range']
        
        # ARPES: binding energy typically -2 to 6 eV or momentum
        # Could be E vs k slice
        if (-3 < x_min < 1 and 4 < x_max < 8) or (-2 < x_min < 0 and 1 < x_max < 3):
            score += 35
        
        # Can have complex peak structure
        if self.details['n_peaks'] >= 1:
            score += 20
        
        # Relatively narrow energy range
        if 2 < self.details['x_span'] < 10:
            score += 15
        
        return score
    
    def _check_thz(self):
        """Check if data matches THz spectroscopy"""
        score = 0.0
        x_min, x_max = self.details['x_range']
        
        # THz: frequency typically 0.1-10 THz or 3-300 cm⁻¹
        if (0.1 < x_min < 2 and 5 < x_max < 15) or (2 < x_min < 50 and 100 < x_max < 400):
            score += 40
        
        # Usually broader features than Raman
        if self.details['avg_peak_width'] > 5:
            score += 25
        
        # Lower frequency range
        if self.details['x_span'] < 500:
            score += 20
        
        return score
    
    def display_results(self, scores):
        """Display identification results"""
        print("\n" + "="*60)
        print("2D MATERIAL SPECTROSCOPY TECHNIQUE IDENTIFICATION")
        print("="*60)
        
        print(f"\n📊 Data Overview:")
        print(f"   Points: {self.details['n_points']}")
        print(f"   X-axis range: {self.details['x_range'][0]:.2f} to {self.details['x_range'][1]:.2f}")
        print(f"   Y-axis range: {self.details['y_range'][0]:.2e} to {self.details['y_range'][1]:.2e}")
        print(f"   Peaks detected: {self.details['n_peaks']}")
        
        print(f"\n🎯 IDENTIFIED TECHNIQUE: {self.technique}")
        print(f"   Confidence: {self.confidence:.1f}/100")
        
        print(f"\n📈 Classification Scores:")
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for i, (tech, score) in enumerate(sorted_scores, 1):
            bar = "█" * int(score/5) + "░" * (20 - int(score/5))
            marker = "👉" if tech == self.technique else "  "
            print(f"   {marker} {i}. {tech:25s} [{bar}] {score:5.1f}")
        
        print(f"\n💡 Interpretation Tips for {self.technique}:")
        self._print_tips()
        
        print("\n" + "="*60)
    
    def _print_tips(self):
        """Print technique-specific interpretation tips"""
        tips = {
            'Raman': [
                "• Look for G, D, 2D peaks in graphene (~1580, ~1350, ~2700 cm⁻¹)",
                "• TMDs: A1g and E2g modes shift with layer number",
                "• Peak width indicates crystal quality",
                "• Use for: layer counting, strain, doping, defects"
            ],
            'PL (Photoluminescence)': [
                "• Monolayer TMDs show strong direct bandgap emission",
                "• Peak position shifts with layer number, strain, temperature",
                "• Peak width indicates exciton lifetime/quality",
                "• Use for: bandgap, excitons, quantum yield"
            ],
            'UV-Vis Absorption': [
                "• Exciton peaks in 2D materials (A, B excitons)",
                "• Background slope gives bandgap estimate",
                "• Interference fringes may appear with substrate",
                "• Use for: bandgap, optical transitions"
            ],
            'FTIR': [
                "• Vibrational modes in mid-IR region",
                "• Different from Raman selection rules",
                "• Sensitive to functional groups, adsorbates",
                "• Use for: chemical bonds, surface chemistry"
            ],
            'XPS': [
                "• Core-level peaks identify elements",
                "• Peak position shifts show oxidation state",
                "• Peak area gives composition",
                "• Use for: elemental analysis, chemical state"
            ],
            'XRD': [
                "• Peak positions give interlayer spacing (d-spacing)",
                "• Peak width indicates crystallite size (Scherrer)",
                "• For 2D: often need grazing incidence (GIWAXS)",
                "• Use for: crystal structure, stacking order"
            ],
            'Reflectance Contrast': [
                "• Interference-enhanced visibility of monolayers",
                "• Substrate-dependent (SiO₂/Si wafer common)",
                "• Quick layer identification",
                "• Use for: layer counting, flake mapping"
            ],
            'ARPES': [
                "• Maps electronic band structure E(k)",
                "• Reveals Dirac cones in graphene",
                "• Shows valley splitting in TMDs",
                "• Use for: band structure, Fermi surface"
            ],
            'THz Spectroscopy': [
                "• Low-energy excitations",
                "• Interlayer vibrations in vdW materials",
                "• Conductivity measurements",
                "• Use for: carrier dynamics, phonons"
            ]
        }
        
        for tip in tips.get(self.technique, ["• No specific tips available"]):
            print(f"   {tip}")
    
    def plot_data(self, save_path=None):
        """Plot the spectroscopy data"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Main plot
        ax1.plot(self.x, self.y, 'b-', linewidth=1.5, label='Data')
        
        # Mark detected peaks
        y_norm = (self.y - self.y.min()) / (self.y.max() - self.y.min() + 1e-10)
        peaks, _ = signal.find_peaks(y_norm, prominence=0.1, width=1)
        if len(peaks) > 0:
            ax1.plot(self.x[peaks], self.y[peaks], 'r*', markersize=12, 
                    label=f'Peaks ({len(peaks)} detected)')
        
        ax1.set_xlabel('X-axis (units depend on technique)', fontsize=11)
        ax1.set_ylabel('Intensity (a.u.)', fontsize=11)
        ax1.set_title(f'Spectroscopy Data - Identified as: {self.technique} '
                     f'(Confidence: {self.confidence:.1f}/100)', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Derivative plot (helps identify subtle features)
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        derivative = dy / (dx + 1e-10)
        ax2.plot(self.x[:-1], derivative, 'g-', linewidth=1, alpha=0.7)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('X-axis', fontsize=11)
        ax2.set_ylabel('Derivative (dI/dx)', fontsize=11)
        ax2.set_title('First Derivative (helps identify peak positions)', fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Plot saved to: {save_path}")
        
        plt.show()


def load_data(filepath):
    """Load data from various file formats"""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    # Try different loading methods
    try:
        # Try comma-separated
        data = np.loadtxt(filepath, delimiter=',')
        print(f"✓ Loaded data with comma delimiter")
    except:
        try:
            # Try tab/space-separated
            data = np.loadtxt(filepath)
            print(f"✓ Loaded data with whitespace delimiter")
        except:
            try:
                # Try with skiprows for headers
                data = np.loadtxt(filepath, skiprows=1)
                print(f"✓ Loaded data (skipped header)")
            except Exception as e:
                raise ValueError(f"Could not load data: {e}")
    
    # Handle 1D or 2D arrays
    if data.ndim == 1:
        x = np.arange(len(data))
        y = data
    else:
        if data.shape[1] >= 2:
            x = data[:, 0]
            y = data[:, 1]
        else:
            x = np.arange(len(data))
            y = data.flatten()
    
    print(f"✓ Loaded {len(x)} data points")
    return x, y


def main():
    """Main function"""
    print("\n" + "="*60)
    print("2D MATERIAL SPECTROSCOPY IDENTIFIER")
    print("="*60)
    
    if len(sys.argv) < 2:
        print("\n❌ Usage: python spectroscopy_identifier.py <data_file>")
        print("\nSupported formats:")
        print("  • CSV: x,y values separated by comma")
        print("  • TXT: x y values separated by space/tab")
        print("  • Two columns: first=x-axis, second=intensity")
        print("  • One column: assumed to be intensity (auto-generates x-axis)")
        print("\nExample:")
        print("  python spectroscopy_identifier.py my_spectrum.txt")
        sys.exit(1)
    
    # Load data
    filepath = sys.argv[1]
    print(f"\n📁 Loading file: {filepath}")
    
    try:
        x, y = load_data(filepath)
    except Exception as e:
        print(f"\n❌ Error loading data: {e}")
        sys.exit(1)
    
    # Analyze
    print("\n🔍 Analyzing spectroscopy data...")
    identifier = SpectroscopyIdentifier(x, y)
    technique, confidence, scores = identifier.analyze()
    
    # Display results
    identifier.display_results(scores)
    
    # Plot
    print("\n📊 Generating plot...")
    plot_path = Path(filepath).stem + "_analysis.png"
    identifier.plot_data(save_path=plot_path)
    
    print("\n✅ Analysis complete!\n")


if __name__ == "__main__":
    main()