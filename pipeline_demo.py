#!/usr/bin/env python3
"""
Demo script to test the automated 2D material spectroscopy pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_test_data():
    """Create synthetic spectroscopy data for testing"""
    
    # Create synthetic Raman spectrum (graphene-like)
    wavenumber = np.linspace(1000, 3000, 500)
    
    # Add characteristic peaks
    intensity = np.zeros_like(wavenumber)
    
    # D-band at ~1350 cm-1
    intensity += 200 * np.exp(-((wavenumber - 1350) / 25)**2)
    
    # G-band at ~1580 cm-1 (stronger)
    intensity += 800 * np.exp(-((wavenumber - 1580) / 20)**2)
    
    # 2D-band at ~2670 cm-1
    intensity += 400 * np.exp(-((wavenumber - 2670) / 30)**2)
    
    # Add baseline and noise
    baseline = 50 + 0.01 * wavenumber
    noise = np.random.normal(0, 10, len(wavenumber))
    
    final_intensity = intensity + baseline + noise
    
    return wavenumber, final_intensity

def create_synthetic_pl_data():
    """Create synthetic PL spectrum (TMD-like)"""
    wavelength = np.linspace(600, 800, 300)
    
    # A exciton peak
    intensity = 1000 * np.exp(-((wavelength - 670) / 15)**2)
    
    # B exciton peak (weaker, blue-shifted)
    intensity += 300 * np.exp(-((wavelength - 620) / 12)**2)
    
    # Add baseline and noise
    baseline = 50 - 0.05 * wavelength
    noise = np.random.normal(0, 5, len(wavelength))
    
    return wavelength, intensity + baseline + noise

def create_synthetic_xrd_data():
    """Create synthetic XRD pattern"""
    two_theta = np.linspace(10, 80, 400)
    
    # Graphite (002) peak at ~26.6°
    intensity = 2000 * np.exp(-((two_theta - 26.6) / 0.8)**2)
    
    # Additional peaks
    intensity += 500 * np.exp(-((two_theta - 42.4) / 1.2)**2)  # (100)
    intensity += 800 * np.exp(-((two_theta - 54.7) / 1.0)**2)  # (004)
    
    # Add background
    background = 100 + 50 * np.exp(-two_theta / 20)
    noise = np.random.normal(0, 8, len(two_theta))
    
    return two_theta, intensity + background + noise

def save_test_data():
    """Save test data files"""
    # Create test data
    raman_x, raman_y = create_test_data()
    pl_x, pl_y = create_synthetic_pl_data()
    xrd_x, xrd_y = create_synthetic_xrd_data()
    
    # Save to files
    np.savetxt('test_raman.txt', np.column_stack((raman_x, raman_y)), 
               header='Wavenumber (cm-1)\tIntensity', fmt='%.2f')
    
    np.savetxt('test_pl.txt', np.column_stack((pl_x, pl_y)),
               header='Wavelength (nm)\tIntensity', fmt='%.2f')
    
    np.savetxt('test_xrd.txt', np.column_stack((xrd_x, xrd_y)),
               header='2Theta (degrees)\tIntensity', fmt='%.2f')
    
    print("✅ Test data files created:")
    print("   - test_raman.txt (synthetic graphene Raman)")
    print("   - test_pl.txt (synthetic TMD PL)")
    print("   - test_xrd.txt (synthetic graphite XRD)")

def demonstrate_pipeline_concepts():
    """Demonstrate the key concepts of the pipeline"""
    
    print("\n" + "="*80)
    print("2D MATERIAL SPECTROSCOPY PIPELINE - KEY CONCEPTS")
    print("="*80)
    
    print("""
🔬 HOW THE PIPELINE WORKS:

1. TECHNIQUE IDENTIFICATION 🎯
   The pipeline first analyzes the data characteristics to identify the spectroscopy technique:
   - X-axis range (wavenumbers, wavelengths, energy, 2θ angles)
   - Y-axis range (intensity patterns, absorption units)
   - Peak characteristics (sharp vs broad, number of peaks)
   - Data span and resolution
   
   Each technique has a scoring system based on typical ranges:
   • Raman: 100-3500 cm⁻¹, sharp peaks
   • PL: 400-1000 nm or 1-4 eV, broader peaks
   • XRD: 10-80° 2θ, very sharp peaks
   • UV-Vis: 200-800 nm, broad features
   • XPS: 0-1200 eV binding energy
   
2. INTELLIGENT DENOISING 🛠️
   Based on the identified technique, different denoising methods are applied:
   
   • RAMAN: Adaptive Gaussian smoothing
     - Less smoothing on peaks (σ=0.8) to preserve G, D, 2D bands
     - More smoothing on baseline (σ=2.0) to reduce noise
     - Preserves peak positions critical for layer counting
   
   • PL: Gentle Gaussian smoothing (σ=2.0)
     - PL peaks are naturally broader than Raman
     - More aggressive smoothing acceptable
     - Preserves exciton linewidths for quality assessment
   
   • XRD: Conservative Gaussian (σ=1.0)
     - Peak width contains crystallite size information
     - Must preserve FWHM for Scherrer analysis
     - Conservative approach to maintain peak integrity
   
   • UV-VIS: Savitzky-Golay filter
     - Good for broad absorption features
     - Preserves overall spectral shape
     - Handles baseline variations well
   
   • XPS: Bilateral filtering
     - Preserves sharp peaks on curved backgrounds
     - Maintains peak positions for chemical shift analysis
   
3. TECHNIQUE-SPECIFIC PEAK EXTRACTION 📊
   Peak finding parameters are optimized for each technique:
   
   • RAMAN: High sensitivity (prominence=0.02)
     - Detects weak D-bands and overtones
     - Small distance requirement for close peaks
     - Identifies G (~1580), D (~1350), 2D (~2670) bands
   
   • PL: Medium sensitivity (prominence=0.05)
     - Broader distance requirement (5 points)
     - Larger width range for exciton peaks
     - Identifies A and B exciton transitions
   
   • XRD: High prominence (0.1) for strong diffraction peaks
     - Small distance for closely spaced reflections
     - Narrow width range for sharp Bragg peaks
     - Calculates d-spacings and crystallite sizes
   
4. MATERIAL-SPECIFIC ANALYSIS 🧬
   Advanced interpretation based on 2D material knowledge:
   
   • GRAPHENE RAMAN:
     - I(D)/I(G) ratio → defect density
     - I(2D)/I(G) ratio → layer number
     - Peak positions → strain, doping
     - Peak widths → crystal quality
   
   • TMD PL:
     - A/B exciton positions → band structure
     - Linewidths → sample quality
     - Intensity ratios → quantum yield
   
   • XRD:
     - d-spacing analysis → interlayer distance
     - Peak width → crystallite size (Scherrer equation)
     - Peak positions → structural identification
   
5. QUALITY ASSESSMENT 📈
   Automated quality metrics:
   - Signal-to-noise ratio
   - Peak resolution
   - Baseline stability
   - Data consistency checks
""")

def show_denoising_comparison():
    """Visual demonstration of different denoising methods"""
    
    # Create noisy synthetic data
    x = np.linspace(1000, 3000, 500)
    clean_signal = 500 * np.exp(-((x - 1580) / 30)**2) + 200 * np.exp(-((x - 1350) / 25)**2)
    noise = np.random.normal(0, 25, len(x))
    noisy_data = clean_signal + 50 + noise
    
    # Apply different denoising methods
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import savgol_filter
    
    gentle_gaussian = gaussian_filter1d(noisy_data, sigma=1.0)
    strong_gaussian = gaussian_filter1d(noisy_data, sigma=3.0)
    savgol = savgol_filter(noisy_data, window_length=21, polyorder=3)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].plot(x, noisy_data, 'lightgray', alpha=0.7, label='Noisy data')
    axes[0, 0].plot(x, clean_signal + 50, 'blue', linewidth=2, label='True signal')
    axes[0, 0].set_title('Original vs True Signal')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(x, noisy_data, 'lightgray', alpha=0.5, label='Noisy')
    axes[0, 1].plot(x, gentle_gaussian, 'green', linewidth=2, label='Gentle Gaussian (σ=1)')
    axes[0, 1].set_title('Gentle Gaussian Smoothing')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(x, noisy_data, 'lightgray', alpha=0.5, label='Noisy')
    axes[1, 0].plot(x, strong_gaussian, 'red', linewidth=2, label='Strong Gaussian (σ=3)')
    axes[1, 0].set_title('Strong Gaussian Smoothing')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(x, noisy_data, 'lightgray', alpha=0.5, label='Noisy')
    axes[1, 1].plot(x, savgol, 'orange', linewidth=2, label='Savitzky-Golay')
    axes[1, 1].set_title('Savitzky-Golay Filter')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('denoising_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📊 Denoising comparison plot saved as 'denoising_comparison.png'")

if __name__ == "__main__":
    # Create test data
    save_test_data()
    
    # Demonstrate concepts
    demonstrate_pipeline_concepts()
    
    # Show visual comparison
    print("\n🔍 Creating denoising method comparison...")
    try:
        show_denoising_comparison()
    except ImportError:
        print("⚠️  Scipy not available for visual demo. Install with: pip install scipy matplotlib")
