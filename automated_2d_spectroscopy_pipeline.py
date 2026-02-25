#!/usr/bin/env python3
"""
Automated 2D Material Spectroscopy Analysis Pipeline
====================================================

This script automatically:
1. Identifies the type of spectroscopy technique
2. Applies technique-specific denoising methods
3. Extracts peaks with appropriate parameters
4. Provides material-specific analysis

Author: Enhanced from original spectroscopy identifier
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths, savgol_filter
from pathlib import Path
import sys
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional


@dataclass
class Peak:
    """Data class for storing peak information"""
    position: float
    intensity: float
    fwhm: float
    left_half: float
    right_half: float
    prominence: float
    band_type: Optional[str] = None


class DenoisingMethods:
    """Collection of denoising/smoothing methods for different spectroscopy techniques"""
    
    @staticmethod
    def gaussian_smooth_gentle(y: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Gentle Gaussian smoothing for preserving peak shapes"""
        return gaussian_filter1d(y, sigma)
    
    @staticmethod
    def gaussian_smooth_adaptive(y: np.ndarray, base_sigma: float = 1.0, 
                                peak_sigma: float = 0.5) -> np.ndarray:
        """Variable Gaussian smoothing - less smoothing on peaks, more on baseline"""
        # Identify regions with high intensity (likely peaks)
        threshold = 0.15 * np.max(y)
        sigma_array = np.where(y > threshold, peak_sigma, base_sigma * 2.0)
        
        result = np.zeros_like(y)
        for i in range(len(y)):
            local_sigma = sigma_array[i]
            window_size = int(6 * local_sigma)
            start_idx = max(0, i - window_size)
            end_idx = min(len(y), i + window_size + 1)
            
            local_y = y[start_idx:end_idx]
            if len(local_y) > 1:
                local_smooth = gaussian_filter1d(local_y, local_sigma)
                result[i] = local_smooth[i - start_idx] if i - start_idx < len(local_smooth) else y[i]
            else:
                result[i] = y[i]
        
        return result
    
    @staticmethod
    def savitzky_golay_adaptive(y: np.ndarray, base_window: int = 11, 
                               polyorder: int = 3) -> np.ndarray:
        """Savitzky-Golay filter with adaptive window length"""
        if base_window % 2 == 0:
            base_window += 1
        if base_window >= len(y):
            base_window = len(y) - 1 if len(y) % 2 == 0 else len(y) - 2
        
        return savgol_filter(y, base_window, polyorder)
    
    @staticmethod
    def moving_average_weighted(y: np.ndarray, window_size: int = 7) -> np.ndarray:
        """Moving average with Bartlett (triangular) weights"""
        weights = np.bartlett(window_size)
        weights = weights / weights.sum()
        
        pad_width = window_size // 2
        y_padded = np.pad(y, pad_width, mode='edge')
        result = np.convolve(y_padded, weights, mode='same')
        
        return result[pad_width:-pad_width]
    
    @staticmethod
    def median_filter_robust(y: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Median filter for spike removal"""
        from scipy.signal import medfilt
        return medfilt(y, kernel_size)
    
    @staticmethod
    def bilateral_filter_1d(y: np.ndarray, d: int = 5, sigma_intensity: float = 0.1, 
                           sigma_spatial: float = 1.0) -> np.ndarray:
        """1D bilateral filter - preserves edges while reducing noise"""
        result = np.zeros_like(y)
        
        for i in range(len(y)):
            # Define neighborhood
            start = max(0, i - d)
            end = min(len(y), i + d + 1)
            
            # Calculate weights
            spatial_weights = np.exp(-((np.arange(start, end) - i) ** 2) / (2 * sigma_spatial ** 2))
            intensity_weights = np.exp(-((y[start:end] - y[i]) ** 2) / (2 * sigma_intensity ** 2 * np.var(y)))
            
            combined_weights = spatial_weights * intensity_weights
            combined_weights = combined_weights / np.sum(combined_weights)
            
            result[i] = np.sum(y[start:end] * combined_weights)
        
        return result


class TechniqueSpecificAnalysis:
    """Technique-specific analysis methods for 2D materials"""
    
    @staticmethod
    def analyze_raman_peaks(peaks: List[Peak], x_range: Tuple[float, float]) -> Dict:
        """Analyze Raman peaks for 2D materials (especially graphene, TMDs)"""
        analysis = {
            'identified_bands': [],
            'material_indicators': [],
            'quality_metrics': {}
        }
        
        # Define characteristic bands for 2D materials
        band_definitions = {
            'D-band': (1200, 1400),      # Defect-induced
            'G-band': (1500, 1650),      # Graphitic
            'D-prime': (1590, 1630),     # Defect-induced
            '2D-band': (2600, 2750),     # Second-order
            'D+G': (2900, 3000),         # Combination
            # TMD bands
            'E2g_TMD': (350, 450),        # TMD in-plane
            'A1g_TMD': (400, 420),        # TMD out-of-plane
        }
        
        for peak in peaks:
            for band_name, (low, high) in band_definitions.items():
                if low <= peak.position <= high:
                    peak.band_type = band_name
                    analysis['identified_bands'].append({
                        'name': band_name,
                        'position': peak.position,
                        'intensity': peak.intensity,
                        'fwhm': peak.fwhm
                    })
        
        # Material-specific analysis
        g_peaks = [p for p in peaks if p.band_type == 'G-band']
        d_peaks = [p for p in peaks if p.band_type == 'D-band']
        td_peaks = [p for p in peaks if p.band_type == '2D-band']
        
        if g_peaks and d_peaks:
            # I(D)/I(G) ratio - defect density indicator
            id_ig_ratio = max([p.intensity for p in d_peaks]) / max([p.intensity for p in g_peaks])
            analysis['quality_metrics']['ID_IG_ratio'] = id_ig_ratio
            
            if id_ig_ratio < 0.1:
                analysis['material_indicators'].append("High-quality graphene (low defect density)")
            elif id_ig_ratio > 1.0:
                analysis['material_indicators'].append("Heavily defected or amorphous carbon")
        
        if td_peaks and g_peaks:
            # 2D/G ratio - layer number indicator for graphene
            i2d_ig_ratio = max([p.intensity for p in td_peaks]) / max([p.intensity for p in g_peaks])
            analysis['quality_metrics']['I2D_IG_ratio'] = i2d_ig_ratio
            
            if i2d_ig_ratio > 2:
                analysis['material_indicators'].append("Likely monolayer graphene")
            elif 0.5 < i2d_ig_ratio < 2:
                analysis['material_indicators'].append("Few-layer graphene")
            else:
                analysis['material_indicators'].append("Multilayer graphene/graphite")
        
        return analysis
    
    @staticmethod
    def analyze_pl_peaks(peaks: List[Peak], x_range: Tuple[float, float]) -> Dict:
        """Analyze PL peaks for 2D materials"""
        analysis = {
            'identified_transitions': [],
            'material_indicators': [],
            'quality_metrics': {}
        }
        
        # Common PL transitions in 2D TMDs (in eV or nm)
        if 1.5 < x_range[0] < 3.0 and 1.5 < x_range[1] < 3.0:  # Energy scale
            transition_definitions = {
                'MoS2_A': (1.8, 1.9),     # MoS2 A exciton
                'MoS2_B': (2.0, 2.1),     # MoS2 B exciton  
                'WSe2_A': (1.6, 1.7),     # WSe2 A exciton
                'WS2_A': (2.0, 2.1),      # WS2 A exciton
            }
        else:  # Wavelength scale
            transition_definitions = {
                'MoS2_A': (650, 700),     # MoS2 A exciton
                'WSe2_A': (750, 800),     # WSe2 A exciton
                'WS2_A': (620, 650),      # WS2 A exciton
            }
        
        for peak in peaks:
            for transition_name, (low, high) in transition_definitions.items():
                if low <= peak.position <= high:
                    peak.band_type = transition_name
                    analysis['identified_transitions'].append({
                        'name': transition_name,
                        'position': peak.position,
                        'intensity': peak.intensity,
                        'fwhm': peak.fwhm
                    })
        
        # Quality analysis based on FWHM
        if peaks:
            avg_fwhm = np.mean([p.fwhm for p in peaks])
            analysis['quality_metrics']['average_fwhm'] = avg_fwhm
            
            if avg_fwhm < 50:  # meV for energy, nm for wavelength
                analysis['material_indicators'].append("High-quality sample (narrow linewidth)")
            elif avg_fwhm > 100:
                analysis['material_indicators'].append("Lower quality or inhomogeneous sample")
        
        return analysis
    
    @staticmethod
    def analyze_xrd_peaks(peaks: List[Peak], x_range: Tuple[float, float]) -> Dict:
        """Analyze XRD peaks for 2D materials"""
        analysis = {
            'identified_reflections': [],
            'material_indicators': [],
            'quality_metrics': {}
        }
        
        # Calculate d-spacing for each peak (Bragg's law: nλ = 2d sinθ)
        # Assuming Cu Kα radiation (λ = 1.5406 Å)
        lambda_cu = 1.5406  # Angstroms
        
        for peak in peaks:
            theta_rad = np.radians(peak.position / 2)  # 2θ to θ
            d_spacing = lambda_cu / (2 * np.sin(theta_rad))
            
            # Common d-spacings for 2D materials
            if 3.2 < d_spacing < 3.4:  # Graphene/graphite (002)
                peak.band_type = "Graphene_002"
                analysis['material_indicators'].append("Graphitic material detected")
            elif 6.0 < d_spacing < 6.5:  # TMD (002) 
                peak.band_type = "TMD_002"
                analysis['material_indicators'].append("TMD material detected")
            
            analysis['identified_reflections'].append({
                'peak_2theta': peak.position,
                'd_spacing': d_spacing,
                'intensity': peak.intensity,
                'fwhm': peak.fwhm,
                'type': peak.band_type or "Unknown"
            })
        
        # Calculate crystallite size using Scherrer equation
        if peaks:
            # Find strongest peak for crystallite size calculation
            strongest_peak = max(peaks, key=lambda p: p.intensity)
            theta_rad = np.radians(strongest_peak.position / 2)
            beta_rad = np.radians(strongest_peak.fwhm)  # FWHM in radians
            
            # Scherrer equation: D = Kλ/(βcosθ), K ≈ 0.9
            crystallite_size = (0.9 * lambda_cu) / (beta_rad * np.cos(theta_rad))
            analysis['quality_metrics']['crystallite_size_nm'] = crystallite_size / 10  # Convert to nm
        
        return analysis


class Enhanced2DSpectroscopyPipeline:
    """Enhanced pipeline combining technique identification with intelligent processing"""
    
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray):
        self.x = np.array(x_data)
        self.y = np.array(y_data)
        self.technique = None
        self.confidence = 0.0
        self.denoised_data = None
        self.peaks = []
        self.analysis_results = {}
        self.processing_log = []
        
        # Initialize technique identifier from original code
        from detect_spectra_type import SpectroscopyIdentifier
        self.identifier = SpectroscopyIdentifier(x_data, y_data)
    
    def run_full_pipeline(self) -> Dict:
        """Execute the complete analysis pipeline"""
        results = {}
        
        # Step 1: Identify technique
        self.log("🔍 Step 1: Identifying spectroscopy technique...")
        technique, confidence, scores = self.identifier.analyze()
        self.technique = technique
        self.confidence = confidence
        results['technique'] = technique
        results['confidence'] = confidence
        results['all_scores'] = scores
        
        # Step 2: Apply technique-specific denoising
        self.log(f"🛠️  Step 2: Applying {technique}-specific denoising...")
        self.denoised_data = self._apply_denoising()
        results['denoising_method'] = self._get_denoising_method()
        
        # Step 3: Extract peaks with technique-appropriate parameters
        self.log("📊 Step 3: Extracting peaks with optimized parameters...")
        self.peaks = self._extract_peaks()
        results['peaks'] = [vars(peak) for peak in self.peaks]
        
        # Step 4: Technique-specific analysis
        self.log("🧬 Step 4: Performing material-specific analysis...")
        self.analysis_results = self._perform_analysis()
        results['material_analysis'] = self.analysis_results
        
        # Step 5: Generate comprehensive report
        self.log("📋 Step 5: Generating analysis report...")
        results['processing_log'] = self.processing_log
        
        return results
    
    def _apply_denoising(self) -> np.ndarray:
        """Apply technique-specific denoising methods"""
        method_map = {
            'Raman': self._denoise_raman,
            'PL (Photoluminescence)': self._denoise_pl,
            'UV-Vis Absorption': self._denoise_uvvis,
            'FTIR': self._denoise_ftir,
            'XPS': self._denoise_xps,
            'XRD': self._denoise_xrd,
            'Reflectance Contrast': self._denoise_reflectance,
            'ARPES': self._denoise_arpes,
            'THz Spectroscopy': self._denoise_thz
        }
        
        denoising_func = method_map.get(self.technique, self._denoise_default)
        return denoising_func()
    
    def _denoise_raman(self) -> np.ndarray:
        """Raman-specific denoising - preserve sharp peaks"""
        self.log("   Using adaptive Gaussian smoothing for Raman (σ=0.8-2.0)")
        # Use adaptive Gaussian - less smoothing on peaks, more on baseline
        return DenoisingMethods.gaussian_smooth_adaptive(self.y, base_sigma=1.5, peak_sigma=0.8)
    
    def _denoise_pl(self) -> np.ndarray:
        """PL-specific denoising - handle broader peaks"""
        self.log("   Using gentle Gaussian smoothing for PL (σ=2.0)")
        # PL peaks are typically broader, can use more smoothing
        return DenoisingMethods.gaussian_smooth_gentle(self.y, sigma=2.0)
    
    def _denoise_uvvis(self) -> np.ndarray:
        """UV-Vis denoising - smooth broad features"""
        self.log("   Using Savitzky-Golay filter for UV-Vis (window=15)")
        # UV-Vis often has broad features, SG filter works well
        return DenoisingMethods.savitzky_golay_adaptive(self.y, base_window=15, polyorder=3)
    
    def _denoise_ftir(self) -> np.ndarray:
        """FTIR denoising - preserve multiple bands"""
        self.log("   Using bilateral filter for FTIR")
        # FTIR has many bands, bilateral filter preserves edges
        return DenoisingMethods.bilateral_filter_1d(self.y, d=3, sigma_intensity=0.05)
    
    def _denoise_xps(self) -> np.ndarray:
        """XPS denoising - handle sharp peaks on curved background"""
        self.log("   Using adaptive Gaussian for XPS")
        return DenoisingMethods.gaussian_smooth_adaptive(self.y, base_sigma=2.0, peak_sigma=1.0)
    
    def _denoise_xrd(self) -> np.ndarray:
        """XRD denoising - preserve peak positions and widths"""
        self.log("   Using gentle Gaussian smoothing for XRD (σ=1.0)")
        # XRD peak width contains information, be conservative
        return DenoisingMethods.gaussian_smooth_gentle(self.y, sigma=1.0)
    
    def _denoise_reflectance(self) -> np.ndarray:
        """Reflectance contrast denoising"""
        self.log("   Using moving average for reflectance contrast")
        return DenoisingMethods.moving_average_weighted(self.y, window_size=9)
    
    def _denoise_arpes(self) -> np.ndarray:
        """ARPES denoising"""
        self.log("   Using Savitzky-Golay for ARPES")
        return DenoisingMethods.savitzky_golay_adaptive(self.y, base_window=7, polyorder=2)
    
    def _denoise_thz(self) -> np.ndarray:
        """THz denoising"""
        self.log("   Using gentle Gaussian for THz (σ=1.5)")
        return DenoisingMethods.gaussian_smooth_gentle(self.y, sigma=1.5)
    
    def _denoise_default(self) -> np.ndarray:
        """Default denoising when technique is uncertain"""
        self.log("   Using conservative Savitzky-Golay as default")
        return DenoisingMethods.savitzky_golay_adaptive(self.y, base_window=9, polyorder=3)
    
    def _extract_peaks(self) -> List[Peak]:
        """Extract peaks with technique-appropriate parameters"""
        # Technique-specific peak finding parameters
        params_map = {
            'Raman': {'prominence': 0.02, 'distance': 3, 'width': (1, 100)},
            'PL (Photoluminescence)': {'prominence': 0.05, 'distance': 5, 'width': (5, 200)},
            'UV-Vis Absorption': {'prominence': 0.1, 'distance': 10, 'width': (10, 500)},
            'FTIR': {'prominence': 0.03, 'distance': 5, 'width': (2, 50)},
            'XPS': {'prominence': 0.05, 'distance': 5, 'width': (2, 20)},
            'XRD': {'prominence': 0.1, 'distance': 3, 'width': (1, 10)},
            'Reflectance Contrast': {'prominence': 0.01, 'distance': 20, 'width': (20, 1000)},
            'ARPES': {'prominence': 0.05, 'distance': 5, 'width': (2, 50)},
            'THz Spectroscopy': {'prominence': 0.05, 'distance': 10, 'width': (5, 200)}
        }
        
        params = params_map.get(self.technique, {'prominence': 0.05, 'distance': 5, 'width': (2, 100)})
        
        # Normalize data for peak finding
        y_norm = (self.denoised_data - self.denoised_data.min()) / \
                 (self.denoised_data.max() - self.denoised_data.min() + 1e-10)
        
        # Dynamic prominence based on noise level
        noise_level = np.std(y_norm[y_norm < 0.1])
        min_prominence = max(params['prominence'], 3 * noise_level)
        
        # Find peaks
        peak_indices, properties = find_peaks(
            y_norm,
            prominence=min_prominence,
            distance=params['distance'],
            width=params['width']
        )
        
        peaks = []
        if len(peak_indices) > 0:
            # Calculate peak widths
            widths, width_heights, left_ips, right_ips = peak_widths(
                y_norm, peak_indices, rel_height=0.5
            )
            
            step_size = np.mean(np.diff(self.x))
            
            for i, (idx, width, left_ip, right_ip) in enumerate(zip(
                peak_indices, widths, left_ips, right_ips
            )):
                peak = Peak(
                    position=float(self.x[idx]),
                    intensity=float(self.denoised_data[idx]),
                    fwhm=float(width * step_size),
                    left_half=float(self.x[int(left_ip)] if int(left_ip) < len(self.x) else self.x[0]),
                    right_half=float(self.x[int(right_ip)] if int(right_ip) < len(self.x) else self.x[-1]),
                    prominence=float(properties['prominences'][i])
                )
                peaks.append(peak)
        
        # Sort by intensity
        peaks.sort(key=lambda p: p.intensity, reverse=True)
        
        self.log(f"   Found {len(peaks)} peaks using {self.technique}-optimized parameters")
        return peaks
    
    def _perform_analysis(self) -> Dict:
        """Perform technique and material-specific analysis"""
        x_range = (self.x.min(), self.x.max())
        
        analysis_map = {
            'Raman': TechniqueSpecificAnalysis.analyze_raman_peaks,
            'PL (Photoluminescence)': TechniqueSpecificAnalysis.analyze_pl_peaks,
            'XRD': TechniqueSpecificAnalysis.analyze_xrd_peaks,
        }
        
        if self.technique in analysis_map:
            return analysis_map[self.technique](self.peaks, x_range)
        else:
            return {
                'message': f'Specific analysis not yet implemented for {self.technique}',
                'identified_peaks': [vars(peak) for peak in self.peaks[:5]]
            }
    
    def _get_denoising_method(self) -> str:
        """Get description of applied denoising method"""
        method_descriptions = {
            'Raman': 'Adaptive Gaussian (σ=0.8-2.0) - preserves sharp peaks',
            'PL (Photoluminescence)': 'Gentle Gaussian (σ=2.0) - handles broad peaks',
            'UV-Vis Absorption': 'Savitzky-Golay (window=15) - smooths broad features',
            'FTIR': 'Bilateral filter - preserves multiple absorption bands',
            'XPS': 'Adaptive Gaussian - handles peaks on curved background',
            'XRD': 'Gentle Gaussian (σ=1.0) - preserves peak widths',
            'Reflectance Contrast': 'Weighted moving average - smooths gentle curves',
            'ARPES': 'Savitzky-Golay (window=7) - preserves band structure',
            'THz Spectroscopy': 'Gentle Gaussian (σ=1.5) - handles broad features'
        }
        return method_descriptions.get(self.technique, 'Conservative Savitzky-Golay')
    
    def log(self, message: str):
        """Add message to processing log"""
        print(message)
        self.processing_log.append(message)
    
    def plot_analysis_results(self, save_path: Optional[str] = None) -> None:
        """Create comprehensive visualization of analysis results"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # Main plot: original vs denoised with peaks
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(self.x, self.y, 'lightgray', alpha=0.7, linewidth=1, label='Original')
        ax1.plot(self.x, self.denoised_data, 'blue', linewidth=2, label='Denoised')
        
        # Mark peaks
        if self.peaks:
            peak_x = [p.position for p in self.peaks]
            peak_y = [p.intensity for p in self.peaks]
            ax1.scatter(peak_x, peak_y, color='red', s=60, zorder=5, 
                       marker='*', label=f'Peaks ({len(self.peaks)})')
            
            # Label significant peaks
            for i, peak in enumerate(self.peaks[:5]):
                label = peak.band_type if peak.band_type else f'Peak {i+1}'
                ax1.annotate(f'{label}\n{peak.position:.1f}', 
                           (peak.position, peak.intensity),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        ax1.set_title(f'{self.technique} Analysis - Confidence: {self.confidence:.1f}%', 
                     fontsize=14, fontweight='bold')
        ax1.set_xlabel('X-axis (units depend on technique)', fontsize=12)
        ax1.set_ylabel('Intensity (a.u.)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Peak analysis plot
        ax2 = fig.add_subplot(gs[1, 0])
        if self.peaks:
            positions = [p.position for p in self.peaks]
            intensities = [p.intensity for p in self.peaks]
            fwhms = [p.fwhm for p in self.peaks]
            
            scatter = ax2.scatter(positions, intensities, c=fwhms, s=100, 
                                cmap='viridis', alpha=0.7, edgecolors='black')
            ax2.set_xlabel('Peak Position', fontsize=11)
            ax2.set_ylabel('Peak Intensity', fontsize=11)
            ax2.set_title('Peak Analysis (color = FWHM)', fontsize=11)
            plt.colorbar(scatter, ax=ax2, label='FWHM')
        
        # Technique scores
        ax3 = fig.add_subplot(gs[1, 1])
        scores = self.identifier.analyze()[2]  # Get all scores
        techniques = list(scores.keys())
        score_values = list(scores.values())
        
        bars = ax3.barh(range(len(techniques)), score_values, 
                       color=['red' if t == self.technique else 'lightblue' for t in techniques])
        ax3.set_yticks(range(len(techniques)))
        ax3.set_yticklabels([t.replace(' ', '\n') for t in techniques], fontsize=9)
        ax3.set_xlabel('Confidence Score', fontsize=11)
        ax3.set_title('Technique Classification', fontsize=11)
        
        # Analysis summary (text)
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        summary_text = f"""
ANALYSIS SUMMARY:
• Technique: {self.technique} (Confidence: {self.confidence:.1f}%)
• Denoising: {self._get_denoising_method()}
• Peaks Found: {len(self.peaks)}
• Data Points: {len(self.x)} ({self.x.min():.1f} to {self.x.max():.1f})
"""
        
        # Add material-specific insights
        if 'material_indicators' in self.analysis_results:
            summary_text += "\nMATERIAL INSIGHTS:\n"
            for indicator in self.analysis_results['material_indicators']:
                summary_text += f"• {indicator}\n"
        
        if 'quality_metrics' in self.analysis_results:
            summary_text += "\nQUALITY METRICS:\n"
            for metric, value in self.analysis_results['quality_metrics'].items():
                summary_text += f"• {metric}: {value:.3f}\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='lightgray', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.log(f"📊 Analysis plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, output_path: str) -> None:
        """Save complete analysis results to JSON"""
        results = self.run_full_pipeline()
        
        # Make results JSON serializable
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, dict):
                results_serializable[key] = value
            else:
                results_serializable[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
        
        self.log(f"💾 Complete results saved to: {output_path}")


def load_data(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
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
    print("\n" + "="*80)
    print("AUTOMATED 2D MATERIAL SPECTROSCOPY ANALYSIS PIPELINE")
    print("="*80)
    print("🔬 Automatically identifies technique, denoises data, and extracts peaks")
    print("🧬 Provides material-specific analysis for 2D materials")
    
    if len(sys.argv) < 2:
        print("\n❌ Usage: python automated_2d_spectroscopy_pipeline.py <data_file> [output_dir]")
        print("\nSupported formats:")
        print("  • CSV: x,y values separated by comma")
        print("  • TXT: x y values separated by space/tab")
        print("  • Two columns: first=x-axis, second=intensity")
        print("  • One column: assumed to be intensity (auto-generates x-axis)")
        print("\nExample:")
        print("  python automated_2d_spectroscopy_pipeline.py my_raman.txt results/")
        sys.exit(1)
    
    # Parse arguments
    filepath = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Generate output filenames
    stem = Path(filepath).stem
    plot_path = output_dir / f"{stem}_automated_analysis.png"
    json_path = output_dir / f"{stem}_analysis_results.json"
    
    print(f"\n📁 Loading file: {filepath}")
    print(f"📂 Output directory: {output_dir}")
    
    try:
        # Load data
        x, y = load_data(filepath)
        
        # Create pipeline and run analysis
        print(f"\n🚀 Starting automated analysis pipeline...")
        pipeline = Enhanced2DSpectroscopyPipeline(x, y)
        
        # Run complete pipeline
        results = pipeline.run_full_pipeline()
        
        # Display summary
        print(f"\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print(f"🎯 Identified Technique: {results['technique']}")
        print(f"📊 Confidence: {results['confidence']:.1f}%")
        print(f"🔧 Denoising Method: {results['denoising_method']}")
        print(f"📈 Peaks Found: {len(results['peaks'])}")
        
        if results['material_analysis'].get('material_indicators'):
            print(f"\n🧬 Material Insights:")
            for indicator in results['material_analysis']['material_indicators']:
                print(f"   • {indicator}")
        
        # Save results and create plots
        pipeline.save_results(str(json_path))
        pipeline.plot_analysis_results(str(plot_path))
        
        print(f"\n✅ Pipeline completed successfully!")
        print(f"📊 Visual analysis: {plot_path}")
        print(f"💾 Detailed results: {json_path}")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
