"""
TeslaQuake Pipeline: ObsPy → SeismicQuake AI → Supabase

End-to-end bridge that:
1. Takes an ObsPy download folder (waveforms.mseed + event.json)
2. Runs SeismicQuake AI (detect, classify, estimate magnitude)
3. Optionally runs TeslaQuake FFT frequency analysis
4. Pushes all results to Supabase

Usage:
    # Analyze a single ObsPy download folder
    python teslaquake_pipeline.py ./obspy_downloads/2025-12-08T14-15-10Z_IU.ANMO
    
    # Analyze all folders in obspy_downloads
    python teslaquake_pipeline.py ./obspy_downloads --batch
    
    # Skip Supabase push (local analysis only)
    python teslaquake_pipeline.py ./obspy_downloads/folder --no-push
    
    # Python API
    from teslaquake_pipeline import TeslaQuakePipeline
    pipeline = TeslaQuakePipeline()
    result = pipeline.analyze_folder("./obspy_downloads/2025-12-08T14-15-10Z_IU.ANMO")
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

# SeismicQuake AI
from seismic_analyzer import SeismicAnalyzer

# Supabase integration
from supabase_bridge import SupabaseBridge, EventContext, FrequencyData


# ═══════════════════════════════════════════════════════════════
# TeslaQuake Frequency Constants (mirrors simple-obspy config)
# ═══════════════════════════════════════════════════════════════
TESLAQUAKE_FREQUENCIES = {
    "sr1":       {"freq": 7.83,  "label": "Schumann SR₁",    "tolerance": 0.5},
    "tesla":     {"freq": 11.78, "label": "Tesla Telluric",   "tolerance": 0.5},
    "sr2":       {"freq": 14.3,  "label": "Schumann SR₂",    "tolerance": 0.5},
    "sr3":       {"freq": 20.8,  "label": "Schumann SR₃",    "tolerance": 0.5},
    "tesla_369": {"freq": 23.5,  "label": "Tesla 3-6-9",     "tolerance": 0.5},
    "sr4":       {"freq": 26.4,  "label": "Schumann SR₄",    "tolerance": 0.5},
    "sr5":       {"freq": 33.8,  "label": "Schumann SR₅",    "tolerance": 0.5},
}

ANOMALY_MODERATE = 2.0
ANOMALY_HIGH = 3.0
ANOMALY_CRITICAL = 4.0


class TeslaQuakePipeline:
    """End-to-end pipeline: ObsPy data → AI analysis → Supabase."""
    
    def __init__(self, push_to_supabase: bool = True, verbose: bool = True):
        self.verbose = verbose
        self.push = push_to_supabase
        
        # Initialize SeismicQuake AI
        self._log("Loading SeismicQuake AI models...")
        self.analyzer = SeismicAnalyzer(verbose=False)
        self._log("✅ AI models loaded")
        
        # Initialize Supabase bridge (optional)
        self.bridge = None
        if self.push:
            try:
                self.bridge = SupabaseBridge()
                health = self.bridge.health_check()
                if health["ok"]:
                    self._log("✅ Supabase connected")
                else:
                    self._log(f"⚠️ Supabase health check failed: {health.get('error')}")
                    self.bridge = None
            except ValueError as e:
                self._log(f"⚠️ Supabase not configured: {e}")
                self.bridge = None
    
    def analyze_folder(self, folder_path: str) -> Dict[str, Any]:
        """Analyze a single ObsPy download folder.
        
        Expected folder structure:
            folder/
            ├── waveforms.mseed
            ├── event.json
            └── station.xml (optional)
        
        Returns:
            Complete analysis result with AI + frequency + push status
        """
        folder = Path(folder_path)
        t_start = time.time()
        
        if not folder.is_dir():
            return {"ok": False, "error": f"Not a directory: {folder}"}
        
        # Find waveform file
        mseed = folder / "waveforms.mseed"
        if not mseed.exists():
            # Try any .mseed file
            mseeds = list(folder.glob("*.mseed"))
            if not mseeds:
                return {"ok": False, "error": f"No .mseed file in {folder}"}
            mseed = mseeds[0]
        
        self._log(f"\n{'='*60}")
        self._log(f"📂 Analyzing: {folder.name}")
        self._log(f"{'='*60}")
        
        # ── Step 1: Load event context ──
        context = self._load_event_context(folder)
        self._log(f"📍 Event: M{context.usgs_magnitude or '?'} at {context.event_time or 'unknown time'}")
        self._log(f"📡 Station: {context.network}.{context.station}")
        
        # ── Step 2: Run SeismicQuake AI ──
        self._log("\n🧠 Running SeismicQuake AI...")
        try:
            ai_result = self.analyzer.analyze_file(str(mseed))
            self._log(f"   Earthquake: {'YES' if ai_result.is_earthquake else 'NO'} "
                      f"({ai_result.earthquake_confidence:.1%} confidence)")
            if ai_result.p_wave_arrival:
                self._log(f"   P-wave arrival: {ai_result.p_wave_arrival:.2f}s")
            if ai_result.s_wave_arrival:
                self._log(f"   S-wave arrival: {ai_result.s_wave_arrival:.2f}s")
            if ai_result.estimated_magnitude:
                self._log(f"   AI Magnitude: {ai_result.estimated_magnitude:.1f}")
                if context.usgs_magnitude:
                    error = ai_result.estimated_magnitude - context.usgs_magnitude
                    self._log(f"   USGS Magnitude: {context.usgs_magnitude:.1f} (error: {error:+.2f})")
        except Exception as e:
            self._log(f"   ❌ AI analysis failed: {e}")
            ai_result = None
        
        # ── Step 3: Run TeslaQuake Frequency Analysis ──
        self._log("\n⚡ Running TeslaQuake frequency analysis...")
        freq_data = self._run_frequency_analysis(str(mseed))
        self._log(f"   SR₁ (7.83 Hz): z={freq_data.sr1_z_score or 0:.1f}")
        self._log(f"   Tesla (11.78 Hz): z={freq_data.tesla_z_score or 0:.1f}")
        self._log(f"   Alert level: {freq_data.frequency_alert_level}")
        if freq_data.precursor_flags:
            for flag in freq_data.precursor_flags:
                self._log(f"   🚩 {flag}")
        
        # ── Step 4: Push to Supabase ──
        push_result = {"ok": False, "skipped": True}
        if self.bridge and ai_result:
            self._log("\n📡 Pushing to Supabase...")
            processing_ms = (time.time() - t_start) * 1000
            push_result = self.bridge.push_analysis(
                result=ai_result,
                context=context,
                frequency=freq_data,
                processing_time_ms=processing_ms,
            )
            if push_result["ok"]:
                self._log(f"   ✅ Pushed: {push_result['seismicquake_results']} result, "
                          f"{push_result['anomaly_detections']} anomalies")
            else:
                self._log(f"   ❌ Push errors: {push_result['errors']}")
        elif not self.bridge:
            self._log("\n⏭️ Supabase push skipped (not configured)")
        
        elapsed = time.time() - t_start
        self._log(f"\n⏱️ Total time: {elapsed:.1f}s")
        
        return {
            "ok": True,
            "folder": str(folder),
            "event_context": {
                "trace_id": context.trace_id,
                "station": f"{context.network}.{context.station}",
                "event_time": context.event_time,
                "usgs_magnitude": context.usgs_magnitude,
            },
            "ai_detection": {
                "is_earthquake": ai_result.is_earthquake if ai_result else None,
                "confidence": ai_result.earthquake_confidence if ai_result else None,
                "ai_magnitude": ai_result.estimated_magnitude if ai_result else None,
                "p_wave": ai_result.p_wave_arrival if ai_result else None,
                "s_wave": ai_result.s_wave_arrival if ai_result else None,
            },
            "frequency_analysis": {
                "sr1_z_score": freq_data.sr1_z_score,
                "tesla_z_score": freq_data.tesla_z_score,
                "alert_level": freq_data.frequency_alert_level,
                "precursor_flags": freq_data.precursor_flags,
            },
            "supabase": push_result,
            "processing_seconds": round(elapsed, 2),
        }
    
    def analyze_batch(self, base_dir: str) -> Dict[str, Any]:
        """Analyze all ObsPy download folders in a directory."""
        base = Path(base_dir)
        folders = sorted([
            d for d in base.iterdir()
            if d.is_dir() and any(d.glob("*.mseed"))
        ])
        
        if not folders:
            return {"ok": False, "error": f"No folders with .mseed files in {base}"}
        
        self._log(f"\n🔬 Batch analysis: {len(folders)} folders in {base}")
        
        results = []
        for folder in folders:
            result = self.analyze_folder(str(folder))
            results.append(result)
        
        # Summary
        total = len(results)
        earthquakes = sum(1 for r in results
                         if r.get("ai_detection", {}).get("is_earthquake"))
        anomalies = sum(1 for r in results
                        if r.get("frequency_analysis", {}).get("alert_level") not in ("NORMAL", None))
        pushed = sum(1 for r in results
                     if r.get("supabase", {}).get("ok"))
        
        self._log(f"\n{'='*60}")
        self._log(f"📊 BATCH SUMMARY")
        self._log(f"{'='*60}")
        self._log(f"   Analyzed: {total}")
        self._log(f"   Earthquakes detected: {earthquakes}")
        self._log(f"   Frequency anomalies: {anomalies}")
        self._log(f"   Pushed to Supabase: {pushed}")
        
        return {
            "ok": True,
            "total_analyzed": total,
            "earthquakes_detected": earthquakes,
            "frequency_anomalies": anomalies,
            "pushed_to_supabase": pushed,
            "results": results,
        }
    
    # ── Internal Methods ─────────────────────────────────────
    
    def _load_event_context(self, folder: Path) -> EventContext:
        """Load event metadata from event.json if present."""
        event_file = folder / "event.json"
        ctx = EventContext()
        
        if event_file.exists():
            try:
                data = json.loads(event_file.read_text())
                ctx.event_time = data.get("time_utc", "")
                ctx.event_latitude = data.get("latitude")
                ctx.event_longitude = data.get("longitude")
                ctx.event_depth_km = data.get("depth_km")
                ctx.usgs_magnitude = data.get("magnitude")
                ctx.network = data.get("network", "")
                ctx.station = data.get("station", "")
                ctx.channel = data.get("channel", "")
                ctx.trace_id = f"{ctx.network}.{ctx.station}.{ctx.channel}"
            except Exception:
                pass
        
        # Try to infer from folder name if no event.json
        if not ctx.station:
            name = folder.name
            parts = name.split("_")
            if len(parts) >= 2:
                station_part = parts[-1]
                if "." in station_part:
                    net_sta = station_part.split(".")
                    ctx.network = net_sta[0]
                    ctx.station = net_sta[1] if len(net_sta) > 1 else ""
                    ctx.trace_id = f"{ctx.network}.{ctx.station}"
        
        return ctx
    
    def _run_frequency_analysis(self, mseed_path: str) -> FrequencyData:
        """Run TeslaQuake FFT analysis on a waveform file."""
        freq = FrequencyData()
        
        try:
            from obspy import read as obspy_read
            st = obspy_read(mseed_path)
            if len(st) == 0:
                return freq
            
            tr = st[0]
            data = tr.data.astype(float)
            sr = tr.stats.sampling_rate
            nyquist = sr / 2.0
            freq_max = min(40.0, nyquist * 0.95)
            
            # FFT
            n = len(data)
            data_centered = data - np.mean(data)
            window = np.hanning(n)
            fft_vals = np.fft.rfft(data_centered * window)
            freqs = np.fft.rfftfreq(n, d=1.0 / sr)
            amps = np.abs(fft_vals) * 2.0 / n
            
            # Filter to band
            mask = (freqs >= 1.0) & (freqs <= freq_max)
            freqs = freqs[mask]
            amps = amps[mask]
            
            # Baseline stats
            mean_amp = float(np.mean(amps))
            std_amp = float(np.std(amps))
            
            # Check each TeslaQuake frequency
            precursor_flags = []
            max_z = 0.0
            
            for key, config in TESLAQUAKE_FREQUENCIES.items():
                target = config["freq"]
                tol = config["tolerance"]
                band_mask = (freqs >= target - tol) & (freqs <= target + tol)
                
                if not np.any(band_mask):
                    continue
                
                band_amps = amps[band_mask]
                band_freqs = freqs[band_mask]
                peak_idx = np.argmax(band_amps)
                peak_amp = float(band_amps[peak_idx])
                peak_freq = float(band_freqs[peak_idx])
                z = (peak_amp - mean_amp) / std_amp if std_amp > 0 else 0.0
                shift = peak_freq - target
                
                if key == "sr1":
                    freq.sr1_amplitude = round(peak_amp, 6)
                    freq.sr1_z_score = round(z, 2)
                    if abs(shift) > 0.15:
                        precursor_flags.append(f"SR₁ shifted {shift:+.3f} Hz")
                elif key == "tesla":
                    freq.tesla_amplitude = round(peak_amp, 6)
                    freq.tesla_z_score = round(z, 2)
                    if abs(shift) > 0.15:
                        precursor_flags.append(f"Tesla shifted {shift:+.3f} Hz")
                
                max_z = max(max_z, z)
            
            # Classify alert level
            if max_z >= ANOMALY_CRITICAL:
                freq.frequency_alert_level = "CRITICAL"
            elif max_z >= ANOMALY_HIGH:
                freq.frequency_alert_level = "HIGH"
            elif max_z >= ANOMALY_MODERATE:
                freq.frequency_alert_level = "MODERATE"
            else:
                freq.frequency_alert_level = "NORMAL"
            
            freq.precursor_flags = precursor_flags if precursor_flags else None
            
        except ImportError:
            # ObsPy not installed — skip frequency analysis
            freq.frequency_alert_level = "SKIPPED"
        except Exception as e:
            freq.frequency_alert_level = f"ERROR: {e}"
        
        return freq
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ═══════════════════════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="TeslaQuake Pipeline: ObsPy → SeismicQuake AI → Supabase"
    )
    parser.add_argument(
        "path",
        help="Path to ObsPy download folder (or parent dir with --batch)"
    )
    parser.add_argument(
        "--batch", action="store_true",
        help="Analyze all subfolders containing .mseed files"
    )
    parser.add_argument(
        "--no-push", action="store_true",
        help="Skip Supabase push (local analysis only)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress console output"
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Save results JSON to this file"
    )
    
    args = parser.parse_args()
    
    pipeline = TeslaQuakePipeline(
        push_to_supabase=not args.no_push,
        verbose=not args.quiet,
    )
    
    if args.batch:
        result = pipeline.analyze_batch(args.path)
    else:
        result = pipeline.analyze_folder(args.path)
    
    # Save output
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, default=str))
        print(f"\n💾 Results saved to {args.output}")
    
    # Print summary JSON
    if not args.quiet:
        print(f"\n{json.dumps(result, indent=2, default=str)}")
    
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
