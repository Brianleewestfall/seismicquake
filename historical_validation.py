"""
TeslaQuake Historical Validation Engine

Backtests SeismicQuake AI against your Supabase earthquake history to calculate:
- Detection accuracy (did AI correctly identify earthquakes?)
- Magnitude accuracy (how close was AI magnitude vs USGS?)
- Wave classification accuracy
- TeslaQuake frequency correlation (were anomalies present before quakes?)
- Prediction validation (did your predictions match AI detections?)

Data sources:
    - earthquakes table (14K+ USGS events)
    - predictions table (163 TeslaQuake predictions, 47 linked)
    - seismicquake_results table (AI analysis history)
    - anomaly_detections table (frequency anomalies)

Usage:
    # Full validation report
    python historical_validation.py --report
    
    # Validate specific magnitude range
    python historical_validation.py --min-mag 5.0 --max-events 50
    
    # Backtest: download + analyze + score (slow, uses FDSN)
    python historical_validation.py --backtest --min-mag 6.0 --max-events 10
    
    # Score existing seismicquake_results against USGS
    python historical_validation.py --score-existing
    
    # Python API
    from historical_validation import ValidationEngine
    engine = ValidationEngine()
    report = engine.generate_report()
"""

import os
import sys
import json
import argparse
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path


SUPABASE_URL = os.environ.get("TESLAQUAKE_SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("TESLAQUAKE_SUPABASE_KEY", "")


class ValidationEngine:
    """Historical validation of SeismicQuake AI against USGS ground truth."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.url = SUPABASE_URL
        self.key = SUPABASE_KEY
        if not self.url or not self.key:
            raise ValueError("Set TESLAQUAKE_SUPABASE_URL and TESLAQUAKE_SUPABASE_KEY")
    
    def _headers(self):
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
        }
    
    def _query(self, table: str, params: str = "") -> list:
        """Query Supabase REST API."""
        url = f"{self.url}/rest/v1/{table}?{params}"
        req = urllib.request.Request(url, headers=self._headers())
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    
    # ══════════════════════════════════════════════════════════
    # Report: Score existing AI results vs USGS
    # ══════════════════════════════════════════════════════════
    
    def score_existing_results(self) -> Dict[str, Any]:
        """Score all seismicquake_results that have USGS magnitude."""
        self._log("📊 Scoring existing SeismicQuake AI results vs USGS...")
        
        try:
            rows = self._query(
                "seismicquake_results",
                "select=ai_is_earthquake,ai_earthquake_confidence,ai_magnitude,"
                "usgs_magnitude,magnitude_error,ai_p_wave_arrival,ai_s_wave_arrival,"
                "frequency_alert_level,sr1_z_score,tesla_z_score,analyzed_at"
                "&usgs_magnitude=not.is.null&order=analyzed_at.desc&limit=500"
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        
        if not rows:
            return {"ok": True, "message": "No results with USGS magnitude to score", "total": 0}
        
        # Detection accuracy
        total = len(rows)
        detected = sum(1 for r in rows if r.get("ai_is_earthquake"))
        detection_rate = detected / total if total > 0 else 0
        
        # Magnitude accuracy
        mag_errors = [
            abs(r["magnitude_error"])
            for r in rows
            if r.get("magnitude_error") is not None
        ]
        within_05 = sum(1 for e in mag_errors if e <= 0.5) / len(mag_errors) if mag_errors else 0
        within_10 = sum(1 for e in mag_errors if e <= 1.0) / len(mag_errors) if mag_errors else 0
        mae = sum(mag_errors) / len(mag_errors) if mag_errors else None
        
        # Confidence distribution
        confidences = [r["ai_earthquake_confidence"] for r in rows if r.get("ai_earthquake_confidence")]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None
        
        # Frequency correlation
        anomaly_before_quake = sum(
            1 for r in rows
            if r.get("ai_is_earthquake") and r.get("frequency_alert_level") not in ("NORMAL", None, "SKIPPED")
        )
        
        report = {
            "ok": True,
            "total_scored": total,
            "detection": {
                "earthquakes_detected": detected,
                "detection_rate": round(detection_rate * 100, 1),
                "avg_confidence": round(avg_confidence * 100, 1) if avg_confidence else None,
            },
            "magnitude": {
                "samples": len(mag_errors),
                "mae": round(mae, 3) if mae else None,
                "within_0.5": round(within_05 * 100, 1),
                "within_1.0": round(within_10 * 100, 1),
                "worst_error": round(max(mag_errors), 2) if mag_errors else None,
            },
            "frequency_correlation": {
                "quakes_with_anomaly": anomaly_before_quake,
                "correlation_rate": round(anomaly_before_quake / detected * 100, 1) if detected > 0 else 0,
            },
        }
        
        self._log(f"\n{'='*60}")
        self._log(f"📊 AI ACCURACY REPORT ({total} events scored)")
        self._log(f"{'='*60}")
        self._log(f"\n🔍 Detection:")
        self._log(f"   Rate: {report['detection']['detection_rate']}%")
        self._log(f"   Avg confidence: {report['detection']['avg_confidence']}%")
        self._log(f"\n📏 Magnitude Accuracy:")
        self._log(f"   MAE: {report['magnitude']['mae']}")
        self._log(f"   Within ±0.5: {report['magnitude']['within_0.5']}%")
        self._log(f"   Within ±1.0: {report['magnitude']['within_1.0']}%")
        self._log(f"\n⚡ Frequency Correlation:")
        self._log(f"   Quakes preceded by anomaly: {report['frequency_correlation']['correlation_rate']}%")
        
        return report
    
    # ══════════════════════════════════════════════════════════
    # Report: TeslaQuake prediction accuracy
    # ══════════════════════════════════════════════════════════
    
    def score_predictions(self) -> Dict[str, Any]:
        """Score TeslaQuake predictions against actual earthquakes."""
        self._log("\n📊 Scoring TeslaQuake predictions...")
        
        try:
            predictions = self._query(
                "predictions",
                "select=id,confidence_score,alert_level,predicted_region,"
                "predicted_latitude,predicted_longitude,predicted_mag_min,"
                "predicted_mag_max,window_start,window_end,status,"
                "actual_earthquake_id,accuracy_score,schumann_anomaly,"
                "tesla_anomaly,solar_trigger,lunar_trigger,pattern_369"
                "&order=created_at.desc&limit=200"
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        
        total = len(predictions)
        linked = sum(1 for p in predictions if p.get("actual_earthquake_id"))
        unlinked = total - linked
        
        # Trigger analysis
        schumann_triggered = sum(1 for p in predictions if p.get("schumann_anomaly"))
        tesla_triggered = sum(1 for p in predictions if p.get("tesla_anomaly"))
        solar_triggered = sum(1 for p in predictions if p.get("solar_trigger"))
        lunar_triggered = sum(1 for p in predictions if p.get("lunar_trigger"))
        pattern_369 = sum(1 for p in predictions if p.get("pattern_369"))
        
        # Accuracy scores for linked predictions
        scores = [p["accuracy_score"] for p in predictions if p.get("accuracy_score")]
        avg_score = sum(scores) / len(scores) if scores else None
        
        # Confidence distribution
        high_conf = sum(1 for p in predictions if (p.get("confidence_score") or 0) >= 80)
        high_conf_linked = sum(
            1 for p in predictions
            if (p.get("confidence_score") or 0) >= 80 and p.get("actual_earthquake_id")
        )
        
        report = {
            "ok": True,
            "total_predictions": total,
            "linked_to_earthquake": linked,
            "hit_rate": round(linked / total * 100, 1) if total > 0 else 0,
            "avg_accuracy_score": round(avg_score, 1) if avg_score else None,
            "high_confidence": {
                "total": high_conf,
                "linked": high_conf_linked,
                "hit_rate": round(high_conf_linked / high_conf * 100, 1) if high_conf > 0 else 0,
            },
            "trigger_breakdown": {
                "schumann_anomaly": schumann_triggered,
                "tesla_anomaly": tesla_triggered,
                "solar_trigger": solar_triggered,
                "lunar_trigger": lunar_triggered,
                "pattern_369": pattern_369,
            },
        }
        
        self._log(f"\n{'='*60}")
        self._log(f"🎯 TESLAQUAKE PREDICTION REPORT ({total} predictions)")
        self._log(f"{'='*60}")
        self._log(f"\n   Linked to actual earthquake: {linked}/{total} ({report['hit_rate']}%)")
        self._log(f"   Avg accuracy score: {report['avg_accuracy_score']}")
        self._log(f"\n   High confidence (≥80):")
        self._log(f"     Total: {high_conf}")
        self._log(f"     Hit rate: {report['high_confidence']['hit_rate']}%")
        self._log(f"\n   Trigger breakdown:")
        self._log(f"     Schumann anomaly: {schumann_triggered}")
        self._log(f"     Tesla anomaly: {tesla_triggered}")
        self._log(f"     Solar trigger: {solar_triggered}")
        self._log(f"     Lunar trigger: {lunar_triggered}")
        self._log(f"     3-6-9 pattern: {pattern_369}")
        
        return report
    
    # ══════════════════════════════════════════════════════════
    # Report: Frequency anomaly → earthquake correlation
    # ══════════════════════════════════════════════════════════
    
    def score_frequency_correlation(self, lookback_hours: int = 72) -> Dict[str, Any]:
        """Check if frequency anomalies preceded actual earthquakes."""
        self._log(f"\n⚡ Checking frequency anomaly → earthquake correlation ({lookback_hours}h window)...")
        
        try:
            # Get recent anomalies
            anomalies = self._query(
                "anomaly_detections",
                "select=metric_name,z_score,severity,detected_at,description"
                "&metric_name=in.(obspy_sr1_amplitude,obspy_tesla_amplitude,"
                "seismicquake_freq_anomaly,schumann_sr1_freq)"
                "&severity=in.(MODERATE,HIGH,CRITICAL)"
                "&order=detected_at.desc&limit=200"
            )
            
            # Get significant earthquakes
            earthquakes = self._query(
                "earthquakes",
                "select=timestamp,magnitude,place,latitude,longitude"
                "&magnitude=gte.5.0"
                "&order=timestamp.desc&limit=500"
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        
        if not anomalies or not earthquakes:
            return {"ok": True, "message": "Insufficient data for correlation", 
                    "anomalies": len(anomalies or []), "earthquakes": len(earthquakes or [])}
        
        # For each M5+ earthquake, check if anomaly was detected within lookback window
        correlations = []
        for eq in earthquakes:
            eq_time = datetime.fromisoformat(eq["timestamp"].replace("Z", "+00:00"))
            window_start = eq_time - timedelta(hours=lookback_hours)
            
            preceding = [
                a for a in anomalies
                if a.get("detected_at") and
                window_start <= datetime.fromisoformat(
                    a["detected_at"].replace("Z", "+00:00")
                ) <= eq_time
            ]
            
            if preceding:
                best = max(preceding, key=lambda x: x.get("z_score", 0))
                correlations.append({
                    "earthquake": f"M{eq['magnitude']} {eq.get('place', '')}",
                    "eq_time": eq["timestamp"],
                    "anomaly_type": best["metric_name"],
                    "anomaly_z": best["z_score"],
                    "anomaly_severity": best["severity"],
                    "lead_time_hours": round(
                        (eq_time - datetime.fromisoformat(
                            best["detected_at"].replace("Z", "+00:00")
                        )).total_seconds() / 3600, 1
                    ),
                })
        
        total_eq = len(earthquakes)
        preceded = len(correlations)
        
        report = {
            "ok": True,
            "lookback_hours": lookback_hours,
            "total_m5_earthquakes": total_eq,
            "preceded_by_anomaly": preceded,
            "correlation_rate": round(preceded / total_eq * 100, 1) if total_eq > 0 else 0,
            "total_anomalies_checked": len(anomalies),
            "correlations": correlations[:20],  # Top 20
        }
        
        self._log(f"   M5+ earthquakes: {total_eq}")
        self._log(f"   Preceded by anomaly: {preceded} ({report['correlation_rate']}%)")
        if correlations:
            self._log(f"\n   Recent correlations:")
            for c in correlations[:5]:
                self._log(f"     {c['earthquake']} ← {c['anomaly_type']} "
                          f"z={c['anomaly_z']} ({c['lead_time_hours']}h before)")
        
        return report
    
    # ══════════════════════════════════════════════════════════
    # Full report
    # ══════════════════════════════════════════════════════════
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        self._log("\n" + "═"*60)
        self._log("🔬 TESLAQUAKE HISTORICAL VALIDATION ENGINE")
        self._log("═"*60)
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ai_accuracy": self.score_existing_results(),
            "prediction_accuracy": self.score_predictions(),
            "frequency_correlation": self.score_frequency_correlation(),
        }
        
        # Overall summary
        ai = report["ai_accuracy"]
        pred = report["prediction_accuracy"]
        freq = report["frequency_correlation"]
        
        self._log(f"\n{'═'*60}")
        self._log(f"📋 OVERALL SUMMARY")
        self._log(f"{'═'*60}")
        
        if ai.get("detection"):
            self._log(f"   AI Detection Rate: {ai['detection']['detection_rate']}%")
        if ai.get("magnitude"):
            self._log(f"   AI Magnitude MAE: {ai['magnitude']['mae']}")
            self._log(f"   AI Within ±1.0: {ai['magnitude']['within_1.0']}%")
        if pred.get("hit_rate"):
            self._log(f"   TeslaQuake Prediction Hit Rate: {pred['hit_rate']}%")
        if freq.get("correlation_rate"):
            self._log(f"   Frequency→Earthquake Correlation: {freq['correlation_rate']}%")
        
        report["ok"] = True
        return report
    
    # ══════════════════════════════════════════════════════════
    # Backtest: Download + Analyze historical events
    # ══════════════════════════════════════════════════════════
    
    def backtest(self, min_magnitude: float = 6.0, max_events: int = 10) -> Dict[str, Any]:
        """Download historical waveforms and run full AI analysis.
        
        WARNING: This is slow — each event requires FDSN download + AI inference.
        Uses ~2-5 min per event depending on station availability.
        """
        self._log(f"\n🔄 BACKTEST: Downloading + analyzing M{min_magnitude}+ events...")
        self._log(f"   Max events: {max_events}")
        self._log(f"   ⚠️ This will take {max_events * 3}-{max_events * 5} minutes\n")
        
        # Get historical earthquakes from Supabase
        try:
            events = self._query(
                "earthquakes",
                f"select=usgs_id,timestamp,latitude,longitude,depth_km,magnitude,place"
                f"&magnitude=gte.{min_magnitude}"
                f"&order=magnitude.desc"
                f"&limit={max_events}"
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}
        
        if not events:
            return {"ok": False, "error": f"No M{min_magnitude}+ earthquakes found"}
        
        self._log(f"   Found {len(events)} events to backtest\n")
        
        # Import pipeline
        try:
            from teslaquake_pipeline import TeslaQuakePipeline
        except ImportError:
            return {"ok": False, "error": "teslaquake_pipeline.py not found"}
        
        pipeline = TeslaQuakePipeline(push_to_supabase=True, verbose=self.verbose)
        
        # For each event, use ObsPy to download + analyze
        results = []
        for i, eq in enumerate(events):
            self._log(f"\n[{i+1}/{len(events)}] M{eq['magnitude']} — {eq.get('place', 'Unknown')}")
            
            try:
                # Use ObsPy via seismic_analyzer to download
                from obspy import UTCDateTime
                from obspy.clients.fdsn import Client
                
                client = Client("IRIS")
                t0 = UTCDateTime(eq["timestamp"])
                lat, lon = float(eq["latitude"]), float(eq["longitude"])
                
                # Find nearby station
                inv = client.get_stations(
                    latitude=lat, longitude=lon,
                    maxradius=3.0, channel="BH?", level="station"
                )
                stations = [
                    (net.code, sta.code)
                    for net in inv for sta in net
                ]
                
                if not stations:
                    self._log(f"   ⏭️ No stations within 3° — skipping")
                    continue
                
                net, sta = stations[0]
                self._log(f"   📡 Station: {net}.{sta}")
                
                # Download waveform
                st = client.get_waveforms(
                    net, sta, "*", "BH?",
                    t0 - 120, t0 + 1200
                )
                
                if len(st) == 0:
                    self._log(f"   ⏭️ No waveform data — skipping")
                    continue
                
                # Save to temp folder
                tmp_dir = Path(f"./backtest_tmp/{eq['usgs_id']}_{net}.{sta}")
                tmp_dir.mkdir(parents=True, exist_ok=True)
                
                mseed_path = tmp_dir / "waveforms.mseed"
                st.write(str(mseed_path), format="MSEED")
                
                event_json = tmp_dir / "event.json"
                event_json.write_text(json.dumps({
                    "time_utc": eq["timestamp"],
                    "latitude": float(eq["latitude"]),
                    "longitude": float(eq["longitude"]),
                    "depth_km": float(eq["depth_km"]) if eq.get("depth_km") else None,
                    "magnitude": float(eq["magnitude"]),
                    "network": net,
                    "station": sta,
                    "channel": "BH?",
                }, indent=2))
                
                # Run pipeline
                result = pipeline.analyze_folder(str(tmp_dir))
                result["usgs_event"] = eq
                results.append(result)
                
            except Exception as e:
                self._log(f"   ❌ Error: {e}")
                results.append({"ok": False, "error": str(e), "usgs_event": eq})
        
        # Score the backtest
        successful = [r for r in results if r.get("ok")]
        detected = [
            r for r in successful
            if r.get("ai_detection", {}).get("is_earthquake")
        ]
        
        mag_errors = []
        for r in successful:
            ai_mag = r.get("ai_detection", {}).get("ai_magnitude")
            usgs_mag = r.get("usgs_event", {}).get("magnitude")
            if ai_mag and usgs_mag:
                mag_errors.append(abs(float(ai_mag) - float(usgs_mag)))
        
        mae = sum(mag_errors) / len(mag_errors) if mag_errors else None
        within_1 = sum(1 for e in mag_errors if e <= 1.0) / len(mag_errors) if mag_errors else 0
        
        summary = {
            "ok": True,
            "total_attempted": len(events),
            "successful_analyses": len(successful),
            "earthquakes_detected": len(detected),
            "detection_rate": round(len(detected) / len(successful) * 100, 1) if successful else 0,
            "magnitude_mae": round(mae, 3) if mae else None,
            "within_1.0": round(within_1 * 100, 1),
            "results": results,
        }
        
        self._log(f"\n{'═'*60}")
        self._log(f"🔬 BACKTEST RESULTS")
        self._log(f"{'═'*60}")
        self._log(f"   Analyzed: {len(successful)}/{len(events)}")
        self._log(f"   Detection rate: {summary['detection_rate']}%")
        self._log(f"   Magnitude MAE: {summary['magnitude_mae']}")
        self._log(f"   Within ±1.0: {summary['within_1.0']}%")
        
        return summary
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="TeslaQuake Historical Validation Engine"
    )
    parser.add_argument("--report", action="store_true", help="Full validation report")
    parser.add_argument("--score-existing", action="store_true", help="Score existing AI results")
    parser.add_argument("--score-predictions", action="store_true", help="Score TeslaQuake predictions")
    parser.add_argument("--frequency-correlation", action="store_true", help="Frequency → earthquake correlation")
    parser.add_argument("--backtest", action="store_true", help="Download + analyze historical events")
    parser.add_argument("--min-mag", type=float, default=6.0, help="Min magnitude for backtest")
    parser.add_argument("--max-events", type=int, default=10, help="Max events for backtest")
    parser.add_argument("--output", type=str, default="", help="Save report to JSON file")
    
    args = parser.parse_args()
    engine = ValidationEngine()
    
    if args.backtest:
        result = engine.backtest(min_magnitude=args.min_mag, max_events=args.max_events)
    elif args.score_existing:
        result = engine.score_existing_results()
    elif args.score_predictions:
        result = engine.score_predictions()
    elif args.frequency_correlation:
        result = engine.score_frequency_correlation()
    else:
        result = engine.generate_report()
    
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, default=str))
        print(f"\n💾 Saved to {args.output}")
    
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
