"""
TeslaQuake Supabase Integration for SeismicQuake
Pushes AI detection results to the TeslaQuake database.

Tables written:
    - seismicquake_results  : Full AI analysis (detection + classification + magnitude)
    - anomaly_detections    : Flagged events (high confidence earthquakes or frequency anomalies)

Environment variables:
    TESLAQUAKE_SUPABASE_URL : Your Supabase project URL
    TESLAQUAKE_SUPABASE_KEY : Service role key

Usage:
    from supabase_bridge import SupabaseBridge
    
    bridge = SupabaseBridge()
    bridge.push_analysis(result, event_context, frequency_data)
"""

import os
import json
import urllib.request
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List


SUPABASE_URL = os.environ.get("TESLAQUAKE_SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("TESLAQUAKE_SUPABASE_KEY", "")


@dataclass
class EventContext:
    """Context from ObsPy download about the earthquake being analyzed."""
    trace_id: str = ""
    network: str = ""
    station: str = ""
    channel: str = ""
    event_time: str = ""          # ISO 8601
    event_latitude: float = None
    event_longitude: float = None
    event_depth_km: float = None
    usgs_magnitude: float = None
    usgs_id: str = ""


@dataclass
class FrequencyData:
    """TeslaQuake frequency analysis results (from ObsPy FFT)."""
    sr1_amplitude: float = None
    sr1_z_score: float = None
    tesla_amplitude: float = None
    tesla_z_score: float = None
    frequency_alert_level: str = "NORMAL"
    precursor_flags: List[str] = None


class SupabaseBridge:
    """Push SeismicQuake AI results to TeslaQuake Supabase."""
    
    def __init__(self, url: str = "", key: str = ""):
        self.url = url or SUPABASE_URL
        self.key = key or SUPABASE_KEY
        if not self.url or not self.key:
            raise ValueError(
                "Supabase not configured. Set TESLAQUAKE_SUPABASE_URL and "
                "TESLAQUAKE_SUPABASE_KEY environment variables."
            )
    
    def _headers(self, prefer: str = "return=minimal"):
        return {
            "apikey": self.key,
            "Authorization": f"Bearer {self.key}",
            "Content-Type": "application/json",
            "Prefer": prefer,
        }
    
    def _post(self, table: str, rows: list) -> Dict[str, Any]:
        """Insert rows into a Supabase table via REST API."""
        url = f"{self.url}/rest/v1/{table}"
        data = json.dumps(rows).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers=self._headers(), method="POST"
        )
        try:
            with urllib.request.urlopen(req) as resp:
                return {"ok": True, "status": resp.status, "rows_sent": len(rows)}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    def push_analysis(
        self,
        result,  # SeismicAnalyzer AnalysisResult object
        context: EventContext = None,
        frequency: FrequencyData = None,
        processing_time_ms: float = None,
    ) -> Dict[str, Any]:
        """Push a complete SeismicQuake analysis to Supabase.
        
        Args:
            result: AnalysisResult from SeismicAnalyzer.analyze_file()
            context: EventContext with ObsPy earthquake metadata
            frequency: FrequencyData from TeslaQuake FFT analysis
            processing_time_ms: How long analysis took
            
        Returns:
            Dict with ok, rows pushed to each table, errors
        """
        ctx = context or EventContext()
        freq = frequency or FrequencyData()
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Build seismicquake_results row
        row = {
            "trace_id": ctx.trace_id or getattr(result, "filename", "unknown"),
            "network": ctx.network or None,
            "station": ctx.station or None,
            "channel": ctx.channel or None,
            "event_time": ctx.event_time or None,
            "event_latitude": ctx.event_latitude,
            "event_longitude": ctx.event_longitude,
            "event_depth_km": ctx.event_depth_km,
            "usgs_magnitude": ctx.usgs_magnitude,
            "usgs_id": ctx.usgs_id or None,
            
            # AI Detection
            "ai_is_earthquake": getattr(result, "is_earthquake", False),
            "ai_earthquake_confidence": getattr(result, "earthquake_confidence", None),
            
            # AI Wave Classification
            "ai_wave_type": self._get_primary_wave_type(result),
            "ai_wave_confidence": self._get_primary_wave_confidence(result),
            "ai_p_wave_arrival": getattr(result, "p_wave_arrival", None),
            "ai_s_wave_arrival": getattr(result, "s_wave_arrival", None),
            "ai_surface_wave_arrival": getattr(result, "surface_wave_arrival", None),
            
            # AI Magnitude
            "ai_magnitude": getattr(result, "estimated_magnitude", None),
            "ai_magnitude_uncertainty": None,
            
            # TeslaQuake Frequencies
            "sr1_amplitude": freq.sr1_amplitude,
            "sr1_z_score": freq.sr1_z_score,
            "tesla_amplitude": freq.tesla_amplitude,
            "tesla_z_score": freq.tesla_z_score,
            "frequency_alert_level": freq.frequency_alert_level,
            "precursor_flags": freq.precursor_flags or [],
            
            # Comparison
            "magnitude_error": self._calc_mag_error(result, ctx),
            "processing_time_ms": processing_time_ms,
            "analyzed_at": now_iso,
        }
        
        # Clean None values for JSON
        row = {k: v for k, v in row.items() if v is not None}
        
        output = {"seismicquake_results": 0, "anomaly_detections": 0, "errors": []}
        
        # 1. Push to seismicquake_results
        res = self._post("seismicquake_results", [row])
        if res["ok"]:
            output["seismicquake_results"] = 1
        else:
            output["errors"].append(f"seismicquake_results: {res['error']}")
        
        # 2. Push to anomaly_detections if significant
        anomaly_rows = self._build_anomaly_rows(result, ctx, freq, now_iso)
        if anomaly_rows:
            res = self._post("anomaly_detections", anomaly_rows)
            if res["ok"]:
                output["anomaly_detections"] = len(anomaly_rows)
            else:
                output["errors"].append(f"anomaly_detections: {res['error']}")
        
        output["ok"] = len(output["errors"]) == 0
        return output
    
    def push_batch(
        self,
        results: list,
        contexts: list = None,
        frequencies: list = None,
    ) -> Dict[str, Any]:
        """Push multiple analysis results at once."""
        contexts = contexts or [EventContext()] * len(results)
        frequencies = frequencies or [FrequencyData()] * len(results)
        
        total = {"ok": True, "total_results": 0, "total_anomalies": 0, "errors": []}
        
        for result, ctx, freq in zip(results, contexts, frequencies):
            out = self.push_analysis(result, ctx, freq)
            total["total_results"] += out["seismicquake_results"]
            total["total_anomalies"] += out["anomaly_detections"]
            total["errors"].extend(out.get("errors", []))
        
        total["ok"] = len(total["errors"]) == 0
        return total
    
    def health_check(self) -> Dict[str, Any]:
        """Verify Supabase connectivity."""
        try:
            url = f"{self.url}/rest/v1/seismicquake_results?select=id&limit=1"
            req = urllib.request.Request(url, headers=self._headers())
            with urllib.request.urlopen(req) as resp:
                return {"ok": resp.status == 200, "status": resp.status}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    
    # ── Helpers ──────────────────────────────────────────────
    
    def _get_primary_wave_type(self, result) -> Optional[str]:
        """Extract the highest-confidence wave type from detections."""
        detections = getattr(result, "detections", [])
        if not detections:
            return None
        # Find highest confidence detection that isn't noise
        best = None
        for d in detections:
            wt = getattr(d, "wave_type", "Noise")
            conf = getattr(d, "confidence", 0)
            if wt != "Noise" and (best is None or conf > best[1]):
                best = (wt, conf)
        return best[0] if best else "Noise"
    
    def _get_primary_wave_confidence(self, result) -> Optional[float]:
        """Get confidence of primary wave detection."""
        detections = getattr(result, "detections", [])
        if not detections:
            return None
        confs = [getattr(d, "confidence", 0) for d in detections
                 if getattr(d, "wave_type", "Noise") != "Noise"]
        return max(confs) if confs else None
    
    def _calc_mag_error(self, result, ctx: EventContext) -> Optional[float]:
        """Calculate magnitude prediction error vs USGS."""
        ai_mag = getattr(result, "estimated_magnitude", None)
        usgs_mag = ctx.usgs_magnitude
        if ai_mag is not None and usgs_mag is not None:
            return round(ai_mag - usgs_mag, 3)
        return None
    
    def _build_anomaly_rows(
        self, result, ctx: EventContext, freq: FrequencyData, now_iso: str
    ) -> list:
        """Build anomaly_detections rows for significant findings."""
        rows = []
        now_dt = datetime.now(timezone.utc)
        
        # High-confidence earthquake detection
        confidence = getattr(result, "earthquake_confidence", 0)
        if getattr(result, "is_earthquake", False) and confidence >= 0.85:
            ai_mag = getattr(result, "estimated_magnitude", None)
            desc = f"SeismicQuake AI detection: {confidence:.1%} confidence"
            if ai_mag:
                desc += f", estimated M{ai_mag:.1f}"
            if ctx.station:
                desc += f" | station={ctx.network}.{ctx.station}"
            
            rows.append({
                "metric_name": "seismicquake_ai_detection",
                "observed_value": confidence,
                "baseline_mean": 0.5,
                "baseline_std": 0.15,
                "z_score": round((confidence - 0.5) / 0.15, 2),
                "severity": "HIGH" if confidence >= 0.95 else "MODERATE",
                "baseline_count": 0,
                "weekday": now_dt.weekday(),
                "month": now_dt.month,
                "description": desc,
                "detected_at": now_iso,
                "acknowledged": False,
                "detection_method": "seismicquake_cnn",
            })
        
        # Frequency anomalies from TeslaQuake analysis
        if freq.frequency_alert_level in ("HIGH", "CRITICAL"):
            desc_parts = [f"Frequency alert: {freq.frequency_alert_level}"]
            if freq.sr1_z_score and freq.sr1_z_score >= 2.0:
                desc_parts.append(f"SR1 z={freq.sr1_z_score:.1f}")
            if freq.tesla_z_score and freq.tesla_z_score >= 2.0:
                desc_parts.append(f"Tesla z={freq.tesla_z_score:.1f}")
            if freq.precursor_flags:
                desc_parts.extend(freq.precursor_flags)
            
            z = max(freq.sr1_z_score or 0, freq.tesla_z_score or 0)
            rows.append({
                "metric_name": "seismicquake_freq_anomaly",
                "observed_value": z,
                "baseline_mean": 0.0,
                "baseline_std": 1.0,
                "z_score": round(z, 2),
                "severity": freq.frequency_alert_level,
                "baseline_count": 0,
                "weekday": now_dt.weekday(),
                "month": now_dt.month,
                "description": " | ".join(desc_parts),
                "detected_at": now_iso,
                "acknowledged": False,
                "detection_method": "seismicquake_fft",
            })
        
        return rows
