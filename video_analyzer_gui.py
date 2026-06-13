#!/usr/bin/env python3
"""Tkinter GUI for MPEG-TS Analyser with TR101-290 P1/P2 Error Display

This GUI wrapper provides a visual interface for the TS analyser showing:
- Priority 1 errors (critical for service availability)
- Priority 2 errors (affecting service quality)
- Detailed analysis results
- Graphs: PCR jitter, PCR accuracy, PTS-PCR difference, instantaneous bitrate
"""
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import os
import json
from video_analyzer import TSAnalyser, get_enhanced_stream_description


class H264SpecValidator:
    """Validates NAL/SEI parameters against H.264/AVC specification constraints.
    
    Returns validation result with spec violation details for GUI highlighting.
    """
    
    # H.264 Specification Constraints
    VALID_NAL_TYPES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 19, 20}
    SEI_TYPES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66}
    VALID_SLICE_TYPES = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}  # 0-4 for P/B, 5-9 for SP/SI
    MAX_SLICE_TYPE = 9
    MAX_FRAME_NUM = (1 << 16)  # 16-bit max
    MAX_PIC_ORDER_CNT = (1 << 32)  # 32-bit max
    
    @staticmethod
    def validate_field(field_name, field_value, field_type, constraints=None):
        """Validate a single NAL/SEI field against spec constraints.
        
        Args:
            field_name: Parameter name (str)
            field_value: Parameter value (can be int, str, etc.)
            field_type: Type of field ('forbidden_zero_bit', 'nal_unit_type', 'nal_ref_idc', 
                       'slice_type', 'frame_num', 'pic_order_cnt', 'cc_valid', etc.)
            constraints: Additional constraints dict (e.g., {'max': 100})
            
        Returns:
            (is_valid: bool, violation_msg: str or None)
        """
        violations = []
        
        try:
            # Convert string values to int if possible for validation
            val = field_value
            if isinstance(val, str):
                # Try to parse hex (0x...)
                if val.startswith('0x'):
                    try:
                        val = int(val, 16)
                    except ValueError:
                        pass
                else:
                    # Try to extract number from format like "5 (0x05)"
                    parts = val.split()
                    if parts and parts[0].isdigit():
                        val = int(parts[0])
            
            if isinstance(val, int):
                # forbidden_zero_bit must be 0
                if field_type == 'forbidden_zero_bit' and val != 0:
                    violations.append(f"{field_name} must be 0 (H.264 spec), got {val}")
                
                # nal_ref_idc constraints
                elif field_type == 'nal_ref_idc':
                    if val < 0 or val > 3:
                        violations.append(f"{field_name} out of range [0-3], got {val}")
                    # ref_idc=0 for non-reference NALs, >0 for reference
                    if field_name == 'nal_ref_idc' and constraints and constraints.get('must_be_nonref') and val != 0:
                        violations.append(f"{field_name} must be 0 for non-reference NAL, got {val}")
                
                # nal_unit_type constraints
                elif field_type == 'nal_unit_type':
                    if val not in H264SpecValidator.VALID_NAL_TYPES:
                        violations.append(f"{field_name} type {val} not valid H.264 NAL type")
                    # Type 0 and 24-31 reserved/invalid
                    if val == 0 or (val >= 24 and val <= 31 and val != 20):
                        violations.append(f"{field_name} type {val} is reserved/undefined in H.264")
                
                # slice_type constraints
                elif field_type == 'slice_type':
                    if val > H264SpecValidator.MAX_SLICE_TYPE:
                        violations.append(f"{field_name} {val} exceeds max {H264SpecValidator.MAX_SLICE_TYPE}")
                
                # frame_num width constraints (depends on log2_max_frame_num, typically 4-16 bits)
                elif field_type == 'frame_num':
                    if val < 0 or val >= H264SpecValidator.MAX_FRAME_NUM:
                        violations.append(f"{field_name} {val} exceeds maximum frame number")
                
                # pic_order_cnt constraints
                elif field_type == 'pic_order_cnt':
                    if val < 0 or val >= H264SpecValidator.MAX_PIC_ORDER_CNT:
                        violations.append(f"{field_name} {val} exceeds maximum POC value")
                
                # cc_valid must be 0 or 1
                elif field_type == 'cc_valid':
                    if val not in (0, 1):
                        violations.append(f"{field_name} must be 0 or 1, got {val}")
                
                # cc_type constraints (0-3)
                elif field_type == 'cc_type':
                    if val < 0 or val > 3:
                        violations.append(f"{field_name} out of range [0-3], got {val}")
                
                # one_bit (in CC block) must be 1
                elif field_type == 'one_bit':
                    if val != 1:
                        violations.append(f"{field_name} must be 1 per H.264 spec, got {val}")
                
                # reserved bits must match spec (usually 0 or specific pattern)
                elif field_type == 'reserved':
                    if constraints and constraints.get('must_be_zero') and val != 0:
                        violations.append(f"{field_name} reserved bits must be 0, got 0x{val:X}")
                
                # pic_struct (pic_timing SEI) valid range 0-12
                elif field_type == 'pic_struct':
                    if val > 12:
                        violations.append(f"{field_name} {val} exceeds max valid pic_struct value (12)")
        
        except Exception as e:
            # If validation fails due to parsing, don't mark as spec violation
            pass
        
        return (len(violations) == 0, violations[0] if violations else None)
    
    @staticmethod
    def validate_nal_header(forbidden_zero_bit, nal_ref_idc, nal_unit_type):
        """Validate NAL header fields collectively."""
        violations = []
        
        # Forbidden zero bit check
        if forbidden_zero_bit != 0:
            violations.append(f"forbidden_zero_bit must be 0, got {forbidden_zero_bit}")
        
        # nal_ref_idc check
        if nal_ref_idc < 0 or nal_ref_idc > 3:
            violations.append(f"nal_ref_idc must be in [0-3], got {nal_ref_idc}")
        
        # nal_unit_type check
        if nal_unit_type not in H264SpecValidator.VALID_NAL_TYPES:
            violations.append(f"nal_unit_type {nal_unit_type} is not a valid H.264 NAL type")
        
        # For SEI (type 6) and type 0, nal_ref_idc must be 0
        if nal_unit_type in (0, 6) and nal_ref_idc != 0:
            violations.append(f"nal_unit_type {nal_unit_type} must have nal_ref_idc=0, got {nal_ref_idc}")
        
        return violations


# Heuristic sniffing for MPEG-TS based on sync byte alignment so we don't rely on file extensions
def is_ts_by_content(path: str, probes: int = 6) -> bool:
    try:
        with open(path, 'rb') as f:
            data = f.read(204 * probes)
    except Exception:
        return False

    if len(data) < 188 * 3:
        return False

    # Check common packet sizes and offsets (TS 188, M2TS 192 with 4-byte timestamp, 204 with FEC)
    candidates = [
        (188, 0),
        (192, 0),
        (192, 4),
        (204, 0),
    ]

    for size, offset in candidates:
        if len(data) < (offset + size * probes):
            continue
        syncs = 0
        for i in range(probes):
            pos = offset + i * size
            if pos < len(data) and data[pos] == 0x47:
                syncs += 1
        if syncs >= max(3, probes - 1):  # tolerate one miss
            return True
    return False


def detect_file_format(path: str) -> str:
    """Detect file format: ts, m2ts, mp4, or mov"""
    try:
        with open(path, 'rb') as f:
            header = f.read(32)
            
            # Check for MP4/MOV signatures
            if len(header) >= 8:
                # ftyp box check
                if header[4:8] == b'ftyp':
                    brand = header[8:12].decode('ascii', errors='ignore')
                    if brand in ('isom', 'iso2', 'avc1', 'mp41', 'mp42'):
                        return 'mp4'
                    elif brand in ('qt  ', 'M4V ', 'M4A '):
                        return 'mov'
                
                # MOV might not have ftyp, check for moov/mdat
                if header[4:8] in (b'moov', b'mdat', b'wide', b'free'):
                    return 'mov'
            
            # Check for MPEG-TS
            if is_ts_by_content(path):
                # Determine if M2TS
                if len(header) >= 192 and header[4] == 0x47:
                    return 'm2ts'
                return 'ts'
    except Exception:
        pass
    
    return 'unknown'

# Debug flag - set to True to enable SEI/timecode debug logging during verification
DEBUG = True

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    np = None
    Figure = None
    FigureCanvasTkAgg = None

try:
    import av
    from PIL import Image, ImageTk
    import io
    AV_AVAILABLE = True
except ImportError:
    AV_AVAILABLE = False
    av = None
    Image = None
    ImageTk = None

try:
    from ndi_streamer import NDIReceiver
    from ndi_analysis import analyze_frame
    from ndi_recorder import NDIRecorder
    NDI_AVAILABLE = True
except Exception:
    NDI_AVAILABLE = False
    NDIReceiver = None
    analyze_frame = None
    NDIRecorder = None

# Optional RTMP analyser (ffprobe wrapper)
try:
    import rtmp_analyser
    RTMP_ANALYSER_AVAILABLE = True
except Exception:
    RTMP_ANALYSER_AVAILABLE = False


class TR101290ErrorClassifier:
    """Classifies errors according to TR101-290 Priority levels (ETSI TR 101 290 V1.2.1)"""
    
    @staticmethod
    def get_all_checks():
        """Returns all TR101-290 error checks with descriptions"""
        return {
            'P1': [
                {'code': 'P1.1', 'name': 'TS_sync_loss', 'description': 'Loss of synchronization'},
                {'code': 'P1.2', 'name': 'Sync_byte_error', 'description': 'Sync byte not equal to 0x47'},
                {'code': 'P1.3', 'name': 'PAT_error', 'description': 'PAT not occurring at least every 0.5s'},
                {'code': 'P1.4', 'name': 'Continuity_count_error', 'description': 'Incorrect packet sequence'},
                {'code': 'P1.5', 'name': 'PMT_error', 'description': 'PMT not occurring at least every 0.5s'},
                {'code': 'P1.6', 'name': 'PID_error', 'description': 'Referenced PID not present'},
            ],
            'P2': [
                {'code': 'P2.1', 'name': 'Transport_error', 'description': 'Transport_error_indicator set'},
                {'code': 'P2.2', 'name': 'CRC_error', 'description': 'CRC error in PSI/SI tables'},
                {'code': 'P2.3', 'name': 'PCR_repetition_error', 'description': 'PCR interval > 40ms'},
                {'code': 'P2.4', 'name': 'PCR_discontinuity', 'description': 'PCR discontinuity > 100ms'},
                {'code': 'P2.5', 'name': 'PCR_accuracy_error', 'description': 'PCR accuracy > 500ns'},
                {'code': 'P2.6', 'name': 'PTS_error', 'description': 'PTS repetition or discontinuity error'},
                {'code': 'P2.7', 'name': 'CAT_error', 'description': 'CAT not occurring at least every 0.5s'},
            ],
            'P3': [
                {'code': 'P3.1', 'name': 'NIT_error', 'description': 'NIT not occurring at least every 10s'},
                {'code': 'P3.2', 'name': 'NIT_actual_error', 'description': 'NIT_actual not present'},
                {'code': 'P3.3', 'name': 'SI_repetition_error', 'description': 'SI table repetition error'},
                {'code': 'P3.4', 'name': 'Unreferenced_PID', 'description': 'PID present but not in PMT'},
                {'code': 'P3.5', 'name': 'SDT_error', 'description': 'SDT not occurring at least every 2s'},
                {'code': 'P3.6', 'name': 'EIT_error', 'description': 'EIT not occurring at least every 2s'},
                {'code': 'P3.7', 'name': 'RST_error', 'description': 'RST error'},
                {'code': 'P3.8', 'name': 'TDT_error', 'description': 'TDT not occurring at least every 30s'},
                {'code': 'P3.9', 'name': 'Empty_buffer_error', 'description': 'Data broadcast buffer empty'},
                {'code': 'P3.10', 'name': 'Data_delay_error', 'description': 'Data delay > threshold'},
            ]
        }
    
    @staticmethod
    def classify_p1_errors(report):
        """Priority 1 errors - Critical errors affecting basic monitoring and rendering"""
        p1_errors = []
        
        # P1.1 & P1.2: TS sync loss / Sync byte error
        sync_errors = report.get('sync_errors', 0)
        p1_errors.append({
            'code': 'P1.1',
            'name': 'TS_sync_loss',
            'count': sync_errors,
            'status': 'FAIL' if sync_errors > 0 else 'PASS',
            'severity': 'CRITICAL' if sync_errors > 0 else 'OK',
            'description': f"{sync_errors} sync loss events detected" if sync_errors > 0 else "No sync loss detected"
        })
        
        p1_errors.append({
            'code': 'P1.2',
            'name': 'Sync_byte_error',
            'count': sync_errors,
            'status': 'FAIL' if sync_errors > 0 else 'PASS',
            'severity': 'CRITICAL' if sync_errors > 0 else 'OK',
            'description': f"{sync_errors} packets with incorrect sync byte (not 0x47)" if sync_errors > 0 else "All sync bytes correct (0x47)"
        })
        
        # P1.3: PAT error (should occur at least every 0.5s)
        pat_present = bool(report.get('programs'))
        pat_interval = report.get('pat_interval_ms', 0) / 1000.0 if report.get('pat_interval_ms') else 0
        pat_error = not pat_present or (pat_interval > 0.5 if pat_interval > 0 else False)
        p1_errors.append({
            'code': 'P1.3',
            'name': 'PAT_error',
            'count': 1 if pat_error else 0,
            'status': 'FAIL' if pat_error else 'PASS',
            'severity': 'CRITICAL' if pat_error else 'OK',
            'description': f"PAT not found or interval > 0.5s (interval: {pat_interval:.3f}s)" if pat_error else f"PAT present, interval: {pat_interval:.3f}s"
        })
        
        # P1.4: Continuity count error
        total_cc_errors = report.get('continuity_errors_total', 0)
        cc_details = []
        if total_cc_errors > 0:
            for pid, info in report.get('continuity_by_pid', {}).items():
                if info['errors'] > 0:
                    cc_details.append(f"PID 0x{pid:04X}: {info['errors']} errors")
        
        p1_errors.append({
            'code': 'P1.4',
            'name': 'Continuity_count_error',
            'count': total_cc_errors,
            'status': 'FAIL' if total_cc_errors > 0 else 'PASS',
            'severity': 'CRITICAL' if total_cc_errors > 0 else 'OK',
            'description': f"{total_cc_errors} continuity errors: " + ", ".join(cc_details[:3]) if total_cc_errors > 0 else "No continuity errors"
        })
        
        # P1.5: PMT error (should occur at least every 0.5s per program)
        pmt_present = bool(report.get('pmts'))
        pmt_interval = report.get('pmt_interval_ms', 0) / 1000.0 if report.get('pmt_interval_ms') else 0
        pmt_error = (pat_present and not pmt_present) or (pmt_interval > 0.5 if pmt_interval > 0 else False)
        p1_errors.append({
            'code': 'P1.5',
            'name': 'PMT_error',
            'count': 1 if pmt_error else 0,
            'status': 'FAIL' if pmt_error else 'PASS',
            'severity': 'CRITICAL' if pmt_error else 'OK',
            'description': f"PMT not found or interval > 0.5s (interval: {pmt_interval:.3f}s)" if pmt_error else f"PMT present, interval: {pmt_interval:.3f}s"
        })
        
        # P1.6: PID error (referenced PID not present)
        missing_pids = report.get('missing_referenced_pids', [])
        pid_error_count = len(missing_pids)
        p1_errors.append({
            'code': 'P1.6',
            'name': 'PID_error',
            'count': pid_error_count,
            'status': 'FAIL' if pid_error_count > 0 else 'PASS',
            'severity': 'CRITICAL' if pid_error_count > 0 else 'OK',
            'description': f"Missing PIDs: {', '.join([f'0x{p:04X}' for p in missing_pids[:5]])}" if pid_error_count > 0 else "All referenced PIDs present"
        })
        
        return p1_errors
    
    @staticmethod
    def classify_p2_errors(report):
        """Priority 2 errors - Errors affecting service quality"""
        p2_errors = []
        
        # P2.1: Transport error indicator
        tei_count = report.get('transport_error_indicators', 0)
        p2_errors.append({
            'code': 'P2.1',
            'name': 'Transport_error',
            'count': tei_count,
            'status': 'FAIL' if tei_count > 0 else 'PASS',
            'severity': 'ERROR' if tei_count > 0 else 'OK',
            'description': f"{tei_count} packets with TEI set ({report.get('tei_percent', 0):.6f}%)" if tei_count > 0 else "No transport errors"
        })
        
        # P2.2: CRC error in PSI/SI tables
        crc_errors = report.get('crc_errors', 0)
        p2_errors.append({
            'code': 'P2.2',
            'name': 'CRC_error',
            'count': crc_errors,
            'status': 'FAIL' if crc_errors > 0 else 'PASS',
            'severity': 'ERROR' if crc_errors > 0 else 'OK',
            'description': f"{crc_errors} CRC errors in PSI/SI tables" if crc_errors > 0 else "No CRC errors"
        })
        
        # P2.3: PCR repetition error (interval > 40ms)
        pcr_rep_errors = report.get('pcr_repetition_errors', 0)
        p2_errors.append({
            'code': 'P2.3',
            'name': 'PCR_repetition_error',
            'count': pcr_rep_errors,
            'status': 'FAIL' if pcr_rep_errors > 0 else 'PASS',
            'severity': 'ERROR' if pcr_rep_errors > 0 else 'OK',
            'description': f"{pcr_rep_errors} PCR intervals > 40ms" if pcr_rep_errors > 0 else "PCR repetition within limits"
        })
        
        # P2.4: PCR discontinuity (> 100ms)
        pcr_issues = report.get('pcr_jitter_issues', {})
        total_pcr_disc = sum(info.get('large_jumps', 0) for info in pcr_issues.values())
        pcr_details = []
        for pid, info in pcr_issues.items():
            if info.get('large_jumps', 0) > 0:
                pcr_details.append(f"PID 0x{pid:04X}: {info['large_jumps']} jumps")
        
        p2_errors.append({
            'code': 'P2.4',
            'name': 'PCR_discontinuity',
            'count': total_pcr_disc,
            'status': 'FAIL' if total_pcr_disc > 0 else 'PASS',
            'severity': 'ERROR' if total_pcr_disc > 0 else 'OK',
            'description': f"{total_pcr_disc} PCR discontinuities > 100ms: " + ", ".join(pcr_details[:3]) if total_pcr_disc > 0 else "No PCR discontinuities"
        })
        
        # P2.5: PCR accuracy error (> 500ns)
        pcr_acc_errors = report.get('pcr_accuracy_errors', 0)
        p2_errors.append({
            'code': 'P2.5',
            'name': 'PCR_accuracy_error',
            'count': pcr_acc_errors,
            'status': 'FAIL' if pcr_acc_errors > 0 else 'PASS',
            'severity': 'ERROR' if pcr_acc_errors > 0 else 'OK',
            'description': f"{pcr_acc_errors} PCR accuracy errors > 500ns" if pcr_acc_errors > 0 else "PCR accuracy within limits"
        })
        
        # P2.6: PTS error
        pts_errors = report.get('pts_errors', 0)
        p2_errors.append({
            'code': 'P2.6',
            'name': 'PTS_error',
            'count': pts_errors,
            'status': 'FAIL' if pts_errors > 0 else 'PASS',
            'severity': 'ERROR' if pts_errors > 0 else 'OK',
            'description': f"{pts_errors} PTS errors detected" if pts_errors > 0 else "No PTS errors"
        })
        
        # P2.7: CAT error (should occur at least every 0.5s)
        cat_interval = report.get('cat_interval_ms', 0) / 1000.0 if report.get('cat_interval_ms') else 0
        cat_error = cat_interval > 0.5 if cat_interval > 0 else False
        p2_errors.append({
            'code': 'P2.7',
            'name': 'CAT_error',
            'count': 1 if cat_error else 0,
            'status': 'FAIL' if cat_error else 'PASS',
            'severity': 'ERROR' if cat_error else 'OK',
            'description': f"CAT interval > 0.5s ({cat_interval:.3f}s)" if cat_error else "CAT not applicable or within limits"
        })
        
        return p2_errors
    
    @staticmethod
    def classify_p3_errors(report):
        """Priority 3 errors - Application-specific errors"""
        p3_errors = []
        
        # P3.1: NIT error (should occur at least every 10s)
        nit_interval = report.get('nit_interval_ms', 0) / 1000.0 if report.get('nit_interval_ms') else 0
        nit_error = nit_interval > 10.0 if nit_interval > 0 else False
        p3_errors.append({
            'code': 'P3.1',
            'name': 'NIT_error',
            'count': 1 if nit_error else 0,
            'status': 'FAIL' if nit_error else 'PASS',
            'severity': 'WARNING' if nit_error else 'OK',
            'description': f"NIT interval > 10s ({nit_interval:.3f}s)" if nit_error else "NIT not applicable or within limits"
        })
        
        # P3.2: NIT_actual error
        nit_actual_present = report.get('nit_actual_present', False)
        p3_errors.append({
            'code': 'P3.2',
            'name': 'NIT_actual_error',
            'count': 0 if nit_actual_present else 1,
            'status': 'PASS' if nit_actual_present else 'FAIL',
            'severity': 'OK' if nit_actual_present else 'WARNING',
            'description': "NIT_actual present" if nit_actual_present else "NIT_actual not found"
        })
        
        # P3.3: SI repetition error
        si_rep_errors = report.get('si_repetition_errors', 0)
        p3_errors.append({
            'code': 'P3.3',
            'name': 'SI_repetition_error',
            'count': si_rep_errors,
            'status': 'FAIL' if si_rep_errors > 0 else 'PASS',
            'severity': 'WARNING' if si_rep_errors > 0 else 'OK',
            'description': f"{si_rep_errors} SI table repetition errors" if si_rep_errors > 0 else "SI repetition within limits"
        })
        
        # P3.4: Unreferenced PID
        unreferenced_pids = report.get('unreferenced_pids', [])
        unref_count = len(unreferenced_pids)
        p3_errors.append({
            'code': 'P3.4',
            'name': 'Unreferenced_PID',
            'count': unref_count,
            'status': 'FAIL' if unref_count > 0 else 'PASS',
            'severity': 'WARNING' if unref_count > 0 else 'OK',
            'description': f"Unreferenced PIDs: {', '.join([f'0x{p:04X}' for p in unreferenced_pids[:5]])}" if unref_count > 0 else "All PIDs referenced in PMT"
        })
        
        # P3.5: SDT error (should occur at least every 2s)
        sdt_interval = report.get('sdt_interval_ms', 0) / 1000.0 if report.get('sdt_interval_ms') else 0
        sdt_error = sdt_interval > 2.0 if sdt_interval > 0 else False
        p3_errors.append({
            'code': 'P3.5',
            'name': 'SDT_error',
            'count': 1 if sdt_error else 0,
            'status': 'FAIL' if sdt_error else 'PASS',
            'severity': 'WARNING' if sdt_error else 'OK',
            'description': f"SDT interval > 2s ({sdt_interval:.3f}s)" if sdt_error else "SDT not applicable or within limits"
        })
        
        # P3.6: EIT error (should occur at least every 2s)
        eit_interval = report.get('eit_interval_ms', 0) / 1000.0 if report.get('eit_interval_ms') else 0
        eit_error = eit_interval > 2.0 if eit_interval > 0 else False
        p3_errors.append({
            'code': 'P3.6',
            'name': 'EIT_error',
            'count': 1 if eit_error else 0,
            'status': 'FAIL' if eit_error else 'PASS',
            'severity': 'WARNING' if eit_error else 'OK',
            'description': f"EIT interval > 2s ({eit_interval:.3f}s)" if eit_error else "EIT not applicable or within limits"
        })
        
        # P3.7: RST error
        rst_errors = report.get('rst_errors', 0)
        p3_errors.append({
            'code': 'P3.7',
            'name': 'RST_error',
            'count': rst_errors,
            'status': 'FAIL' if rst_errors > 0 else 'PASS',
            'severity': 'WARNING' if rst_errors > 0 else 'OK',
            'description': f"{rst_errors} RST errors" if rst_errors > 0 else "No RST errors"
        })
        
        # P3.8: TDT error (should occur at least every 30s)
        tdt_interval = report.get('tdt_interval_ms', 0) / 1000.0 if report.get('tdt_interval_ms') else 0
        tdt_error = tdt_interval > 30.0 if tdt_interval > 0 else False
        p3_errors.append({
            'code': 'P3.8',
            'name': 'TDT_error',
            'count': 1 if tdt_error else 0,
            'status': 'FAIL' if tdt_error else 'PASS',
            'severity': 'WARNING' if tdt_error else 'OK',
            'description': f"TDT interval > 30s ({tdt_interval:.3f}s)" if tdt_error else "TDT not applicable or within limits"
        })
        
        # P3.9: Empty buffer error
        empty_buffer_errors = report.get('empty_buffer_errors', 0)
        p3_errors.append({
            'code': 'P3.9',
            'name': 'Empty_buffer_error',
            'count': empty_buffer_errors,
            'status': 'FAIL' if empty_buffer_errors > 0 else 'PASS',
            'severity': 'WARNING' if empty_buffer_errors > 0 else 'OK',
            'description': f"{empty_buffer_errors} empty buffer errors" if empty_buffer_errors > 0 else "No buffer underruns"
        })
        
        # P3.10: Data delay error
        data_delay_errors = report.get('data_delay_errors', 0)
        p3_errors.append({
            'code': 'P3.10',
            'name': 'Data_delay_error',
            'count': data_delay_errors,
            'status': 'FAIL' if data_delay_errors > 0 else 'PASS',
            'severity': 'WARNING' if data_delay_errors > 0 else 'OK',
            'description': f"{data_delay_errors} data delay errors" if data_delay_errors > 0 else "Data delay within limits"
        })
        
        return p3_errors


class TSAnalyserGUI:
    def on_program_selected(self, event=None):
        """Handle program dropdown selection change."""
        # TODO: Implement logic to update stream dropdown and graphs based on selected program
        pass

    def on_stream_selected(self, event=None):
        """Handle stream dropdown selection change."""
        # TODO: Implement logic to update graphs based on selected stream
        pass
    
    def create_tr101290_tab(self):
        """Create comprehensive TR101-290 analysis tab"""
        tr_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(tr_frame, text="TR101-290")
        
        tr_frame.columnconfigure(0, weight=1)
        tr_frame.rowconfigure(1, weight=1)
        
        # Summary header
        summary_frame = ttk.LabelFrame(tr_frame, text="TR101-290 Compliance Summary (ETSI TR 101 290 V1.2.1)", padding="10")
        summary_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.tr_summary_label = ttk.Label(summary_frame, text="Analysis pending...", font=('TkDefaultFont', 10))
        self.tr_summary_label.pack()
        
        # Notebook for P1/P2/P3 tabs
        tr_notebook = ttk.Notebook(tr_frame)
        tr_notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # P1 Priority tab
        p1_tab = ttk.Frame(tr_notebook, padding="5")
        tr_notebook.add(p1_tab, text="Priority 1 (Critical)")
        p1_tab.columnconfigure(0, weight=1)
        p1_tab.rowconfigure(0, weight=1)
        
        self.tr_p1_tree = ttk.Treeview(p1_tab, columns=('code', 'name', 'count', 'status', 'description'), 
                                        show='headings', height=12)
        self.tr_p1_tree.heading('code', text='Error Code')
        self.tr_p1_tree.heading('name', text='Check Name')
        self.tr_p1_tree.heading('count', text='Count')
        self.tr_p1_tree.heading('status', text='Status')
        self.tr_p1_tree.heading('description', text='Description')
        
        self.tr_p1_tree.column('code', width=80, anchor='center')
        self.tr_p1_tree.column('name', width=200)
        self.tr_p1_tree.column('count', width=80, anchor='center')
        self.tr_p1_tree.column('status', width=80, anchor='center')
        self.tr_p1_tree.column('description', width=450)
        
        p1_scroll = ttk.Scrollbar(p1_tab, orient=tk.VERTICAL, command=self.tr_p1_tree.yview)
        self.tr_p1_tree.configure(yscrollcommand=p1_scroll.set)
        self.tr_p1_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        p1_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # P2 Priority tab
        p2_tab = ttk.Frame(tr_notebook, padding="5")
        tr_notebook.add(p2_tab, text="Priority 2 (Quality)")
        p2_tab.columnconfigure(0, weight=1)
        p2_tab.rowconfigure(0, weight=1)
        
        self.tr_p2_tree = ttk.Treeview(p2_tab, columns=('code', 'name', 'count', 'status', 'description'), 
                                        show='headings', height=12)
        self.tr_p2_tree.heading('code', text='Error Code')
        self.tr_p2_tree.heading('name', text='Check Name')
        self.tr_p2_tree.heading('count', text='Count')
        self.tr_p2_tree.heading('status', text='Status')
        self.tr_p2_tree.heading('description', text='Description')
        
        self.tr_p2_tree.column('code', width=80, anchor='center')
        self.tr_p2_tree.column('name', width=200)
        self.tr_p2_tree.column('count', width=80, anchor='center')
        self.tr_p2_tree.column('status', width=80, anchor='center')
        self.tr_p2_tree.column('description', width=450)
        
        p2_scroll = ttk.Scrollbar(p2_tab, orient=tk.VERTICAL, command=self.tr_p2_tree.yview)
        self.tr_p2_tree.configure(yscrollcommand=p2_scroll.set)
        self.tr_p2_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        p2_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # P3 Priority tab
        p3_tab = ttk.Frame(tr_notebook, padding="5")
        tr_notebook.add(p3_tab, text="Priority 3 (Application)")
        p3_tab.columnconfigure(0, weight=1)
        p3_tab.rowconfigure(0, weight=1)
        
        self.tr_p3_tree = ttk.Treeview(p3_tab, columns=('code', 'name', 'count', 'status', 'description'), 
                                        show='headings', height=12)
        self.tr_p3_tree.heading('code', text='Error Code')
        self.tr_p3_tree.heading('name', text='Check Name')
        self.tr_p3_tree.heading('count', text='Count')
        self.tr_p3_tree.heading('status', text='Status')
        self.tr_p3_tree.heading('description', text='Description')
        
        self.tr_p3_tree.column('code', width=80, anchor='center')
        self.tr_p3_tree.column('name', width=200)
        self.tr_p3_tree.column('count', width=80, anchor='center')
        self.tr_p3_tree.column('status', width=80, anchor='center')
        self.tr_p3_tree.column('description', width=450)
        
        p3_scroll = ttk.Scrollbar(p3_tab, orient=tk.VERTICAL, command=self.tr_p3_tree.yview)
        self.tr_p3_tree.configure(yscrollcommand=p3_scroll.set)
        self.tr_p3_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        p3_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def __init__(self, root):
        self.root = root
        self.root.title("VideoAnalyzer - Professional Edition")
        self.root.geometry("1000x700")

        # Theme toggle
        self.USE_DARK_THEME = False
        self.MS_BLUE = "#0078D4"
        self.DARK_BG = "#121212"
        self.DARK_PANEL = "#1E1E1E"
        self.DARK_BORDER = "#2A2A2A"
        self.TEXT_PRIMARY = "#E6E6E6"
        self.TEXT_SECONDARY = "#B0B0B0"
        self.ACCENT = "#1890FF"

        if self.USE_DARK_THEME:
            try:
                self.root.configure(bg=self.DARK_BG)
            except Exception:
                pass

            style = ttk.Style()
            try:
                style.theme_use('clam')
            except Exception:
                style.theme_use('default')
            style.configure('TFrame', background=self.DARK_BG)
            style.configure('TLabel', background=self.DARK_BG, foreground=self.TEXT_PRIMARY)
            style.configure('TLabelFrame', background=self.DARK_BG, foreground=self.TEXT_PRIMARY)
            style.configure('TButton', background=self.DARK_PANEL, foreground=self.TEXT_PRIMARY, font=('Segoe UI', 10))
            style.map('TButton', background=[('active', self.ACCENT)], foreground=[('active', 'white')])
            style.configure('TNotebook', background=self.DARK_BG, bordercolor=self.DARK_BORDER)
            style.configure('TNotebook.Tab', background=self.DARK_PANEL, foreground=self.TEXT_PRIMARY, font=('Segoe UI', 10))
            style.map('TNotebook.Tab', 
                       background=[('selected', self.DARK_BG), ('disabled', '#444444')],
                       foreground=[('selected', self.TEXT_PRIMARY), ('disabled', '#888888')])
            style.configure('Treeview', background=self.DARK_PANEL, fieldbackground=self.DARK_PANEL, foreground=self.TEXT_PRIMARY)
            style.map('Treeview', background=[('selected', self.ACCENT)], foreground=[('selected', 'white')])
            style.configure('TProgressbar', background=self.ACCENT)

        # Default (light) UI accent: light-blue tabs and buttons
        if not self.USE_DARK_THEME:
            try:
                self.apply_file_theme(None)
            except Exception:
                pass


        self.analyser = None
        self.current_file = None
        self.analysis_thread = None
        self.last_report = None
        
        # Initialize NAL extraction caches
        self._nal_cache = {}
        self._all_nals_unlimited = None
        self._frame_nals_grouped = None

        self.setup_menu()
        self.setup_ui()
    
    def setup_menu(self):
        """Create menu bar"""
        menu_bg = self.MS_BLUE
        menu_fg = "white"
        menu_active_bg = self.ACCENT
        menu_active_fg = "white"

        menubar = tk.Menu(
            self.root,
            bg=menu_bg,
            fg=menu_fg,
            activebackground=menu_active_bg,
            activeforeground=menu_active_fg,
            relief=tk.FLAT,
            bd=0
        )
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=menu_bg,
            fg=menu_fg,
            activebackground=menu_active_bg,
            activeforeground=menu_active_fg,
            relief=tk.FLAT,
            bd=0
        )
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Media File...", command=self.browse_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Open Network Stream...", command=self.open_network_stream)
        file_menu.add_command(label="Open URL Stream...", command=self.open_url_stream)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit, accelerator="Ctrl+Q")
        
        # Help menu
        help_menu = tk.Menu(
            menubar,
            tearoff=0,
            bg=menu_bg,
            fg=menu_fg,
            activebackground=menu_active_bg,
            activeforeground=menu_active_fg,
            relief=tk.FLAT,
            bd=0
        )
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Bind keyboard shortcuts
        self.root.bind('<Control-o>', lambda e: self.browse_file())
        self.root.bind('<Control-q>', lambda e: self.root.quit())

        # Global option defaults for dark palette
        if self.USE_DARK_THEME:
            try:
                self.root.option_add('*background', self.DARK_BG)
                self.root.option_add('*foreground', self.TEXT_PRIMARY)
                self.root.option_add('*activeBackground', self.ACCENT)
                self.root.option_add('*activeForeground', 'white')
                self.root.option_add('*selectBackground', self.ACCENT)
                self.root.option_add('*selectForeground', 'white')
            except Exception:
                pass

    def open_url_stream(self):
        """Open a streaming URL via PyAV (RTMP/HLS/SRT and others)."""
        if not AV_AVAILABLE:
            messagebox.showerror("PyAV Missing", "PyAV is required to open URL streams.")
            return
        dlg = tk.Toplevel(self.root)
        dlg.title("Open URL Stream (RTMP/HLS/SRT)")
        dlg.resizable(False, False)
        ttk.Label(dlg, text="URL").grid(row=0, column=0, padx=8, pady=6, sticky=tk.W)
        url_var = tk.StringVar(value="rtmp://example/live/stream")
        ttk.Entry(dlg, textvariable=url_var, width=48).grid(row=0, column=1, padx=8, pady=6, sticky=tk.W)
        # TS Ring Buffer Mode (for direct MPEG-TS URLs)
        ttk.Label(dlg, text="MPEG-TS Ring Mode?").grid(row=1, column=0, padx=8, pady=6, sticky=tk.W)
        ts_ring_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dlg, variable=ts_ring_var).grid(row=1, column=1, padx=8, pady=6, sticky=tk.W)
        ttk.Label(dlg, text="Auto-refresh (s)").grid(row=1, column=0, padx=8, pady=6, sticky=tk.W)
        refresh_var = tk.StringVar(value="5")
        ttk.Entry(dlg, textvariable=refresh_var, width=10).grid(row=1, column=1, padx=8, pady=6, sticky=tk.W)
        auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dlg, text="Enable", variable=auto_var).grid(row=1, column=2, padx=8, pady=6, sticky=tk.W)
        # RTMP analyser options
        rtmp_var = tk.BooleanVar(value=False)
        rtmp_chk = ttk.Checkbutton(dlg, text="Use RTMP Inspector (ffprobe)", variable=rtmp_var)
        rtmp_chk.grid(row=2, column=0, columnspan=2, padx=8, pady=6, sticky=tk.W)
        rtmp_frames_var = tk.BooleanVar(value=False)
        rtmp_packets_var = tk.BooleanVar(value=False)
        rtmp_frames_cb = ttk.Checkbutton(dlg, text="Show frames", variable=rtmp_frames_var)
        rtmp_frames_cb.grid(row=3, column=0, padx=8, pady=6, sticky=tk.W)
        rtmp_packets_cb = ttk.Checkbutton(dlg, text="Show packets", variable=rtmp_packets_var)
        rtmp_packets_cb.grid(row=3, column=1, padx=8, pady=6, sticky=tk.W)
        if not RTMP_ANALYSER_AVAILABLE:
            rtmp_chk.state(['disabled'])
            rtmp_frames_cb.state(['disabled'])
            rtmp_packets_cb.state(['disabled'])
        ttk.Label(dlg, text="Packets per snapshot").grid(row=2, column=0, padx=8, pady=6, sticky=tk.W)
        pkt_var = tk.StringVar(value="2000")
        ttk.Entry(dlg, textvariable=pkt_var, width=10).grid(row=2, column=1, padx=8, pady=6, sticky=tk.W)
        btns = ttk.Frame(dlg)
        btns.grid(row=3, column=0, columnspan=3, pady=10)
        def do_cancel():
            dlg.destroy()
        def do_open():
            dlg.destroy()
            url = url_var.get().strip()
            try:
                refresh_s = max(1, int(refresh_var.get())) if auto_var.get() else 0
                pkt_snapshot = max(188, int(pkt_var.get()))
            except Exception:
                refresh_s = 0
                pkt_snapshot = 2000
            if rtmp_var.get() and RTMP_ANALYSER_AVAILABLE:
                # run RTMP inspector in background and show output
                threading.Thread(target=self._run_rtmp_inspect, args=(url, rtmp_frames_var.get(), rtmp_packets_var.get()), daemon=True).start()
                return
            if ts_ring_var.get():
                threading.Thread(target=self._url_ts_ring_capture, args=(url, pkt_snapshot, refresh_s), daemon=True).start()
            else:
                threading.Thread(target=self._open_url_and_analyze, args=(url, refresh_s), daemon=True).start()
        ttk.Button(btns, text="Cancel", command=do_cancel).pack(side=tk.RIGHT, padx=6)
        ttk.Button(btns, text="Open", command=do_open).pack(side=tk.RIGHT, padx=6)
        dlg.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dlg.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

    def _open_url_and_analyze(self, url, refresh_s):
        """Open a streaming URL via PyAV and analyze according to container; periodically refresh if requested."""
        try:
            self.root.after(0, lambda: self.status_label.config(text=f"Opening {url}...", foreground="orange"))
            container = av.open(url)
            # Attempt to detect if content is MPEG-TS
            fmt_name = getattr(container, 'format', None).name if getattr(container, 'format', None) else ''
            # Fallback: infer TS if any stream has id-like PID and packet size
            is_ts_like = fmt_name and ('mpegts' in fmt_name or 'ts' in fmt_name)
            self.container_type = 'ts' if is_ts_like else 'mp4'
            # Close container for probing; follow standard display flow
            try:
                container.close()
            except Exception:
                pass
            if self.container_type == 'ts':
                # For URL streams, we cannot pass a filename; instead, do a lightweight MP4/MOV analysis fallback
                # and keep thumbnails via PyAV container usage in thumbnail extractor by opening self.current_file.
                # Here set current_file to URL for PyAV operations.
                self.current_file = url
                # Run TS-like analysis is file-based; use MP4/MOV path for URL, but keep tabs enabled only if TS when we later capture to file.
                self.root.after(0, self.run_mp4_mov_analysis)
            else:
                self.current_file = url
                self.root.after(0, self.run_mp4_mov_analysis)
            # Auto-refresh loop: re-run analysis periodically
            if refresh_s > 0:
                def refresher():
                    self.status_label.config(text=f"Auto-refresh every {refresh_s}s", foreground="blue")
                    while True:
                        import time
                        time.sleep(refresh_s)
                        try:
                            self.run_mp4_mov_analysis()
                        except Exception:
                            break
                threading.Thread(target=refresher, daemon=True).start()
        except Exception as e:
            self.root.after(0, self.show_error, f"URL open failed: {e}")

    def _url_ts_ring_capture(self, url, pkt_snapshot, refresh_s):
        """Read a direct MPEG-TS URL via PyAV into a ring buffer and analyze snapshots so TS tabs stay active."""
        import tempfile
        import time
        try:
            self.root.after(0, lambda: self.status_label.config(text=f"Opening TS URL {url}...", foreground="orange"))
            container = av.open(url)
            # Confirm format is MPEG-TS
            fmt_name = getattr(container, 'format', None).name if getattr(container, 'format', None) else ''
            if not fmt_name or ('mpegts' not in fmt_name and 'ts' not in fmt_name):
                self.root.after(0, self.show_error, "URL does not appear to be MPEG-TS. Disable TS ring mode or use a TS URL.")
                try:
                    container.close()
                except Exception:
                    pass
                return
            max_bytes = pkt_snapshot * 188
            ring = bytearray()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ts")
            tmp_path = tmp.name
            tmp.close()

    
        except Exception as e:
            self.root.after(0, self.show_error, f"TS URL ring capture failed: {e}")

    def _run_rtmp_inspect(self, url: str, show_frames: bool, show_packets: bool):
        """Run RTMP/URL inspection via the bundled `rtmp_analyser` (ffprobe) and show output."""
        if not RTMP_ANALYSER_AVAILABLE:
            self.root.after(0, lambda: self.show_error("RTMP analyser not available (missing ffprobe or module)."))
            return

        # Create output window
        def make_win():
            w = tk.Toplevel(self.root)
            w.title(f"RTMP Inspector: {url}")
            txt = scrolledtext.ScrolledText(w, width=100, height=40)
            txt.pack(fill=tk.BOTH, expand=True)
            return w, txt

        win, txt = None, None
        try:
            self.root.after(0, lambda: self.status_label.config(text=f"Inspecting {url} with ffprobe...", foreground="orange"))
            win, txt = make_win()
            # Query basic format/streams
            data = rtmp_analyser.ffprobe_json(url, ["-show_format", "-show_streams"]) or {}
            out = {
                "url": url,
                "format": data.get("format"),
                "streams": data.get("streams", []),
            }
            if show_frames:
                frames = rtmp_analyser.ffprobe_json(url, ["-show_frames"]) or {}
                out["frames"] = frames.get("frames", [])
            if show_packets:
                packets = rtmp_analyser.ffprobe_json(url, ["-show_packets"]) or {}
                out["packets"] = packets.get("packets", [])

            s = json.dumps(out, indent=2)
            def write_text():
                try:
                    txt.delete('1.0', tk.END)
                    txt.insert(tk.END, s)
                except Exception:
                    pass
            self.root.after(0, write_text)
            self.root.after(0, lambda: self.status_label.config(text=f"RTMP inspect completed", foreground="green"))
        except Exception as e:
            self.root.after(0, self.show_error, f"RTMP inspect failed: {e}")
            if win:
                try:
                    win.destroy()
                except Exception:
                    pass

    # --- NDI support methods ---
    def ndi_refresh_sources(self):
        if not NDI_AVAILABLE:
            messagebox.showerror("NDI Missing", "NDI support not available. Install SDK and Python binding.")
            return
        try:
            recv = NDIReceiver()
            sources = recv.list_sources()
            if not sources:
                sources = ["(no sources)"]
            try:
                self.ndi_source_combo['values'] = sources
            except Exception:
                try:
                    self.ndi_source_combo.delete(0, tk.END)
                    self.ndi_source_combo.insert(0, sources[0])
                except Exception:
                    pass
        except Exception as e:
            messagebox.showerror("NDI Error", f"Failed to enumerate NDI sources: {e}")

    def ndi_start_receive(self):
        if not NDI_AVAILABLE:
            messagebox.showerror("NDI Missing", "NDI support not available. Install SDK and Python binding.")
            return
        src = self.ndi_source_var.get().strip()
        if not src:
            # Try refresh to pick first
            self.ndi_refresh_sources()
            src = self.ndi_source_var.get().strip() or (self.ndi_source_combo.get() if hasattr(self.ndi_source_combo, 'get') else '')
        try:
            self._ndi_receiver = NDIReceiver()
            self._ndi_prev_frame = None
            self._ndi_recorder = None
            if self.ndi_record_var.get():
                # default filename with timestamp
                import datetime
                name = f"ndi_record_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                try:
                    self._ndi_recorder = NDIRecorder(name)
                except Exception as e:
                    messagebox.showwarning("Recorder", f"Failed to initialise recorder: {e}")

            def cb(frame):
                # Schedule GUI update on main thread
                try:
                    self.root.after(0, lambda f=frame: self.ndi_on_frame(f))
                except Exception:
                    pass

            self._ndi_receiver.start(source_name=src or None, frame_callback=cb)
            self.live_status_var.set(f"NDI live: {src}")
            self.ndi_preview_label.config(text="NDI preview running")
        except Exception as e:
            messagebox.showerror("NDI Start Failed", str(e))

    def ndi_stop_receive(self):
        try:
            if getattr(self, '_ndi_receiver', None):
                try:
                    self._ndi_receiver.stop()
                except Exception:
                    pass
                self._ndi_receiver = None
            if getattr(self, '_ndi_recorder', None):
                try:
                    self._ndi_recorder.close()
                except Exception:
                    pass
                self._ndi_recorder = None
            self.live_status_var.set("")
            self.ndi_preview_label.config(text="NDI preview stopped")
        except Exception:
            pass

    def ndi_on_frame(self, frame):
        # Display small preview and analysis
        try:
            # Optionally record
            if getattr(self, '_ndi_recorder', None):
                try:
                    self._ndi_recorder.write_frame(frame)
                except Exception:
                    pass

            # Run analysis
            try:
                res = analyze_frame(frame, getattr(self, '_ndi_prev_frame', None)) if analyze_frame else {}
            except Exception:
                res = {}

            self._ndi_prev_frame = frame

            # Update analysis text
            try:
                txt = json.dumps(res, indent=2)
            except Exception:
                txt = str(res)
            try:
                self.ndi_analysis_text.delete('1.0', tk.END)
                self.ndi_analysis_text.insert(tk.END, txt)
            except Exception:
                pass

            # Update preview image if PIL available
            if Image and ImageTk is not None:
                try:
                    import numpy as _np
                    arr = _np.asarray(frame)
                    # Convert BGR to RGB for PIL
                    if arr.ndim == 3 and arr.shape[2] >= 3:
                        img = Image.fromarray(arr[:, :, ::-1])
                    else:
                        img = Image.fromarray(arr)
                    img.thumbnail((320, 180))
                    imgtk = ImageTk.PhotoImage(img)
                    self.ndi_preview_label.config(image=imgtk, text='')
                    self.ndi_preview_label.image = imgtk
                except Exception:
                    # Fallback: show simple text
                    try:
                        self.ndi_preview_label.config(text=f"NDI frame {res.get('width', '?')}x{res.get('height', '?')}")
                    except Exception:
                        pass

        except Exception:
            pass

    def open_network_stream(self):
        """Prompt for a UDP/RTP unicast/multicast address and start live capture with optional auto-refresh."""
        import tempfile
        import socket
        import struct
        # Simple dialog using Toplevel
        dlg = tk.Toplevel(self.root)
        dlg.title("Open Network Stream (UDP/RTP TS)")
        dlg.resizable(False, False)
        ttk.Label(dlg, text="Address (IP)").grid(row=0, column=0, padx=8, pady=6, sticky=tk.W)
        addr_var = tk.StringVar(value="239.0.0.1")
        ttk.Entry(dlg, textvariable=addr_var, width=20).grid(row=0, column=1, padx=8, pady=6)
        ttk.Label(dlg, text="Port").grid(row=1, column=0, padx=8, pady=6, sticky=tk.W)
        port_var = tk.StringVar(value="1234")
        ttk.Entry(dlg, textvariable=port_var, width=10).grid(row=1, column=1, padx=8, pady=6, sticky=tk.W)
        ttk.Label(dlg, text="Multicast?").grid(row=2, column=0, padx=8, pady=6, sticky=tk.W)
        mcast_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dlg, variable=mcast_var).grid(row=2, column=1, padx=8, pady=6, sticky=tk.W)
        ttk.Label(dlg, text="RTP payload?").grid(row=2, column=2, padx=8, pady=6, sticky=tk.W)
        rtp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(dlg, variable=rtp_var).grid(row=2, column=3, padx=8, pady=6, sticky=tk.W)
        ttk.Label(dlg, text="Interface (IP)").grid(row=3, column=0, padx=8, pady=6, sticky=tk.W)
        iface_var = tk.StringVar(value="0.0.0.0")
        ttk.Entry(dlg, textvariable=iface_var, width=16).grid(row=3, column=1, padx=8, pady=6, sticky=tk.W)
        ttk.Label(dlg, text="Packets per snapshot").grid(row=4, column=0, padx=8, pady=6, sticky=tk.W)
        pkt_var = tk.StringVar(value="1000")
        ttk.Entry(dlg, textvariable=pkt_var, width=10).grid(row=4, column=1, padx=8, pady=6, sticky=tk.W)
        ttk.Label(dlg, text="Auto-refresh (s)").grid(row=5, column=0, padx=8, pady=6, sticky=tk.W)
        refresh_var = tk.StringVar(value="5")
        ttk.Entry(dlg, textvariable=refresh_var, width=10).grid(row=5, column=1, padx=8, pady=6, sticky=tk.W)
        auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(dlg, text="Enable", variable=auto_var).grid(row=5, column=2, padx=8, pady=6, sticky=tk.W)
        ttk.Label(dlg, text="Capture seconds").grid(row=3, column=0, padx=8, pady=6, sticky=tk.W)
        secs_var = tk.StringVar(value="5")
        ttk.Entry(dlg, textvariable=secs_var, width=10).grid(row=3, column=1, padx=8, pady=6, sticky=tk.W)
        btns = ttk.Frame(dlg)
        btns.grid(row=4, column=0, columnspan=2, pady=10)
        def do_cancel():
            dlg.destroy()
        def do_capture():
            dlg.destroy()
            try:
                addr = addr_var.get().strip()
                port = int(port_var.get())
                secs = max(1, int(secs_var.get()))
                iface_ip = iface_var.get().strip()
                pkt_snapshot = max(188, int(pkt_var.get()))
                refresh_s = max(1, int(refresh_var.get())) if auto_var.get() else 0
            except Exception:
                messagebox.showerror("Invalid Input", "Please enter valid address, port, and seconds.")
                return
            # Start background capture
            threading.Thread(target=self._live_capture_ring_udp, args=(addr, port, mcast_var.get(), iface_ip, secs, pkt_snapshot, refresh_s, rtp_var.get()), daemon=True).start()
        ttk.Button(btns, text="Cancel", command=do_cancel).pack(side=tk.RIGHT, padx=6)
        ttk.Button(btns, text="Capture & Analyze", command=do_capture).pack(side=tk.RIGHT, padx=6)
        # Center dialog
        dlg.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dlg.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dlg.winfo_height()) // 2
        dlg.geometry(f"+{x}+{y}")

    def _live_capture_ring_udp(self, addr, port, is_multicast, iface_ip, seconds, pkt_snapshot, refresh_s, rtp_payload):
        """Live UDP/RTP TS capture with fixed-size ring buffer; periodic snapshot analysis."""
        import time
        import tempfile
        import socket
        import struct
        try:
            self.root.after(0, lambda: self.status_label.config(text=f"Capturing {addr}:{port} (iface {iface_ip})...", foreground="orange"))
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            except Exception:
                pass
            # Bind
            sock.bind((iface_ip if iface_ip else "", port))
            if is_multicast:
                mreq = struct.pack("=4s4s", socket.inet_aton(addr), socket.inet_aton(iface_ip if iface_ip else "0.0.0.0"))
                sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
            sock.settimeout(0.5)

            # Ring buffer of TS bytes (limit by packets*188 approx)
            max_bytes = pkt_snapshot * 188
            ring = bytearray()
            start_time = time.time()
            last_snapshot = 0
            total_bytes = 0
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".ts")
            tmp_path = tmp.name
            tmp.close()
            while time.time() - start_time < seconds:
                try:
                    data, _src = sock.recvfrom(65536)
                except socket.timeout:
                    data = b""
                except Exception:
                    break
                if data:
                    total_bytes += len(data)
                    # If RTP payload is indicated, try to extract aligned TS packets from payload
                    if rtp_payload:
                        # Parse RTP header to locate payload start
                        payload = data
                        try:
                            if len(data) >= 12:
                                vpxcc = data[0]
                                version = (vpxcc >> 6) & 0x03
                                pad = (vpxcc >> 5) & 0x01
                                ext = (vpxcc >> 4) & 0x01
                                csrc_count = vpxcc & 0x0F
                                # Basic header length
                                offset = 12 + (csrc_count * 4)
                                if ext and len(data) >= offset + 4:
                                    # RTP header extension: 16-bit profile + 16-bit length (in 32-bit words)
                                    ext_len_words = (data[offset+2] << 8) | data[offset+3]
                                    offset += 4 + (ext_len_words * 4)
                                # Skip padding at end if present
                                if offset < len(data):
                                    payload = data[offset:]
                            # Align TS packets from payload using sync byte at 188 boundaries
                            sync_offset = -1
                            search_limit = min(128, len(payload))
                            for i in range(0, search_limit):
                                if payload[i] == 0x47:
                                    # Confirm next boundary
                                    if i + 188 <= len(payload) and payload[i+188] == 0x47:
                                        sync_offset = i
                                        break
                            if sync_offset >= 0:
                                payload = payload[sync_offset:]
                                full_packets = (len(payload) // 188)
                                if full_packets > 0:
                                    ring.extend(payload[:full_packets*188])
                            else:
                                # Fallback: append payload as-is
                                ring.extend(payload)
                        except Exception:
                            ring.extend(payload)
                    else:
                        ring.extend(data)
                    if len(ring) > max_bytes:
                        # Trim from the front to keep size
                        del ring[:len(ring)-max_bytes]
                now = time.time()
                if refresh_s > 0 and (now - last_snapshot) >= refresh_s and len(ring) >= 188*10:
                    # Snapshot and analyze
                    with open(tmp_path, "wb") as f:
                        f.write(ring)
                    self.current_file = tmp_path
                    self.container_type = "ts"
                    self.root.after(0, lambda: self.status_label.config(text=f"Analyzing snapshot ({len(ring)/1_000_000:.2f} MB)...", foreground="orange"))
                    self.root.after(0, self.start_analysis)
                    try:
                        self.root.after(0, lambda: self.live_status_var.set(f"Live: TS URL {url} | Last snapshot: {__import__('time').strftime('%H:%M:%S')}"))
                    except Exception:
                        pass
                    last_snapshot = now
            # Final snapshot
            if len(ring) >= 188*10:
                with open(tmp_path, "wb") as f:
                    f.write(ring)
                self.current_file = tmp_path
                self.container_type = "ts"
                self.root.after(0, lambda: self.status_label.config(text=f"Final snapshot analyze ({len(ring)/1_000_000:.2f} MB)", foreground="orange"))
                self.root.after(0, self.start_analysis)
                try:
                    self.root.after(0, lambda: self.live_status_var.set(f"Live: TS URL {url} | Last snapshot: {__import__('time').strftime('%H:%M:%S')}"))
                except Exception:
                    pass
            try:
                sock.close()
            except Exception:
                pass
            self.root.after(0, lambda: self.status_label.config(text=f"Capture complete ({total_bytes/1_000_000:.2f} MB)", foreground="green"))
        except Exception as e:
            self.root.after(0, self.show_error, f"Network live capture failed: {e}")
    
    def show_about(self):
        """Show About dialog"""
        # Create custom dialog with normal font
        dialog = tk.Toplevel(self.root)
        dialog.title("About VideoAnalyzer")
        dialog.geometry("500x450")
        dialog.resizable(False, False)
        
        # Center the dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Create text widget with scrollbar
        frame = ttk.Frame(dialog, padding="20")
        frame.pack(fill=tk.BOTH, expand=True)
        
        text_widget = tk.Text(frame, wrap=tk.WORD, font=('TkDefaultFont', 10), 
                             relief=tk.FLAT, background='#f0f0f0')
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        about_text = """VideoAnalyzer - Professional Edition

Version: 1.0

Developed by Sachin Chandrashekar, a seasoned Video Engineer.

Developed using Python 3 with the following technologies:

Video & Audio Analysis:
• PyAV - Pythonic bindings for FFmpeg libraries
• FFmpeg - Multimedia framework for video/audio processing
• Matplotlib - Data visualization and graphing

Stream Analysis:
• MP4/MOV probing via PyAV
• TR 101 290 - DVB measurement guidelines compliance (TS only)
• ISO/IEC 13818-1 - MPEG-2 Transport Stream specification (TS only)
• SCTE-35 - Digital Program Insertion Cueing (TS only)
• HRD/T-STD - Buffer compliance analysis (TS only)

GUI Framework:
• Tkinter - Python's standard GUI library
• PIL/Pillow - Image processing

© 2025 - Professional Video Analysis Tool"""
        
        text_widget.insert('1.0', about_text)
        text_widget.config(state=tk.DISABLED)
        
        # OK button
        button_frame = ttk.Frame(dialog, padding="0 10 0 10")
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="OK", command=dialog.destroy, width=10).pack()
        
        # Center dialog on parent
        dialog.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - dialog.winfo_width()) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
    
    def setup_ui(self):
        # Main PanedWindow for resizable sections
        main_pane = tk.PanedWindow(self.root, orient=tk.VERTICAL, sashrelief=tk.RAISED)
        main_pane.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Bottom pane: Tabs (notebook)
        tabs_frame = ttk.Frame(main_pane, padding="0")
        tabs_frame.columnconfigure(0, weight=1)
        main_pane.add(tabs_frame, stretch='always')

        # Create notebook before any tab frames
        self.notebook = ttk.Notebook(tabs_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Graphs tab and related widgets removed as requested
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Top pane: File selection, progress, status, summary
        top_frame = ttk.Frame(main_pane, padding="10")
        top_frame.columnconfigure(0, weight=1)
        main_pane.add(top_frame, stretch='always')

        # Middle pane: Preview & Thumbnails (horizontal paned)
        preview_pane = tk.PanedWindow(main_pane, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_pane.add(preview_pane, stretch='always')

        # Bottom pane: Tabs (notebook)
        tabs_frame = ttk.Frame(main_pane, padding="0")
        tabs_frame.columnconfigure(0, weight=1)
        main_pane.add(tabs_frame, stretch='always')

        # Create notebook before any tab frames
        self.notebook = ttk.Notebook(tabs_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Progress bar (hidden by default) - determinate mode for percentage
        self.progress = ttk.Progressbar(top_frame, mode='determinate', maximum=100)
        # Don't grid it yet - will be shown when analysis starts
        
        # Progress label for percentage display
        self.progress_label = ttk.Label(top_frame, text="", foreground="blue")

        # Status label
        self.status_label = ttk.Label(top_frame, text="Ready - Use File > Open to select a TS file", foreground="blue")
        self.status_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        # Live capture indicator (mode + last snapshot time)
        self.live_status_var = tk.StringVar(value="")
        self.live_status_label = ttk.Label(top_frame, textvariable=self.live_status_var, foreground="#555")
        self.live_status_label.grid(row=0, column=1, sticky=tk.E)

        # Summary section
        summary_frame = ttk.LabelFrame(top_frame, text="Analysis Summary", padding="5")
        summary_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        summary_frame.columnconfigure(1, weight=1)
        summary_frame.columnconfigure(3, weight=1)
        
        ttk.Label(summary_frame, text="Media File:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.media_file_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.media_file_var, font=('TkDefaultFont', 9)).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(summary_frame, text="Total Packets:").grid(row=0, column=2, sticky=tk.W, padx=(20, 5))
        self.total_packets_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.total_packets_var, font=('TkDefaultFont', 9)).grid(row=0, column=3, sticky=tk.W)
        
        ttk.Label(summary_frame, text="Duration:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.duration_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.duration_var, font=('TkDefaultFont', 9)).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(summary_frame, text="Bitrate:").grid(row=1, column=2, sticky=tk.W, padx=(20, 5))
        self.bitrate_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.bitrate_var, font=('TkDefaultFont', 9)).grid(row=1, column=3, sticky=tk.W)
        
        ttk.Label(summary_frame, text="PIDs:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self.pids_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.pids_var, font=('TkDefaultFont', 9)).grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(summary_frame, text="GOP Structure:").grid(row=2, column=2, sticky=tk.W, padx=(20, 5))
        self.gop_structure_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.gop_structure_var, font=('TkDefaultFont', 9)).grid(row=2, column=3, sticky=tk.W)
        
        ttk.Label(summary_frame, text="GOP Length (Min-Max):").grid(row=3, column=0, sticky=tk.W, padx=(0, 5))
        self.gop_length_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.gop_length_var, font=('TkDefaultFont', 9)).grid(row=3, column=1, sticky=tk.W)
        
        ttk.Label(summary_frame, text="GOP Type:").grid(row=3, column=2, sticky=tk.W, padx=(20, 5))
        self.gop_type_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.gop_type_var, font=('TkDefaultFont', 9)).grid(row=3, column=3, sticky=tk.W)
        
        ttk.Label(summary_frame, text="Resolution:").grid(row=4, column=0, sticky=tk.W, padx=(0, 5))
        self.resolution_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.resolution_var, font=('TkDefaultFont', 9)).grid(row=4, column=1, sticky=tk.W)
        
        ttk.Label(summary_frame, text="Frame Rate:").grid(row=4, column=2, sticky=tk.W, padx=(20, 5))
        self.frame_rate_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.frame_rate_var, font=('TkDefaultFont', 9)).grid(row=4, column=3, sticky=tk.W)
        
        ttk.Label(summary_frame, text="Scan Type:").grid(row=5, column=0, sticky=tk.W, padx=(0, 5))
        self.scan_type_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.scan_type_var, font=('TkDefaultFont', 9)).grid(row=5, column=1, sticky=tk.W)
        
        # NDI Controls
        ndi_frame = ttk.LabelFrame(top_frame, text="NDI Live Capture", padding="5")
        ndi_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(8, 10))
        ndi_frame.columnconfigure(1, weight=1)

        ttk.Label(ndi_frame, text="Source:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.ndi_source_var = tk.StringVar(value="")
        try:
            from tkinter import ttk as _ttk
            self.ndi_source_combo = _ttk.Combobox(ndi_frame, textvariable=self.ndi_source_var, state='readonly', width=48)
        except Exception:
            self.ndi_source_combo = ttk.Entry(ndi_frame, textvariable=self.ndi_source_var, width=48)
        self.ndi_source_combo.grid(row=0, column=1, sticky=(tk.W, tk.E))

        self.ndi_refresh_btn = ttk.Button(ndi_frame, text="Refresh", command=self.ndi_refresh_sources)
        self.ndi_refresh_btn.grid(row=0, column=2, padx=6)

        self.ndi_start_btn = ttk.Button(ndi_frame, text="Start", command=self.ndi_start_receive)
        self.ndi_start_btn.grid(row=0, column=3, padx=6)

        self.ndi_stop_btn = ttk.Button(ndi_frame, text="Stop", command=self.ndi_stop_receive)
        self.ndi_stop_btn.grid(row=0, column=4, padx=6)

        self.ndi_record_var = tk.BooleanVar(value=False)
        self.ndi_record_chk = ttk.Checkbutton(ndi_frame, text="Record to file", variable=self.ndi_record_var)
        self.ndi_record_chk.grid(row=1, column=1, sticky=tk.W, pady=(6,0))

        # Live preview and analysis summary
        self.ndi_preview_label = ttk.Label(ndi_frame, text="NDI preview not running", width=32)
        self.ndi_preview_label.grid(row=1, column=2, columnspan=3, sticky=tk.E)

        self.ndi_analysis_text = scrolledtext.ScrolledText(ndi_frame, height=4)
        self.ndi_analysis_text.grid(row=2, column=0, columnspan=5, sticky=(tk.W, tk.E), pady=(6,0))
        
        # Video Frames Preview section (Preview & Thumbnails in horizontal paned window)
        preview_frame = ttk.LabelFrame(preview_pane, text="Video Frames & Audio Preview", padding="5")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=0)  # Navigation row
        preview_frame.rowconfigure(1, weight=1)  # Thumbnails row
        preview_frame.rowconfigure(2, weight=0)  # Scrollbar row
        preview_pane.add(preview_frame, stretch='always')

        # Navigation controls (use grid)
        nav_frame = ttk.Frame(preview_frame)
        nav_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        nav_frame.columnconfigure(0, weight=0)
        nav_frame.columnconfigure(1, weight=0)
        nav_frame.columnconfigure(2, weight=1)
        nav_frame.columnconfigure(3, weight=0)
        nav_frame.columnconfigure(4, weight=0)
        nav_frame.columnconfigure(5, weight=0)
        nav_frame.columnconfigure(6, weight=0)
        nav_frame.columnconfigure(7, weight=0)
        nav_frame.columnconfigure(8, weight=0)
        nav_frame.columnconfigure(9, weight=0)
        nav_frame.columnconfigure(10, weight=0)

        self.prev_10_btn = ttk.Button(nav_frame, text="◀◀ Prev 10", command=self.navigate_prev_10, state=tk.DISABLED)
        self.prev_10_btn.grid(row=0, column=0, padx=5)

        self.next_10_btn = ttk.Button(nav_frame, text="Next 10 ▶▶", command=self.navigate_next_10, state=tk.DISABLED)
        self.next_10_btn.grid(row=0, column=1, padx=5)

        # I/IDR frame navigation buttons
        self.prev_idr_btn = ttk.Button(nav_frame, text="◀ Prev I/IDR", command=self.navigate_prev_idr, state=tk.DISABLED)
        self.prev_idr_btn.grid(row=0, column=6, padx=5)

        self.next_idr_btn = ttk.Button(nav_frame, text="Next I/IDR ▶", command=self.navigate_next_idr, state=tk.DISABLED)
        self.next_idr_btn.grid(row=0, column=7, padx=5)

        self.current_position_var = tk.StringVar(value="No frames loaded")
        ttk.Label(nav_frame, textvariable=self.current_position_var).grid(row=0, column=2, padx=10, sticky=tk.W)

        ttk.Label(nav_frame, text="Jump to frame:").grid(row=0, column=3, padx=5, sticky=tk.W)
        self.jump_frame_var = tk.StringVar()
        jump_entry = ttk.Entry(nav_frame, textvariable=self.jump_frame_var, width=8)
        jump_entry.grid(row=0, column=4, padx=5)

        self.jump_btn = ttk.Button(nav_frame, text="Go", command=self.jump_to_frame, state=tk.DISABLED)
        self.jump_btn.grid(row=0, column=5, padx=5)
        
        # Frame type filter (show All, I-frames only, or IDR-frames only)
        ttk.Label(nav_frame, text="Show:").grid(row=0, column=8, padx=(15, 5), sticky=tk.W)
        self.frame_filter_var = tk.StringVar(value="all")
        filter_frame = ttk.Frame(nav_frame)
        filter_frame.grid(row=0, column=9, columnspan=2, padx=5)
        ttk.Radiobutton(filter_frame, text="All Frames", variable=self.frame_filter_var, 
                       value="all", command=self.apply_frame_filter).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(filter_frame, text="I-Frames Only", variable=self.frame_filter_var, 
                       value="i_frames", command=self.apply_frame_filter).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(filter_frame, text="IDR-Frames Only", variable=self.frame_filter_var, 
                       value="idr_frames", command=self.apply_frame_filter).pack(side=tk.LEFT, padx=2)

        # Frame order toggle (PTS / DTS)
        ttk.Label(nav_frame, text="Order:").grid(row=0, column=11, padx=(10, 5), sticky=tk.W)
        self.frame_order_var = tk.StringVar(value="pts")
        order_frame = ttk.Frame(nav_frame)
        order_frame.grid(row=0, column=12, padx=5)
        ttk.Radiobutton(order_frame, text="PTS", variable=self.frame_order_var,
                   value="pts", command=self.apply_frame_order).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(order_frame, text="DTS", variable=self.frame_order_var,
                   value="dts", command=self.apply_frame_order).pack(side=tk.LEFT, padx=2)

        # Thumbnails display area with both horizontal and vertical scrollbars (use grid)
        thumbnails_canvas = tk.Canvas(preview_frame, bg='white', height=500)
        thumbnails_scroll_y = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=thumbnails_canvas.yview)
        thumbnails_scroll_x = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=thumbnails_canvas.xview)

        self.thumbnails_inner_frame = ttk.Frame(thumbnails_canvas)
        self.thumbnails_inner_frame.bind("<Configure>", lambda e: thumbnails_canvas.configure(scrollregion=thumbnails_canvas.bbox("all")))

        thumbnails_canvas.create_window((0, 0), window=self.thumbnails_inner_frame, anchor="nw")
        thumbnails_canvas.configure(yscrollcommand=thumbnails_scroll_y.set, xscrollcommand=thumbnails_scroll_x.set)

        thumbnails_canvas.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        thumbnails_scroll_y.grid(row=1, column=1, sticky=(tk.N, tk.S))
        thumbnails_scroll_x.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        # Store thumbnail images and navigation state
        self.thumbnail_images = []
        self.current_frame_start = 0
        self.total_video_frames = 0
        self.current_media_type = None  # 'video' or 'audio'
        self.video_stream_info = None
        self.audio_stream_info = None
        self.num_frames_var = tk.StringVar(value="10")  # Default 10 frames
        
        # Notebook for P1/P2 errors and details
        self.notebook = ttk.Notebook(tabs_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Bind tab change event for lazy graph loading
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_changed)
        self.graphs_loaded = False  # Track if graphs have been rendered
        
        self.create_tr101290_tab()
        
        # Stream Structure tab (Tree view of PAT/PMT/PIDs)
        structure_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(structure_frame, text="Stream Structure")
        
        structure_frame.columnconfigure(0, weight=1)
        structure_frame.rowconfigure(0, weight=1)
        
        # Tree view for hierarchical structure
        self.structure_tree = ttk.Treeview(structure_frame, show='tree headings', height=20)
        self.structure_tree.heading('#0', text='Stream Structure')
        self.structure_tree.column('#0', width=400)
        
        # Configure columns for additional info
        self.structure_tree['columns'] = ('value', 'packets', 'type')
        self.structure_tree.heading('value', text='Value')
        self.structure_tree.heading('packets', text='Packets')
        self.structure_tree.heading('type', text='Type/Info')
        
        self.structure_tree.column('value', width=120)
        self.structure_tree.column('packets', width=100)
        self.structure_tree.column('type', width=300)
        
        structure_scroll = ttk.Scrollbar(structure_frame, orient=tk.VERTICAL, command=self.structure_tree.yview)
        self.structure_tree.configure(yscrollcommand=structure_scroll.set)
        
        self.structure_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        structure_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind right-click context menu for TS/PES header inspection
        self.structure_tree.bind("<Button-3>", self.show_packet_header_menu)
        
        # Graphs tab
        if MATPLOTLIB_AVAILABLE:
            graphs_frame = ttk.Frame(self.notebook, padding="5")
            self.notebook.add(graphs_frame, text="Graphs")
            graphs_frame.columnconfigure(0, weight=1)
            graphs_frame.rowconfigure(0, weight=1)

            # Create canvas for graphs
            self.graphs_canvas = tk.Canvas(graphs_frame, bg='white')
            graphs_scroll_y = ttk.Scrollbar(graphs_frame, orient=tk.VERTICAL, command=self.graphs_canvas.yview)
            graphs_scroll_x = ttk.Scrollbar(graphs_frame, orient=tk.HORIZONTAL, command=self.graphs_canvas.xview)

            self.graphs_inner_frame = ttk.Frame(self.graphs_canvas)
            self.graphs_inner_frame.bind("<Configure>", 
                lambda e: self.graphs_canvas.configure(scrollregion=self.graphs_canvas.bbox("all")))

            self.graphs_canvas.create_window((0, 0), window=self.graphs_inner_frame, anchor="nw")
            self.graphs_canvas.configure(yscrollcommand=graphs_scroll_y.set, xscrollcommand=graphs_scroll_x.set)

            self.graphs_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            graphs_scroll_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
            graphs_scroll_x.grid(row=1, column=0, sticky=(tk.W, tk.E))

            self.graph_figures = []
        
        # Create SCTE35 and Elementary Streams tabs during initialization
        self.create_scte35_tab()
        self.create_es_tab()
        self.create_buffer_analysis_tab()
        self.create_captions_tab()
        self.create_klv_stanag_tab()
    
    def on_tab_changed(self, event):
        """Handle notebook tab change - lazy load graphs on first view"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # Get selected tab name
        try:
            selected_tab = self.notebook.tab(self.notebook.select(), "text")
        except:
            return
        
        # If Graphs tab selected and not yet loaded, load them now
        if selected_tab == "Graphs" and not self.graphs_loaded and hasattr(self, 'prepared_graph_figures'):
            if self.prepared_graph_figures:
                # Show progress briefly
                self.status_label.config(text="Rendering graphs...", foreground="orange")
                self.root.update_idletasks()
                
                # Attach graphs to GUI (this is the slow part ~5s for 6 graphs)
                self.attach_graphs_to_gui(self.prepared_graph_figures)
                self.graphs_loaded = True
                
                self.status_label.config(text="Analysis complete", foreground="green")
    
    def create_buffer_analysis_tab(self):
        """Create Buffer Analysis tab"""
        buffer_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(buffer_frame, text="Buffer Analysis")
        buffer_frame.columnconfigure(0, weight=1)
        buffer_frame.rowconfigure(3, weight=1)
        
        # Info text explaining buffer vs transport errors
        info_frame = ttk.Frame(buffer_frame, padding="5")
        info_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        info_text = ("ℹ️  Buffer Analysis checks decoder buffer compliance (T-STD/HRD) for overflow/underflow.\n"
                    "This is independent of transport errors (CC errors, sync loss). A stream can have CC errors\n"
                    "but still be buffer compliant if the bitrate and timing constraints are met.")
        info_label = ttk.Label(info_frame, text=info_text, foreground="blue", 
                              font=('TkDefaultFont', 9), wraplength=900, justify=tk.LEFT)
        info_label.pack(side=tk.LEFT, padx=5)
        
        # Summary frame at top
        summary_frame = ttk.LabelFrame(buffer_frame, text="Buffer Compliance Summary", padding="10")
        summary_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        summary_frame.columnconfigure(1, weight=1)
        
        ttk.Label(summary_frame, text="Status:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.buffer_status_var = tk.StringVar(value="Not analyzed")
        self.buffer_status_label = ttk.Label(summary_frame, textvariable=self.buffer_status_var, 
                                             font=('TkDefaultFont', 10))
        self.buffer_status_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(summary_frame, text="PIDs Analyzed:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.buffer_pids_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.buffer_pids_var).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(summary_frame, text="Overflows:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self.buffer_overflows_var = tk.StringVar(value="-")
        self.buffer_overflows_label = ttk.Label(summary_frame, textvariable=self.buffer_overflows_var)
        self.buffer_overflows_label.grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(summary_frame, text="Underflows:").grid(row=3, column=0, sticky=tk.W, padx=(0, 5))
        self.buffer_underflows_var = tk.StringVar(value="-")
        self.buffer_underflows_label = ttk.Label(summary_frame, textvariable=self.buffer_underflows_var)
        self.buffer_underflows_label.grid(row=3, column=1, sticky=tk.W)
        
        # Detailed analysis note
        detail_frame = ttk.Frame(buffer_frame, padding="5")
        detail_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5, 10))
        
        detail_text = ("Click on any PID row below to view detailed 3-stage buffer analysis graphs:\n"
                      "   Stage 1: ES Buffer (Decoder) ← Stage 2: Multiplex Buffer (Demux) ← Stage 3: Transport Buffer (TS packets)")
        detail_label = ttk.Label(detail_frame, text=detail_text, foreground="green", 
                                font=('TkDefaultFont', 9), wraplength=900, justify=tk.LEFT)
        detail_label.pack(side=tk.LEFT, padx=5)
        
        # Per-PID buffer details tree
        self.buffer_tree = ttk.Treeview(buffer_frame, 
                                        columns=("pid", "stream_type", "buffer_size", "max_util", "overflows", "underflows", "compliant"),
                                        show='headings', height=12)
        
        for col, label, w in [
            ("pid", "PID", 80),
            ("stream_type", "Stream Type", 180),
            ("buffer_size", "Buffer Size (KB)", 120),
            ("max_util", "Max Util %", 100),
            ("overflows", "Overflows", 80),
            ("underflows", "Underflows", 80),
            ("compliant", "Compliant", 80)
        ]:
            self.buffer_tree.heading(col, text=label)
            self.buffer_tree.column(col, width=w)
        
        buffer_scroll = ttk.Scrollbar(buffer_frame, orient=tk.VERTICAL, command=self.buffer_tree.yview)
        self.buffer_tree.configure(yscrollcommand=buffer_scroll.set)
        
        self.buffer_tree.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        buffer_scroll.grid(row=3, column=1, sticky=(tk.N, tk.S))
        
        # Bind double-click to show buffer graph
        self.buffer_tree.bind("<Double-1>", self.show_buffer_graph)

    def create_thumbnails_tab(self):
        """Legacy method - thumbnails are now in main UI"""
        pass

    def create_scte35_tab(self):
        scte35_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(scte35_frame, text="SCTE-35 Events")
        scte35_frame.columnconfigure(0, weight=1)
        scte35_frame.rowconfigure(0, weight=1)
        # Tree breakdown: parent = message, children = parsed fields (similar to NAL tree)
        self.scte35_tree = ttk.Treeview(
            scte35_frame,
            columns=("value", "info", "hex"),
            show='tree headings',
            height=20
        )
        self.scte35_tree.heading('#0', text='Item')
        self.scte35_tree.column('#0', width=240)
        for col, label, w in [
            ("value", "Value", 160),
            ("info", "Info", 240),
            ("hex", "Hex / Notes", 260)
        ]:
            self.scte35_tree.heading(col, text=label)
            self.scte35_tree.column(col, width=w)
        scte35_scroll = ttk.Scrollbar(scte35_frame, orient=tk.VERTICAL, command=self.scte35_tree.yview)
        self.scte35_tree.configure(yscrollcommand=scte35_scroll.set)
        self.scte35_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scte35_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

    def display_scte35_events(self, report):
        self.scte35_tree.delete(*self.scte35_tree.get_children())
        
        # Display PAT warnings if any
        pat_warnings = report.get('pat_warnings', [])
        if pat_warnings:
            for warning in pat_warnings:
                self.scte35_tree.insert('', 'end', values=(
                    "⚠ WARNING", "-", "PAT Corruption", "-", "-", warning
                ), tags=('warning',))
            self.scte35_tree.tag_configure('warning', background='#fff3cd', foreground='#856404')
        self.scte35_tree.tag_configure('error', background='#f8d7da', foreground='#721c24')
        
        # Display PMT warnings if any
        pmt_warnings = report.get('pmt_warnings', [])
        if pmt_warnings:
            for warning in pmt_warnings:
                self.scte35_tree.insert('', 'end', values=(
                    "⚠ WARNING", "-", "PMT Issue", "-", "-", warning
                ), tags=('warning',))
            self.scte35_tree.tag_configure('warning', background='#fff3cd', foreground='#856404')
        
        scte35_messages = report.get("scte35_messages", {})
        if not scte35_messages:
            if not pat_warnings and not pmt_warnings:  # Only show "not found" if there are no warnings
                self.scte35_tree.insert('', 'end', values=("-", "No SCTE-35 found", "-"))
            return

        def insert_tree(parent, nodes):
            for n in nodes:
                node_id = self.scte35_tree.insert(parent, 'end', text=n.get("label", ""),
                                                  values=(n.get("value", ""), n.get("info", ""), n.get("hex", "")))
                if n.get("children"):
                    insert_tree(node_id, n.get("children"))

        for pid, events in scte35_messages.items():
            for idx, msg in enumerate(events):
                if "error" in msg:
                    self.scte35_tree.insert('', 'end', text=f"PID {pid} evt {idx}",
                                             values=(f"ERROR: {msg['error']}", "", msg.get("raw_hex", "-")))
                    continue

                root = self.scte35_tree.insert('', 'end', text=f"PID {pid} evt {idx}",
                                                values=(msg.get("command_name", "-"), "", msg.get("raw_hex", "")))
                # Display validation errors/warnings if present
                validation = msg.get("validation", {})
                if validation.get("errors"):
                    for err in validation["errors"]:
                        err_node = self.scte35_tree.insert(root, 'end',
                            text=f"❌ ERROR: {err.field}",
                            values=(err.message, err.spec_ref, ""), tags=('error',))
                if validation.get("warnings"):
                    for warn in validation["warnings"]:
                        warn_node = self.scte35_tree.insert(root, 'end',
                            text=f"⚠ WARNING: {warn.field}",
                            values=(warn.message, warn.spec_ref, ""), tags=('warning',))
                
                # If we have a parsed tree, render it
                if msg.get("tree"):
                    insert_tree(root, msg["tree"])
                else:
                    # Fallback flat fields
                    for key in ["table_id", "section_length", "protocol_version", "encrypted_packet", "pts_adjustment", "tier", "splice_command_type", "command_name"]:
                        if key in msg:
                            self.scte35_tree.insert(root, 'end', text=key, values=(msg[key], "", ""))

    def create_es_tab(self):
        """Create Elementary Streams tab"""
        es_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(es_frame, text="Elementary Streams")
        es_frame.columnconfigure(0, weight=1)
        es_frame.rowconfigure(0, weight=1)
        
        # Create paned window to split tree and pie chart
        es_paned = tk.PanedWindow(es_frame, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        es_paned.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left side: Tree view
        tree_frame = ttk.Frame(es_paned)
        es_paned.add(tree_frame, stretch='always')
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        self.es_tree = ttk.Treeview(tree_frame, columns=("pid", "type", "pes_packets", "payload_bytes", "bitrate", "pts_range", "dts_range", "syntax_errors"), show='headings', height=12)
        for col, label, w in [
            ("pid", "PID", 80),
            ("type", "Type", 120),
            ("pes_packets", "PES", 60),
            ("payload_bytes", "Payload Bytes", 100),
            ("bitrate", "Bitrate (kbps)", 110),
            ("pts_range", "PTS Range", 140),
            ("dts_range", "DTS Range", 140),
            ("syntax_errors", "Syntax Errors", 200)
        ]:
            self.es_tree.heading(col, text=label)
            self.es_tree.column(col, width=w)
        es_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.es_tree.yview)
        self.es_tree.configure(yscrollcommand=es_scroll.set)
        self.es_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        es_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Right side: Pie chart with scrollbar
        chart_frame = ttk.Frame(es_paned)
        es_paned.add(chart_frame, stretch='always')
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        # Create scrollable canvas for the pie chart
        self.es_pie_canvas = tk.Canvas(chart_frame, bg='white', highlightthickness=0)
        pie_scrollbar = ttk.Scrollbar(chart_frame, orient=tk.VERTICAL, command=self.es_pie_canvas.yview)
        self.es_pie_canvas.configure(yscrollcommand=pie_scrollbar.set)
        
        self.es_pie_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        pie_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Bind mousewheel to canvas for scrolling
        def _on_mousewheel(event):
            self.es_pie_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.es_pie_canvas.bind("<MouseWheel>", _on_mousewheel)
        
        # Bind double-click event to show PES details
        self.es_tree.bind("<Double-1>", self.show_pes_details)
        
        # PES Details window (top-level)
        self.pes_detail_window = None
        self.pes_detail_tree = None

        
    def create_captions_tab(self):
        """Create Captions (CEA-608/CEA-708) display tab"""
        captions_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(captions_frame, text="Captions")
        captions_frame.columnconfigure(0, weight=1)
        captions_frame.rowconfigure(1, weight=1)
        
        # Info section
        info_frame = ttk.LabelFrame(captions_frame, text="Closed Captions (CEA-608/CEA-708)", padding="5")
        info_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        ttk.Label(info_frame, text="Decoded closed caption text from video stream:").pack(anchor=tk.W)
        
        # Main notebook for caption types
        captions_notebook = ttk.Notebook(captions_frame)
        captions_notebook.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # CEA-608 (Field 1 & 2) tab
        cea608_frame = ttk.Frame(captions_notebook, padding="5")
        captions_notebook.add(cea608_frame, text="CEA-608 Captions")
        cea608_frame.columnconfigure(0, weight=1)
        cea608_frame.rowconfigure(0, weight=1)
        
        self.caption_cea608_text = tk.Text(cea608_frame, wrap=tk.WORD, font=('Courier', 10), bg='white', fg='black')
        cea608_scroll = ttk.Scrollbar(cea608_frame, orient=tk.VERTICAL, command=self.caption_cea608_text.yview)
        self.caption_cea608_text.configure(yscrollcommand=cea608_scroll.set)
        self.caption_cea608_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        cea608_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # CEA-708 (DTVCC) tab
        cea708_frame = ttk.Frame(captions_notebook, padding="5")
        captions_notebook.add(cea708_frame, text="CEA-708 DTVCC")
        cea708_frame.columnconfigure(0, weight=1)
        cea708_frame.rowconfigure(0, weight=1)
        
        self.caption_cea708_text = tk.Text(cea708_frame, wrap=tk.WORD, font=('Courier', 10), bg='white', fg='black')
        cea708_scroll = ttk.Scrollbar(cea708_frame, orient=tk.VERTICAL, command=self.caption_cea708_text.yview)
        self.caption_cea708_text.configure(yscrollcommand=cea708_scroll.set)
        self.caption_cea708_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        cea708_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # SEI Summary tab
        sei_frame = ttk.Frame(captions_notebook, padding="5")
        captions_notebook.add(sei_frame, text="SEI Summary")
        sei_frame.columnconfigure(0, weight=1)
        sei_frame.rowconfigure(0, weight=1)
        
        self.caption_sei_text = tk.Text(sei_frame, wrap=tk.WORD, font=('Courier', 10), bg='white', fg='black')
        sei_scroll = ttk.Scrollbar(sei_frame, orient=tk.VERTICAL, command=self.caption_sei_text.yview)
        self.caption_sei_text.configure(yscrollcommand=sei_scroll.set)
        self.caption_sei_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        sei_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))

        
    def create_klv_stanag_tab(self):
        """Create KLV Metadata and STANAG 4609 Compliance tab"""
        klv_frame = ttk.Frame(self.notebook, padding="5")
        self.notebook.add(klv_frame, text="KLV / STANAG 4609")
        klv_frame.columnconfigure(0, weight=1)
        klv_frame.rowconfigure(3, weight=2)  # Telemetry section gets more space
        klv_frame.rowconfigure(5, weight=1)  # Issues section
        
        # Info text explaining KLV and STANAG 4609
        info_frame = ttk.Frame(klv_frame, padding="5")
        info_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        info_text = ("ℹ️  KLV Metadata & MISB Telemetry - Double-click any row in the telemetry table to see all packet samples")
        info_label = ttk.Label(info_frame, text=info_text, foreground="blue", 
                              font=('TkDefaultFont', 9, 'bold'), wraplength=900, justify=tk.LEFT)
        info_label.pack(side=tk.LEFT, padx=5)
        
        # Compliance summary frame
        summary_frame = ttk.LabelFrame(klv_frame, text="STANAG 4609 Compliance Summary", padding="10")
        summary_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        summary_frame.columnconfigure(1, weight=1)
        
        ttk.Label(summary_frame, text="Compliant:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.stanag_compliant_var = tk.StringVar(value="Not analyzed")
        self.stanag_compliant_label = ttk.Label(summary_frame, textvariable=self.stanag_compliant_var, 
                                                font=('TkDefaultFont', 10, 'bold'))
        self.stanag_compliant_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(summary_frame, text="KLV Detected:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5))
        self.klv_detected_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.klv_detected_var).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(summary_frame, text="Asynchronous PIDs:").grid(row=2, column=0, sticky=tk.W, padx=(0, 5))
        self.klv_async_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.klv_async_var).grid(row=2, column=1, sticky=tk.W)
        
        ttk.Label(summary_frame, text="Synchronous (in video):").grid(row=3, column=0, sticky=tk.W, padx=(0, 5))
        self.klv_sync_var = tk.StringVar(value="-")
        ttk.Label(summary_frame, textvariable=self.klv_sync_var).grid(row=3, column=1, sticky=tk.W)
        
        # KLV Details tree (compact)
        self.klv_tree = ttk.Treeview(klv_frame, 
                                     columns=("type", "pid", "sync_type", "packet_count", "misb_standard", "stream_type"),
                                     show='headings', height=4)
        
        for col, label, w in [
            ("type", "Type", 100),
            ("pid", "PID", 80),
            ("sync_type", "Synchronization", 200),
            ("packet_count", "Packet Count", 100),
            ("misb_standard", "MISB Standard", 180),
            ("stream_type", "Stream Type", 100)
        ]:
            self.klv_tree.heading(col, text=label)
            self.klv_tree.column(col, width=w)
        
        klv_scroll = ttk.Scrollbar(klv_frame, orient=tk.VERTICAL, command=self.klv_tree.yview)
        self.klv_tree.configure(yscrollcommand=klv_scroll.set)
        
        self.klv_tree.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        klv_scroll.grid(row=2, column=1, sticky=(tk.N, tk.S))
        
        # Telemetry (MISB ST 0601) decoded fields - MAIN SECTION
        telemetry_frame = ttk.LabelFrame(klv_frame, text="📡 Decoded MISB ST 0601 Telemetry (Double-click for packet details)", padding="5")
        telemetry_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10,0))
        telemetry_frame.columnconfigure(0, weight=1)
        telemetry_frame.rowconfigure(1, weight=1)
        
        # Map button for GPS visualization
        map_btn_frame = ttk.Frame(telemetry_frame)
        map_btn_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.map_btn = ttk.Button(map_btn_frame, text="🗺️ Show GPS Flight Path Map", 
                                   command=self.show_gps_map, state=tk.DISABLED)
        self.map_btn.pack(side=tk.LEFT, padx=5)
        ttk.Label(map_btn_frame, text="(Plots Sensor/Frame Center coordinates)", 
                 foreground='#666', font=('TkDefaultFont', 8)).pack(side=tk.LEFT, padx=5)
        
        self.telemetry_tree = ttk.Treeview(telemetry_frame, 
                                          columns=("field","value","min","max","avg","samples"), 
                                          show='headings', height=15)
        for col, label, w in [
            ("field","Telemetry Field",280),
            ("value","Latest Value",140),
            ("min","Min",100),
            ("max","Max",100),
            ("avg","Average",100),
            ("samples","Samples",80)
        ]:
            self.telemetry_tree.heading(col, text=label)
            self.telemetry_tree.column(col, width=w)
        
        telemetry_scroll = ttk.Scrollbar(telemetry_frame, orient=tk.VERTICAL, command=self.telemetry_tree.yview)
        self.telemetry_tree.configure(yscrollcommand=telemetry_scroll.set)
        self.telemetry_tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        telemetry_scroll.grid(row=1, column=1, sticky=(tk.N, tk.S))
        
        # Bind double-click to show packet details
        self.telemetry_tree.bind("<Double-1>", self.show_klv_packet_details)
        
        # Store telemetry data for packet viewer
        self.klv_telemetry_data = {}

        # Compliance status (compact)
        compliance_frame = ttk.LabelFrame(klv_frame, text="⚠ Compliance Status", padding="5")
        self.klv_compliance_frame = compliance_frame
        # Do not display the compliance status section in the GUI
        compliance_frame.grid_remove()
        compliance_frame.columnconfigure(0, weight=1)
        
        self.klv_issues_text = tk.Text(compliance_frame, wrap=tk.WORD, height=4, font=('TkDefaultFont', 9))
        issues_scroll = ttk.Scrollbar(compliance_frame, orient=tk.VERTICAL, command=self.klv_issues_text.yview)
        self.klv_issues_text.configure(yscrollcommand=issues_scroll.set)
        
        self.klv_issues_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        issues_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
    
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Media File",
            filetypes=[
                ("Transport Stream", "*.ts"),
                ("MP4", "*.mp4"),
                ("QuickTime MOV", "*.mov"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.current_file = filename
            # Prefer content sniffing over extension so TS-in-MPG is handled correctly
            if is_ts_by_content(filename):
                self.container_type = "ts"
            else:
                ext = os.path.splitext(filename)[1].lower()
                if ext == ".ts":
                    self.container_type = "ts"
                elif ext == ".mp4":
                    self.container_type = "mp4"
                elif ext == ".mov":
                    self.container_type = "mov"
                else:
                    self.container_type = "unknown"

            self.status_label.config(text=f"File loaded: {os.path.basename(filename)}", foreground="blue")
            # Automatically start analysis
            self.start_analysis()
    
    def start_analysis(self):
        if not self.current_file or not os.path.isfile(self.current_file):
            messagebox.showerror("Error", "Please select a valid media file first")
            return

        # Re-sniff on start in case a non-TS extension actually contains TS (e.g., .mpg)
        if is_ts_by_content(self.current_file):
            self.container_type = "ts"
        
        # Show and start progress bar (insert between status and summary)
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        self.progress['value'] = 0
        self.progress_label.grid(row=2, column=0, sticky=tk.W, pady=(0, 5))
        self.progress_label.config(text="Analyzing: 0%")
        
        # Move summary down temporarily
        summary_frame = None
        for child in self.progress.master.winfo_children():
            if isinstance(child, ttk.LabelFrame) and "Analysis Summary" in str(child.cget('text')):
                summary_frame = child
                break
        if summary_frame:
            summary_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label.config(text="Analyzing...", foreground="orange")
        
        # Clear previous results
        self.clear_results()
        
        # Run analysis in separate thread
        target = self.run_analysis if getattr(self, 'container_type', 'ts') == 'ts' else self.run_mp4_mov_analysis
        self.analysis_thread = threading.Thread(target=target, daemon=True)
        self.analysis_thread.start()
    
    def run_analysis(self):
        import time  # For timing debug
        try:
            # Get file size for progress calculation
            file_size = os.path.getsize(self.current_file)
            
            # Throttle progress updates to avoid GUI lag
            last_update_time = [0]  # Use list to allow modification in lambda
            def throttled_progress_callback(pct):
                import time
                current_time = time.time()
                # Only update GUI every 100ms to avoid excessive events
                if current_time - last_update_time[0] >= 0.1:
                    last_update_time[0] = current_time
                    self.root.after(0, self.update_progress, int(pct * 0.7))
            
            # Create analyser with progress callback (file parsing is 0-70%)
            t0 = time.time()
            self.analyser = TSAnalyser(self.current_file, progress_callback=throttled_progress_callback)
            self.analyzer = self.analyser  # Keep reference for on-demand NAL parsing
            self._nal_sei_cache = {}  # Clear NAL cache for new file
            self.analyser.analyze()
            
            # Report generation (70-75%)
            self.root.after(0, self.update_progress, 73)
            t1 = time.time()
            report = self.analyser.report()
            
            # Store report immediately for frame details
            self.last_report = report
            # Pre-extract NALs for H.264 so per-frame grouping and SEI are available.
            # This can be heavy for long files so run in a background thread
            try:
                def _preextract_nals():
                    try:
                        h264_pid = None
                        for pid, codec_type in getattr(self.analyser, 'video_pids', {}).items():
                            if codec_type == 0x1B:  # H.264
                                h264_pid = pid
                                break
                        if h264_pid is not None:
                            if DEBUG: print(f"[Pre-extract] extracting unlimited NALs for PID {h264_pid}")
                            all_nals = None
                            try:
                                all_nals = self.analyser.extract_nal_sei_unlimited(h264_pid)
                            except Exception as _e:
                                if DEBUG: print(f"[Pre-extract] extract_nal_sei_unlimited failed: {_e}")
                            if all_nals:
                                # Cache into GUI structures used by _extract_nals_by_pts
                                self._all_nals_unlimited = all_nals
                                try:
                                    self._frame_nals_grouped = self._group_nals_by_frame_correct(self._all_nals_unlimited)
                                except Exception as _e:
                                    if DEBUG: print(f"[Pre-extract] grouping failed: {_e}")
                                if DEBUG:
                                    try:
                                        gn = len(self._all_nals_unlimited) if hasattr(self, '_all_nals_unlimited') and self._all_nals_unlimited else 0
                                        fg = len(self._frame_nals_grouped) if hasattr(self, '_frame_nals_grouped') and self._frame_nals_grouped else 0
                                        print(f"[Pre-extract] got {gn} NALs, grouped into {fg} frames")
                                    except Exception:
                                        pass
                    except Exception:
                        if DEBUG: import traceback; traceback.print_exc()

                threading.Thread(target=_preextract_nals, daemon=True).start()
            except Exception:
                pass
            # Apply file-specific UI theme (light blue tabs/buttons) when supported
            try:
                self.root.after(0, self.apply_file_theme, report)
            except Exception:
                pass
            
            # Check if MPTS (multiple programs) (75-80%)
            self.root.after(0, self.update_progress, 75)
            pat_info = report.get('pat', {})
            programs = pat_info.get('programs', {})
            # Filter out program 0 (network PID)
            service_programs = {p: pid for p, pid in programs.items() if p != 0}
            
            if DEBUG: print(f"[MPTS Detection] Total programs: {len(programs)}, Service programs: {len(service_programs)}")
            if DEBUG: print(f"[MPTS Detection] Programs: {programs}")
            
            # Display results immediately without graphs (80-100%)
            # Graph generation will happen in background thread
            self.root.after(0, self.update_progress, 80)
            if len(service_programs) > 1:
                # MPTS detected - ask user to select program
                if DEBUG: print(f"[MPTS] Multiple programs detected, showing selector")
                self.root.after(0, self.show_program_selector, report, service_programs, None)
            else:
                # SPTS or single program - proceed normally
                if DEBUG: print(f"[SPTS] Single program detected, proceeding normally")
                self.root.after(0, self.display_results, report, None)
            
            # Generate matplotlib figures in background thread (async, after UI is responsive)
            if MATPLOTLIB_AVAILABLE:
                def generate_graphs_async():
                    """Generate graphs in background and render them progressively"""
                    t2 = time.time()
                    try:
                        graph_figures = self.prepare_graphs(report)
                        # Render prepared graphs to GUI using lambda to pass arguments
                        self.root.after(0, lambda: self.render_prepared_graphs(graph_figures))
                    except Exception as e:
                        print(f"[ERROR] Graph generation failed: {e}")
                        import traceback
                        traceback.print_exc()
                
                graph_thread = threading.Thread(target=generate_graphs_async, daemon=True)
                graph_thread.start()
            
            
        except Exception as e:
            self.root.after(0, self.show_error, str(e))
    
    def update_progress(self, percentage):
        """Update progress bar with percentage and phase"""
        self.progress['value'] = percentage
        
        # Determine phase based on percentage
        if percentage < 70:
            phase = "Parsing file"
        elif percentage < 80:
            phase = "Generating report"
        elif percentage < 90:
            phase = "Building UI"
        else:
            phase = "Generating graphs" if percentage < 100 else "Complete"
        
        self.progress_label.config(text=f"{phase}: {percentage}%")
    
    def display_results(self, report, graph_figures=None, full_report=None):
        # Display rendering (80-90%)
        self.update_progress(85)
        
        # Store report and pre-generated figures for later use
        self.last_report = report
        self.prepared_graph_figures = graph_figures or []
        
        # Progress update (90%)
        self.update_progress(90)
        
        # Don't hide progress bar yet - graphs will be loading in background
        
        # Move summary back to row 1
        summary_frame = None
        for child in self.progress.master.winfo_children():
            if isinstance(child, ttk.LabelFrame) and "Analysis Summary" in str(child.cget('text')):
                summary_frame = child
                break
        if summary_frame:
            summary_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Check file type - handle MP4/MOV differently from TS
        file_type = report.get('file_type', 'TS')
        is_mp4_format = file_type in ['MP4/MOV', 'MP4', 'MOV']
        
        # Disable TS-specific tabs for MP4/MOV
        if is_mp4_format:
            try:
                self.disable_ts_tabs()
            except:
                pass
        else:
            # Ensure TS tabs are enabled when analyzing TS files after MP4/MOV
            try:
                self.enable_ts_tabs()
            except:
                pass
        
        self.status_label.config(text="Analysis complete - graphs loading...", foreground="green")
        
        # Update summary
        # Media file name
        if self.current_file:
            file_name = os.path.basename(self.current_file)
            self.media_file_var.set(file_name)
        else:
            self.media_file_var.set("-")
        
        # Update metrics based on format
        if is_mp4_format:
            # MP4/MOV metrics
            track_count = len(report.get('video_tracks', []))
            self.total_packets_var.set(f"{track_count} track(s)")
            self.pids_var.set("N/A (MP4/MOV)")
        else:
            # TS metrics
            self.total_packets_var.set(f"{report['total_packets']:,}")
            self.pids_var.set(str(report['pid_count']))
        
        if report['approx_duration_s']:
            self.duration_var.set(f"{report['approx_duration_s']:.2f} s")
        else:
            self.duration_var.set("N/A")
        
        if report['approx_bitrate_bps']:
            bitrate_mbps = report['approx_bitrate_bps'] / 1_000_000
            self.bitrate_var.set(f"{bitrate_mbps:.2f} Mbps")
        else:
            self.bitrate_var.set("N/A")
        
        # Only display TR101-290 and SCTE-35 for TS files
        if not is_mp4_format:
            # Classify and display TR101-290 errors
            p1_errors = TR101290ErrorClassifier.classify_p1_errors(report)
            p2_errors = TR101290ErrorClassifier.classify_p2_errors(report)
            p3_errors = TR101290ErrorClassifier.classify_p3_errors(report)
            
            # Display in TR101-290 tab
            self.display_tr101290_results(p1_errors, p2_errors, p3_errors)
            


            # Display SCTE-35 events
            self.display_scte35_events(report)
        
        # Display stream structure: always use full report if provided, else current report
        if full_report is not None:
            self.display_stream_structure(full_report)
        else:
            self.display_stream_structure(report)
        
        # Store pre-generated figures for lazy loading
        # Don't attach to GUI yet - wait for user to click Graphs tab
        if MATPLOTLIB_AVAILABLE and self.prepared_graph_figures:
            self.graphs_loaded = False  # Mark as not loaded yet
        
        # Display elementary streams
        self.display_elementary_streams(report)
        
        # Display buffer analysis
        self.display_buffer_analysis(report)
        
        # Display captions (CEA-608/CEA-708)
        self.display_captions(report)
        
        # Display KLV and STANAG 4609 compliance
        self.display_klv_stanag(report)
        
        # Auto-load first 10 video frames with audio
        # Skip auto-load if file has suspicious stream structure (many unknown streams)
        # which might indicate a corrupted/unusual file
        elementary_streams = report.get('elementary_streams', {})
        unknown_stream_count = sum(1 for info in elementary_streams.values() 
                                   if info.get('stream_type') is None or info.get('stream_type_name') == 'Unknown')
        total_streams = len(elementary_streams)
        
        # If more than 50% of streams are unknown, skip auto-thumbnail loading
        # User can still manually extract thumbnails by clicking the Video tab
        skip_auto_load = (total_streams > 100 and unknown_stream_count / max(total_streams, 1) > 0.5)
        
        if DEBUG and skip_auto_load:
            print(f"[Auto-load] Skipping auto-thumbnail loading - suspicious file structure: {unknown_stream_count}/{total_streams} unknown streams")
        
        if AV_AVAILABLE and self.current_file and not skip_auto_load:
            threading.Thread(target=self._auto_load_thumbnails, daemon=True).start()
        elif skip_auto_load:
            self.status_label.config(text="Analysis complete - Manual video extraction recommended", foreground="orange")
        
        # Load GOP and video analysis in background thread (after thumbnails start loading)
        # This allows thumbnails to render immediately without waiting for ffprobe calls
        threading.Thread(target=self._load_ffprobe_analysis, args=(report,), daemon=True).start()

    def _load_ffprobe_analysis(self, report):
        """Load GOP and video analysis in background thread without blocking UI.
        
        This method runs ffprobe analysis (which can be slow) in a background thread
        after display_results() has already updated the fast-to-compute summary values
        and started auto-loading thumbnails. This allows the GUI to be responsive while
        awaiting ffprobe results.
        """
        try:
            if DEBUG: print("[Background] Starting GOP and video analysis...")
            
            # Update GOP structure information using ffprobe
            self.update_gop_summary(report)
            
            # Update video information (resolution, frame rate, scan type) using ffprobe
            self.update_video_summary(report)
            
            if DEBUG: print("[Background] GOP and video analysis complete")
        except Exception as e:
            if DEBUG: print(f"[Background] Error during ffprobe analysis: {e}")
            # Silently ignore errors - the fields will remain as N/A

    def enable_ts_tabs(self):
        """Enable TS-specific tabs when analyzing a TS file."""
        try:
            for tab_name in ["TR101-290", "Stream Structure", "SCTE-35 Events", "Buffer Analysis", "KLV / STANAG 4609", "Graphs"]:
                for i in range(self.notebook.index('end')):
                    if self.notebook.tab(i, 'text') == tab_name:
                        self.notebook.tab(i, state='normal')
        except:
            pass

    def run_mp4_mov_analysis(self):
        """Run full analysis for MP4/MOV: parse container, extract NALs, generate report."""
        try:
            # Progress for analysis (0-70%)
            self.root.after(0, self.update_progress, 10)
            
            # Create analyser and run full analysis
            self.analyser = TSAnalyser(self.current_file)
            self.analyzer = self.analyser  # Keep reference for on-demand NAL parsing
            self._nal_sei_cache = {}  # Clear NAL cache for new file
            
            self.root.after(0, self.update_progress, 40)
            self.analyser.analyze()
            
            # Generate report (70-80%)
            self.root.after(0, self.update_progress, 75)
            report = self.analyser.report()
            
            # Store report for frame details
            self.last_report = report
            # Apply file-specific UI theme (light blue tabs/buttons) when supported
            try:
                self.root.after(0, self.apply_file_theme, report)
            except Exception:
                pass
            
            # Display results (80-100%)
            self.root.after(0, self.update_progress, 85)
            self.root.after(0, self.display_results, report, None)
            
            # Final status
            self.root.after(0, self.update_progress, 100)
            self.root.after(0, lambda: self.status_label.config(text="Analysis complete", foreground="green"))
            
            # Auto-load thumbnails if available
            if AV_AVAILABLE:
                threading.Thread(target=self._auto_load_thumbnails, daemon=True).start()
        except Exception as e:
            self.root.after(0, self.show_error, str(e))

    def disable_ts_tabs(self):
        """Disable TS-specific tabs when analyzing MP4/MOV."""
        try:
            for tab_name in ["TR101-290", "Stream Structure", "SCTE-35 Events", "Buffer Analysis", "KLV / STANAG 4609", "Graphs"]:
                for i in range(self.notebook.index('end')):
                    if self.notebook.tab(i, 'text') == tab_name:
                        self.notebook.tab(i, state='disabled')
        except:
            pass

    def display_tr101290_results(self, p1_errors, p2_errors, p3_errors):
        """Display TR101-290 analysis results in dedicated tab"""
        # Clear existing entries
        self.tr_p1_tree.delete(*self.tr_p1_tree.get_children())
        self.tr_p2_tree.delete(*self.tr_p2_tree.get_children())
        self.tr_p3_tree.delete(*self.tr_p3_tree.get_children())
        
        # Count failures
        p1_fails = sum(1 for e in p1_errors if e['status'] == 'FAIL')
        p2_fails = sum(1 for e in p2_errors if e['status'] == 'FAIL')
        p3_fails = sum(1 for e in p3_errors if e['status'] == 'FAIL')
        
        # Update summary
        if p1_fails == 0 and p2_fails == 0 and p3_fails == 0:
            self.tr_summary_label.config(text="✓ COMPLIANT - All TR101-290 checks passed", foreground='green')
        elif p1_fails > 0:
            self.tr_summary_label.config(text=f"✗ NON-COMPLIANT - {p1_fails} Priority 1 errors (Critical)", foreground='red')
        elif p2_fails > 0:
            self.tr_summary_label.config(text=f"⚠ WARNING - {p2_fails} Priority 2 errors (Quality issues)", foreground='orange')
        else:
            self.tr_summary_label.config(text=f"⚠ INFO - {p3_fails} Priority 3 warnings (Application-specific)", foreground='blue')
        
        # Populate P1 tree
        for error in p1_errors:
            if error['status'] == 'FAIL':
                tag = 'fail'
            else:
                tag = 'pass'
            self.tr_p1_tree.insert('', tk.END, values=(
                error['code'],
                error['name'],
                error['count'],
                error['status'],
                error['description']
            ), tags=(tag,))
        
        self.tr_p1_tree.tag_configure('fail', background='#ffcccc', foreground='red')
        self.tr_p1_tree.tag_configure('pass', background='#ccffcc', foreground='green')
        
        # Populate P2 tree
        for error in p2_errors:
            if error['status'] == 'FAIL':
                tag = 'fail'
            else:
                tag = 'pass'
            self.tr_p2_tree.insert('', tk.END, values=(
                error['code'],
                error['name'],
                error['count'],
                error['status'],
                error['description']
            ), tags=(tag,))
        
        self.tr_p2_tree.tag_configure('fail', background='#fff4cc', foreground='orange')
        self.tr_p2_tree.tag_configure('pass', background='#ccffcc', foreground='green')
        
        # Populate P3 tree
        for error in p3_errors:
            if error['status'] == 'FAIL':
                tag = 'fail'
            else:
                tag = 'pass'
            self.tr_p3_tree.insert('', tk.END, values=(
                error['code'],
                error['name'],
                error['count'],
                error['status'],
                error['description']
            ), tags=(tag,))
        
        self.tr_p3_tree.tag_configure('fail', background='#e6f2ff', foreground='blue')
        self.tr_p3_tree.tag_configure('pass', background='#ccffcc', foreground='green')
    
    def display_stream_structure(self, report):
        """Display hierarchical stream structure in tree view with comprehensive PAT/PMT details"""
        # Clear existing items
        for item in self.structure_tree.get_children():
            self.structure_tree.delete(item)
        
        # Root node - Transport Stream
        pat_info = report.get('pat', {})
        ts_id = pat_info.get('transport_stream_id', 'N/A')
        ts_id_hex = f"0x{ts_id:04X}" if isinstance(ts_id, int) else ts_id
        root = self.structure_tree.insert('', 'end', text=f'Transport Stream', 
                                          values=(f'TS ID: {ts_id_hex}', f"{report['total_packets']:,} packets", ''), 
                                          open=True, tags=('header',))
        
        # Configure tag colors
        self.structure_tree.tag_configure('header', font=('TkDefaultFont', 9, 'bold'))
        self.structure_tree.tag_configure('pat_node', foreground='#1976D2')
        self.structure_tree.tag_configure('pmt_node', foreground='#388E3C')
        self.structure_tree.tag_configure('descriptor', foreground='#F57C00')
        self.structure_tree.tag_configure('stream', foreground='#7B1FA2')
        
        # PAT node with comprehensive details
        if pat_info:
            pat_version = pat_info.get('version', 'N/A')
            pat_current_next = pat_info.get('current_next', 'N/A')
            pat_status = "Current" if pat_current_next == 1 else "Next" if pat_current_next == 0 else "N/A"
            
            pat_count = report.get('pid_info', {}).get(0, {}).get('count', 'N/A')
            pat_label = f'PAT (Program Association Table) - Version {pat_version} [{pat_status}]'
            
            pat_node = self.structure_tree.insert(root, 'end', text=pat_label, 
                                                   values=(f"PID 0x0000", 
                                                          f"{pat_count:,} packets" if isinstance(pat_count, int) else pat_count,
                                                          f"TS ID: {ts_id_hex}"), 
                                                   open=True, tags=('pat_node',))
            
            # PAT Details sub-node
            details_node = self.structure_tree.insert(pat_node, 'end', text='Table Details',
                                                      values=('', '', ''))
            self.structure_tree.insert(details_node, 'end', text=f'Table ID: 0x00 (PAT)',
                                      values=('', '', ''))
            self.structure_tree.insert(details_node, 'end', text=f'Version Number: {pat_version}',
                                      values=('', '', 'Increments when PAT changes'))
            self.structure_tree.insert(details_node, 'end', text=f'Current/Next Indicator: {pat_current_next}',
                                      values=('', '', pat_status))
            self.structure_tree.insert(details_node, 'end', text=f'Transport Stream ID: {ts_id_hex}',
                                      values=('', '', 'Uniquely identifies this TS'))
            
            # PAT Warnings
            pat_warnings = pat_info.get('warnings', [])
            if pat_warnings:
                warn_node = self.structure_tree.insert(pat_node, 'end', 
                                                       text=f'⚠ Warnings ({len(pat_warnings)})',
                                                       values=('', '', ''))
                for warning in pat_warnings:
                    self.structure_tree.insert(warn_node, 'end', text=f'• {warning}',
                                              values=('', '', ''))
            
            # Programs
            programs = pat_info.get('programs', {})
            prog_count_text = f"Programs ({len(programs)})" if programs else "Programs (0)"
            programs_node = self.structure_tree.insert(pat_node, 'end', text=prog_count_text,
                                                       values=('', '', ''), open=True)
            
            for prog_num, pmt_pid in programs.items():
                if prog_num == 0:
                    # Network PID
                    prog_node = self.structure_tree.insert(programs_node, 'end', 
                                                           text=f'Network Information Table (NIT)',
                                                           values=(f"PID 0x{pmt_pid:04X}", '', 'Network-specific data'))
                else:
                    # Regular program
                    prog_label = f'Program #{prog_num}'
                    prog_node = self.structure_tree.insert(programs_node, 'end', 
                                                           text=prog_label,
                                                           values=(f"PMT PID: 0x{pmt_pid:04X}", '', ''), 
                                                           open=True)
                    
                    # PMT details
                    pmt_info = report.get('pmts', {}).get(pmt_pid, {})
                    if pmt_info:
                        pmt_version = pmt_info.get('version', 'N/A')
                        pmt_current_next = pmt_info.get('current_next', 'N/A')
                        pmt_status = "Current" if pmt_current_next == 1 else "Next" if pmt_current_next == 0 else "N/A"
                        pmt_count = report.get('pid_info', {}).get(pmt_pid, {}).get('count', 'N/A')
                        
                        pmt_label = f'PMT (Program Map Table) - Version {pmt_version} [{pmt_status}]'
                        pmt_node = self.structure_tree.insert(prog_node, 'end', 
                                                              text=pmt_label,
                                                              values=(f"PID 0x{pmt_pid:04X}",
                                                                     f"{pmt_count:,} packets" if isinstance(pmt_count, int) else pmt_count,
                                                                     f"Program {prog_num}"),
                                                              open=True, tags=('pmt_node',))
                        
                        # PMT Table Details
                        pmt_details_node = self.structure_tree.insert(pmt_node, 'end', text='Table Details',
                                                                      values=('', '', ''))
                        self.structure_tree.insert(pmt_details_node, 'end', text=f'Table ID: 0x02 (PMT)',
                                                  values=('', '', ''))
                        self.structure_tree.insert(pmt_details_node, 'end', text=f'Version Number: {pmt_version}',
                                                  values=('', '', 'Increments when PMT changes'))
                        self.structure_tree.insert(pmt_details_node, 'end', text=f'Current/Next Indicator: {pmt_current_next}',
                                                  values=('', '', pmt_status))
                        self.structure_tree.insert(pmt_details_node, 'end', text=f'Program Number: {prog_num}',
                                                  values=('', '', 'Matches PAT entry'))
                        
                        # PCR PID
                        pcr_pid = pmt_info.get('pcr_pid')
                        if pcr_pid is not None:
                            pcr_count = len(report.get('pcr_records', {}).get(pcr_pid, []))
                            pcr_invalid = " (⚠ Invalid - NULL PID)" if pcr_pid == 0x1FFF else ""
                            self.structure_tree.insert(pmt_details_node, 'end',
                                                      text=f'PCR PID: 0x{pcr_pid:04X}{pcr_invalid}',
                                                      values=('', '', f"{pcr_count} PCR values" if pcr_count > 0 else 'Clock reference'))
                        
                        # PMT Warnings
                        pmt_warnings = pmt_info.get('warnings', [])
                        if pmt_warnings:
                            warn_node = self.structure_tree.insert(pmt_node, 'end', 
                                                                   text=f'⚠ Warnings ({len(pmt_warnings)})',
                                                                   values=('', '', ''))
                            for warning in pmt_warnings:
                                self.structure_tree.insert(warn_node, 'end', text=f'• {warning}',
                                                          values=('', '', ''))
                        
                        # Program Descriptors (Program-level)
                        prog_descriptors = pmt_info.get('program_descriptors', [])
                        if prog_descriptors:
                            desc_label = f'Program Descriptors ({len(prog_descriptors)})'
                            desc_node = self.structure_tree.insert(pmt_node, 'end',
                                                                  text=desc_label,
                                                                  values=('', '', 'Apply to entire program'), 
                                                                  tags=('descriptor',))
                            for desc in prog_descriptors:
                                desc_tag = desc['tag']
                                desc_name = desc['tag_name']
                                desc_len = desc['length']
                                desc_data = desc.get('data', '')
                                
                                desc_text = f"[0x{desc_tag:02X}] {desc_name}"
                                desc_item = self.structure_tree.insert(desc_node, 'end',
                                                          text=desc_text,
                                                          values=(f"{desc_len} bytes", 
                                                                 f"Data: {desc_data[:32]}{'...' if len(desc_data) > 32 else ''}", 
                                                                 ''))
                        
                        # Elementary Streams
                        streams = pmt_info.get('streams', [])
                        if streams:
                            streams_label = f'Elementary Streams ({len(streams)})'
                            streams_node = self.structure_tree.insert(pmt_node, 'end',
                                                                     text=streams_label,
                                                                     values=('', '', 'Audio, Video, Data'), 
                                                                     open=True)
                            
                            for idx, stream in enumerate(streams, 1):
                                stream_pid = stream['pid']
                                stream_type = stream['type']
                                stream_type_name = stream['type_name']
                                stream_count = report.get('pid_info', {}).get(stream_pid, {}).get('count', 'N/A')
                                stream_descriptors = stream.get('descriptors', [])
                                
                                # Get enhanced stream description
                                enhanced_description = get_enhanced_stream_description(
                                    stream_type, stream_descriptors
                                )
                                
                                stream_label = f"ES #{idx}: {enhanced_description}"
                                stream_node = self.structure_tree.insert(streams_node, 'end',
                                                                        text=stream_label,
                                                                        values=(f"PID 0x{stream_pid:04X}",
                                                                               f"{stream_count:,} packets" if isinstance(stream_count, int) else stream_count,
                                                                               f"Type: 0x{stream_type:02X}"),
                                                                        open=True, tags=('stream',))
                                
                                # Stream Type Details
                                type_node = self.structure_tree.insert(stream_node, 'end', text='Stream Type Info',
                                                                      values=('', '', ''))
                                self.structure_tree.insert(type_node, 'end', text=f'Stream Type: 0x{stream_type:02X}',
                                                          values=('', '', stream_type_name))
                                self.structure_tree.insert(type_node, 'end', text=f'Elementary PID: 0x{stream_pid:04X}',
                                                          values=('', '', f"{stream_pid} (decimal)"))
                                
                                # ES Info Length
                                es_info_len = stream.get('info_len', 0)
                                if es_info_len > 0:
                                    self.structure_tree.insert(type_node, 'end', text=f'ES Info Length: {es_info_len} bytes',
                                                              values=('', '', f"{len(stream.get('descriptors', []))} descriptor(s)"))
                                
                                # Stream Descriptors (ES-specific) with detailed parsing
                                stream_descriptors = stream.get('descriptors', [])
                                if stream_descriptors:
                                    es_desc_label = f'ES Descriptors ({len(stream_descriptors)})'
                                    es_desc_node = self.structure_tree.insert(stream_node, 'end',
                                                                          text=es_desc_label,
                                                                          values=('', '', 'Stream-specific metadata'),
                                                                          tags=('descriptor',))
                                    for desc in stream_descriptors:
                                        desc_tag = desc['tag']
                                        desc_name = desc['tag_name']
                                        desc_len = desc['length']
                                        desc_data = desc.get('data', '')
                                        
                                        desc_text = f"[0x{desc_tag:02X}] {desc_name}"
                                        desc_item = self.structure_tree.insert(es_desc_node, 'end',
                                                                  text=desc_text,
                                                                  values=(f"{desc_len} bytes",
                                                                         f"Data: {desc_data[:32]}{'...' if len(desc_data) > 32 else ''}",
                                                                         ''))
                                        
                                        # Parse descriptor details
                                        parsed = self._parse_descriptor_details(desc_tag, desc_data)
                                        if parsed:
                                            for key, val in parsed.items():
                                                self.structure_tree.insert(desc_item, 'end',
                                                                          text=f"  {key}: {val}",
                                                                          values=('', '', ''))
                                
                                # PTS/DTS Timing info if available
                                pts_records = report.get('pts_records', {}).get(stream_pid, [])
                                dts_records = report.get('dts_records', {}).get(stream_pid, [])
                                if pts_records or dts_records:
                                    timing_node = self.structure_tree.insert(stream_node, 'end',
                                                                            text='Timing Information',
                                                                            values=('', '', ''))
                                    if pts_records:
                                        # PTS records are tuples (packet_num, timestamp)
                                        pts_first = pts_records[0][1] if isinstance(pts_records[0], tuple) else pts_records[0]
                                        pts_last = pts_records[-1][1] if isinstance(pts_records[-1], tuple) else pts_records[-1]
                                        self.structure_tree.insert(timing_node, 'end',
                                                                  text=f'PTS: {len(pts_records)} timestamps',
                                                                  values=('', '', 
                                                                         f"Range: {pts_first:.3f}s - {pts_last:.3f}s"))
                                    if dts_records:
                                        # DTS records are tuples (packet_num, timestamp)
                                        dts_first = dts_records[0][1] if isinstance(dts_records[0], tuple) else dts_records[0]
                                        dts_last = dts_records[-1][1] if isinstance(dts_records[-1], tuple) else dts_records[-1]
                                        self.structure_tree.insert(timing_node, 'end',
                                                                  text=f'DTS: {len(dts_records)} timestamps',
                                                                  values=('', '', 
                                                                         f"Range: {dts_first:.3f}s - {dts_last:.3f}s"))
        
        # KLV Metadata PIDs (from KLV detection)
        klv_metadata = report.get('klv_metadata', {})
        klv_pids = set()
        for klv_type, klv_streams in klv_metadata.items():
            if klv_type in ['asynchronous_klv', 'synchronous_klv']:
                for stream_info in klv_streams:
                    if 'pid' in stream_info:
                        klv_pids.add(stream_info['pid'])
                    if 'video_pid' in stream_info:  # synchronous in video
                        klv_pids.add(stream_info['video_pid'])
        
        if klv_pids:
            klv_label = f'KLV Metadata PIDs ({len(klv_pids)})'
            klv_node = self.structure_tree.insert(root, 'end',
                                                  text=klv_label,
                                                  values=('', '', 'MISB Motion Imagery Metadata'),
                                                  open=True)
            self.structure_tree.tag_configure('klv', foreground='#D32F2F', font=('TkDefaultFont', 9, 'bold'))
            
            for klv_pid in sorted(klv_pids):
                klv_count = report.get('pid_info', {}).get(klv_pid, {}).get('count', 'N/A')
                
                # Find KLV details
                klv_sync_type = 'Unknown'
                klv_standards = []
                for async_info in klv_metadata.get('asynchronous_klv', []):
                    if async_info.get('pid') == klv_pid:
                        klv_sync_type = 'Asynchronous (separate PID)'
                        klv_standards = async_info.get('standards', [])
                        break
                for sync_info in klv_metadata.get('synchronous_klv', []):
                    if sync_info.get('video_pid') == klv_pid:
                        klv_sync_type = 'Synchronous (embedded in video)'
                        klv_standards = sync_info.get('standards', [])
                        break
                
                standards_str = ', '.join(klv_standards) if klv_standards else 'Generic KLV'
                klv_pid_node = self.structure_tree.insert(klv_node, 'end',
                                                          text=f"PID 0x{klv_pid:04X} - KLV",
                                                          values=(klv_sync_type,
                                                                 f"{klv_count:,} packets" if isinstance(klv_count, int) else klv_count,
                                                                 standards_str),
                                                          tags=('klv',))
                
                # Add MISB standards detail
                if klv_standards:
                    std_node = self.structure_tree.insert(klv_pid_node, 'end',
                                                         text='MISB Standards',
                                                         values=('', '', ''))
                    for std in klv_standards:
                        self.structure_tree.insert(std_node, 'end',
                                                  text=f'• {std}',
                                                  values=('', '', ''))
        
        # All PIDs node (for unassigned/other PIDs)
        pid_info = report.get('pid_info', {})
        assigned_pids = set([0])  # PAT
        assigned_pids.update(klv_pids)  # KLV PIDs
        if pat_info:
            assigned_pids.update(pat_info.get('programs', {}).values())  # PMT PIDs
            for pmt in report.get('pmts', {}).values():
                if pmt.get('pcr_pid') is not None:
                    assigned_pids.add(pmt['pcr_pid'])
                for stream in pmt.get('streams', []):
                    assigned_pids.add(stream['pid'])
        
        unassigned_pids = {pid: info for pid, info in pid_info.items() 
                          if pid not in assigned_pids and pid != 0x1FFF}
        
        if unassigned_pids:
            other_label = f'Other/Unknown PIDs ({len(unassigned_pids)})'
            other_node = self.structure_tree.insert(root, 'end',
                                                   text=other_label,
                                                   values=('', '', 'Not in PAT/PMT'))
            for pid, info in sorted(unassigned_pids.items()):
                pid_type = info.get('type', 'Unknown')
                self.structure_tree.insert(other_node, 'end',
                                          text=f"PID 0x{pid:04X}",
                                          values=(pid_type, f"{info['count']:,} packets", ''))
        
        # NULL packets
        null_count = report.get('null_packets', 0)
        if null_count > 0:
            null_percent = report.get('null_percent', 0)
            self.structure_tree.insert(root, 'end',
                                      text='NULL Packets (Stuffing)',
                                      values=('PID 0x1FFF', 
                                             f"{null_count:,} packets", 
                                             f"{null_percent:.2f}% of stream"))
    
    def _parse_descriptor_details(self, tag, data_hex):
        """Parse common descriptor types and return human-readable details"""
        if not data_hex or len(data_hex) < 2:
            return {}
        
        try:
            data_bytes = bytes.fromhex(data_hex)
        except:
            return {}
        
        parsed = {}
        
        # ISO 639 Language Descriptor (0x0A)
        if tag == 0x0A and len(data_bytes) >= 4:
            lang_code = data_bytes[0:3].decode('ascii', errors='ignore')
            audio_type = data_bytes[3]
            audio_type_map = {
                0: 'Undefined',
                1: 'Clean effects',
                2: 'Hearing impaired',
                3: 'Visual impaired commentary'
            }
            parsed['Language'] = lang_code
            parsed['Audio Type'] = audio_type_map.get(audio_type, f'Reserved ({audio_type})')
        
        # Registration Descriptor (0x05)
        elif tag == 0x05 and len(data_bytes) >= 4:
            format_id = data_bytes[0:4].decode('ascii', errors='ignore')
            parsed['Format Identifier'] = format_id
            if len(data_bytes) > 4:
                parsed['Additional Info'] = data_bytes[4:].hex()
        
        # AC-3 Audio Descriptor (0x81 - ATSC)
        elif tag == 0x81 and len(data_bytes) >= 3:
            sample_rate_code = (data_bytes[0] >> 5) & 0x07
            bsid = data_bytes[0] & 0x1F
            bit_rate_code = (data_bytes[1] >> 2) & 0x3F
            surround_mode = data_bytes[1] & 0x03
            
            sample_rates = {0: '48 kHz', 1: '44.1 kHz', 2: '32 kHz'}
            surround_modes = {0: 'Not indicated', 1: 'Not Dolby surround', 2: 'Dolby surround', 3: 'Reserved'}
            
            parsed['Bit Stream ID'] = str(bsid)
            parsed['Sample Rate'] = sample_rates.get(sample_rate_code, f'Reserved ({sample_rate_code})')
            parsed['Bit Rate Code'] = str(bit_rate_code)
            parsed['Surround Mode'] = surround_modes.get(surround_mode, f'Unknown ({surround_mode})')
        
        # Enhanced AC-3 Descriptor (0xCC - ATSC)
        elif tag == 0xCC and len(data_bytes) >= 1:
            component_type = data_bytes[0]
            parsed['Component Type'] = f'0x{component_type:02X}'
            if len(data_bytes) >= 2:
                bsid = data_bytes[1] & 0x1F
                parsed['Bit Stream ID'] = str(bsid)
        
        # Teletext Descriptor (0x56)
        elif tag == 0x56 and len(data_bytes) >= 5:
            lang_code = data_bytes[0:3].decode('ascii', errors='ignore')
            teletext_type = (data_bytes[3] >> 3) & 0x1F
            mag_num = data_bytes[3] & 0x07
            page_num = data_bytes[4]
            parsed['Language'] = lang_code
            parsed['Type'] = f'{teletext_type} ({"Initial" if teletext_type == 1 else "Subtitle" if teletext_type == 2 else "Other"})'
            parsed['Magazine'] = str(mag_num)
            parsed['Page'] = f'{page_num:02X}'
        
        # Subtitling Descriptor (0x59)
        elif tag == 0x59 and len(data_bytes) >= 8:
            lang_code = data_bytes[0:3].decode('ascii', errors='ignore')
            subtitling_type = data_bytes[3]
            composition_page = int.from_bytes(data_bytes[4:6], 'big')
            ancillary_page = int.from_bytes(data_bytes[6:8], 'big')
            parsed['Language'] = lang_code
            parsed['Subtitling Type'] = f'0x{subtitling_type:02X}'
            parsed['Composition Page'] = str(composition_page)
            parsed['Ancillary Page'] = str(ancillary_page)
        
        # Stream Identifier Descriptor (0x52)
        elif tag == 0x52 and len(data_bytes) >= 1:
            component_tag = data_bytes[0]
            parsed['Component Tag'] = f'0x{component_tag:02X}'
        
        # Maximum Bitrate Descriptor (0x0E)
        elif tag == 0x0E and len(data_bytes) >= 3:
            max_bitrate = int.from_bytes(data_bytes[0:3], 'big') & 0x3FFFFF
            parsed['Max Bitrate'] = f'{max_bitrate * 50} bytes/s ({max_bitrate * 50 * 8 / 1000000:.2f} Mbps)'
        
        return parsed
    
    def show_packet_header_menu(self, event):
        """Show context menu for TS/PES header inspection"""
        # Get clicked item
        item_id = self.structure_tree.identify_row(event.y)
        if not item_id:
            return
        
        # Get item text and values
        item = self.structure_tree.item(item_id)
        item_text = item['text']
        item_values = item['values']
        
        # Extract PID from item (look for "PID 0x" pattern)
        pid = None
        if 'PID 0x' in item_text:
            try:
                pid_str = item_text.split('PID 0x')[1].split()[0][:4]
                pid = int(pid_str, 16)
            except:
                pass
        elif len(item_values) > 0 and isinstance(item_values[0], str) and 'PID 0x' in item_values[0]:
            try:
                pid_str = item_values[0].split('0x')[1][:4]
                pid = int(pid_str, 16)
            except:
                pass
        
        if pid is None:
            return
        
        # Create context menu
        menu = tk.Menu(self.root, tearoff=0)
        menu.add_command(label=f"Show TS Header for PID 0x{pid:04X}", 
                        command=lambda: self.show_ts_header_analysis(pid))
        menu.add_command(label=f"Show PES Header for PID 0x{pid:04X}", 
                        command=lambda: self.show_pes_header_analysis(pid))
        menu.add_separator()
        menu.add_command(label="Cancel")
        
        menu.post(event.x_root, event.y_root)
    
    def show_ts_header_analysis(self, pid):
        """Show TS packet header analysis for a PID"""
        if not self.current_file or not self.last_report:
            messagebox.showinfo("No Data", "Please analyze a file first.")
            return
        
        # Read packets for this PID (limit to 1000 for performance)
        try:
            packets_analyzed = []
            max_packets = 1000
            with open(self.current_file, 'rb') as f:
                packet_num = 0
                while len(packets_analyzed) < max_packets:
                    packet = f.read(188)
                    if len(packet) < 188:
                        break
                    
                    if packet[0] != 0x47:
                        continue
                    
                    packet_pid = ((packet[1] & 0x1F) << 8) | packet[2]
                    if packet_pid == pid:
                        # Parse TS header & adaptation field for richer detail
                        tei = (packet[1] & 0x80) >> 7
                        pusi = (packet[1] & 0x40) >> 6
                        priority = (packet[1] & 0x20) >> 5
                        scrambling = (packet[3] & 0xC0) >> 6
                        adaptation_field = (packet[3] & 0x30) >> 4
                        continuity_counter = packet[3] & 0x0F

                        af_len = 0
                        pcr_val = None
                        opcr_val = None
                        rai = None
                        disc = None
                        af_flags = ""
                        if adaptation_field in (2, 3) and len(packet) >= 5:
                            af_len = packet[4]
                            if af_len > 0 and 5 + af_len <= len(packet):
                                flags = packet[5]
                                disc = bool(flags & 0x80)
                                rai = bool(flags & 0x40)
                                pcr_flag = bool(flags & 0x10)
                                opcr_flag = bool(flags & 0x08)
                                splicing_flag = bool(flags & 0x04)
                                transport_private_flag = bool(flags & 0x02)
                                extension_flag = bool(flags & 0x01)
                                
                                af_parts = []
                                if disc:
                                    af_parts.append("Disc")
                                if rai:
                                    af_parts.append("RAI")
                                if pcr_flag:
                                    af_parts.append("PCR")
                                if opcr_flag:
                                    af_parts.append("OPCR")
                                if splicing_flag:
                                    af_parts.append("Splice")
                                if transport_private_flag:
                                    af_parts.append("Private")
                                if extension_flag:
                                    af_parts.append("Ext")
                                af_flags = ", ".join(af_parts) if af_parts else "None"
                                
                                # PCR parsing if present
                                if pcr_flag and af_len >= 7:
                                    if 6 + 6 <= len(packet):
                                        b = packet[6:12]
                                        pcr_base = ((b[0] << 25) | (b[1] << 17) | (b[2] << 9) | (b[3] << 1) | (b[4] >> 7))
                                        pcr_ext = ((b[4] & 0x01) << 8) | b[5]
                                        pcr_val = pcr_base / 90000.0 + pcr_ext / 27000000.0
                                # OPCR parsing if present (immediately after PCR if both)
                                opcr_offset = 6
                                if flags & 0x10:  # pcr_flag
                                    opcr_offset += 6
                                if opcr_flag and af_len >= opcr_offset + 6:
                                    b = packet[5 + opcr_offset:5 + opcr_offset + 6]
                                    opcr_base = ((b[0] << 25) | (b[1] << 17) | (b[2] << 9) | (b[3] << 1) | (b[4] >> 7))
                                    opcr_ext = ((b[4] & 0x01) << 8) | b[5]
                                    opcr_val = opcr_base / 90000.0 + opcr_ext / 27000000.0
                        
                        scrambling_map = {0: 'Not', 1: 'Reserved', 2: 'Even', 3: 'Odd'}
                        scrambling_text = scrambling_map.get(scrambling, 'Unknown')
                        tei_text = 'Error' if tei else 'OK'
                        
                        packets_analyzed.append({
                            'packet_num': packet_num,
                            'tei': tei_text,
                            'pusi': 'Y' if pusi else '-',
                            'priority': 'Y' if priority else '-',
                            'scrambling': scrambling_text,
                            'adaptation_field': adaptation_field,
                            'af_len': af_len,
                            'af_flags': af_flags,
                            'pcr': pcr_val,
                            'opcr': opcr_val,
                            'continuity_counter': continuity_counter,
                            'raw': packet[:24].hex()  # First 24 bytes
                        })
                    
                    packet_num += 1
            
            if not packets_analyzed:
                messagebox.showinfo("No Data", f"No packets found for PID 0x{pid:04X}")
                return
            
            # Create analysis window
            win = tk.Toplevel(self.root)
            win.title(f"TS Header Analysis - PID 0x{pid:04X}")
            win.geometry("900x600")
            
            # Header info
            header_frame = ttk.Frame(win, padding="10")
            header_frame.pack(fill=tk.X)
            ttk.Label(header_frame, text=f"TS Packet Headers for PID 0x{pid:04X} ({pid})", 
                     font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W)
            limit_msg = f" (limited to first {max_packets})" if len(packets_analyzed) >= max_packets else ""
            ttk.Label(header_frame, text=f"Showing {len(packets_analyzed)} packets{limit_msg}", 
                     foreground='#666').pack(anchor=tk.W)
            
            # Tree view
            tree_frame = ttk.Frame(win)
            tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            tree = ttk.Treeview(tree_frame, 
                               columns=("packet", "tei", "pusi", "priority", "scrambling", "adaptation", "af_len", "af_flags", "pcr", "opcr", "cc", "raw"),
                               show='headings', height=20)
            
            tree.heading("packet", text="Packet #")
            tree.heading("tei", text="TEI")
            tree.heading("pusi", text="PUSI")
            tree.heading("priority", text="Priority")
            tree.heading("scrambling", text="Scrambling")
            tree.heading("adaptation", text="Adapt")
            tree.heading("af_len", text="AF Len")
            tree.heading("af_flags", text="AF Flags")
            tree.heading("pcr", text="PCR (s)")
            tree.heading("opcr", text="OPCR (s)")
            tree.heading("cc", text="CC")
            tree.heading("raw", text="Raw Header (hex)")
            
            tree.column("packet", width=70)
            tree.column("tei", width=50)
            tree.column("pusi", width=50)
            tree.column("priority", width=60)
            tree.column("scrambling", width=80)
            tree.column("adaptation", width=70)
            tree.column("af_len", width=60)
            tree.column("af_flags", width=180)
            tree.column("pcr", width=110)
            tree.column("opcr", width=110)
            tree.column("cc", width=40)
            tree.column("raw", width=350)
            
            scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scroll.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Populate data
            adaptation_map = {0: 'Reserved', 1: 'Payload', 2: 'AF only', 3: 'Both'}
            for pkt in packets_analyzed:
                tree.insert('', 'end', values=(
                    pkt['packet_num'],
                    pkt['tei'],
                    pkt['pusi'],
                    pkt['priority'],
                    pkt['scrambling'],
                    adaptation_map.get(pkt['adaptation_field'], 'Unknown'),
                    pkt['af_len'],
                    pkt['af_flags'],
                    f"{pkt['pcr']:.6f}" if pkt['pcr'] is not None else '-',
                    f"{pkt['opcr']:.6f}" if pkt['opcr'] is not None else '-',
                    pkt['continuity_counter'],
                    pkt['raw']
                ))
            
            # Legend
            legend_frame = ttk.LabelFrame(win, text="Field Descriptions", padding="10")
            legend_frame.pack(fill=tk.X, padx=10, pady=5)
            legend_text = ("TEI = Transport Error Indicator | PUSI = Payload Unit Start Indicator | "
                          "Priority = Transport Priority | Scrambling: Not/Even/Odd/Reserved | AF Flags: Disc(Discontinuity), "
                          "RAI(Random Access), PCR/OPCR, Splice, Private, Ext | CC = Continuity Counter (0-15)")
            ttk.Label(legend_frame, text=legend_text, wraplength=900, foreground='#666').pack()
            
            # Close button
            btn_frame = ttk.Frame(win, padding="10")
            btn_frame.pack(fill=tk.X)
            ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze TS headers: {str(e)}")
    
    def show_pes_header_analysis(self, pid):
        """Show PES packet header analysis for a PID"""
        if not self.current_file or not self.last_report:
            messagebox.showinfo("No Data", "Please analyze a file first.")
            return
        
        # Read all PES packets for this PID (limit to 500 for performance)
        try:
            pes_packets = []
            pes_buffer = bytearray()
            max_pes = 500
            with open(self.current_file, 'rb') as f:
                packet_num = 0
                while len(pes_packets) < max_pes:
                    packet = f.read(188)
                    if len(packet) < 188:
                        break
                    
                    if packet[0] != 0x47:
                        continue
                    
                    packet_pid = ((packet[1] & 0x1F) << 8) | packet[2]
                    if packet_pid == pid:
                        pusi = (packet[1] & 0x40) >> 6
                        adaptation_field = (packet[3] & 0x30) >> 4
                        payload_start = 4
                        
                        if adaptation_field == 2:
                            continue  # No payload
                        elif adaptation_field == 3:
                            adaptation_len = packet[4]
                            payload_start = 5 + adaptation_len
                        
                        if pusi and len(pes_buffer) > 0:
                            # Start of new PES packet - parse previous one
                            if len(pes_buffer) >= 9 and pes_buffer[0:3] == b'\x00\x00\x01':
                                stream_id = pes_buffer[3]
                                pes_length = (pes_buffer[4] << 8) | pes_buffer[5]
                                flags1 = pes_buffer[6]
                                flags2 = pes_buffer[7]
                                pts_dts_flags = (flags2 & 0xC0) >> 6
                                escr_flag = (flags2 & 0x20) >> 5
                                es_rate_flag = (flags2 & 0x10) >> 4
                                dsm_trick_flag = (flags2 & 0x08) >> 3
                                add_copy_flag = (flags2 & 0x04) >> 2
                                pes_crc_flag = (flags2 & 0x02) >> 1
                                pes_ext_flag = flags2 & 0x01
                                header_len = pes_buffer[8]
                                
                                pts, dts = None, None
                                if pts_dts_flags >= 2 and len(pes_buffer) >= 14:
                                    pts = (((pes_buffer[9] & 0x0E) << 29) | (pes_buffer[10] << 22) |
                                           ((pes_buffer[11] & 0xFE) << 14) | (pes_buffer[12] << 7) |
                                           ((pes_buffer[13] & 0xFE) >> 1))
                                    pts = pts / 90000.0
                                if pts_dts_flags == 3 and len(pes_buffer) >= 19:
                                    dts = (((pes_buffer[14] & 0x0E) << 29) | (pes_buffer[15] << 22) |
                                           ((pes_buffer[16] & 0xFE) << 14) | (pes_buffer[17] << 7) |
                                           ((pes_buffer[18] & 0xFE) >> 1))
                                    dts = dts / 90000.0
                                
                                flags_parts = [f"scr={(flags1 & 0x30) >> 4}"]
                                if flags1 & 0x08:
                                    flags_parts.append("prio")
                                if flags1 & 0x04:
                                    flags_parts.append("align")
                                if flags1 & 0x02:
                                    flags_parts.append("copyright")
                                if flags1 & 0x01:
                                    flags_parts.append("orig_copy")
                                if pts_dts_flags == 2:
                                    flags_parts.append("PTS")
                                elif pts_dts_flags == 3:
                                    flags_parts.append("PTS/DTS")
                                if escr_flag:
                                    flags_parts.append("ESCR")
                                if es_rate_flag:
                                    flags_parts.append("ES rate")
                                if dsm_trick_flag:
                                    flags_parts.append("DSM trick")
                                if add_copy_flag:
                                    flags_parts.append("add copy")
                                if pes_crc_flag:
                                    flags_parts.append("CRC")
                                if pes_ext_flag:
                                    flags_parts.append("ext")
                                flags_text = ", ".join(flags_parts)
                                
                                pes_packets.append({
                                    'stream_id': f'0x{stream_id:02X}',
                                    'pes_length': pes_length,
                                    'header_len': header_len,
                                    'flags': flags_text,
                                    'pts': f'{pts:.3f}s' if pts else '-',
                                    'dts': f'{dts:.3f}s' if dts else '-',
                                    'raw': pes_buffer[:24].hex()
                                })
                            pes_buffer = bytearray()
                        
                        if payload_start < len(packet):
                            pes_buffer.extend(packet[payload_start:])
                    
                    packet_num += 1
            
            if not pes_packets:
                messagebox.showinfo("No Data", f"No PES packets found for PID 0x{pid:04X}")
                return
            
            # Create analysis window
            win = tk.Toplevel(self.root)
            win.title(f"PES Header Analysis - PID 0x{pid:04X}")
            win.geometry("900x500")
            
            # Header info
            header_frame = ttk.Frame(win, padding="10")
            header_frame.pack(fill=tk.X)
            ttk.Label(header_frame, text=f"PES Packet Headers for PID 0x{pid:04X} ({pid})", 
                     font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W)
            limit_msg = f" (limited to first {max_pes})" if len(pes_packets) >= max_pes else ""
            ttk.Label(header_frame, text=f"Showing {len(pes_packets)} PES packets{limit_msg}", 
                     foreground='#666').pack(anchor=tk.W)
            
            # Tree view
            tree_frame = ttk.Frame(win)
            tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            
            tree = ttk.Treeview(tree_frame, 
                               columns=("stream_id", "pes_length", "header_len", "flags", "pts", "dts", "raw"),
                               show='headings', height=15)
            
            tree.heading("stream_id", text="Stream ID")
            tree.heading("pes_length", text="PES Length")
            tree.heading("header_len", text="Header Len")
            tree.heading("flags", text="Flags")
            tree.heading("pts", text="PTS")
            tree.heading("dts", text="DTS")
            tree.heading("raw", text="Raw Header (hex)")
            
            tree.column("stream_id", width=90)
            tree.column("pes_length", width=90)
            tree.column("header_len", width=90)
            tree.column("flags", width=220)
            tree.column("pts", width=110)
            tree.column("dts", width=110)
            tree.column("raw", width=360)
            
            scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scroll.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scroll.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Populate data
            for pes in pes_packets:
                tree.insert('', 'end', values=(
                    pes['stream_id'],
                    pes['pes_length'],
                    pes['header_len'],
                    pes['flags'],
                    pes['pts'],
                    pes['dts'],
                    pes['raw']
                ))
            
            # Close button
            btn_frame = ttk.Frame(win, padding="10")
            btn_frame.pack(fill=tk.X)
            ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side=tk.RIGHT)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze PES headers: {str(e)}")
    
    def prepare_graphs(self, report):
        """Generate matplotlib Figure objects in background thread (no GUI operations).
        Returns list of (figure, title) tuples."""
        import time
        t_total_start = time.time()
        figures = []
        
        # Graph 1: PCR Jitter (inter-arrival time)
        t0 = time.time()
        pcr_records = report.get('pcr_records', {})
        if pcr_records:
            for pid, records in list(pcr_records.items())[:3]:  # Limit to 3 PIDs
                if len(records) < 2:
                    continue
                
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # Calculate inter-arrival times (jitter)
                times = [r[1] for r in records]
                intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
                time_points = times[1:]
                
                ax.plot(time_points, [iv * 1000 for iv in intervals], 'b-', linewidth=0.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('PCR Interval (ms)')
                ax.set_title(f'PCR Jitter - PID 0x{pid:04X}')
                ax.grid(True, alpha=0.3)
                
                # TR101-290 P2.3: PCR repetition interval threshold (40ms max)
                ax.axhline(y=40, color='red', linestyle=':', alpha=0.7, linewidth=2, label='TR101-290 Limit (40ms)')
                
                # Expected interval line (if close to constant bitrate)
                if intervals:
                    avg_interval = np.mean(intervals) * 1000
                    ax.axhline(y=avg_interval, color='g', linestyle='--', alpha=0.5, label=f'Mean: {avg_interval:.2f}ms')
                    ax.legend()
                
                figures.append(fig)
        
        
        # Graph 2: PCR Accuracy (deviation from expected)
        t1 = time.time()
        if pcr_records:
            for pid, records in list(pcr_records.items())[:3]:
                if len(records) < 3:
                    continue
                
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # Calculate PCR accuracy (difference between actual and expected PCR)
                packet_indices = [r[0] for r in records]
                pcr_values = [r[1] for r in records]
                
                # Expected PCR based on linear fit
                z = np.polyfit(packet_indices, pcr_values, 1)
                expected_pcr = np.poly1d(z)
                pcr_errors = [(pcr_values[i] - expected_pcr(packet_indices[i])) * 1000 for i in range(len(pcr_values))]
                
                # Detect if stream is CBR by checking PCR interval variation
                times = [r[1] for r in records]
                intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
                interval_std = np.std(intervals) * 1000 if len(intervals) > 1 else 0
                is_cbr = interval_std < 5  # If std dev < 5ms, likely CBR
                
                ax.plot(pcr_values, pcr_errors, 'g-', linewidth=0.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('PCR Error (ms)')
                
                # Update title and note based on CBR detection
                cbr_note = " [CBR Stream]" if is_cbr else " [VBR Stream - Note: PCR accuracy applies to CBR only]"
                ax.set_title(f'PCR Accuracy (Deviation from Linear) - PID 0x{pid:04X}{cbr_note}')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='b', linestyle='--', alpha=0.5, label='Expected (0ms)')
                
                # TR101-290 P2.5: PCR accuracy threshold (±500ns = ±0.0005ms) - only applicable for CBR
                if is_cbr:
                    ax.axhline(y=0.0005, color='red', linestyle=':', alpha=0.7, linewidth=2, label='TR101-290 Limit (±500ns, CBR)')
                    ax.axhline(y=-0.0005, color='red', linestyle=':', alpha=0.7, linewidth=2)
                else:
                    # Add informational text for VBR
                    ax.text(0.5, 0.95, 'Note: PCR accuracy (P2.5) applies only to CBR streams', 
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                           fontsize=9)
                ax.legend()
                
                figures.append(fig)
        
        
        # Graph 2b: PCR Frequency Offset (deviation from nominal 27 MHz clock)
        t1b = time.time()
        if pcr_records:
            for pid, records in list(pcr_records.items())[:3]:
                if len(records) < 3:
                    continue
                
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # Extract packet indices and PCR values
                packet_indices = [r[0] for r in records]
                pcr_values = [r[1] for r in records]
                
                # Linear fit to determine actual PCR increment per packet
                z = np.polyfit(packet_indices, pcr_values, 1)
                actual_increment = z[0]  # seconds per packet
                
                # Nominal increment per packet for 27 MHz clock:
                # Each packet is 188 bytes, so packets per second = bitrate_bps / (8*188)
                bitrate_bps = report.get('approx_bitrate_bps', None)
                freq_offset_ppm = None
                
                if bitrate_bps and bitrate_bps > 0:
                    packets_per_sec = bitrate_bps / (8 * 188)
                    nominal_increment = 1.0 / packets_per_sec
                    freq_offset_ppm = ((actual_increment - nominal_increment) / nominal_increment * 1e6)
                    if abs(freq_offset_ppm) > 100000:
                        freq_offset_ppm = None
                
                # Plot actual vs nominal frequency lines
                ax.plot(packet_indices, pcr_values, 'b-', linewidth=1, label='Actual PCR', alpha=0.7)
                
                # Expected PCR line (nominal)
                expected_pcr = np.poly1d(z)
                if freq_offset_ppm is not None:
                    ax.plot(packet_indices, expected_pcr(packet_indices), 'g--', linewidth=1.5, 
                           label=f'Linear Fit (Offset: {freq_offset_ppm:.2f} ppm)', alpha=0.8)
                else:
                    ax.plot(packet_indices, expected_pcr(packet_indices), 'g--', linewidth=1.5, 
                           label=f'Linear Fit', alpha=0.8)
                
                ax.set_xlabel('Packet Index')
                ax.set_ylabel('PCR Time (s)')
                if freq_offset_ppm is not None:
                    ax.set_title(f'PCR Frequency Offset - PID 0x{pid:04X} ({freq_offset_ppm:.2f} ppm)')
                else:
                    ax.set_title(f'PCR Frequency Offset - PID 0x{pid:04X} (N/A)')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Determine compliance status
                # ITU-R BT.656 / DVB: ±100 ppm typical limit for broadcast
                # Stricter applications: ±50 ppm
                freq_limit_ppm = 100  # Standard broadcast limit
                if freq_offset_ppm is None:
                    offset_str = 'N/A'
                    status = 'N/A'
                    status_color = 'yellow'
                else:
                    offset_str = f'{freq_offset_ppm:.3f}'
                    status = "✓ PASS" if abs(freq_offset_ppm) <= freq_limit_ppm else "✗ FAIL"
                    status_color = 'lightgreen' if abs(freq_offset_ppm) <= freq_limit_ppm else 'lightcoral'
                
                # Add comprehensive info box
                info_text = (f'Detected Offset: {offset_str} ppm\n'
                            f'Nominal Clock: 27.000 MHz\n'
                            f'Limit (DVB): ±{freq_limit_ppm} ppm\n'
                            f'Status: {status}')
                ax.text(0.02, 0.98, info_text,
                       transform=ax.transAxes, ha='left', va='top',
                       bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8, edgecolor='black', linewidth=1.5),
                       fontsize=9, fontweight='bold')
                
                figures.append(fig)
        
        
        # Graph 2c: PCR Drift Rate (cumulative deviation from linear trend)
        t1c = time.time()
        if pcr_records:
            for pid, records in list(pcr_records.items())[:3]:
                if len(records) < 3:
                    continue
                
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # Extract packet indices and PCR values
                packet_indices = [r[0] for r in records]
                pcr_values = [r[1] for r in records]
                
                # Linear fit to get expected PCR
                z = np.polyfit(packet_indices, pcr_values, 1)
                expected_pcr_func = np.poly1d(z)
                
                # Calculate drift (residuals) in milliseconds
                drift_ms = [(pcr_values[i] - expected_pcr_func(packet_indices[i])) * 1000 
                           for i in range(len(pcr_values))]
                
                # Plot drift over time
                ax.plot(pcr_values, drift_ms, 'r-', linewidth=1, alpha=0.7)
                ax.fill_between(pcr_values, drift_ms, alpha=0.2, color='red')
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('PCR Drift (ms)')
                ax.set_title(f'PCR Drift Rate - PID 0x{pid:04X}')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='b', linestyle='--', alpha=0.5, linewidth=1)
                
                # Calculate drift statistics
                max_drift = max(drift_ms) if drift_ms else 0
                min_drift = min(drift_ms) if drift_ms else 0
                drift_range = max_drift - min_drift
                
                # Reference limits for drift (typically ±100ms for broadcast, ±500ms for streaming)
                drift_limit_ms = 100  # DVB standard
                status = "✓ PASS" if abs(max_drift) <= drift_limit_ms and abs(min_drift) <= drift_limit_ms else "✗ FAIL"
                status_color = 'lightgreen' if abs(max_drift) <= drift_limit_ms and abs(min_drift) <= drift_limit_ms else 'lightcoral'
                
                # Add comprehensive info box with limits
                info_text = (f'Max Drift: {max_drift:.3f} ms\n'
                            f'Min Drift: {min_drift:.3f} ms\n'
                            f'Range: {drift_range:.3f} ms\n'
                            f'Limit: ±{drift_limit_ms} ms\n'
                            f'Status: {status}')
                ax.text(0.02, 0.98, info_text,
                       transform=ax.transAxes, ha='left', va='top',
                       bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.8, edgecolor='black', linewidth=1.5),
                       fontsize=9, fontweight='bold')
                
                figures.append(fig)
        
        
        # Graph 3: PTS-PCR Difference (for any PID with PTS records)
        t2 = time.time()
        pts_records = report.get('pts_records', {})
        if pts_records:
            es_info = report.get('elementary_streams', {})
            
            # Get any available PCR records (can be on a dedicated PID, video PID, or any other PID)
            # PCR PID can be separate from PTS PIDs - we use any PCR PID as timing reference
            pcr_ref = None
            first_pcr_pid = None
            if pcr_records:
                # Use the first available PCR PID as reference for all PTS comparisons
                # This works regardless of whether PCR is on its own PID or shared with video/audio
                first_pcr_pid = list(pcr_records.keys())[0]
                pcr_ref = pcr_records[first_pcr_pid]
            
            if DEBUG:
                print(f"[PTS-PCR] PTS PIDs: {set(pts_records.keys())}, PCR Ref PID: 0x{first_pcr_pid:04X}" if first_pcr_pid else f"[PTS-PCR] PTS PIDs: {set(pts_records.keys())}, PCR Ref PID: None")
            
            # Separate into video and audio PIDs to prioritize audio if needed
            video_pids = []
            audio_pids = []
            other_pids = []
            
            for pid in pts_records.keys():
                stream_type = es_info.get(pid, {}).get('type', '').lower() if pid in es_info else ''
                if stream_type == 'video':
                    video_pids.append(pid)
                elif stream_type == 'audio':
                    audio_pids.append(pid)
                else:
                    other_pids.append(pid)
            
            # Process all audio PIDs first, then video, then others
            pids_to_process = audio_pids + video_pids + other_pids
            
            if DEBUG:
                print(f"[PTS-PCR] Audio: {audio_pids}, Video: {video_pids}, Other: {other_pids}")
            
            for pid in pids_to_process:
                pts_list = pts_records[pid]
                if not pts_list or not pcr_ref:
                    if DEBUG:
                        print(f"[PTS-PCR] Skipping PID 0x{pid:04X}: pts_list empty={not pts_list}, no PCR ref={not pcr_ref}")
                    continue
                # Determine stream type for color/label
                stream_type = es_info.get(pid, {}).get('type', '').lower() if pid in es_info else ''
                if stream_type == 'audio':
                    color = 'c'
                    label = f'PTS-PCR Difference (Audio) - PID 0x{pid:04X}'
                elif stream_type == 'video':
                    color = 'm'
                    label = f'PTS-PCR Difference (Video) - PID 0x{pid:04X}'
                else:
                    color = 'k'
                    label = f'PTS-PCR Difference - PID 0x{pid:04X}'
                    
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # PERFORMANCE FIX: Use numpy arrays for O(log N) binary search instead of O(N) linear search
                # Use PCR reference (typically from video PID) for all PTS comparisons
                pcr_indices_arr = np.array([r[0] for r in pcr_ref])
                pcr_times_arr = np.array([r[1] for r in pcr_ref])
                
                pts_pcr_diffs = []
                pts_plot_times = []
                for pts_idx, pts_val in pts_list:
                    # Binary search for insertion point, then check neighbors
                    insert_pos = np.searchsorted(pcr_indices_arr, pts_idx)
                    candidates = [i for i in [insert_pos-1, insert_pos, insert_pos+1] 
                                 if 0 <= i < len(pcr_indices_arr)]
                    if candidates:
                        closest_idx = min(candidates, key=lambda i: abs(pcr_indices_arr[i] - pts_idx))
                        diff = (pts_val - pcr_times_arr[closest_idx]) * 1000  # ms
                        pts_pcr_diffs.append(diff)
                        pts_plot_times.append(pts_val)
                if pts_pcr_diffs:
                    ax.plot(pts_plot_times, pts_pcr_diffs, color+'-', linewidth=0.5, marker='o', markersize=2)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('PTS - PCR (ms)')
                    ax.set_title(label)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    figures.append(fig)
        
        pts_pcr_count = len([f for f in figures if 'PTS-PCR' in f.axes[0].get_title() if f.axes and f.axes[0].get_title()])
        
        # Graph 3b: DTS-PCR Difference (for any PID with DTS records)
        t2b = time.time()
        dts_records = report.get('dts_records', {})
        if dts_records:
            es_info = report.get('elementary_streams', {})
            
            # Get any available PCR records (can be on a dedicated PID, video PID, or any other PID)
            # PCR PID can be separate from DTS PIDs - we use any PCR PID as timing reference
            pcr_ref = None
            first_pcr_pid = None
            if pcr_records:
                # Use the first available PCR PID as reference for all DTS comparisons
                # This works regardless of whether PCR is on its own PID or shared with video/audio
                first_pcr_pid = list(pcr_records.keys())[0]
                pcr_ref = pcr_records[first_pcr_pid]
            
            if DEBUG:
                print(f"[DTS-PCR] DTS PIDs: {set(dts_records.keys())}, PCR Ref PID: 0x{first_pcr_pid:04X}" if first_pcr_pid else f"[DTS-PCR] DTS PIDs: {set(dts_records.keys())}, PCR Ref PID: None")
            
            # Separate into video and audio PIDs to prioritize audio if needed
            video_pids = []
            audio_pids = []
            other_pids = []
            
            for pid in dts_records.keys():
                stream_type = es_info.get(pid, {}).get('type', '').lower() if pid in es_info else ''
                if stream_type == 'video':
                    video_pids.append(pid)
                elif stream_type == 'audio':
                    audio_pids.append(pid)
                else:
                    other_pids.append(pid)
            
            # Process all audio PIDs first, then video, then others
            pids_to_process = audio_pids + video_pids + other_pids
            
            if DEBUG:
                print(f"[DTS-PCR] Audio: {audio_pids}, Video: {video_pids}, Other: {other_pids}")
            
            for pid in pids_to_process:
                dts_list = dts_records[pid]
                if not dts_list or not pcr_ref:
                    if DEBUG:
                        print(f"[DTS-PCR] Skipping PID 0x{pid:04X}: dts_list empty={not dts_list}, no pcr_ref={not pcr_ref}")
                    continue
                # Determine stream type for color/label
                stream_type = es_info.get(pid, {}).get('type', '').lower() if pid in es_info else ''
                if stream_type == 'audio':
                    color = 'b'
                    label = f'DTS-PCR Difference (Audio) - PID 0x{pid:04X}'
                elif stream_type == 'video':
                    color = 'orange'
                    label = f'DTS-PCR Difference (Video) - PID 0x{pid:04X}'
                else:
                    color = 'k'
                    label = f'DTS-PCR Difference - PID 0x{pid:04X}'
                    
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # PERFORMANCE FIX: Use numpy arrays for O(log N) binary search
                # Use PCR reference (typically from video PID) for all DTS comparisons
                pcr_indices_arr = np.array([r[0] for r in pcr_ref])
                pcr_times_arr = np.array([r[1] for r in pcr_ref])
                
                dts_pcr_diffs = []
                dts_plot_times = []
                for dts_idx, dts_val in dts_list:
                    # Binary search for insertion point, then check neighbors
                    insert_pos = np.searchsorted(pcr_indices_arr, dts_idx)
                    candidates = [i for i in [insert_pos-1, insert_pos, insert_pos+1] 
                                 if 0 <= i < len(pcr_indices_arr)]
                    if candidates:
                        closest_idx = min(candidates, key=lambda i: abs(pcr_indices_arr[i] - dts_idx))
                        diff = (dts_val - pcr_times_arr[closest_idx]) * 1000  # ms
                        dts_pcr_diffs.append(diff)
                        dts_plot_times.append(dts_val)
                if dts_pcr_diffs:
                    ax.plot(dts_plot_times, dts_pcr_diffs, color+'-', linewidth=0.5, marker='s', markersize=2)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('DTS - PCR (ms)')
                    ax.set_title(label)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    figures.append(fig)
                elif DEBUG:
                    print(f"[DTS-PCR] PID 0x{pid:04X}: No valid DTS-PCR differences (dts_list has {len(dts_list)} entries)")
        
        dts_pcr_count = len([f for f in figures if 'DTS-PCR' in f.axes[0].get_title() if f.axes and f.axes[0].get_title()])
        
        # Graph 4: PCR Discontinuity Detection (highlight jumps > threshold)
        t3 = time.time()
        pcr_jitter_issues = report.get('pcr_jitter_issues', {})
        if pcr_records and pcr_jitter_issues:
            for pid, issue_info in pcr_jitter_issues.items():
                if issue_info.get('large_jumps', 0) == 0:
                    continue  # Skip PIDs with no discontinuities
                
                records = pcr_records.get(pid, [])
                if len(records) < 2:
                    continue
                
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # Calculate PCR differences between consecutive samples
                times = [r[1] for r in records]
                intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
                time_points = times[1:]
                
                # Identify discontinuities (typically > 100ms as per TR101-290 P2.4)
                discontinuity_threshold_ms = 100  # 100ms threshold for visualization
                normal_intervals = []
                normal_times = []
                disc_intervals = []
                disc_times = []
                
                for i, interval in enumerate(intervals):
                    interval_ms = interval * 1000
                    if abs(interval_ms) > discontinuity_threshold_ms:
                        disc_intervals.append(interval_ms)
                        disc_times.append(time_points[i])
                    else:
                        normal_intervals.append(interval_ms)
                        normal_times.append(time_points[i])
                
                # Plot normal intervals
                if normal_times:
                    ax.plot(normal_times, normal_intervals, 'b.', markersize=2, label='Normal', alpha=0.6)
                
                # Highlight discontinuities
                if disc_times:
                    ax.plot(disc_times, disc_intervals, 'ro', markersize=5, label=f'Discontinuities (>{discontinuity_threshold_ms}ms)', zorder=5)
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('PCR Interval (ms)')
                ax.set_title(f'PCR Discontinuity Detection - PID 0x{pid:04X} ({issue_info["large_jumps"]} discontinuities)')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=discontinuity_threshold_ms, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Threshold (+100ms)')
                ax.axhline(y=-discontinuity_threshold_ms, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Threshold (-100ms)')
                ax.legend()
                
                figures.append(fig)
        
        
        # Graph 5: Instantaneous Bitrate (with per-stream-type breakdown)
        t4 = time.time()
        if pcr_records:
            # Use first PCR PID for bitrate calculation
            pid = list(pcr_records.keys())[0]
            records = pcr_records[pid]
            
            if len(records) > 10:
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # Get elementary streams to identify video/audio PIDs
                elementary_streams = report.get('elementary_streams', {})
                pid_to_type = {}  # pid -> 'video', 'audio', or 'other'
                for es_pid, es_info in elementary_streams.items():
                    stream_type_name = es_info.get('stream_type_name', 'Unknown').lower()
                    if 'video' in stream_type_name or stream_type_name.startswith('h.'):
                        pid_to_type[es_pid] = 'video'
                    elif 'audio' in stream_type_name or stream_type_name.startswith('mp') or 'aac' in stream_type_name:
                        pid_to_type[es_pid] = 'audio'
                    else:
                        pid_to_type[es_pid] = 'other'

                # Get PID info for bitrate calculation per stream
                pid_info = report.get('pid_info', {})
                # Precompute total packets (exclude NULL pid)
                total_packets_all = sum(info.get('count', 0) for pid_x, info in pid_info.items() if pid_x != 0x1FFF)
                # Prepare per-PID bitrate series
                pid_bitrates = {pid_x: [] for pid_x in pid_info if pid_x != 0x1FFF}
                
                # Calculate instantaneous bitrate over sliding window
                window_size = max(5, len(records) // 50)  # Adaptive window
                bitrates_total = []
                time_points = []
                
                # Get actual packet size from report (188, 192 for m2ts, or 204 for FEC)
                packet_size = report.get('packet_size', 188)
                
                for i in range(window_size, len(records)):
                    start_idx, start_pcr = records[i - window_size]
                    end_idx, end_pcr = records[i]
                    
                    packets_in_window = end_idx - start_idx
                    time_diff = end_pcr - start_pcr
                    
                    if time_diff > 0:
                        # Total bitrate
                        bitrate_bps = (packets_in_window * packet_size * 8) / time_diff
                        bitrates_total.append(bitrate_bps / 1_000_000)  # Mbps
                        # Per-PID bitrates (approximate proportional allocation based on overall packet counts)
                        if total_packets_all > 0:
                            for pid_x, info in pid_info.items():
                                if pid_x == 0x1FFF:
                                    continue
                                ratio = info.get('count', 0) / total_packets_all
                                pid_bitrates[pid_x].append((bitrate_bps * ratio) / 1_000_000)
                        else:
                            for pid_x in pid_bitrates:
                                pid_bitrates[pid_x].append(0)

                        time_points.append(end_pcr)
                
                if bitrates_total:
                    # Plot total bitrate and per-stream-type breakdown
                    ax.plot(time_points, bitrates_total, 'k-', linewidth=1.5, label='Total Bitrate', alpha=0.7)

                    # Plot per-PID lines with colors
                    import itertools
                    color_cycle = itertools.cycle(plt.cm.Set2.colors)
                    for pid_x, series in pid_bitrates.items():
                        if not any(series):
                            continue
                        pid_type = pid_to_type.get(pid_x, 'other')
                        color = next(color_cycle)
                        label = f"PID 0x{pid_x:04X} ({'Audio' if pid_type=='audio' else 'Video' if pid_type=='video' else 'Other'})"
                        ax.plot(time_points, series, linewidth=0.9, alpha=0.8, label=label, color=color)
                    
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Bitrate (Mbps)')
                    ax.grid(True, alpha=0.3)
                    
                    # Calculate variability metrics for total bitrate
                    avg_bitrate = np.mean(bitrates_total)
                    std_bitrate = np.std(bitrates_total)
                    cv = (std_bitrate / avg_bitrate * 100) if avg_bitrate > 0 else 0  # Coefficient of Variation in %
                    min_bitrate = np.min(bitrates_total)
                    max_bitrate = np.max(bitrates_total)
                    
                    # Determine if CBR or VBR based on CV
                    # CV < 5%: CBR, 5-20%: low VBR, > 20%: high VBR
                    if cv < 5:
                        stream_type = "CBR"
                        type_color = 'lightgreen'
                    elif cv < 20:
                        stream_type = "Low VBR"
                        type_color = 'lightyellow'
                    else:
                        stream_type = "High VBR"
                        type_color = 'lightcoral'
                    
                    ax.set_title(f'Instantaneous Bitrate - {stream_type} (CV: {cv:.1f}%)')
                    
                    # Average bitrate line
                    ax.axhline(y=avg_bitrate, color='g', linestyle='--', alpha=0.7, linewidth=1.5, label=f'Avg: {avg_bitrate:.2f} Mbps')
                    
                    # Info box with variability metrics
                    info_text = (f'Stream Type: {stream_type}\n'
                                f'Coefficient of Variation: {cv:.1f}%\n'
                                f'Average: {avg_bitrate:.2f} Mbps\n'
                                f'Min: {min_bitrate:.2f} Mbps\n'
                                f'Max: {max_bitrate:.2f} Mbps\n'
                                f'Std Dev: {std_bitrate:.2f} Mbps')
                    ax.text(0.02, 0.98, info_text,
                           transform=ax.transAxes, ha='left', va='top',
                           bbox=dict(boxstyle='round', facecolor=type_color, alpha=0.85, edgecolor='black', linewidth=1.5),
                           fontsize=9, fontweight='bold')
                    
                    ax.legend(loc='upper right', fontsize=8, ncol=2)
                    figures.append(fig)
        
        
        # Graph 6: SCTE-35 Splice Opportunities Timeline
        t5 = time.time()
        scte35_messages = report.get('scte35_messages', {})
        if scte35_messages and pcr_records:
            # Get duration from PCR if available
            stream_duration = report.get('approx_duration_s', 0)
            if stream_duration and stream_duration > 0:
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # Extract splice events with actual timing from PTS
                splice_events = []
                print(f"DEBUG: Processing SCTE-35 messages from {len(scte35_messages)} PIDs")
                print(f"DEBUG: Stream duration: {stream_duration:.2f}s")
                for pid, messages in scte35_messages.items():
                    print(f"DEBUG: PID {pid} has {len(messages)} messages")
                    for idx, msg in enumerate(messages):
                        command_type = msg.get('command_name', 'unknown')
                        packet_pts = msg.get('packet_pts_seconds')  # When message arrived
                        splice_time_adjusted = msg.get('splice_time_seconds')  # When splice should occur (adjusted PTS)
                        msg_duration = msg.get('duration_seconds')
                        out_of_network = msg.get('out_of_network_indicator')  # 1=out (ad start), 0=return (ad end)
                        splice_immediate = msg.get('splice_immediate_flag')  # 1=immediate, 0=use splice_time
                        
                        # Determine when the splice opportunity occurs
                        # IMPORTANT: splice_time_adjusted is in raw PTS space and often doesn't align with packet_pts timeline
                        # For graphing, we should use packet_pts (when the message arrived) as the reference point
                        # The splice happens either immediately or within the duration window from that point
                        
                        if packet_pts is not None:
                            # Use packet arrival time as the splice opportunity time (most reliable)
                            splice_opportunity_time = packet_pts
                        else:
                            # No packet PTS available - must estimate from message order
                            # Do NOT use splice_time_adjusted as it's in an unrelated PTS timeline
                            splice_opportunity_time = (idx / max(1, len(messages))) * stream_duration
                        
                        # Calculate offset for display
                        offset_text = ""
                        # Determine if immediate based on actual flag and timing
                        is_immediate = (splice_immediate == 1) or (splice_time_adjusted is None) or (splice_time_adjusted == packet_pts)
                        if splice_time_adjusted is not None and packet_pts is not None:
                            offset = splice_time_adjusted - packet_pts
                            offset_text = f", offset={offset:.2f}s"
                            if abs(offset) < 0.1:
                                is_immediate = True
                        
                        # Format debug output with None handling
                        packet_pts_str = f"{packet_pts:.2f}s" if packet_pts is not None else "None"
                        splice_time_str = f"{splice_time_adjusted:.2f}s" if splice_time_adjusted is not None else "None"
                        duration_str = f"{msg_duration:.2f}s" if msg_duration is not None else "None"
                        
                        # Show raw PTS values for debugging
                        splice_pts_raw = msg.get('splice_pts_time')  # Raw PTS ticks
                        pts_adj = msg.get('pts_adjustment', 0)
                        print(f"DEBUG:   Raw splice_pts_time={splice_pts_raw} ticks, pts_adjustment={pts_adj}, adjusted={(splice_pts_raw + pts_adj) & 0x1FFFFFFFF if splice_pts_raw else None} ticks")
                        
                        out_of_net_str = "OUT" if out_of_network == 1 else "RETURN" if out_of_network == 0 else "N/A"
                        print(f"DEBUG: Message {idx}: type={command_type}, arrival={packet_pts_str}, splice_time={splice_time_str}{offset_text}, duration={duration_str}, immediate={is_immediate}, out_of_network={out_of_net_str}")
                        
                        splice_events.append({
                            'arrival_time': packet_pts,  # When message arrived
                            'splice_time': splice_opportunity_time,  # When splice occurs
                            'type': command_type,
                            'duration': msg_duration,
                            'is_immediate': is_immediate,
                            'out_of_network': out_of_network,  # 1=ad start, 0=ad end
                            'pid': pid
                        })
                
                print(f"DEBUG: Total splice_events collected: {len(splice_events)}")
                for ev in splice_events[:5]:  # Show first 5
                    dur_str = f"{ev['duration']:.1f}s" if ev['duration'] is not None else "None"
                    print(f"  Event: type={ev['type']}, splice_time={ev['splice_time']:.2f}s, duration={dur_str}")
                
                # Sort by splice time
                splice_events.sort(key=lambda x: x['splice_time'])
                
                # Plot timeline
                splice_insert_arrivals = []  # Message arrival times
                splice_insert_times = []  # Actual splice opportunity times
                splice_insert_durations = []
                splice_insert_immediates = []  # Track if immediate or delayed
                splice_insert_out_of_networks = []  # Track if out (ad start) or return (ad end)
                splice_null_times = []
                other_times = []
                
                for event in splice_events:
                    print(f"DEBUG: Processing event type='{event['type']}', splice_time={event['splice_time']:.2f}s, out_of_network={event.get('out_of_network', 'N/A')}")
                    if event['type'] == 'splice_insert':
                        if event['arrival_time'] is not None:
                            splice_insert_arrivals.append(event['arrival_time'])
                        splice_insert_times.append(event['splice_time'])
                        splice_insert_durations.append(event['duration'] if event['duration'] is not None else 30.0)
                        splice_insert_immediates.append(event['is_immediate'])
                        splice_insert_out_of_networks.append(event['out_of_network'])
                    elif event['type'] == 'null':
                        splice_null_times.append(event['splice_time'])
                    else:
                        other_times.append(event['splice_time'])
                
                print(f"DEBUG: Found {len(splice_insert_times)} splice_insert events at times: {splice_insert_times}")
                print(f"DEBUG: Found {len(splice_null_times)} null events")
                print(f"DEBUG: Stream duration: {stream_duration:.2f}s, X-axis range: -5 to {stream_duration + 5:.2f}s")
                
                # Plot message arrivals as green vertical lines (when SCTE-35 message arrived)
                for i, arrival_t in enumerate(splice_insert_arrivals):
                    label = 'Message Arrival' if i == 0 else ''
                    ax.axvline(x=arrival_t, color='green', linewidth=1.5, alpha=0.6, linestyle='--', label=label)
                
                # Plot splice opportunities as vertical lines
                # Red = Out of network (ad start), Orange = Return to network (ad end)
                for i, t in enumerate(splice_insert_times):
                    ad_dur = splice_insert_durations[i]
                    is_immediate = splice_insert_immediates[i]
                    out_of_net = splice_insert_out_of_networks[i]
                    
                    # Choose color based on out_of_network indicator
                    if out_of_net == 1:
                        color = 'red'
                        net_label = 'Ad Insertion Opportunity'
                    elif out_of_net == 0:
                        color = 'orange'
                        net_label = 'Ad End Opportunity'
                    else:
                        color = 'purple'
                        net_label = 'Unknown'
                    
                    if is_immediate:
                        label = f'{net_label} - Immediate' if i == 0 or (i == 1 and out_of_net != splice_insert_out_of_networks[0]) else ''
                    else:
                        label = f'{net_label} - Delayed' if i == 0 or (i == 1 and out_of_net != splice_insert_out_of_networks[0]) else ''
                    
                    ax.axvline(x=t, color=color, linewidth=2.5, alpha=0.8, label=label)
                    
                    # Add shaded region for ad break duration in dark blue
                    ax.axvspan(t, min(t + ad_dur, stream_duration), alpha=0.25, color='darkblue', 
                              label='Ad Break Duration' if i == 0 else '')
                    
                    # Add text annotation showing duration and offset from arrival
                    if i < len(splice_insert_arrivals):
                        arrival_t = splice_insert_arrivals[i]
                        offset = t - arrival_t
                        
                        # Clarify the meaning based on out_of_network_indicator
                        if out_of_net == 1:
                            net_text = "AD START\nOPP"
                            desc_text = f"within {ad_dur:.0f}s"
                        elif out_of_net == 0:
                            net_text = "AD END\nOPP"
                            desc_text = f"within {ad_dur:.0f}s"
                        else:
                            net_text = "UNKNOWN"
                            desc_text = f"{ad_dur:.0f}s"
                        
                        if abs(offset) > 0.1:  # Show offset if > 0.1 second
                            ax.text(t, 0.8, f"{net_text}\n{desc_text}\n+{offset:.1f}s", 
                                   ha='center', va='bottom', fontsize=7.5, 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                        else:
                            ax.text(t, 0.8, f"{net_text}\n{desc_text}\nImmed", 
                                   ha='center', va='bottom', fontsize=7.5, 
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
                
                # Plot null commands as blue dots (timing markers)
                if splice_null_times:
                    ax.scatter(splice_null_times, [0.5] * len(splice_null_times), 
                              color='blue', marker='|', s=100, alpha=0.4, 
                              label=f'Null Commands ({len(splice_null_times)})')
                
                # Plot other commands if any
                if other_times:
                    ax.scatter(other_times, [0.7] * len(other_times), 
                              color='orange', marker='D', s=50, alpha=0.7, label='Other Commands')
                
                # Format X-axis to show time in mm:ss format for better readability
                def format_time(x, pos):
                    minutes = int(x // 60)
                    seconds = int(x % 60)
                    return f'{minutes}:{seconds:02d}'
                
                from matplotlib.ticker import FuncFormatter
                ax.xaxis.set_major_formatter(FuncFormatter(format_time))
                
                ax.set_xlabel('Time (mm:ss)', fontsize=12, fontweight='bold')
                ax.set_ylabel('Event Type', fontsize=12, fontweight='bold')
                
                # Calculate total ad duration
                total_ad_duration = sum(splice_insert_durations)
                out_count = sum(1 for x in splice_insert_out_of_networks if x == 1)
                return_count = sum(1 for x in splice_insert_out_of_networks if x == 0)
                ax.set_title(f'SCTE-35 Splice Events Timeline - {out_count} Ad Start + {return_count} Ad End Opportunities\nRed=Ad Insertion (OUT=1), Orange=Ad End (OUT=0) | Stream Duration: {int(stream_duration//60)}m {int(stream_duration%60)}s', 
                           fontsize=12, fontweight='bold')
                ax.set_ylim(-0.3, 1.3)
                ax.set_xlim(-5, stream_duration + 5)  # Add padding
                ax.grid(True, alpha=0.3, axis='x', linestyle='--')
                ax.set_yticks([])
                
                # Add summary text at the top
                summary = f"Total Events: {len(splice_events)} | Splice Inserts: {len(splice_insert_times)} | Null: {len(splice_null_times)}"
                ax.text(0.5, 0.92, summary, transform=ax.transAxes, ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, edgecolor='blue', linewidth=1.5),
                       fontsize=11, fontweight='bold')
                
                # Legend with unique labels only
                handles, labels = ax.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10, framealpha=0.9)
                
                # Ensure tight layout with extra padding at bottom for X-axis labels
                fig.tight_layout(pad=2.0)
                fig.subplots_adjust(bottom=0.12)
                figures.append(fig)
        self.graphs_inner_frame.columnconfigure(0, weight=1)
        self.graphs_inner_frame.columnconfigure(1, weight=1)
        
        # Return figures for async rendering
        return figures
    
    def render_prepared_graphs(self, figures):
        """Render pre-prepared matplotlib figures to GUI in 2-column grid layout with navigation toolbar.
        Called from background thread via after() to update GUI safely."""
        # Clear previous graphs
        for fig in self.graph_figures:
            plt.close(fig)
        self.graph_figures.clear()
        
        for widget in self.graphs_inner_frame.winfo_children():
            widget.destroy()
        
        # Render figures and attach to GUI in 2-column grid layout with navigation toolbar
        # Note: FigureCanvasTkAgg + draw() is slow (~0.85s per graph)
        # So we use the figures directly as canvas widgets
        for idx, fig in enumerate(figures):
            row = idx // 2  # 2 graphs per row
            col = idx % 2   # Alternate between column 0 and 1
            
            # Create a frame to hold both the toolbar and canvas
            graph_frame = ttk.Frame(self.graphs_inner_frame)
            graph_frame.grid(row=row, column=col, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5, padx=5)
            graph_frame.columnconfigure(0, weight=1)
            graph_frame.rowconfigure(1, weight=1)
            
            # Create canvas for the figure
            canvas = FigureCanvasTkAgg(fig, master=graph_frame)
            canvas.draw()
            
            # Add navigation toolbar (with zoom, pan, home, back, forward buttons)
            try:
                from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
                toolbar = NavigationToolbar2Tk(canvas, graph_frame)
                toolbar.update()
                toolbar.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=2, pady=2)
            except ImportError:
                pass  # If toolbar not available, skip it
            
            # Add canvas widget below toolbar
            canvas.get_tk_widget().grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            
            self.graph_figures.append(fig)
        
        # Update progress to 100% and hide progress bar
        self.update_progress(100)
        self.progress['value'] = 0
        self.progress.grid_forget()
        self.progress_label.grid_forget()
        self.status_label.config(text="Analysis complete", foreground="green")
    
    def generate_graphs(self, report):
        """Generate matplotlib graphs for PCR jitter, accuracy, PTS-PCR, and bitrate"""
        # Clear previous graphs
        for fig in self.graph_figures:
            plt.close(fig)
        self.graph_figures.clear()
        
        for widget in self.graphs_inner_frame.winfo_children():
            widget.destroy()
        
        row = 0
        
        # Graph 1: PCR Jitter (inter-arrival time)
        pcr_records = report.get('pcr_records', {})
        if pcr_records:
            for pid, records in list(pcr_records.items())[:3]:  # Limit to 3 PIDs
                if len(records) < 2:
                    continue
                
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # Calculate inter-arrival times (jitter)
                times = [r[1] for r in records]
                intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
                time_points = times[1:]
                
                ax.plot(time_points, [iv * 1000 for iv in intervals], 'b-', linewidth=0.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('PCR Interval (ms)')
                ax.set_title(f'PCR Jitter - PID 0x{pid:04X}')
                ax.grid(True, alpha=0.3)
                
                # TR101-290 P2.3: PCR repetition interval threshold (40ms max)
                ax.axhline(y=40, color='red', linestyle=':', alpha=0.7, linewidth=2, label='TR101-290 Limit (40ms)')
                
                # Expected interval line (if close to constant bitrate)
                if intervals:
                    avg_interval = np.mean(intervals) * 1000
                    ax.axhline(y=avg_interval, color='g', linestyle='--', alpha=0.5, label=f'Mean: {avg_interval:.2f}ms')
                    ax.legend()
                
                canvas = FigureCanvasTkAgg(fig, master=self.graphs_inner_frame)
                canvas.draw()
                canvas.get_tk_widget().grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10, padx=10)
                self.graph_figures.append(fig)
                row += 1
        
        # Graph 2: PCR Accuracy (deviation from expected)
        if pcr_records:
            for pid, records in list(pcr_records.items())[:3]:
                if len(records) < 3:
                    continue
                
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # Calculate PCR accuracy (difference between actual and expected PCR)
                packet_indices = [r[0] for r in records]
                pcr_values = [r[1] for r in records]
                
                # Expected PCR based on linear fit
                z = np.polyfit(packet_indices, pcr_values, 1)
                expected_pcr = np.poly1d(z)
                pcr_errors = [(pcr_values[i] - expected_pcr(packet_indices[i])) * 1000 for i in range(len(pcr_values))]
                
                # Detect if stream is CBR by checking PCR interval variation
                times = [r[1] for r in records]
                intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
                interval_std = np.std(intervals) * 1000 if len(intervals) > 1 else 0
                is_cbr = interval_std < 5  # If std dev < 5ms, likely CBR
                
                ax.plot(pcr_values, pcr_errors, 'g-', linewidth=0.5)
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('PCR Error (ms)')
                
                # Update title and note based on CBR detection
                cbr_note = " [CBR Stream]" if is_cbr else " [VBR Stream - Note: PCR accuracy applies to CBR only]"
                ax.set_title(f'PCR Accuracy (Deviation from Linear) - PID 0x{pid:04X}{cbr_note}')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='b', linestyle='--', alpha=0.5, label='Expected (0ms)')
                
                # TR101-290 P2.5: PCR accuracy threshold (±500ns = ±0.0005ms) - only applicable for CBR
                if is_cbr:
                    ax.axhline(y=0.0005, color='red', linestyle=':', alpha=0.7, linewidth=2, label='TR101-290 Limit (±500ns, CBR)')
                    ax.axhline(y=-0.0005, color='red', linestyle=':', alpha=0.7, linewidth=2)
                else:
                    # Add informational text for VBR
                    ax.text(0.5, 0.95, 'Note: PCR accuracy (P2.5) applies only to CBR streams', 
                           transform=ax.transAxes, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                           fontsize=9)
                ax.legend()
                
                canvas = FigureCanvasTkAgg(fig, master=self.graphs_inner_frame)
                canvas.draw()
                canvas.get_tk_widget().grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10, padx=10)
                self.graph_figures.append(fig)
                row += 1
        
        # Graph 3: PTS-PCR Difference (for any PID with both records)
        pts_records = report.get('pts_records', {})
        if pcr_records and pts_records:
            es_info = report.get('elementary_streams', {})
            common_pids = set(pcr_records.keys()) & set(pts_records.keys())
            for pid in list(common_pids):
                pcr_list = pcr_records[pid]
                pts_list = pts_records[pid]
                if not pcr_list or not pts_list:
                    continue
                # Determine stream type for color/label
                stream_type = es_info.get(pid, {}).get('type', '').lower() if pid in es_info else ''
                if stream_type == 'audio':
                    color = 'c'
                    label = f'PTS-PCR Difference (Audio) - PID 0x{pid:04X}'
                elif stream_type == 'video':
                    color = 'm'
                    label = f'PTS-PCR Difference (Video) - PID 0x{pid:04X}'
                else:
                    color = 'k'
                    label = f'PTS-PCR Difference - PID 0x{pid:04X}'
                    
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # PERFORMANCE FIX: Use numpy for O(log N) search instead of O(N)
                pcr_indices_arr = np.array([r[0] for r in pcr_list])
                pcr_times_arr = np.array([r[1] for r in pcr_list])
                
                pts_pcr_diffs = []
                pts_plot_times = []
                for pts_idx, pts_val in pts_list:
                    insert_pos = np.searchsorted(pcr_indices_arr, pts_idx)
                    candidates = [i for i in [insert_pos-1, insert_pos, insert_pos+1] 
                                 if 0 <= i < len(pcr_indices_arr)]
                    if candidates:
                        closest_idx = min(candidates, key=lambda i: abs(pcr_indices_arr[i] - pts_idx))
                        diff = (pts_val - pcr_times_arr[closest_idx]) * 1000  # ms
                        pts_pcr_diffs.append(diff)
                        pts_plot_times.append(pts_val)
                if pts_pcr_diffs:
                    ax.plot(pts_plot_times, pts_pcr_diffs, color+'-', linewidth=0.5, marker='o', markersize=2)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('PTS - PCR (ms)')
                    ax.set_title(label)
                    ax.grid(True, alpha=0.3)
                    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
                    canvas = FigureCanvasTkAgg(fig, master=self.graphs_inner_frame)
                    canvas.draw()
                    canvas.get_tk_widget().grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10, padx=10)
                    self.graph_figures.append(fig)
                    row += 1
        
        # Graph 4: PCR Discontinuity Detection (highlight jumps > threshold)
        pcr_jitter_issues = report.get('pcr_jitter_issues', {})
        if pcr_records and pcr_jitter_issues:
            for pid, issue_info in pcr_jitter_issues.items():
                if issue_info.get('large_jumps', 0) == 0:
                    continue  # Skip PIDs with no discontinuities
                
                records = pcr_records.get(pid, [])
                if len(records) < 2:
                    continue
                
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # Calculate PCR differences between consecutive samples
                times = [r[1] for r in records]
                intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
                time_points = times[1:]
                
                # Identify discontinuities (typically > 100ms as per TR101-290 P2.4)
                discontinuity_threshold_ms = 100  # 100ms threshold for visualization
                normal_intervals = []
                normal_times = []
                disc_intervals = []
                disc_times = []
                
                for i, interval in enumerate(intervals):
                    interval_ms = interval * 1000
                    if abs(interval_ms) > discontinuity_threshold_ms:
                        disc_intervals.append(interval_ms)
                        disc_times.append(time_points[i])
                    else:
                        normal_intervals.append(interval_ms)
                        normal_times.append(time_points[i])
                
                # Plot normal intervals
                if normal_times:
                    ax.plot(normal_times, normal_intervals, 'b.', markersize=2, label='Normal', alpha=0.6)
                
                # Highlight discontinuities
                if disc_times:
                    ax.plot(disc_times, disc_intervals, 'ro', markersize=5, label=f'Discontinuities (>{discontinuity_threshold_ms}ms)', zorder=5)
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('PCR Interval (ms)')
                ax.set_title(f'PCR Discontinuity Detection - PID 0x{pid:04X} ({issue_info["large_jumps"]} discontinuities)')
                ax.grid(True, alpha=0.3)
                ax.axhline(y=discontinuity_threshold_ms, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Threshold (+100ms)')
                ax.axhline(y=-discontinuity_threshold_ms, color='orange', linestyle='--', alpha=0.5, linewidth=1, label='Threshold (-100ms)')
                ax.legend()
                
                canvas = FigureCanvasTkAgg(fig, master=self.graphs_inner_frame)
                canvas.draw()
                canvas.get_tk_widget().grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10, padx=10)
                self.graph_figures.append(fig)
                row += 1
        
        # Graph 5: Instantaneous Bitrate
        if pcr_records:
            # Use first PCR PID for bitrate calculation
            pid = list(pcr_records.keys())[0]
            records = pcr_records[pid]
            
            if len(records) > 10:
                fig = Figure(figsize=(10, 4), dpi=80)
                ax = fig.add_subplot(111)
                
                # Calculate instantaneous bitrate over sliding window
                window_size = max(5, len(records) // 50)  # Adaptive window
                bitrates = []
                time_points = []
                
                # Get actual packet size from report (188, 192 for m2ts, or 204 for FEC)
                packet_size = report.get('packet_size', 188)
                
                for i in range(window_size, len(records)):
                    start_idx, start_pcr = records[i - window_size]
                    end_idx, end_pcr = records[i]
                    
                    packets_in_window = end_idx - start_idx
                    time_diff = end_pcr - start_pcr
                    
                    if time_diff > 0:
                        # Use actual packet size to match analysis summary bitrate
                        bitrate_bps = (packets_in_window * packet_size * 8) / time_diff
                        bitrates.append(bitrate_bps / 1_000_000)  # Mbps
                        time_points.append(end_pcr)
                
                if bitrates:
                    ax.plot(time_points, bitrates, 'r-', linewidth=1)
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Bitrate (Mbps)')
                    ax.set_title(f'Instantaneous Bitrate (Window={window_size} PCRs, Packet Size={packet_size}B)')
                    ax.grid(True, alpha=0.3)
                    
                    # Average bitrate line
                    avg_bitrate = np.mean(bitrates)
                    ax.axhline(y=avg_bitrate, color='g', linestyle='--', alpha=0.5, label=f'Avg: {avg_bitrate:.2f} Mbps')
                    ax.legend()
                    
                    canvas = FigureCanvasTkAgg(fig, master=self.graphs_inner_frame)
                    canvas.draw()
                    canvas.get_tk_widget().grid(row=row, column=0, sticky=(tk.W, tk.E), pady=10, padx=10)
                    self.graph_figures.append(fig)
                    row += 1
        
        if row == 0:
            # No graphs generated
            label = ttk.Label(self.graphs_inner_frame, 
                            text="No sufficient PCR/PTS data available for graphing.\nPCRs are required for timing analysis.",
                            foreground="gray", font=('TkDefaultFont', 10))
            label.grid(row=0, column=0, pady=50)
    
    def clear_results(self):
        # Clear old NAL cache
        if hasattr(self, '_nal_sei_cache'):
            self._nal_sei_cache = {}
        
        # Clear new NAL caches (unlimited extraction)
        self._nal_cache = {}
        self._all_nals_unlimited = None
        self._frame_nals_grouped = None
        
        # Clear TR101-290 trees
        if hasattr(self, 'tr_p1_tree'):
            for item in self.tr_p1_tree.get_children():
                self.tr_p1_tree.delete(item)
        
        if hasattr(self, 'tr_p2_tree'):
            for item in self.tr_p2_tree.get_children():
                self.tr_p2_tree.delete(item)
        
        if hasattr(self, 'tr_p3_tree'):
            for item in self.tr_p3_tree.get_children():
                self.tr_p3_tree.delete(item)
        
        # Clear stream structure
        for item in self.structure_tree.get_children():
            self.structure_tree.delete(item)
        
        # Clear elementary streams
        if hasattr(self, 'es_tree') and self.es_tree:
            for item in self.es_tree.get_children():
                self.es_tree.delete(item)
        
        # Reset summary
        self.media_file_var.set("-")
        self.total_packets_var.set("-")
        self.duration_var.set("-")
        self.bitrate_var.set("-")
        self.pids_var.set("-")
        self.gop_structure_var.set("-")
        self.gop_length_var.set("-")
        self.gop_type_var.set("-")
        self.resolution_var.set("-")
        self.frame_rate_var.set("-")
        self.scan_type_var.set("-")
    
    def show_program_selector(self, report, programs, graph_figures=None):
        """Show dialog to select program for MPTS"""
        # Keep progress bar visible for graph loading
        self.update_progress(85)
        
        # Store pre-generated figures
        self.prepared_graph_figures = graph_figures or []
        
        # Move summary back to row 1
        summary_frame = None
        for child in self.progress.master.winfo_children():
            if isinstance(child, ttk.LabelFrame) and "Analysis Summary" in str(child.cget('text')):
                summary_frame = child
                break
        if summary_frame:
            summary_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label.config(text="MPTS detected - Select program", foreground="orange")
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Program (MPTS)")
        dialog.geometry("500x300")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Multiple programs detected. Select one:", 
                 font=('TkDefaultFont', 10)).pack(pady=10)
        
        # Program list
        program_frame = ttk.Frame(dialog, padding="10")
        program_frame.pack(fill=tk.BOTH, expand=True)
        
        program_listbox = tk.Listbox(program_frame, height=8, font=('TkDefaultFont', 10))
        program_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(program_frame, orient=tk.VERTICAL, command=program_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        program_listbox.configure(yscrollcommand=scrollbar.set)
        
        # Populate programs
        program_list = []
        for prog_num, pmt_pid in sorted(programs.items()):
            pmt_info = report.get('pmts', {}).get(pmt_pid, {})
            stream_count = len(pmt_info.get('streams', []))
            display_text = f"Program {prog_num} (PMT PID: 0x{pmt_pid:04X}, {stream_count} streams)"
            program_listbox.insert(tk.END, display_text)
            program_list.append(prog_num)
        
        program_listbox.selection_set(0)
        
        selected_program = [None]
        
        def on_select():
            selection = program_listbox.curselection()
            if selection:
                selected_program[0] = program_list[selection[0]]
                dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
            self.status_label.config(text="Analysis cancelled", foreground="red")
        
        button_frame = ttk.Frame(dialog, padding="10")
        button_frame.pack()
        
        ttk.Button(button_frame, text="Select", command=on_select).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
        
        dialog.wait_window()
        
        if selected_program[0] is not None:
            # Filter report to selected program
            filtered_report = self.filter_report_by_program(report, selected_program[0])
            # Regenerate graphs for the filtered program
            program_graph_figures = []
            if MATPLOTLIB_AVAILABLE:
                self.status_label.config(text=f"Generating graphs for Program {selected_program[0]}...", foreground="orange")
                self.root.update_idletasks()
                program_graph_figures = self.prepare_graphs(filtered_report)
            # Pass both filtered and full report
            self.display_results(filtered_report, program_graph_figures, full_report=report)
    
    def filter_report_by_program(self, report, program_number):
        """Filter MPTS report to show only selected program (SPTS-like)"""
        import copy
        filtered = copy.deepcopy(report)
        
        # Get the PMT PID for this program
        pmt_pid = report['pat']['programs'].get(program_number)
        if not pmt_pid:
            if DEBUG: print(f"[Filter] Program {program_number} not found in PAT")
            return report
        
        # Get PMT info
        pmt_info = report.get('pmts', {}).get(pmt_pid)
        if not pmt_info:
            if DEBUG: print(f"[Filter] PMT not found for PID {pmt_pid}")
            return report
        
        # Get all PIDs for this program
        program_pids = {0, pmt_pid}  # PAT and PMT
        if pmt_info.get('pcr_pid') is not None:
            program_pids.add(pmt_info['pcr_pid'])
        for stream in pmt_info.get('streams', []):
            program_pids.add(stream['pid'])
        
        if DEBUG: print(f"[Filter] Program {program_number}: PMT PID={pmt_pid}, Total PIDs={len(program_pids)}")
        if DEBUG: print(f"[Filter] Program PIDs: {sorted(program_pids)}")
        
        # Filter PID-related data - preserve type information
        original_pid_info = report.get('pid_info', {})
        filtered['pid_info'] = {}
        for pid in program_pids:
            if pid in original_pid_info:
                filtered['pid_info'][pid] = original_pid_info[pid].copy()
                if DEBUG: print(f"[Filter] PID {pid} (0x{pid:04X}): type={original_pid_info[pid].get('type', 'Unknown')}")
        
        # Also keep NULL packets
        if 0x1FFF in original_pid_info:
            filtered['pid_info'][0x1FFF] = original_pid_info[0x1FFF].copy()
        
        filtered['pid_count'] = len(filtered['pid_info'])
        
        # Filter PAT to show only selected program
        filtered['pat'] = report['pat'].copy()
        filtered['pat']['programs'] = {program_number: pmt_pid}
        
        # Filter PMTs to show only selected program's PMT
        filtered['pmts'] = {pmt_pid: pmt_info}
        
        # Filter elementary streams
        filtered['elementary_streams'] = {pid: info for pid, info in report.get('elementary_streams', {}).items()
                                         if pid in program_pids}
        
        # Filter PCR records
        filtered['pcr_records'] = {pid: records for pid, records in report.get('pcr_records', {}).items()
                                  if pid in program_pids}
        
        # Filter PTS records
        filtered['pts_records'] = {pid: records for pid, records in report.get('pts_records', {}).items()
                                  if pid in program_pids}
        
        # Filter DTS records
        filtered['dts_records'] = {pid: records for pid, records in report.get('dts_records', {}).items()
                                  if pid in program_pids}
        
        # Filter continuity errors
        filtered['continuity_errors_per_pid'] = {pid: errors for pid, errors in report.get('continuity_errors_per_pid', {}).items()
                                                 if pid in program_pids}
        
        # Note: NAL/SEI per-frame data is now parsed on-demand, not included in report
        
        # Filter SCTE-35 messages
        filtered['scte35_messages'] = {pid: msgs for pid, msgs in report.get('scte35_messages', {}).items()
                                      if pid in program_pids}
        
        # Update program reference
        filtered['programs'] = filtered['pat']['programs']
        
        if DEBUG: print(f"[Filter] Filtered report: {len(filtered['pid_info'])} PIDs, {len(filtered['elementary_streams'])} elementary streams")
        
        return filtered
    
    def show_error(self, error_msg):
        self.progress['value'] = 0
        self.progress.grid_forget()
        self.progress_label.grid_forget()
        
        # Move summary back to row 1
        summary_frame = None
        for child in self.progress.master.winfo_children():
            if isinstance(child, ttk.LabelFrame) and "Analysis Summary" in str(child.cget('text')):
                summary_frame = child
                break
        if summary_frame:
            summary_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.status_label.config(text=f"Error: {error_msg}", foreground="red")
        messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n\n{error_msg}")
    
    def show_pes_details(self, event):
        """Show PES details in a new window on double-clicking an elementary stream"""
        if not self.pes_detail_window or not self.pes_detail_window.winfo_exists():
            self.pes_detail_window = tk.Toplevel(self.root)
            self.pes_detail_window.title("PES Details")
            self.pes_detail_window.geometry("800x600")
            
            # Treeview for PES details
            self.pes_detail_tree = ttk.Treeview(self.pes_detail_window, columns=("field", "value"), show='headings')
            self.pes_detail_tree.heading("field", text="Field")
            self.pes_detail_tree.heading("value", text="Value")
            self.pes_detail_tree.column("field", width=200)
            self.pes_detail_tree.column("value", width=600)
            
            pes_scroll_y = ttk.Scrollbar(self.pes_detail_window, orient=tk.VERTICAL, command=self.pes_detail_tree.yview)
            self.pes_detail_tree.configure(yscrollcommand=pes_scroll_y.set)
            
            self.pes_detail_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            pes_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Get selected ES PID
        selected_item = self.es_tree.selection()
        if not selected_item:
            return
        es_pid = self.es_tree.item(selected_item, "values")[0]
        es_pid = int(es_pid, 16)  # Convert from hex string to int
        
        # Find PES info in report
        pes_info = None
        report = self.analyser.report()
        for pid, info in report.get('elementary_streams', {}).items():
            if pid == es_pid:
                pes_info = info
                break
        
        if not pes_info:
            return
        
        # Clear existing PES details
        for item in self.pes_detail_tree.get_children():
            self.pes_detail_tree.delete(item)
        
        # PES details
        self.pes_detail_tree.insert('', tk.END, values=("PID", f"0x{pes_info['pid']:04X}"))
        self.pes_detail_tree.insert('', tk.END, values=("Type", pes_info.get('type', "Unknown")))
        self.pes_detail_tree.insert('', tk.END, values=("PES Packet Count", pes_info.get('pes_packets', "-")))
        self.pes_detail_tree.insert('', tk.END, values=("Payload Bytes", pes_info.get('payload_bytes', "-")))
        self.pes_detail_tree.insert('', tk.END, values=("Approx. Bitrate (bps)", pes_info.get('approx_bitrate_bps', "-")))
        
        # PTS/DTS information
        pts_range = "N/A"
        if pes_info.get('pts_first') is not None and pes_info.get('pts_last') is not None:
            pts_range = f"{pes_info['pts_first']:.3f} - {pes_info['pts_last']:.3f}"
        dts_range = "N/A"
        if pes_info.get('dts_first') is not None and pes_info.get('dts_last') is not None:
            dts_range = f"{pes_info['dts_first']:.3f} - {pes_info['dts_last']:.3f}"
        self.pes_detail_tree.insert('', tk.END, values=("PTS Range", pts_range))
        self.pes_detail_tree.insert('', tk.END, values=("DTS Range", dts_range))
        
        # Syntax errors
        errors = pes_info.get('syntax_errors', [])
        err_text = f"{len(errors)}" if errors else "0"
        if errors:
            err_text += ": " + "; ".join(errors[:3])
            if len(errors) > 3:
                err_text += " ..."
        self.pes_detail_tree.insert('', tk.END, values=("Syntax Errors", err_text))
        
        # Show PES detail window
        self.pes_detail_window.deiconify()
    
    def hide_pes_details(self):
        """Hide PES details window if open"""
        if self.pes_detail_window and self.pes_detail_window.winfo_exists():
            self.pes_detail_window.withdraw()
    
    def display_elementary_streams(self, report):
        self.es_tree.delete(*self.es_tree.get_children())
        es = report.get('elementary_streams', {})
        if not es:
            self.es_tree.insert('', 'end', values=("-", "No ES found", "-", "-", "-", "-", "-", "-"))
            return
        
        # Check if this is MP4/MOV format
        file_type = report.get('file_type', 'TS')
        is_mp4_format = file_type in ['MP4/MOV', 'MP4', 'MOV']
        
        if is_mp4_format:
            # Display MP4/MOV tracks
            for track_key, info in sorted(es.items()):
                track_id = info.get('track_id', track_key.replace('track_', ''))
                codec = info.get('codec', 'Unknown')
                stream_type = info.get('stream_type', 'Unknown')
                nal_count = info.get('nal_count', 0)
                
                # Get resolution from SPS
                resolution = "N/A"
                if 'h264_sps' in info:
                    sps = info['h264_sps']
                    resolution = f"{sps.get('width', '?')}x{sps.get('height', '?')}"
                elif 'hevc_sps' in info:
                    sps = info['hevc_sps']
                    resolution = f"{sps.get('width', '?')}x{sps.get('height', '?')}"
                
                # Build codec info string
                codec_info = f"{codec} ({stream_type})"
                
                self.es_tree.insert('', 'end', values=(
                    f"Track {track_id}",
                    codec_info,
                    f"{nal_count} NALs",
                    resolution,
                    "-",
                    "-",
                    "-",
                    "-"
                ))
            
            # Skip pie chart for MP4
            return
        
        # Build a map of PID to stream type from PMT for fallback
        pid_to_stream_type = {}
        for pmt_pid, pmt_info in report.get('pmts', {}).items():
            for stream in pmt_info.get('streams', []):
                pid_to_stream_type[stream['pid']] = stream.get('type_name', 'Unknown')
        
        for pid, info in sorted(es.items()):
            # Get type - fallback to PMT stream type if elementary_stream type is Unknown
            es_type = info.get('type', "Unknown")
            if es_type == "Unknown" and pid in pid_to_stream_type:
                es_type = pid_to_stream_type[pid]
            
            pts_range = "N/A"
            if info.get('pts_first') is not None and info.get('pts_last') is not None:
                pts_range = f"{info['pts_first']:.3f} - {info['pts_last']:.3f}"
            dts_range = "N/A"
            if info.get('dts_first') is not None and info.get('dts_last') is not None:
                dts_range = f"{info['dts_first']:.3f} - {info['dts_last']:.3f}"
            errors = info.get('syntax_errors', [])
            err_text = f"{len(errors)}" if errors else "0"
            if errors:
                err_text += ": " + "; ".join(errors[:3])
                if len(errors) > 3:
                    err_text += " ..."
            bitrate_bps = info.get('approx_bitrate_bps')
            bitrate_text = f"{bitrate_bps/1_000:.1f}" if isinstance(bitrate_bps, (int, float)) and bitrate_bps > 0 else "-"
            self.es_tree.insert('', 'end', values=(
                f"0x{pid:04X}",
                es_type,
                info.get('pes_packets', "-"),
                info.get('payload_bytes', "-"),
                bitrate_text,
                pts_range,
                dts_range,
                err_text
            ))
        
        # Update pie chart
        self.update_es_pie_chart(report)

    def update_es_pie_chart(self, report):
        """Update Elementary Streams pie chart showing PIDs occupancy by payload bytes"""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
            import matplotlib.pyplot as plt
            
            es = report.get('elementary_streams', {})
            if not es:
                # Clear canvas if no data
                self.es_pie_canvas.delete("all")
                return
            
            # Collect PIDs and payload bytes for pie chart
            pids = []
            sizes = []
            for pid, info in sorted(es.items()):
                payload = info.get('payload_bytes', 0)
                if payload > 0:
                    pids.append(f"0x{pid:04X}")
                    sizes.append(payload)
            
            if not sizes or sum(sizes) == 0:
                # Clear canvas if no data
                self.es_pie_canvas.delete("all")
                return
            
            # Create pie chart without labels on slices
            fig = Figure(figsize=(6, 5), dpi=80)
            ax = fig.add_subplot(111)
            # Use a color palette
            colors = plt.cm.Set3(range(len(pids)))
            # No labels on slices, only percentages
            wedges, texts, autotexts = ax.pie(
                sizes, labels=None, autopct='%1.1f%%',
                colors=colors, startangle=90, textprops={'fontsize': 9}
            )
            ax.set_title('PIDs Occupancy (by Payload Bytes)', fontsize=11, fontweight='bold')
            # Make percentage text bold and white
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
            # Add a legend mapping color to PID - position outside pie chart
            legend_labels = [f"{pid}" for pid in pids]
            ax.legend(wedges, legend_labels, title="PID", loc='center left', bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=8, title_fontsize=9, frameon=True)
            fig.tight_layout()
            
            # Clear previous widgets from canvas
            for widget in self.es_pie_canvas.winfo_children():
                widget.destroy()

            # Frame to hold toolbar + canvas
            frame = ttk.Frame(self.es_pie_canvas)
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(1, weight=1)

            # Embed new figure in the frame
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.draw()

            # Add navigation toolbar (zoom/pan)
            toolbar = NavigationToolbar2Tk(canvas, frame)
            toolbar.update()
            toolbar.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=2)

            # Canvas widget below toolbar
            tk_widget = canvas.get_tk_widget()
            tk_widget.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

            # Create a window in the scrollable canvas to hold the frame
            self.es_pie_canvas.create_window(0, 0, window=frame, anchor='nw')

            # Update the scroll region to match the frame size
            self.es_pie_canvas.update_idletasks()
            self.es_pie_canvas.config(scrollregion=self.es_pie_canvas.bbox('all'))
            
            # Keep reference to prevent garbage collection
            if not hasattr(self, '_es_pie_figures'):
                self._es_pie_figures = []
            self._es_pie_figures.append((fig, canvas))
            
        except Exception as e:
            # Silently handle errors (matplotlib not available or other issues)
            pass

    def update_gop_summary(self, report):
        """Update GOP structure summary information using ffprobe.
        
        This method runs ffprobe to analyze GOP structure and updates GUI variables.
        It's designed to be called from background threads and uses root.after() for thread-safe GUI updates.
        """
        try:
            if not self.current_file:
                self.root.after(0, lambda: self.gop_structure_var.set("N/A"))
                self.root.after(0, lambda: self.gop_length_var.set("N/A"))
                self.root.after(0, lambda: self.gop_type_var.set("N/A"))
                return
            
            # Use ffprobe to get frame data and analyze GOP structure
            gop_info = self.extract_gop_from_ffprobe(self.current_file)
            
            if gop_info:
                # GOP Type (Fixed/Variable)
                gop_type = gop_info.get('gop_type', 'N/A')
                self.root.after(0, lambda gt=gop_type: self.gop_structure_var.set(gt))
                
                # GOP Length (Min-Max)
                min_gop = gop_info.get('min_gop_length')
                max_gop = gop_info.get('max_gop_length')
                if min_gop is not None and max_gop is not None:
                    if min_gop == max_gop:
                        gop_len = f"{min_gop}"
                    else:
                        gop_len = f"{min_gop} - {max_gop}"
                else:
                    gop_len = "N/A"
                self.root.after(0, lambda gl=gop_len: self.gop_length_var.set(gl))
                
                # GOP Frame Structure (e.g., IBBP, IP, IBP)
                gop_pattern = gop_info.get('gop_pattern', 'N/A')
                self.root.after(0, lambda gp=gop_pattern: self.gop_type_var.set(gp))
            else:
                self.root.after(0, lambda: self.gop_structure_var.set("N/A"))
                self.root.after(0, lambda: self.gop_length_var.set("N/A"))
                self.root.after(0, lambda: self.gop_type_var.set("N/A"))
            
        except Exception as e:
            # Silently handle errors
            self.root.after(0, lambda: self.gop_structure_var.set("N/A"))
            self.root.after(0, lambda: self.gop_length_var.set("N/A"))
            self.root.after(0, lambda: self.gop_type_var.set("N/A"))

    def apply_file_theme(self, report: dict | None = None):
        """Apply default light-blue tabs and buttons with black text.

        Note: ttk widget background colors are theme-dependent; we prefer `clam`
        because it typically honors background/foreground mappings.
        """
        try:
            style = ttk.Style()
            try:
                style.theme_use('clam')
            except Exception:
                pass

            bg = '#BBDEFB'  # light blue
            active = '#90CAF9'
            pressed = '#64B5F6'
            fg = '#000000'

            style.configure('TButton', background=bg, foreground=fg)
            style.map('TButton',
                      background=[('active', active), ('pressed', pressed)],
                      foreground=[('disabled', 'gray'), ('active', fg), ('pressed', fg)])

            style.configure('TNotebook.Tab', background=bg, foreground=fg)
            style.map('TNotebook.Tab',
                      background=[('selected', active), ('active', active)],
                      foreground=[('selected', fg), ('active', fg)])

            # Progressbar (fill + trough)
            # Note: styling support varies by platform/theme; `clam` usually honors these.
            style.configure('TProgressbar',
                            background=active,
                            troughcolor=bg,
                            bordercolor=bg,
                            lightcolor=active,
                            darkcolor=active)
            style.configure('Horizontal.TProgressbar',
                            background=active,
                            troughcolor=bg,
                            bordercolor=bg,
                            lightcolor=active,
                            darkcolor=active)
        except Exception:
            pass

    def extract_gop_from_ffprobe(self, filepath):
        """Extract GOP information using ffprobe - analyzes entire file between first and last key frame"""
        try:
            import subprocess
            import json
            
            # Use ffprobe to get frame data with pict_type for entire file
            cmd = [
                "ffprobe", "-v", "error", "-of", "json",
                "-show_frames",
                "-show_entries", "frame=pict_type,pkt_pts_time,media_type",
                filepath
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # Increased timeout for full file
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            frames = data.get("frames", [])
            
            if not frames:
                return None
            
            # Filter video frames only
            video_frames = [f for f in frames if f.get("media_type") == "video"]
            if not video_frames:
                return None
            
            # Find first and last key frame (I frame)
            first_key_idx = None
            last_key_idx = None
            
            for i, frame in enumerate(video_frames):
                pict_type = frame.get("pict_type", "?")
                if pict_type == 'I':
                    if first_key_idx is None:
                        first_key_idx = i
                    last_key_idx = i
            
            # If no key frames found, return None
            if first_key_idx is None or last_key_idx is None:
                return None
            
            # Analyze only frames between first and last key frame (inclusive)
            frames_to_analyze = video_frames[first_key_idx:last_key_idx + 1]
            
            # Extract frame types and calculate GOP info
            gop_lengths = []
            frame_types = []
            current_gop_length = 0
            
            # Analyze frames between first and last key frame
            for frame in frames_to_analyze:
                pict_type = frame.get("pict_type", "?")
                frame_types.append(pict_type)
                current_gop_length += 1
                
                # I or IDR frame marks start of new GOP
                if pict_type in ['I']:
                    if current_gop_length > 1:  # Don't count first I frame
                        gop_lengths.append(current_gop_length - 1)
                    current_gop_length = 1
            
            # Add last GOP if exists
            if current_gop_length > 1:
                gop_lengths.append(current_gop_length)
            
            if not gop_lengths:
                return None
            
            # Calculate GOP statistics
            min_gop = min(gop_lengths)
            max_gop = max(gop_lengths)
            
            # Determine if GOP is fixed or variable
            if min_gop == max_gop:
                gop_type = "Fixed"
            else:
                gop_type = "Variable"
            
            # Determine GOP pattern from first few frames (e.g., IBBP, IBBBP, IP)
            pattern = ''.join(frame_types[:min(12, len(frame_types))])
            
            return {
                'gop_type': gop_type,
                'min_gop_length': min_gop,
                'max_gop_length': max_gop,
                'gop_pattern': pattern if pattern else 'N/A'
            }
            
        except Exception as e:
            # Silently handle errors (ffprobe not available, file issues, etc.)
            return None

    def update_video_summary(self, report):
        """Update video information using ffprobe: resolution, frame rate, and scan type.
        
        This method runs ffprobe to extract video metadata and updates GUI variables.
        It's designed to be called from background threads and uses root.after() for thread-safe GUI updates.
        """
        try:
            if not self.current_file:
                self.root.after(0, lambda: self.resolution_var.set("N/A"))
                self.root.after(0, lambda: self.frame_rate_var.set("N/A"))
                self.root.after(0, lambda: self.scan_type_var.set("N/A"))
                return
            
            # Use ffprobe to get video stream information
            video_info = self.extract_video_info_from_ffprobe(self.current_file)
            
            if video_info:
                resolution = video_info.get('resolution', 'N/A')
                frame_rate = video_info.get('frame_rate', 'N/A')
                scan_type = video_info.get('scan_type', 'N/A')
            else:
                # Fallback to N/A if ffprobe fails
                resolution = 'N/A'
                frame_rate = 'N/A'
                scan_type = 'N/A'
            
            self.root.after(0, lambda r=resolution: self.resolution_var.set(r))
            self.root.after(0, lambda f=frame_rate: self.frame_rate_var.set(f))
            self.root.after(0, lambda s=scan_type: self.scan_type_var.set(s))
            
        except Exception as e:
            # Silently handle errors - use root.after for thread-safe updates
            self.root.after(0, lambda: self.resolution_var.set("N/A"))
            self.root.after(0, lambda: self.frame_rate_var.set("N/A"))
            self.root.after(0, lambda: self.scan_type_var.set("N/A"))

    def extract_video_info_from_ffprobe(self, filepath):
        """Extract video information using ffprobe: resolution, frame rate, scan type"""
        try:
            import subprocess
            import json
            
            # Use ffprobe to get stream information
            cmd = [
                "ffprobe", "-v", "error", "-of", "json",
                "-show_streams",
                "-select_streams", "v:0",  # Select first video stream
                filepath
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            
            if not streams:
                return None
            
            stream = streams[0]
            video_info = {}
            
            # Extract resolution
            if 'width' in stream and 'height' in stream:
                width = stream['width']
                height = stream['height']
                video_info['resolution'] = f"{width}x{height}"
            
            # Extract frame rate
            if 'r_frame_rate' in stream:
                # r_frame_rate is in "num/den" format (e.g., "30000/1001")
                fr_str = stream['r_frame_rate']
                try:
                    if '/' in fr_str:
                        num, den = map(int, fr_str.split('/'))
                        frame_rate = num / den
                        video_info['frame_rate'] = f"{frame_rate:.2f} fps"
                    else:
                        video_info['frame_rate'] = f"{float(fr_str):.2f} fps"
                except:
                    if 'avg_frame_rate' in stream:
                        avg_fr_str = stream['avg_frame_rate']
                        try:
                            if '/' in avg_fr_str:
                                num, den = map(int, avg_fr_str.split('/'))
                                frame_rate = num / den
                                video_info['frame_rate'] = f"{frame_rate:.2f} fps"
                            else:
                                video_info['frame_rate'] = f"{float(avg_fr_str):.2f} fps"
                        except:
                            pass
            elif 'avg_frame_rate' in stream:
                avg_fr_str = stream['avg_frame_rate']
                try:
                    if '/' in avg_fr_str:
                        num, den = map(int, avg_fr_str.split('/'))
                        frame_rate = num / den
                        video_info['frame_rate'] = f"{frame_rate:.2f} fps"
                    else:
                        video_info['frame_rate'] = f"{float(avg_fr_str):.2f} fps"
                except:
                    pass
            
            # Extract scan type (interlaced vs progressive)
            if 'field_order' in stream:
                field_order = stream['field_order']
                if field_order and field_order.lower() != 'progressive':
                    video_info['scan_type'] = 'Interlaced'
                else:
                    video_info['scan_type'] = 'Progressive'
            else:
                # Default to progressive if not specified
                video_info['scan_type'] = 'Progressive'
            
            return video_info if video_info else None
            
        except Exception as e:
            # Silently handle errors (ffprobe not available, file issues, etc.)
            return None

    def display_buffer_analysis(self, report):
        """Display HRD/T-STD buffer analysis results"""
        # Clear existing data
        self.buffer_tree.delete(*self.buffer_tree.get_children())
        
        # Check if buffer analysis data exists
        buffer_data = report.get('buffer_analysis')
        if not buffer_data:
            self.buffer_status_var.set("Not available - buffer_analyzer module not found")
            self.buffer_status_label.config(foreground='gray')
            self.buffer_pids_var.set("-")
            self.buffer_overflows_var.set("-")
            self.buffer_underflows_var.set("-")
            self.buffer_tree.insert('', 'end', values=("-", "Buffer analysis not performed", "-", "-", "-", "-", "-"))
            return
        
        # Extract summary statistics
        summary = buffer_data.get('summary', {})
        total_pids = summary.get('total_pids', 0)
        compliant_pids = summary.get('compliant_pids', 0)
        total_overflows = summary.get('total_overflows', 0)
        total_underflows = summary.get('total_underflows', 0)
        
        # Update summary display
        self.buffer_pids_var.set(f"{compliant_pids}/{total_pids} compliant")
        self.buffer_overflows_var.set(str(total_overflows))
        self.buffer_underflows_var.set(str(total_underflows))
        
        # Set overflow/underflow label colors
        if total_overflows > 0:
            self.buffer_overflows_label.config(foreground='red')
        else:
            self.buffer_overflows_label.config(foreground='green')
            
        if total_underflows > 0:
            self.buffer_underflows_label.config(foreground='red')
        else:
            self.buffer_underflows_label.config(foreground='green')
        
        # Set overall compliance status
        if compliant_pids == total_pids and total_pids > 0:
            self.buffer_status_var.set("✓ All buffers compliant")
            self.buffer_status_label.config(foreground='green')
        elif compliant_pids > 0:
            self.buffer_status_var.set(f"⚠ Partial compliance ({compliant_pids}/{total_pids})")
            self.buffer_status_label.config(foreground='orange')
        else:
            self.buffer_status_var.set("✗ Buffer violations detected")
            self.buffer_status_label.config(foreground='red')
        
        # Build a map of PID to stream type from PMT for display
        pid_to_stream_type = {}
        for pmt_pid, pmt_info in report.get('pmts', {}).items():
            for stream in pmt_info.get('streams', []):
                pid_to_stream_type[stream['pid']] = stream.get('type_name', 'Unknown')
        
        # Display per-PID buffer statistics
        per_pid = buffer_data.get('per_pid', {})
        for pid_key, stats in sorted(per_pid.items()):
            # pid_key can be int or string
            if isinstance(pid_key, int):
                pid = pid_key
            elif isinstance(pid_key, str):
                pid = int(pid_key, 16) if pid_key.startswith('0x') else int(pid_key)
            else:
                pid = int(pid_key)
            stream_type = pid_to_stream_type.get(pid, stats.get('stream_type', 'Unknown'))
            
            buffer_size_kb = stats.get('buffer_size', 0) / 1024
            max_level = stats.get('max_level', 0)
            max_util_pct = (max_level / stats.get('buffer_size', 1)) * 100 if stats.get('buffer_size', 0) > 0 else 0
            overflows = stats.get('overflows', 0)
            underflows = stats.get('underflows', 0)
            is_compliant = stats.get('compliant', True)
            
            # Determine compliance display
            compliant_text = "✓ Yes" if is_compliant else "✗ No"
            
            # Set tag for coloring based on compliance
            tag = '' if is_compliant else 'non_compliant'
            
            self.buffer_tree.insert('', 'end', values=(
                f"0x{pid:04X}",
                stream_type,
                f"{buffer_size_kb:.1f}",
                f"{max_util_pct:.1f}",
                overflows,
                underflows,
                compliant_text
            ), tags=(tag,))
        
        # Configure tag colors
        self.buffer_tree.tag_configure('non_compliant', background='#ffcccc')
        
        if not per_pid:
            self.buffer_tree.insert('', 'end', values=("-", "No buffer data", "-", "-", "-", "-", "-"))
    
    def display_captions(self, report):
        """Display decoded closed captions (CEA-608/CEA-708) from video streams"""
        # Clear existing text
        self.caption_cea608_text.configure(state='normal')
        self.caption_cea608_text.delete(1.0, tk.END)
        
        self.caption_cea708_text.configure(state='normal')
        self.caption_cea708_text.delete(1.0, tk.END)
        
        self.caption_sei_text.configure(state='normal')
        self.caption_sei_text.delete(1.0, tk.END)
        
        # Extract captions from elementary streams
        cea608_content = []
        cea708_content = []
        sei_content = []

        def _format_caption_lines(raw_lines):
            """Collapse raw caption tokens into readable wrapped lines and drop immediate duplicates."""
            if not raw_lines:
                return []
            lines = []
            buffer = []
            last_part = None

            def flush():
                if buffer:
                    joined = "".join(buffer).strip()
                    if joined:
                        if not lines or lines[-1] != joined:
                            lines.append(joined)
                    buffer.clear()

            for item in raw_lines:
                if not isinstance(item, str):
                    continue

                token = item.replace('\r', '')
                if token == "\n" or token.strip() == "":
                    flush()
                    continue

                parts = token.split('\n')
                for idx, part in enumerate(parts):
                    part_clean = part.strip()
                    if not part_clean:
                        continue
                    # Skip immediate repeated parts
                    if last_part is not None and part_clean == last_part:
                        continue
                    last_part = part_clean
                    # Add a space separator so words don't glue together; trimmed on flush
                    if buffer and not part_clean.startswith(' '):
                        buffer.append(' ')
                    buffer.append(part_clean)
                    if idx < len(parts) - 1:
                        flush()

            flush()
            return lines
        
        elementary_streams = report.get('elementary_streams', {})
        for pid, stream_info in elementary_streams.items():
            # Look for video NAL stats with closed captions
            if 'h264_sps' in stream_info or 'mpeg2_sequence_header' in stream_info:
                # This is a video stream that may have captions in SEI messages
                pass
        
        # Extract from video_nal_stats if available
        video_nal_stats = report.get('video_nal_stats', {})
        for pid, stats in video_nal_stats.items():
            # Normalize caption_lines which may be stored as tuples (text, field)
            raw_caption_lines = []
            for item in stats.get('caption_lines', []):
                if isinstance(item, str):
                    raw_caption_lines.append(item)
                elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], str):
                    raw_caption_lines.append(item[0])

            # CEA-608 captions
            formatted_608 = _format_caption_lines(raw_caption_lines)
            if formatted_608:
                cea608_content.append(f"PID 0x{pid:04X} CEA-608 Captions:")
                cea608_content.extend(formatted_608)
                cea608_content.append("")

            # CEA-708 DTVCC captions - normalize similarly
            raw_708_lines = []
            for item in stats.get('caption_708_lines', []):
                if isinstance(item, str):
                    raw_708_lines.append(item)
                elif isinstance(item, tuple) and len(item) > 0 and isinstance(item[0], str):
                    raw_708_lines.append(item[0])

            formatted_708 = _format_caption_lines(raw_708_lines)
            if formatted_708:
                cea708_content.append(f"PID 0x{pid:04X} CEA-708 DTVCC:")
                cea708_content.extend(formatted_708)
                cea708_content.append("")
            
            # SEI summary: handled after gathering all PIDs so we can sort by PTS
            # (collection happens below after iterating all PIDs)
        
        # Display content
        if cea608_content:
            self.caption_cea608_text.insert(tk.END, "\n".join(cea608_content))
        else:
            self.caption_cea608_text.insert(tk.END, "(No CEA-608 captions found)")
        
        if cea708_content:
            self.caption_cea708_text.insert(tk.END, "\n".join(cea708_content))
        else:
            self.caption_cea708_text.insert(tk.END, "(No CEA-708 DTVCC captions found)")
        
        # Always include SEI closed_caption blocks from video_nal_stats.
        # Build combined SEI events across all PIDs and sort by PTS (None at end)
        all_sei_events = []
        for pid, stats in video_nal_stats.items():
            for cc_block in stats.get('closed_captions', []):
                all_sei_events.append({'pid': pid, 'pts': cc_block.get('pts'), 'block': cc_block})

        # Sort: entries with pts None go to the end
        all_sei_events.sort(key=lambda e: (e['pts'] is None, e['pts'] if e['pts'] is not None else 0))

        sei_lines = ["Closed Caption SEI Blocks (from video_nal_stats):"]
        if not all_sei_events:
            sei_lines.append("(No closed caption SEI blocks found in video_nal_stats)")
        else:
            for evt in all_sei_events:
                pid = evt['pid']
                pts = evt['pts']
                block = evt['block']
                if pts is not None:
                    sei_lines.append(f"PID 0x{pid:04X}  PTS {pts:.3f}s")
                else:
                    sei_lines.append(f"PID 0x{pid:04X}  PTS (unknown)")
                country = block.get('country_code')
                provider = block.get('provider_code')
                user_id = block.get('user_id')
                sei_lines.append(f"  Country Code: 0x{country:02X}")
                sei_lines.append(f"  Provider Code: 0x{provider:04X}")
                if user_id:
                    sei_lines.append(f"  User ID: {user_id}")
                blocks = block.get('blocks', [])
                if blocks:
                    sei_lines.append(f"  Blocks ({len(blocks)}):")
                    # Show up to 500 blocks to avoid UI overload
                    for blk in blocks[:500]:
                        cc_valid = "✓" if blk.get('valid') else "✗"
                        cc_type = blk.get('type', -1)
                        text = blk.get('text', '')
                        sei_lines.append(f"    [{cc_valid}] Type {cc_type}: {text}")
                    if len(blocks) > 500:
                        sei_lines.append(f"    ... and {len(blocks)-500} more blocks")
                sei_lines.append("")

        self.caption_sei_text.insert(tk.END, "\n".join(sei_lines))
        
        # Make text read-only
        self.caption_cea608_text.configure(state='disabled')
        self.caption_cea708_text.configure(state='disabled')
        self.caption_sei_text.configure(state='disabled')
    
    def display_klv_stanag(self, report):
        """Display KLV metadata and STANAG 4609 compliance results"""
        # Clear existing data
        self.klv_tree.delete(*self.klv_tree.get_children())
        self.klv_issues_text.delete('1.0', tk.END)
        if hasattr(self, 'telemetry_tree'):
            self.telemetry_tree.delete(*self.telemetry_tree.get_children())
        
        # Get KLV and STANAG compliance data
        klv_data = report.get('klv_metadata', {})
        stanag_compliance = report.get('stanag_4609_compliance', {})
        
        # Update summary
        klv_detected = stanag_compliance.get('klv_detected', False)
        is_compliant = stanag_compliance.get('compliant', False)
        
        self.klv_detected_var.set("Yes" if klv_detected else "No")
        
        if is_compliant:
            self.stanag_compliant_var.set("✅ YES")
            self.stanag_compliant_label.config(foreground='green')
        elif klv_detected:
            self.stanag_compliant_var.set("⚠ PARTIAL")
            self.stanag_compliant_label.config(foreground='orange')
        else:
            self.stanag_compliant_var.set("❌ NO")
            self.stanag_compliant_label.config(foreground='red')
        
        # Display asynchronous KLV PIDs
        async_klv = stanag_compliance.get('asynchronous_klv', [])
        self.klv_async_var.set(str(len(async_klv)))
        
        for async_info in async_klv:
            misb_standards = ", ".join(async_info.get('standards', [])) or "None"
            self.klv_tree.insert('', 'end', values=(
                "Asynchronous",
                async_info.get('pid', '-'),
                async_info.get('sync_type', '-'),
                async_info.get('packet_count', '-'),
                misb_standards,
                async_info.get('stream_type', '-')
            ), tags=('async',))
        
        # Display synchronous KLV (embedded in video)
        sync_klv = stanag_compliance.get('synchronous_klv', [])
        self.klv_sync_var.set(str(len(sync_klv)))
        
        for sync_info in sync_klv:
            misb_standards = ", ".join(sync_info.get('standards', [])) or "None"
            self.klv_tree.insert('', 'end', values=(
                "Synchronous",
                sync_info.get('video_pid', '-'),
                sync_info.get('sync_type', '-'),
                sync_info.get('klv_packet_count', '-'),
                misb_standards,
                "Embedded in Video"
            ), tags=('sync',))
        
        # Configure tags for color coding
        self.klv_tree.tag_configure('async', foreground='#1976D2')
        self.klv_tree.tag_configure('sync', foreground='#7B1FA2')
        
        # Display issues and recommendations
        issues = stanag_compliance.get('issues', [])
        recommendations = stanag_compliance.get('recommendations', [])
        
        # Populate details only when we have concrete findings; avoid a generic "no KLV" message
        if issues:
            self.klv_issues_text.insert(tk.END, "⚠️  ISSUES:\n", 'warning')
            for issue in issues:
                self.klv_issues_text.insert(tk.END, f"  • {issue}\n", 'issue')
            self.klv_issues_text.insert(tk.END, "\n")
        
        if recommendations:
            self.klv_issues_text.insert(tk.END, "💡 RECOMMENDATIONS:\n", 'info')
            for rec in recommendations:
                self.klv_issues_text.insert(tk.END, f"  • {rec}\n", 'recommendation')
            self.klv_issues_text.insert(tk.END, "\n")
        
        if klv_detected and not issues and not recommendations and is_compliant:
            self.klv_issues_text.insert(tk.END, "✅ No issues found. Stream is STANAG 4609 compliant.\n", 'success')
        
        # Configure text tags for color coding
        self.klv_issues_text.tag_configure('warning', foreground='#ff6600', font=('TkDefaultFont', 9, 'bold'))
        self.klv_issues_text.tag_configure('issue', foreground='#cc0000')
        self.klv_issues_text.tag_configure('info', foreground='#0066cc', font=('TkDefaultFont', 9, 'bold'))
        self.klv_issues_text.tag_configure('recommendation', foreground='#666666')
        self.klv_issues_text.tag_configure('success', foreground='#009900', font=('TkDefaultFont', 9, 'bold'))
        self.klv_issues_text.tag_configure('normal', foreground='#000000')
        
        # Make text widget read-only
        self.klv_issues_text.config(state=tk.DISABLED)

        # Populate telemetry decoded data with statistics
        self.telemetry_tree.delete(*self.telemetry_tree.get_children())
        telemetry = report.get('misb_telemetry', {})
        
        # Store for packet detail viewer
        self.klv_telemetry_data = telemetry.get('field_history', {})
        
        if telemetry.get('total_samples', 0) == 0:
            self.telemetry_tree.insert('', 'end', values=("No MISB ST 0601 telemetry decoded", "-", "-", "-", "-", "-"))
        else:
            latest = telemetry.get('latest_values', {})
            stats = telemetry.get('field_stats', {})
            counts = telemetry.get('field_counts', {})
            
            for field in sorted(latest.keys()):
                field_stats = stats.get(field, {})
                min_val = field_stats.get('min', '-')
                max_val = field_stats.get('max', '-')
                avg_val = field_stats.get('avg', '-')
                
                # Format numbers
                if isinstance(min_val, (int, float)):
                    min_val = f"{min_val:.4f}" if isinstance(min_val, float) else str(min_val)
                if isinstance(max_val, (int, float)):
                    max_val = f"{max_val:.4f}" if isinstance(max_val, float) else str(max_val)
                if isinstance(avg_val, (int, float)):
                    avg_val = f"{avg_val:.4f}" if isinstance(avg_val, float) else str(avg_val)
                
                latest_val = latest[field]
                if isinstance(latest_val, float):
                    latest_val = f"{latest_val:.4f}"
                
                self.telemetry_tree.insert('', 'end', values=(
                    field, 
                    latest_val, 
                    min_val, 
                    max_val, 
                    avg_val, 
                    counts.get(field, 0)
                ))
            
            # Summary row
            self.telemetry_tree.insert('', 'end', values=(
                f"📊 SUMMARY: {telemetry.get('fields_present', 0)} fields tracked", 
                f"{telemetry.get('total_samples', 0)} total samples",
                "-", "-", "-", "-"
            ), tags=('summary',))
            self.telemetry_tree.tag_configure('summary', background='#e8f4f8', font=('TkDefaultFont', 9, 'bold'))

        # Always enable map button; show_gps_map will handle missing/zero data gracefully
        self.map_btn.config(state=tk.NORMAL)

        # Store last report for map plotting
        self.last_report = report

    def show_gps_map(self):
        """Plot telemetry map: lat/lon path if available; else Sensor FOVs."""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        except Exception:
            messagebox.showinfo("Feature Unavailable", "Matplotlib is not available. Install it to view the map.")
            return
        if not self.last_report:
            return
        tele = self.last_report.get('misb_telemetry', {})
        hist = tele.get('field_history', {})

        # Create window and figure
        win = tk.Toplevel(self.root)
        win.title("Telemetry Map / FOV Viewer")
        win.geometry("900x650")
        fig = Figure(figsize=(9, 6.5), dpi=100)

        # Attempt geopath plot
        lat_key, lon_key = None, None
        if 'Frame Center Latitude (deg)' in hist and 'Frame Center Longitude (deg)' in hist:
            lat_key = 'Frame Center Latitude (deg)'
            lon_key = 'Frame Center Longitude (deg)'
        elif 'Platform Location Latitude (deg)' in hist and 'Platform Location Longitude (deg)' in hist:
            lat_key = 'Platform Location Latitude (deg)'
            lon_key = 'Platform Location Longitude (deg)'

        plotted_geopath = False
        if lat_key and lon_key:
            lats = [v for v in hist.get(lat_key, []) if isinstance(v, (int, float))]
            lons = [v for v in hist.get(lon_key, []) if isinstance(v, (int, float))]
            pts = [(la, lo) for la, lo in zip(lats, lons) if not (la == 0 and lo == 0)]
            if pts:
                # If folium is available, open a real map in browser
                try:
                    import folium, webbrowser, tempfile, os, subprocess, shutil
                    mid = pts[len(pts)//2]
                    m = folium.Map(location=[mid[0], mid[1]], zoom_start=16, tiles='OpenStreetMap')
                    folium.PolyLine(locations=[[la, lo] for la, lo in pts], color='red', weight=3, opacity=0.8).add_to(m)
                    folium.Marker(location=[pts[0][0], pts[0][1]], tooltip='Start').add_to(m)
                    folium.Marker(location=[pts[-1][0], pts[-1][1]], tooltip='End').add_to(m)
                    tmpdir = tempfile.gettempdir()
                    map_path = os.path.join(tmpdir, 'ts_analyser_telemetry_map.html')
                    m.save(map_path)
                    # Try to open the generated map in a browser. webbrowser may delegate to gio/xdg.
                    opened = False
                    try:
                        opened = webbrowser.open('file://' + map_path)
                    except Exception:
                        opened = False

                    if not opened:
                        # Try xdg-open
                        if shutil.which('xdg-open'):
                            try:
                                subprocess.run(['xdg-open', map_path], check=False)
                                opened = True
                            except Exception:
                                opened = False
                        # Try sensible-browser as fallback
                        if not opened and shutil.which('sensible-browser'):
                            try:
                                subprocess.run(['sensible-browser', map_path], check=False)
                                opened = True
                            except Exception:
                                opened = False

                    if not opened:
                        print(f"[INFO] Unable to open map automatically. Open the file manually: {map_path}")
                    plotted_geopath = True
                except Exception:
                    # Fall back to simple lat/lon plot in matplotlib
                    lats, lons = zip(*pts)
                    ax1 = fig.add_subplot(211)
                    ax1.plot(lons, lats, marker='o', markersize=2, linewidth=1)
                    ax1.set_xlabel('Longitude (deg)')
                    ax1.set_ylabel('Latitude (deg)')
                    ax1.set_title('Telemetry Path')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_xlim(min(lons) - 0.001, max(lons) + 0.001)
                    ax1.set_ylim(min(lats) - 0.001, max(lats) + 0.001)
                    plotted_geopath = True

        # Fallback/complement: plot Sensor HFOV/VFOV over samples
        # Standardized FOV field names
        hfov = [v for v in hist.get('Sensor Horizontal FOV (deg)', []) if isinstance(v, (int, float))]
        vfov = [v for v in hist.get('Sensor Vertical FOV (deg)', []) if isinstance(v, (int, float))]
        if hfov or vfov:
            ax2 = fig.add_subplot(212 if plotted_geopath else 111)
            if hfov:
                ax2.plot(range(1, len(hfov) + 1), hfov, label='Sensor Horizontal FOV (deg)', color='tab:blue')
            if vfov:
                ax2.plot(range(1, len(vfov) + 1), vfov, label='Sensor Vertical FOV (deg)', color='tab:orange')
            ax2.set_xlabel('Sample #')
            ax2.set_ylabel('Degrees')
            ax2.set_title('Sensor Field of View over Time')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        else:
            if not plotted_geopath:
                messagebox.showinfo("No Data", "No Lat/Lon or FOV telemetry available to plot.")
                win.destroy()
                return

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        try:
            toolbar = NavigationToolbar2Tk(canvas, win)
            toolbar.update()
        except Exception:
            pass
    
    def show_klv_packet_details(self, event):
        """Show detailed packet-by-packet view of selected telemetry field"""
        selection = self.telemetry_tree.selection()
        if not selection:
            return
        
        item = self.telemetry_tree.item(selection[0])
        values = item['values']
        field_name = values[0]
        
        # Skip summary row
        if field_name.startswith('📊'):
            return
        
        # Get packet history for this field
        history = self.klv_telemetry_data.get(field_name, [])
        if not history:
            messagebox.showinfo("No Data", f"No packet history available for {field_name}")
            return
        
        # Create packet detail window
        detail_win = tk.Toplevel(self.root)
        detail_win.title(f"KLV Packet Details: {field_name}")
        detail_win.geometry("700x500")
        
        # Header
        header_frame = ttk.Frame(detail_win, padding="10")
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text=f"Telemetry Field: {field_name}", 
                 font=('TkDefaultFont', 11, 'bold')).pack(anchor=tk.W)
        ttk.Label(header_frame, text=f"Total Samples: {len(history)}", 
                 foreground='#666').pack(anchor=tk.W)
        
        # Packet list
        tree_frame = ttk.Frame(detail_win)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        packet_tree = ttk.Treeview(tree_frame, 
                                   columns=("packet", "value", "change"), 
                                   show='headings', height=20)
        packet_tree.heading("packet", text="Packet #")
        packet_tree.heading("value", text="Value")
        packet_tree.heading("change", text="Change from Previous")
        
        packet_tree.column("packet", width=100)
        packet_tree.column("value", width=200)
        packet_tree.column("change", width=200)
        
        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=packet_tree.yview)
        packet_tree.configure(yscrollcommand=scroll.set)
        
        packet_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Populate packet data
        prev_val = None
        for idx, value in enumerate(history, 1):
            if prev_val is not None and isinstance(value, (int, float)) and isinstance(prev_val, (int, float)):
                change = value - prev_val
                change_str = f"{change:+.4f}" if isinstance(change, float) else f"{change:+d}"
            else:
                change_str = "-"
            
            val_str = f"{value:.4f}" if isinstance(value, float) else str(value)
            packet_tree.insert('', 'end', values=(idx, val_str, change_str))
            prev_val = value
        
        # Close button
        btn_frame = ttk.Frame(detail_win, padding="10")
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="Close", command=detail_win.destroy).pack(side=tk.RIGHT)
    
    def show_buffer_graph(self, event):
        """Show buffer occupancy graph for selected PID"""
        if not MATPLOTLIB_AVAILABLE:
            messagebox.showinfo("Feature Unavailable", "Matplotlib is not available. Install it to view buffer graphs.")
            return
        
        selection = self.buffer_tree.selection()
        if not selection:
            return
        
        # Get selected PID
        item = self.buffer_tree.item(selection[0])
        values = item['values']
        pid_str = values[0]  # e.g., "0x0100"
        
        if not self.last_report or pid_str == "-":
            return
        
        # Get buffer history from report
        buffer_data = self.last_report.get('buffer_analysis', {})
        per_pid = buffer_data.get('per_pid', {})
        
        # Parse PID from string (e.g., "0x0100" -> 256)
        try:
            pid_int = int(pid_str, 16) if isinstance(pid_str, str) and pid_str.startswith('0x') else int(pid_str)
        except (ValueError, AttributeError):
            return
        
        # Find matching PID in buffer data
        pid_data = None
        for pid_key, stats in per_pid.items():
            # pid_key can be int or string
            if isinstance(pid_key, int):
                if pid_key == pid_int:
                    pid_data = stats
                    break
            elif isinstance(pid_key, str):
                try:
                    key_int = int(pid_key, 16) if pid_key.startswith('0x') else int(pid_key)
                    if key_int == pid_int:
                        pid_data = stats
                        break
                except ValueError:
                    continue
        
        if not pid_data or 'history' not in pid_data:
            messagebox.showinfo("No Graph Data", 
                              f"No buffer history available for PID {pid_str}.\n"
                              "Buffer history tracking may not be enabled.")
            return
        
        # Create graph window
        graph_window = tk.Toplevel(self.root)
        graph_window.title(f"T-STD 3-Stage Buffer Analysis - PID {pid_str}")
        graph_window.geometry("1200x800")
        
        # Check if 3-stage breakdown is available
        has_3stage = ('transport_buffer' in pid_data and 
                     'multiplex_buffer' in pid_data and 
                     'elementary_buffer' in pid_data)
        
        if has_3stage:
            # Create 3 subplots for each stage
            fig = Figure(figsize=(12, 8), dpi=100)
            
            # Extract history data with all buffer levels
            history = pid_data['history']
            if history and len(history) > 0:
                times = [entry['time'] for entry in history]
                tb_levels = [entry.get('tb', 0) for entry in history]
                mb_levels = [entry.get('mb', 0) for entry in history]
                eb_levels = [entry.get('level', entry.get('eb', 0)) for entry in history]
                print(f"[DEBUG] Buffer graph for PID {pid_str}: {len(history)} history entries")
                print(f"[DEBUG] TB range: {min(tb_levels) if tb_levels else 0} - {max(tb_levels) if tb_levels else 0}")
                print(f"[DEBUG] MB range: {min(mb_levels) if mb_levels else 0} - {max(mb_levels) if mb_levels else 0}")
                print(f"[DEBUG] EB range: {min(eb_levels) if eb_levels else 0} - {max(eb_levels) if eb_levels else 0}")
            else:
                times, tb_levels, mb_levels, eb_levels = [], [], [], []
                print(f"[DEBUG] Buffer graph for PID {pid_str}: NO HISTORY DATA")
            
            # Stage 1: Transport Buffer (TB)
            ax1 = fig.add_subplot(311)
            tb_data = pid_data['transport_buffer']
            
            if times and tb_levels:
                # Use step plot to show instantaneous transitions clearly
                ax1.plot(times, tb_levels, 'g-', linewidth=0.5, drawstyle='steps-post', alpha=0.9, label='TB Level')
                ax1.fill_between(times, 0, tb_levels, step='post', alpha=0.2, color='green')
            
            ax1.axhline(y=tb_data['size'], color='r', linestyle='--', alpha=0.7, 
                       label=f"TB Size: {tb_data['size']} bytes")
            if tb_data['max_level'] > 0:
                ax1.axhline(y=tb_data['max_level'], color='orange', linestyle=':', alpha=0.5,
                           label=f"Max: {tb_data['max_level']} bytes")
            
            ax1.set_ylabel('TB Level (bytes)')
            ax1.set_title(f'Stage 1: Transport Buffer (TB) - Instantaneous (Always ≈188 bytes per packet)\n'
                         f'Max: {tb_data["max_level"]} bytes of {tb_data["size"]} bytes capacity, '
                         f'Overflows: {tb_data["overflows"]}')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper right')
            ax1.set_ylim(0, max(tb_data['size'] * 1.1, 1))
            
            # Stage 2: Multiplex Buffer (MB)
            ax2 = fig.add_subplot(312)
            mb_data = pid_data['multiplex_buffer']
            
            if times and mb_levels:
                # Use step plot to show sawtooth pattern (accumulation + drops at PUSI)
                ax2.plot(times, mb_levels, 'm-', linewidth=0.5, drawstyle='steps-post', alpha=0.9, label='MB Level')
                ax2.fill_between(times, 0, mb_levels, step='post', alpha=0.2, color='magenta')
            
            ax2.axhline(y=mb_data['size'], color='r', linestyle='--', alpha=0.7,
                       label=f"MB Size: {mb_data['size']/1024:.1f} KB")
            if mb_data['max_level'] > 0:
                ax2.axhline(y=mb_data['max_level'], color='orange', linestyle=':', alpha=0.5,
                           label=f"Max: {mb_data['max_level']/1024:.1f} KB")
            
            ax2.set_ylabel('MB Level (bytes)')
            ax2.set_title(f'Stage 2: Multiplex Buffer (MB) - PES Packet Demux\n'
                         f'Max PES buffered: {mb_data["max_level"]/1024:.1f} KB of {mb_data["size"]/1024:.1f} KB '
                         f'({mb_data["utilization_percent"]:.1f}%), Overflows: {mb_data["overflows"]}')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right')
            ax2.set_ylim(0, max(mb_data['size'] * 1.1, 1))
            
            # Stage 3: Elementary Buffer (EB) - main plot with history
            ax3 = fig.add_subplot(313)
            eb_data = pid_data['elementary_buffer']
            
            # Extract history data (EB level over time)
            if times and eb_levels:
                ax3.plot(times, eb_levels, 'b-', linewidth=1.5, label='EB Level')
                ax3.fill_between(times, 0, eb_levels, alpha=0.3)
            
            ax3.axhline(y=eb_data['size'], color='r', linestyle='--', alpha=0.7,
                       label=f"EB Size: {eb_data['size']/1024:.1f} KB")
            if eb_data['max_level'] > 0:
                ax3.axhline(y=eb_data['max_level'], color='orange', linestyle=':', alpha=0.5,
                           label=f"Max: {eb_data['max_level']/1024:.1f} KB")
            ax3.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Empty')
            
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('EB Level (bytes)')
            ax3.set_title(f'Stage 3: Elementary Buffer (EB) - Decoder Buffer\n'
                         f'Max: {eb_data["max_level"]/1024:.1f} KB of {eb_data["size"]/1024:.1f} KB '
                         f'({eb_data["utilization_percent"]:.1f}%), '
                         f'Overflows: {eb_data["overflows"]}, Underflows: {eb_data["underflows"]}')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper right')
            ax3.set_ylim(0, eb_data['size'] * 1.1)
            
            fig.suptitle(f'ISO/IEC 13818-1 T-STD Three-Stage Buffer Model - PID {pid_str} ({values[1]})', 
                        fontsize=14, fontweight='bold')
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            
        else:
            # Fallback to single plot (legacy mode)
            fig = Figure(figsize=(12, 6), dpi=100)
            ax = fig.add_subplot(111)
            
            # Extract history data
            history = pid_data['history']
            times = [entry['time'] for entry in history]
            levels = [entry['level'] for entry in history]
            buffer_size = pid_data.get('buffer_size', max(levels) if levels else 1)
            
            # Plot buffer level
            ax.plot(times, levels, 'b-', linewidth=1, label='Buffer Level')
            ax.axhline(y=buffer_size, color='r', linestyle='--', alpha=0.7, 
                      label=f'Buffer Size ({buffer_size/1024:.1f} KB)')
            ax.axhline(y=0, color='g', linestyle='--', alpha=0.5, label='Empty')
            
            # Mark overflow/underflow points
            overflows = [i for i, entry in enumerate(history) if entry.get('overflow', False)]
            underflows = [i for i, entry in enumerate(history) if entry.get('underflow', False)]
            
            if overflows:
                overflow_times = [times[i] for i in overflows]
                overflow_levels = [levels[i] for i in overflows]
                ax.scatter(overflow_times, overflow_levels, color='red', marker='x', s=100, 
                          label=f'Overflows ({len(overflows)})', zorder=5)
            
            if underflows:
                underflow_times = [times[i] for i in underflows]
                underflow_levels = [levels[i] for i in underflows]
                ax.scatter(underflow_times, underflow_levels, color='orange', marker='v', s=100,
                          label=f'Underflows ({len(underflows)})', zorder=5)
            
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Buffer Occupancy (bytes)')
            ax.set_title(f'HRD/T-STD Buffer Occupancy - PID {pid_str} ({values[1]})')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add statistics text
            stats_text = (f"Max Utilization: {values[3]}%\n"
                         f"Overflows: {values[4]}\n"
                         f"Underflows: {values[5]}\n"
                         f"Compliant: {values[6]}")
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        canvas = FigureCanvasTkAgg(fig, master=graph_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _get_frame_type(self, frame):
        """Determine the frame type (I, IDR, P, B, Bold B)"""
        try:
            # Check picture type - PyAV can return numeric values or character strings
            # AV_PICTURE_TYPE_NONE = 0, I = 1, P = 2, B = 3, S = 4, SI = 5, SP = 6, BI = 7
            if hasattr(frame, 'pict_type') and frame.pict_type is not None:
                pict_type = frame.pict_type
                
                # Convert to int if possible
                try:
                    pict_type_num = int(pict_type)
                except (ValueError, TypeError):
                    # Try string conversion
                    pict_type_str = str(pict_type).upper()
                    if pict_type_str == 'P':
                        return "P"
                    elif pict_type_str == 'B':
                        if hasattr(frame, 'is_reference') and frame.is_reference:
                            return "Bold B"
                        return "B"
                    elif pict_type_str == 'I':
                        return "I"
                    else:
                        # Fallback to keyframe check
                        if frame.key_frame:
                            return "I"
                        return "Unknown"
                else:
                    # Handle numeric values
                    if pict_type_num == 1:  # I-frame
                        return "I"
                    elif pict_type_num == 2:  # P-frame
                        return "P"
                    elif pict_type_num == 3:  # B-frame
                        # Check if it's a hierarchical B-frame (reference B-frame)
                        if hasattr(frame, 'is_reference') and frame.is_reference:
                            return "Bold B"
                        return "B"
                    elif pict_type_num == 4:  # S-frame (switching frame)
                        return "S"
                    elif pict_type_num == 5:  # SI-frame
                        return "SI"
                    elif pict_type_num == 6:  # SP-frame
                        return "SP"
                    elif pict_type_num == 7:  # BI-frame
                        return "BI"
            
            # Fallback: check if frame is a keyframe
            if frame.key_frame:
                return "I"
            
            # If no pict_type, return unknown
            return "Unknown"
            
        except Exception as e:
            # Fallback for any errors
            if hasattr(frame, 'key_frame') and frame.key_frame:
                return "I"
            return "Unknown"

    def _auto_load_thumbnails(self):
        """Auto-load first 10 video frames with audio after analysis completes"""
        try:
            # Wait for analysis thread to complete and report to be set
            if hasattr(self, 'analysis_thread') and self.analysis_thread:
                self.analysis_thread.join(timeout=60)  # Wait up to 60 seconds
            
            # Additional delay to ensure everything is fully ready
            import time
            time.sleep(0.5)
            
            # Call extract_video_thumbnails with default start_frame=0
            self.root.after(0, lambda: self.extract_video_thumbnails(start_frame=0))
        except Exception as e:
            if DEBUG: print(f"Error auto-loading thumbnails: {e}")
    
    def _extract_sei_timecode(self, packet_data, codec_name=None):
        """Extract timecode from H.264 SEI or MPEG-2 user data"""
        try:
            data = bytes(packet_data)
            pos = 0
            
            # Handle H.264/AVC
            if codec_name in ['h264', 'avc']:
                while pos < len(data) - 4:
                    # Find start code (0x00 0x00 0x01 or 0x00 0x00 0x00 0x01)
                    if data[pos:pos+3] == b'\x00\x00\x01':
                        nal_start = pos + 3
                        start_code_len = 3
                    elif data[pos:pos+4] == b'\x00\x00\x00\x01':
                        nal_start = pos + 4
                        start_code_len = 4
                    else:
                        pos += 1
                        continue
                    
                    if nal_start >= len(data):
                        break
                    
                    nal_header = data[nal_start]
                    nal_type = nal_header & 0x1F
                    
                    # SEI NAL unit type is 6
                    if nal_type == 6:
                        # Find next start code to get SEI payload
                        nal_end = nal_start + 1
                        while nal_end < len(data) - 3:
                            if data[nal_end:nal_end+3] == b'\x00\x00\x01' or data[nal_end:nal_end+4] == b'\x00\x00\x00\x01':
                                break
                            nal_end += 1
                        
                        sei_data = data[nal_start+1:nal_end]
                        sei_pos = 0
                        
                        # Parse SEI messages
                        while sei_pos + 2 < len(sei_data):
                            # Read payload type
                            payload_type = 0
                            while sei_pos < len(sei_data) and sei_data[sei_pos] == 0xFF:
                                payload_type += 255
                                sei_pos += 1
                            if sei_pos >= len(sei_data):
                                break
                            payload_type += sei_data[sei_pos]
                            sei_pos += 1
                            
                            # Read payload size
                            payload_size = 0
                            while sei_pos < len(sei_data) and sei_data[sei_pos] == 0xFF:
                                payload_size += 255
                                sei_pos += 1
                            if sei_pos >= len(sei_data):
                                break
                            payload_size += sei_data[sei_pos]
                            sei_pos += 1
                            
                            # SEI timecode type is 136 (time_code)
                            if payload_type == 136 and sei_pos + payload_size <= len(sei_data):
                                payload = sei_data[sei_pos:sei_pos+payload_size]
                                if len(payload) >= 4:
                                    # Parse timecode from SEI payload
                                    time_offset = (payload[0] << 24) | (payload[1] << 16) | (payload[2] << 8) | payload[3]
                                    # Bit 24 indicates drop-frame
                                    drop_frame = (time_offset >> 24) & 0x01
                                    hours = (time_offset >> 19) & 0x1F
                                    minutes = (time_offset >> 13) & 0x3F
                                    seconds = (time_offset >> 6) & 0x3F
                                    frames = time_offset & 0x3F
                                    # Use semicolon for drop-frame, colon for non-drop-frame
                                    separator = ';' if drop_frame else ':'
                                    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{frames:02d}"
                            
                            sei_pos += payload_size
                    
                    pos = nal_start
            
            # Handle MPEG-2
            elif codec_name in ['mpeg2video', 'mpeg2']:
                while pos < len(data) - 4:
                    # Find start code 0x00 0x00 0x01
                    if data[pos:pos+3] == b'\x00\x00\x01':
                        start_code = data[pos+3]
                        
                        # User data start code is 0xB2
                        if start_code == 0xB2:
                            user_data_start = pos + 4
                            
                            # Find next start code
                            user_data_end = user_data_start
                            while user_data_end < len(data) - 3:
                                if data[user_data_end:user_data_end+3] == b'\x00\x00\x01':
                                    break
                                user_data_end += 1
                            
                            user_data = data[user_data_start:user_data_end]
                            
                            # Look for timecode in user data
                            # SMPTE 12M timecode structure: 4 bytes
                            # Format varies, but common patterns:
                            # - ATSC/SCTE format
                            # - SMPTE RP 188 / VITC
                            
                            # Check for ATSC timecode identifier (GA94)
                            if len(user_data) >= 8 and user_data[0:4] == b'GA94':
                                # Parse ATSC timecode
                                tc_offset = 4
                                if tc_offset + 4 <= len(user_data):
                                    # Timecode bytes
                                    tc_data = user_data[tc_offset:tc_offset+4]
                                    # Bit 7 of hours byte indicates drop-frame
                                    drop_frame = (tc_data[0] & 0x80) != 0
                                    hours = ((tc_data[0] & 0x3F) >> 4) * 10 + (tc_data[0] & 0x0F)
                                    minutes = ((tc_data[1] & 0x7F) >> 4) * 10 + (tc_data[1] & 0x0F)
                                    seconds = ((tc_data[2] & 0x7F) >> 4) * 10 + (tc_data[2] & 0x0F)
                                    frames = ((tc_data[3] & 0x3F) >> 4) * 10 + (tc_data[3] & 0x0F)
                                    # Use semicolon for drop-frame, colon for non-drop-frame
                                    separator = ';' if drop_frame else ':'
                                    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{frames:02d}"
                            
                            # Check for raw SMPTE 12M timecode (4 bytes BCD format)
                            elif len(user_data) >= 4:
                                # Try to parse as BCD timecode
                                tc_data = user_data[0:4]
                                # Check if values look like valid BCD
                                if all((b & 0x0F) < 10 and ((b >> 4) & 0x0F) < 10 for b in tc_data):
                                    # Bit 7 of hours byte indicates drop-frame
                                    drop_frame = (tc_data[0] & 0x80) != 0
                                    hours = ((tc_data[0] & 0x3F) >> 4) * 10 + (tc_data[0] & 0x0F)
                                    minutes = ((tc_data[1] & 0x7F) >> 4) * 10 + (tc_data[1] & 0x0F)
                                    seconds = ((tc_data[2] & 0x7F) >> 4) * 10 + (tc_data[2] & 0x0F)
                                    frames = ((tc_data[3] & 0x3F) >> 4) * 10 + (tc_data[3] & 0x0F)
                                    
                                    # Sanity check
                                    if hours < 24 and minutes < 60 and seconds < 60 and frames < 60:
                                        # Use semicolon for drop-frame, colon for non-drop-frame
                                        separator = ';' if drop_frame else ':'
                                        return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{frames:02d}"
                        
                        pos += 4
                    else:
                        pos += 1
            
            return None
        except Exception as e:
            if DEBUG: print(f"[Timecode Extract] Error: {e}")
            return None
    
    def extract_video_thumbnails(self, start_frame=0):
        """Extract video frame thumbnails using PyAV"""
        if not self.current_file or not os.path.isfile(self.current_file):
            return  # Silently return if no file (e.g., during auto-load before analysis)
        
        # Clear previous thumbnails
        for widget in self.thumbnails_inner_frame.winfo_children():
            widget.destroy()
        self.thumbnail_images.clear()
        
        # Clear NAL frame cache when navigating to new frame window
        # This ensures fresh NAL extraction for the new visible frames
        self._clear_nal_cache()
        
        try:
            num_frames = int(self.num_frames_var.get())
        except ValueError:
            num_frames = 10
        
        self.current_media_type = 'video'
        self.current_frame_start = start_frame
        
        self.status_label.config(text="Extracting video frames...", foreground="blue")
        
        # Run extraction in separate thread
        threading.Thread(target=self._extract_video_worker, args=(num_frames, start_frame), daemon=True).start()
    
    def _clear_nal_cache(self):
        """Clear the cached NAL data for current frame window.
        
        Called when navigating to a new set of frames to ensure fresh extraction.
        """
        self._nal_cache = {}
        self._all_nals_unlimited = None
        self._frame_nals_grouped = None
        if DEBUG: print("[NAL Cache] Cleared all NAL caches on navigation")
        # Clear thumbnail timecode label references when navigating
        try:
            self._thumb_timecode_labels = []
            self._thumb_frames = []
        except Exception:
            self._thumb_timecode_labels = []
            self._thumb_frames = []
    
    def _extract_video_worker(self, num_frames, start_frame=0):
        """Worker thread to extract video frames"""
        container = None
        try:
            container = av.open(self.current_file)
            
            # For MPTS, find the correct video stream based on selected program's video PID
            video_stream = None
            if self.last_report:
                # Check if this is an MP4/MOV file or TS file
                file_type = self.last_report.get('file_type', 'TS')
                is_mp4_format = file_type in ['MP4/MOV', 'MP4', 'MOV']
                
                if is_mp4_format:
                    # For MP4/MOV files, use first video stream (no PID concept)
                    if DEBUG: print(f"[Video Extract] MP4/MOV file detected, using first video stream")
                    video_stream = next(iter(container.streams.video), None)
                else:
                    # Find video PID from elementary streams in the (possibly filtered) report
                    video_pid = None
                    for pid, stream_info in self.last_report.get('elementary_streams', {}).items():
                        stream_type = stream_info.get('stream_type')
                        # H.264/AVC = 0x1B, H.265/HEVC = 0x24, MPEG-2 = 0x02
                        if stream_type in [0x1B, 0x24, 0x02]:
                            video_pid = pid
                            if DEBUG: print(f"[Video Extract] Found video PID: 0x{video_pid:04X} (type 0x{stream_type:02X})")
                            break
                    
                    # Match PyAV stream by PID (stream.id in PyAV corresponds to TS PID)
                    if video_pid is not None:
                        if DEBUG:
                            print(f"[Video Extract] Available video streams:")
                            for s in container.streams.video:
                                print(f"  Stream index={s.index}, id={s.id}, type={s.type}, codec={s.codec_context.name if s.codec_context else 'N/A'}")
                        
                        video_stream = next((s for s in container.streams.video 
                                           if s.id == video_pid), None)
                        if video_stream:
                            if DEBUG: print(f"[Video Extract] Matched PyAV stream with PID 0x{video_pid:04X}")
                        else:
                            if DEBUG: print(f"[Video Extract] PyAV stream not found for PID 0x{video_pid:04X}, trying by index...")
                            # Try alternate method: match by stream index in the program
                            for pmt in self.last_report.get('pmts', {}).values():
                                for idx, stream in enumerate(pmt.get('streams', [])):
                                    if stream['pid'] == video_pid:
                                        # Try to find video stream by relative index
                                        video_streams = list(container.streams.video)
                                        if idx < len(video_streams):
                                            video_stream = video_streams[idx]
                                            if DEBUG: print(f"[Video Extract] Using video stream at index {idx}")
                                        break
            
            # Fallback: use first video stream (SPTS or if PID matching failed)
            if not video_stream:
                video_stream = next(iter(container.streams.video), None)
                if DEBUG and video_stream: print(f"[Video Extract] Using fallback (first video stream)")
            
            if not video_stream:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No video stream found"))
                return
            
            # Try to get total frames from stream metadata
            total_frames = video_stream.frames
            if total_frames == 0 or total_frames is None:
                # Estimate from duration and frame rate
                if video_stream.duration and video_stream.average_rate:
                    duration_sec = float(video_stream.duration * video_stream.time_base)
                    total_frames = int(duration_sec * float(video_stream.average_rate))
            
            # Store stream info for navigation
            self.video_stream_info = {
                'total_frames': total_frames,
                'time_base': video_stream.time_base,
                'average_rate': video_stream.average_rate
            }
            self.total_video_frames = total_frames
            
            frames = []
            
            # Get frame filter setting
            frame_filter = self.frame_filter_var.get() if hasattr(self, 'frame_filter_var') else "all"

            # For MP4/MOV files, seeking can be unreliable due to sparse keyframes
            # Always decode from the beginning for accurate frame extraction
            target_start_sec = None
            seeked = False
            
            # Note: Seeking disabled for MP4 to ensure accurate frame extraction
            # MP4 files may have very sparse keyframes (e.g., only at start and end)
            # which causes seeks to land at wrong positions
            
            # Extract consecutive frames starting from start_frame
            frame_count = start_frame if seeked else 0
            extracted = 0
            max_frames_to_scan = 10000  # Safety limit: stop scanning after 10k frames to prevent infinite loops
            frames_scanned = 0  # Track how many frames we've examined (for filter feedback)
            
            for packet in container.demux(video_stream):
                # Decode packet to list first to avoid iterator corruption on errors
                try:
                    decoded_frames = list(packet.decode())
                except av.error.EOFError:
                    # Reached end of file while decoding - this is normal
                    if DEBUG: print(f"[Video Extract] Reached EOF after scanning {frames_scanned} frames, extracted {extracted}")
                    break
                except av.error.InvalidDataError:
                    # Skip corrupted packets
                    if DEBUG: print(f"[Video Extract] Invalid data in packet, skipping")
                    continue
                except Exception as e:
                    # Skip other decoding errors
                    if DEBUG: print(f"[Video Extract] Decode error: {e}")
                    continue
                
                # Now iterate through decoded frames safely
                try:
                    for frame in decoded_frames:
                        # Skip frames that land before the requested window when we seeked
                        if target_start_sec is not None and frame.pts is not None:
                            try:
                                if float(frame.pts * video_stream.time_base) + 1e-6 < target_start_sec - 0.05:
                                    frame_count += 1
                                    continue
                            except Exception:
                                pass

                        if frame_count >= start_frame:
                            frames_scanned += 1

                            # Safety check: if we've scanned too many frames without finding matches, stop
                            if frame_filter != "all" and frames_scanned > max_frames_to_scan and extracted == 0:
                                if DEBUG: print(f"[Video Extract] No {frame_filter} found in first {max_frames_to_scan} frames")
                                break

                            try:
                                # Determine frame type
                                frame_type = self._get_frame_type(frame)
                                
                                # Apply filter
                                should_include = False
                                if frame_filter == "all":
                                    should_include = True
                                elif frame_filter == "i_frames":
                                    # Include all I-frames (I, IDR, I-P, I-B)
                                    should_include = frame_type in ["I", "IDR", "I-P", "I-B"]
                                elif frame_filter == "idr_frames":
                                    # Include only IDR frames
                                    should_include = (frame_type == "IDR")
                                
                                if should_include and extracted < num_frames:
                                    # Convert frame to PIL Image
                                    img = frame.to_image()
                                    # Resize to thumbnail (max 200x150)
                                    img.thumbnail((200, 150), Image.Resampling.LANCZOS)
                                    
                                    # Calculate PTS safely
                                    if frame.pts is not None:
                                        pts_sec = float(frame.pts * video_stream.time_base)
                                    elif video_stream.average_rate is not None:
                                        pts_sec = frame_count / float(video_stream.average_rate)
                                    else:
                                        pts_sec = 0.0
                                    
                                    dts_sec = float(frame.dts * video_stream.time_base) if frame.dts else None
                                    
                                    # Extract SEI/user data timecode from packet if available
                                    timecode = None
                                    try:
                                        codec_name = video_stream.codec_context.name if video_stream.codec_context else None
                                        if hasattr(packet, 'buffer_ptr') and packet.buffer_ptr:
                                            timecode = self._extract_sei_timecode(bytes(packet), codec_name)
                                    except:
                                        pass
                                    
                                    # Store frame data: (frame_num, img, pts, dts, frame_type, raw_pts, raw_dts, timecode)
                                    display_index = start_frame + extracted if start_frame and seeked else frame_count
                                    frames.append((display_index, img, pts_sec, dts_sec, frame_type, frame.pts, frame.dts, timecode))
                                    extracted += 1
                            except Exception:
                                # Skip corrupted frames
                                pass
                        frame_count += 1
                        if extracted >= num_frames:
                            break
                        # Another safety check: stop if we've scanned way more than needed
                        if frame_filter != "all" and frames_scanned > max_frames_to_scan:
                            break
                except Exception as e:
                    # Catch any frame processing errors
                    if DEBUG: print(f"[Video Extract] Frame processing error: {e}")
                    continue
                if extracted >= num_frames:
                    break
                if frame_filter != "all" and frames_scanned > max_frames_to_scan:
                    break
            
            if not frames:
                # Show helpful message based on filter type
                filter_type = self.frame_filter_var.get() if hasattr(self, 'frame_filter_var') else "all"
                scanned_info = f"scanned {frames_scanned} frames" if frames_scanned > 0 else "no frames scanned"
                
                if filter_type == "i_frames":
                    msg = f"ℹ️  No I-frames found in the scanned range ({scanned_info}).\n\n" \
                          f"The video might not contain I-frames, or they are very sparse.\n" \
                          f"Consider switching to 'All Frames' filter to view other frame types."
                    self.root.after(0, lambda m=msg: messagebox.showwarning("I-Frames Not Found", m))
                elif filter_type == "idr_frames":
                    msg = f"ℹ️  No IDR frames found in the scanned range ({scanned_info}).\n\n" \
                          f"This video might use open GOP structure without IDR frames,\n" \
                          f"or IDR frames are very sparse in this section.\n\n" \
                          f"Suggestions:\n" \
                          f"• Try 'I-Frames Only' to see all I-frames (including non-IDR)\n" \
                          f"• Switch to 'All Frames' to view all frame types"
                    self.root.after(0, lambda m=msg: messagebox.showwarning("IDR Frames Not Found", m))
                else:
                    msg = "No frames could be extracted from the video stream."
                    self.root.after(0, lambda m=msg: messagebox.showinfo("No Frames Found", m))
                return
            
            # Display thumbnails in UI thread
            self.root.after(0, self._display_video_thumbnails, frames, total_frames if total_frames else 0)
            
        except Exception as e:
            import traceback
            error_msg = str(e)
            trace_msg = traceback.format_exc()
            if DEBUG: print(f"[Video Extract] Exception:\n{trace_msg}")
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to extract video frames:\n{msg}"))
        finally:
            if container is not None:
                try:
                    container.close()
                except:
                    pass
            self.root.after(0, lambda: self.status_label.config(text="Ready", foreground="green"))
    
    def _display_video_thumbnails(self, frames, total_frames=0):
        """Display video thumbnails in the GUI"""
        # Safety check: don't display if no frames
        if not frames or len(frames) == 0:
            self.status_label.config(text="No frames to display", foreground="orange")
            return
        
        col = 0
        
        # Prepare frames ordering based on selected order (PTS or DTS)
        frames_sorted = list(frames)
        try:
            order = self.frame_order_var.get() if hasattr(self, 'frame_order_var') else 'pts'
        except Exception:
            order = 'pts'

        if order == 'dts':
            # Sort by DTS (index 3), missing DTS values go to the end
            frames_sorted.sort(key=lambda f: (f[3] if (len(f) > 3 and f[3] is not None) else float('inf')))
        else:
            # Default: sort by PTS (index 2)
            frames_sorted.sort(key=lambda f: (f[2] if (len(f) > 2 and f[2] is not None) else float('inf')))

        # Store frames for later access
        self.current_frames_data = frames_sorted
        # Prepare storage for per-thumbnail frame widgets and timecode label references
        self._thumb_timecode_labels = []
        self._thumb_frames = []
        
        for frame_data in self.current_frames_data:
            if len(frame_data) == 8:
                idx, img, pts, dts, frame_type, raw_pts, raw_dts, timecode = frame_data
            elif len(frame_data) == 7:
                idx, img, pts, dts, frame_type, raw_pts, raw_dts = frame_data
                timecode = None
            elif len(frame_data) == 4:
                idx, img, pts, frame_type = frame_data
                dts, raw_pts, raw_dts, timecode = None, None, None, None
            else:
                idx, img, pts = frame_data
                frame_type, dts, raw_pts, raw_dts, timecode = "Unknown", None, None, None, None
            
            # Convert PIL image to PhotoImage
            photo = ImageTk.PhotoImage(img)
            self.thumbnail_images.append(photo)  # Keep reference
            
            # Create frame for thumbnail with click binding
            thumb_frame = ttk.Frame(self.thumbnails_inner_frame, relief=tk.RIDGE, borderwidth=2, cursor="hand2")
            thumb_frame.grid(row=0, column=col, padx=5, pady=5)
            
            # Display image with click handler
            label = tk.Label(thumb_frame, image=photo, cursor="hand2")
            label.pack()
            label.bind("<Button-1>", lambda e, f=frame_data: self.show_frame_details(f))
            
            # Display frame type (with color coding)
            frame_type_colors = {
                "I": "#4CAF50",      # Green for I-frames
                "IDR": "#2196F3",    # Blue for IDR frames
                "P": "#FF9800",      # Orange for P-frames
                "B": "#9C27B0",      # Purple for B-frames
                "Bold B": "#E91E63", # Pink for hierarchical B-frames
                "P/B": "#FFC107",    # Amber for ambiguous P/B
            }
            frame_type_color = frame_type_colors.get(frame_type, "#757575")  # Gray for unknown
            
            type_label = tk.Label(thumb_frame, text=frame_type, foreground=frame_type_color, 
                                   font=('TkDefaultFont', 9), justify=tk.CENTER)
            type_label.pack()
            
            # Display frame number, then PTS and DTS stacked below it
            frame_label_text = f"Frame {idx}"
            if total_frames > 0:
                frame_label_text += f" / {total_frames}"
            frame_label = ttk.Label(thumb_frame, text=frame_label_text, justify=tk.CENTER)
            frame_label.pack()

            # PTS label (one per line)
            try:
                if raw_pts is not None:
                    pts_str = f"PTS={int(raw_pts)}"
                else:
                    # Try to use estimated PTS from grouped NALs if available
                    est_pts = None
                    try:
                        if hasattr(self, '_frame_nals_grouped') and self._frame_nals_grouped and idx < len(self._frame_nals_grouped):
                            nal_group = self._frame_nals_grouped[idx]
                            for n in nal_group:
                                if n.get('estimated_pts') is not None:
                                    est_pts = n.get('estimated_pts')
                                    break
                    except Exception:
                        est_pts = None
                    pts_str = f"PTS~{int(est_pts)}" if est_pts is not None else "PTS=None"
            except Exception:
                pts_str = f"PTS={raw_pts}"
            pts_label = ttk.Label(thumb_frame, text=pts_str, justify=tk.CENTER)
            pts_label.pack()

            # DTS label (below PTS)
            try:
                dts_str = f"DTS={int(raw_dts)}" if raw_dts is not None else "DTS=None"
            except Exception:
                dts_str = f"DTS={raw_dts}"
            dts_label = ttk.Label(thumb_frame, text=dts_str, justify=tk.CENTER)
            dts_label.pack()
            
            # Create a timecode label for the thumbnail (empty if not yet known)
            tc_text = f"TC: {timecode}" if timecode else ""
            timecode_label = ttk.Label(thumb_frame, text=tc_text, font=('TkDefaultFont', 8, 'bold'),
                                       foreground="#1976D2", justify=tk.CENTER)
            timecode_label.pack()
            # Keep direct reference to the timecode label for reliable updates
            self._thumb_timecode_labels.append(timecode_label)
            # Always store the thumbnail frame reference for on-demand label creation
            self._thumb_frames.append(thumb_frame)
            
            col += 1
        
        # Extract and display audio waveforms for ALL audio streams
        if frames and len(frames) > 0:
            threading.Thread(target=self._extract_all_audio_streams, args=(frames,), daemon=True).start()
            # Also extract SEI timecodes in background (after audio to avoid excessive background threads)
            threading.Thread(target=self._extract_sei_timecodes_for_frames, daemon=True).start()
        
        interval = frames[1][0] - frames[0][0] if len(frames) > 1 else 0
        status_msg = f"Extracted {len(frames)} video frames"
        if interval > 0:
            status_msg += f" (every {interval} frames)"
        if total_frames > 0:
            status_msg += f" from {total_frames} total"
        self.status_label.config(text=status_msg, foreground="green")
        
        # Update navigation state
        self.current_position_var.set(f"Position: Frame {self.current_frame_start}")
        
        # Enable/disable navigation buttons
        if self.current_frame_start > 0:
            self.prev_10_btn.config(state=tk.NORMAL)
        else:
            self.prev_10_btn.config(state=tk.DISABLED)
        
        # Enable Next 10 if: (1) we don't know total frames (enable to allow exploration)
        # or (2) we know total frames and haven't reached the end
        if total_frames == 0 or (total_frames > 0 and self.current_frame_start + len(frames) < total_frames):
            self.next_10_btn.config(state=tk.NORMAL)
        else:
            self.next_10_btn.config(state=tk.DISABLED)
        
        self.jump_btn.config(state=tk.NORMAL)
        
        # Enable I/IDR navigation buttons (always enabled when frames are loaded)
        self.prev_idr_btn.config(state=tk.NORMAL)
        self.next_idr_btn.config(state=tk.NORMAL)
    
    def _extract_all_audio_streams(self, frames):
        """Extract audio waveforms for ALL audio streams and display them"""
        container = None
        try:
            container = av.open(self.current_file)
            
            # Find all audio PIDs from elementary streams (filtered report for selected program)
            audio_pids = []
            if self.last_report:
                for pid, stream_info in self.last_report.get('elementary_streams', {}).items():
                    stream_type = stream_info.get('stream_type')
                    # MP3 = 0x03, AAC = 0x0F, AC3 = 0x81, etc.
                    if stream_type in [0x03, 0x04, 0x0F, 0x11, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87]:
                        audio_pids.append((pid, stream_type, stream_info.get('type_name', 'Audio')))
                        if DEBUG: print(f"[Audio Extract All] Found audio PID: 0x{pid:04X} (type 0x{stream_type:02X})")
            
            # If still no audio PIDs, check PyAV streams directly
            if not audio_pids:
                if DEBUG: print("[Audio Extract All] No audio PIDs in report, discovering from PyAV")
                for idx, stream in enumerate(container.streams.audio):
                    # Use stream.id as PID, or generate a fake one
                    pid = stream.id if stream.id else (0x100 + idx)
                    codec_name = stream.codec_context.name if stream.codec_context else 'Audio'
                    audio_pids.append((pid, 0x0F, codec_name))
                    if DEBUG: print(f"[Audio Extract All] Found PyAV audio stream: index={idx}, PID 0x{pid:04X} ({codec_name})")
            
            if not audio_pids:
                if DEBUG: print("[Audio Extract All] No audio streams found")
                return
            
            if DEBUG: print(f"[Audio Extract All] Total audio PIDs: {len(audio_pids)}")
            
            # Get all audio streams from container
            all_audio_streams = []
            
            # If we have PIDs from the report, try to match them
            if self.last_report and self.last_report.get('elementary_streams'):
                for pid, stream_type, type_name in audio_pids:
                    audio_stream = next((s for s in container.streams.audio 
                                       if s.id == pid), None)
                    
                    if not audio_stream:
                        # Try alternate method by stream index
                        for pmt in self.last_report.get('pmts', {}).values():
                            for idx, stream in enumerate(pmt.get('streams', [])):
                                if stream['pid'] == pid:
                                    audio_streams_list = list(container.streams.audio)
                                    if idx < len(audio_streams_list):
                                        audio_stream = audio_streams_list[idx]
                                    break
                    
                    if audio_stream:
                        all_audio_streams.append((pid, type_name, audio_stream))
                        if DEBUG: print(f"[Audio Extract All] Matched stream for PID 0x{pid:04X}")
            else:
                # No report data, just use all PyAV audio streams
                for pid, stream_type, type_name in audio_pids:
                    audio_stream = next((s for s in container.streams.audio 
                                       if s.id == pid), None)
                    if audio_stream:
                        all_audio_streams.append((pid, type_name, audio_stream))
                        if DEBUG: print(f"[Audio Extract All] Using PyAV stream for PID 0x{pid:04X}")
            
            if not all_audio_streams:
                if DEBUG: print("[Audio Extract All] No PyAV audio streams matched")
                return
            
            # Check for problematic audio configurations
            # Filter out E-AC-3 streams with >6 channels that cause segfaults
            safe_audio_streams = []
            for pid, type_name, audio_stream in all_audio_streams:
                if audio_stream.codec_context and audio_stream.codec_context.name == 'eac3' and audio_stream.channels > 6:
                    if DEBUG: print(f"[Audio Extract All] Skipping PID 0x{pid:04X}: {audio_stream.channels}-channel E-AC-3 (unstable)")
                    continue
                safe_audio_streams.append((pid, type_name, audio_stream))
            
            if not safe_audio_streams:
                self.root.after(0, lambda: messagebox.showwarning(
                    "Audio Format Not Supported",
                    "All audio streams in this file use configurations that may cause stability issues.\\n\\n"
                    "Audio waveform extraction has been disabled.\\n\\n"
                    "Video thumbnails will still be displayed."
                ))
                return
            
            all_audio_streams = safe_audio_streams
            if DEBUG: print(f"[Audio Extract All] Using {len(all_audio_streams)} safe audio streams")
            
            # Calculate frame duration
            frame_duration = 1.0 / 30.0
            if self.video_stream_info and self.video_stream_info.get('average_rate'):
                frame_duration = 1.0 / float(self.video_stream_info['average_rate'])
            
            # Extract audio data for all streams in a single pass
            all_streams_data = []
            
            # Create a mapping of stream objects to PIDs
            stream_map = {}
            for pid, type_name, stream_obj in all_audio_streams:
                stream_map[stream_obj] = (pid, type_name)
            
            if DEBUG: print(f"[Audio Extract All] Extracting {len(all_audio_streams)} audio streams in single pass")
            
            # Extract all audio data in one container pass
            audio_data_by_pid = {pid: [] for pid, _, _ in all_audio_streams}
            sample_rates = {}
            
            for packet in container.demux(*[s for _, _, s in all_audio_streams]):
                if packet.stream in stream_map:
                    pid, type_name = stream_map[packet.stream]
                    if pid not in sample_rates:
                        sample_rates[pid] = packet.stream.sample_rate
                    
                    try:
                        decoded_frames = list(packet.decode())
                    except:
                        continue
                    
                    try:
                        for audio_frame in decoded_frames:
                            try:
                                if audio_frame.pts is not None:
                                    frame_time = float(audio_frame.pts * packet.stream.time_base)
                                    arr = audio_frame.to_ndarray()
                                    if len(arr.shape) > 1:
                                        arr = arr.mean(axis=0)  # Average channels to mono
                                    audio_data_by_pid[pid].append((frame_time, arr))
                            except Exception as e:
                                if DEBUG: print(f"Error decoding audio frame for PID 0x{pid:04X}: {e}")
                    except Exception as e:
                        if DEBUG: print(f"Error processing packet for PID 0x{pid:04X}: {e}")
                        continue
            
            # Now align each audio stream's data with video frames
            for pid, type_name, _ in all_audio_streams:
                if DEBUG: print(f"[Audio Extract All] Aligning PID 0x{pid:04X} ({type_name}), {len(audio_data_by_pid.get(pid, []))} audio frames")
                
                all_audio_data = audio_data_by_pid.get(pid, [])
                sample_rate = sample_rates.get(pid, 48000)
                
                # Align audio with video frames
                audio_segments = []
                for frame_data in frames:
                    if len(frame_data) >= 3:
                        frame_idx, img, pts = frame_data[0], frame_data[1], frame_data[2]
                    else:
                        continue
                    
                    start_time = pts
                    end_time = pts + frame_duration
                    
                    # Collect samples within this time range
                    samples = []
                    for audio_time, audio_arr in all_audio_data:
                        if audio_time >= start_time and audio_time < end_time:
                            samples.extend(audio_arr)
                    
                    audio_segments.append((frame_idx, pts, samples, sample_rate))
                
                all_streams_data.append({
                    'pid': pid,
                    'type_name': type_name,
                    'segments': audio_segments,
                    'sample_rate': sample_rate
                })
            
            # Display all audio streams in UI thread
            self.root.after(0, self._display_all_audio_streams, all_streams_data)
            
        except Exception as e:
            if DEBUG: print(f"Error extracting all audio streams: {e}")
        finally:
            if container is not None:
                try:
                    container.close()
                except:
                    pass
    
    def _extract_aligned_audio(self, frames):
        """Extract audio waveform segments aligned with each video frame"""
        container = None
        try:
            container = av.open(self.current_file)
            
            # For MPTS, find the correct audio stream based on selected program's audio PID
            audio_stream = None
            if self.last_report:
                # Find audio PID from elementary streams
                audio_pid = None
                for pid, stream_info in self.last_report.get('elementary_streams', {}).items():
                    stream_type = stream_info.get('stream_type')
                    # MP3 = 0x03, AAC = 0x0F, AC3 = 0x81, etc.
                    if stream_type in [0x03, 0x04, 0x0F, 0x11, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87]:
                        audio_pid = pid
                        if DEBUG: print(f"[Audio Extract] Found audio PID: 0x{audio_pid:04X} (type 0x{stream_type:02X})")
                        break
                
                # Match PyAV stream by PID
                if audio_pid is not None:
                    if DEBUG:
                        print(f"[Audio Extract] Available audio streams:")
                        for s in container.streams.audio:
                            print(f"  Stream index={s.index}, id={s.id}, type={s.type}, codec={s.codec_context.name if s.codec_context else 'N/A'}")
                    
                    audio_stream = next((s for s in container.streams.audio 
                                       if s.id == audio_pid), None)
                    if audio_stream:
                        if DEBUG: print(f"[Audio Extract] Matched audio stream with PID 0x{audio_pid:04X}")
                    else:
                        if DEBUG: print(f"[Audio Extract] No audio stream found with PID 0x{audio_pid:04X}, trying by index...")
                        # Try alternate method: match by stream index in the program
                        # In some cases, PyAV uses stream index rather than PID
                        pmt_info = None
                        for pmt in self.last_report.get('pmts', {}).values():
                            for idx, stream in enumerate(pmt.get('streams', [])):
                                if stream['pid'] == audio_pid:
                                    # Try to find audio stream by relative index
                                    audio_streams = list(container.streams.audio)
                                    if idx < len(audio_streams):
                                        audio_stream = audio_streams[idx]
                                        if DEBUG: print(f"[Audio Extract] Using audio stream at index {idx}")
                                    break
            
            # Fallback: use first audio stream
            if not audio_stream:
                audio_stream = next(iter(container.streams.audio), None)
                if DEBUG and audio_stream: print(f"[Audio Extract] Using fallback (first audio stream)")
            
            if not audio_stream:
                if DEBUG: print("No audio stream found")
                return
            
            # Check for problematic audio configurations that cause segfaults
            # E-AC-3 with >6 channels can cause PyAV crashes
            if audio_stream.codec_context and audio_stream.codec_context.name == 'eac3' and audio_stream.channels > 6:
                self.root.after(0, lambda: messagebox.showwarning(
                    "Audio Format Not Supported",
                    f"This file contains {audio_stream.channels}-channel E-AC-3 audio which may cause stability issues.\\n\\n"
                    f"Audio waveform extraction has been disabled for this stream.\\n\\n"
                    f"Video thumbnails will still be displayed."
                ))
                if DEBUG: print(f"Skipping {audio_stream.channels}-channel E-AC-3 audio extraction")
                return
            
            sample_rate = audio_stream.sample_rate
            if DEBUG: print(f"Audio sample rate: {sample_rate}")
            
            # Calculate frame duration
            frame_duration = 1.0 / 30.0
            if self.video_stream_info and self.video_stream_info.get('average_rate'):
                frame_duration = 1.0 / float(self.video_stream_info['average_rate'])
            
            if DEBUG: print(f"Frame duration: {frame_duration:.6f}s")
            
            # First, extract all audio frames with timestamps
            all_audio_data = []
            for packet in container.demux(audio_stream):
                try:
                    decoded_frames = list(packet.decode())
                except:
                    continue
                
                try:
                    for audio_frame in decoded_frames:
                        try:
                            if audio_frame.pts is not None:
                                frame_time = float(audio_frame.pts * audio_stream.time_base)
                                arr = audio_frame.to_ndarray()
                                if len(arr.shape) > 1:
                                    arr = arr.mean(axis=0)  # Average channels to mono
                                all_audio_data.append((frame_time, arr))
                        except Exception as e:
                            if DEBUG: print(f"Error decoding audio frame: {e}")
                except Exception as e:
                    if DEBUG: print(f"Error processing packet: {e}")
                    continue
            
            if DEBUG: print(f"Extracted {len(all_audio_data)} audio frames")
            
            # Now align audio with each video frame
            audio_segments = []
            
            for frame_data in frames:
                if len(frame_data) >= 3:
                    frame_idx, img, pts = frame_data[0], frame_data[1], frame_data[2]
                else:
                    continue
                
                start_time = pts
                end_time = pts + frame_duration
                
                if DEBUG: print(f"Aligning audio for frame {frame_idx}: {start_time:.6f}s to {end_time:.6f}s")
                
                # Collect samples within this time range
                samples = []
                for audio_time, audio_arr in all_audio_data:
                    if audio_time >= start_time and audio_time < end_time:
                        samples.extend(audio_arr)
                
                if DEBUG: print(f"  Collected {len(samples)} audio samples for frame {frame_idx}")
                audio_segments.append((frame_idx, pts, samples, sample_rate))
            
            if DEBUG: print(f"Total audio segments: {len(audio_segments)}")
            
            # Display audio waveforms in UI thread
            self.root.after(0, self._display_aligned_audio, audio_segments)
            
        except Exception as e:
            if DEBUG: print(f"Error extracting aligned audio: {e}")
        finally:
            if container is not None:
                try:
                    container.close()
                except:
                    pass
    
    def _display_all_audio_streams(self, all_streams_data):
        """Display audio waveform segments for all audio streams below video thumbnails"""
        if not all_streams_data:
            return
        
        if DEBUG: print(f"Displaying {len(all_streams_data)} audio streams")
        
        # Start row counter (row 0 is video thumbnails)
        current_row = 1
        
        for stream_data in all_streams_data:
            pid = stream_data['pid']
            type_name = stream_data['type_name']
            segments = stream_data['segments']
            
            # Add header label for this audio stream
            header_frame = ttk.Frame(self.thumbnails_inner_frame)
            header_frame.grid(row=current_row, column=0, columnspan=len(segments), 
                            sticky=(tk.W, tk.E), padx=5, pady=(10, 2))
            
            header_label = ttk.Label(header_frame, 
                                    text=f"Audio PID 0x{pid:04X} ({type_name})",
                                    font=('TkDefaultFont', 9, 'bold'),
                                    foreground='#1976D2')
            header_label.pack(anchor=tk.W)
            
            current_row += 1
            
            # Display waveforms for this stream
            col = 0
            for frame_idx, pts, samples, sample_rate in segments:
                # Create frame for audio waveform with fixed size
                audio_frame = ttk.Frame(self.thumbnails_inner_frame, relief=tk.RIDGE, 
                                       borderwidth=1, width=200, height=150)
                audio_frame.grid(row=current_row, column=col, padx=5, pady=5, sticky=(tk.N, tk.W))
                audio_frame.grid_propagate(False)
                audio_frame.pack_propagate(False)
                
                if samples and len(samples) > 0:
                    # Create small waveform plot
                    samples_array = np.array(samples, dtype=np.float32)
                    
                    # Normalize audio to -1.0 to 1.0 range if needed
                    max_val = np.abs(samples_array).max()
                    if max_val > 0:
                        samples_array = samples_array / max_val
                    
                    fig = Figure(figsize=(2.5, 1.8), dpi=80)
                    ax = fig.add_subplot(111)
                    
                    # Downsample for display if needed
                    display_points = 300
                    if len(samples_array) > display_points:
                        step = len(samples_array) // display_points
                        samples_display = samples_array[::step]
                    else:
                        samples_display = samples_array
                    
                    # Calculate proper time axis
                    duration = len(samples_array) / sample_rate
                    time_axis = np.linspace(0, duration, len(samples_display))
                    
                    ax.plot(time_axis, samples_display, linewidth=0.8, color='#2196F3')
                    ax.set_xlim(0, duration)
                    ax.set_ylim(-1.1, 1.1)
                    ax.grid(True, alpha=0.3, linewidth=0.5)
                    ax.set_xlabel('Time (s)', fontsize=7)
                    ax.set_ylabel('Amplitude', fontsize=7)
                    ax.tick_params(labelsize=6)
                    fig.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.20)
                    
                    # Add to canvas
                    canvas = FigureCanvasTkAgg(fig, master=audio_frame)
                    canvas.draw()
                    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(3, 0))
                    
                    self.thumbnail_images.append(canvas)  # Keep reference
                    
                    # Add time label
                    time_label = ttk.Label(audio_frame, text=f"{pts:.3f}s", 
                                          font=('TkDefaultFont', 7))
                    time_label.pack()
                else:
                    # No audio for this frame
                    ttk.Label(audio_frame, text="No audio", 
                            font=('TkDefaultFont', 7), foreground="gray").pack(expand=True)
                
                col += 1
            
            current_row += 1
    
    def _display_aligned_audio(self, audio_segments):
        """Display audio waveform segments below video thumbnails"""
        col = 0
        
        if DEBUG: print(f"Displaying {len(audio_segments)} audio segments")
        
        for frame_idx, pts, samples, sample_rate in audio_segments:
            # Create frame for audio waveform with fixed size
            audio_frame = ttk.Frame(self.thumbnails_inner_frame, relief=tk.RIDGE, borderwidth=1, width=200, height=180)
            audio_frame.grid(row=1, column=col, padx=5, pady=5, sticky=(tk.N, tk.W))
            audio_frame.grid_propagate(False)  # Don't let children resize the frame
            audio_frame.pack_propagate(False)  # Don't let children resize the frame
            
            if samples and len(samples) > 0:
                if DEBUG: print(f"  Frame {frame_idx}: {len(samples)} samples")
                
                # Create small waveform plot
                samples_array = np.array(samples, dtype=np.float32)
                
                # Normalize audio to -1.0 to 1.0 range if needed
                max_val = np.abs(samples_array).max()
                if max_val > 0:
                    samples_array = samples_array / max_val
                
                if DEBUG: print(f"    After normalization: min={samples_array.min():.3f}, max={samples_array.max():.3f}")
                
                fig = Figure(figsize=(2.5, 2.2), dpi=80)
                ax = fig.add_subplot(111)
                
                # Downsample for display if needed
                display_points = 300
                if len(samples_array) > display_points:
                    step = len(samples_array) // display_points
                    samples_display = samples_array[::step]
                else:
                    samples_display = samples_array
                
                # Calculate proper time axis
                duration = len(samples_array) / sample_rate
                time_axis = np.linspace(0, duration, len(samples_display))
                
                ax.plot(time_axis, samples_display, linewidth=0.8, color='#2196F3')
                ax.set_xlim(0, duration)
                ax.set_ylim(-1.1, 1.1)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.set_xlabel('Time (s)', fontsize=7)
                ax.set_ylabel('Amplitude', fontsize=7)
                ax.tick_params(labelsize=6)
                fig.subplots_adjust(left=0.12, right=0.95, top=0.90, bottom=0.18)
                
                # Add to canvas
                canvas = FigureCanvasTkAgg(fig, master=audio_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(5, 0))
                
                self.thumbnail_images.append(canvas)  # Keep reference
                
                # Add time label
                time_label = ttk.Label(audio_frame, text=f"{pts:.3f}s", font=('TkDefaultFont', 7))
                time_label.pack()
            else:
                if DEBUG: print(f"  Frame {frame_idx}: No samples")
                # No audio for this frame
                ttk.Label(audio_frame, text="No audio", font=('TkDefaultFont', 7), foreground="gray").pack()
            
            col += 1
    
    def _refresh_thumbnail_timecodes(self):
        """Refresh timecode display for all thumbnails based on updated frame data.
        
        This is called when SEI timecode extraction updates frame_data with new timecodes.
        """
        try:
            if not hasattr(self, 'thumbnails_inner_frame') or not hasattr(self, 'current_frames_data'):
                return
            
            # Update timecode labels using stored references to avoid widget scanning
            for idx, lbl in enumerate(getattr(self, '_thumb_timecode_labels', []) ):
                try:
                    if idx >= len(self.current_frames_data):
                        continue
                    timecode = self.current_frames_data[idx][7] if len(self.current_frames_data[idx]) > 7 else None
                    if lbl and timecode:
                        lbl.config(text=f"TC: {timecode}")
                        if DEBUG: print(f"[Refresh TC] Updated thumbnail {idx} label to: {timecode}")
                    elif lbl and not timecode:
                        # Clear label if timecode removed
                        lbl.config(text="")
                        if DEBUG: print(f"[Refresh TC] Cleared thumbnail {idx} label")
                    elif (not lbl) and timecode:
                        # No label exists yet — create one and attach to stored thumbnail frame
                        try:
                            frame_widget = None
                            if hasattr(self, '_thumb_frames') and idx < len(self._thumb_frames):
                                frame_widget = self._thumb_frames[idx]
                            if frame_widget is not None:
                                new_lbl = ttk.Label(frame_widget, text=f"TC: {timecode}", 
                                                    font=('TkDefaultFont', 8, 'bold'),
                                                    foreground="#1976D2", justify=tk.CENTER)
                                new_lbl.pack()
                                self._thumb_timecode_labels[idx] = new_lbl
                                if DEBUG: print(f"[Refresh TC] Created new thumbnail label for {idx}: {timecode}")
                        except Exception as e:
                            if DEBUG: print(f"[Refresh TC] Failed to create label for {idx}: {e}")
                except Exception as e:
                    if DEBUG: print(f"[Refresh TC] Error updating thumb {idx}: {e}")
        except Exception as e:
            if DEBUG: print(f"[Refresh TC] Error refreshing thumbnail timecodes: {e}")
    
    def _extract_sei_timecodes_for_frames(self):
        """Background thread to extract SEI timecodes for all displayed frames.
        
        This is called after thumbnails are displayed to fill in timecode information
        from SEI messages for each frame.
        """
        try:
            if not hasattr(self, 'current_frames_data') or not hasattr(self, 'analyzer'):
                if DEBUG: print("[Background TC] No frame data or analyzer available")
                return
            
            if DEBUG: print(f"[Background TC] Starting timecode extraction for {len(self.current_frames_data)} frames")
            
            updated_count = 0
            
            # Process each frame
            for frame_idx, frame_data in enumerate(self.current_frames_data):
                try:
                    # Skip if already has timecode
                    if len(frame_data) > 7 and frame_data[7]:
                        if DEBUG: print(f"[Background TC] Frame {frame_idx} already has timecode: {frame_data[7]}")
                        continue
                    
                    # Get PTS and absolute frame number for this frame
                    frame_pts = frame_data[2] if len(frame_data) > 2 else None
                    abs_frame_num = frame_data[0] if len(frame_data) > 0 else frame_idx  # Use absolute frame number
                    if not frame_pts:
                        continue
                    
                    # Extract NALs for this frame using absolute frame number
                    frame_nals = self._extract_nals_by_pts(frame_pts, abs_frame_num)
                    if DEBUG:
                        try:
                            ncount = len(frame_nals) if frame_nals else 0
                            first_sei = None
                            if ncount > 0:
                                for nn in frame_nals:
                                    if nn.get('nal_type') == 6:
                                        seih = nn.get('sei_headers') or []
                                        if seih:
                                            first_sei = seih[0].get('payload_hex')
                                            break
                            print(f"[Background TC] Frame {frame_idx}: pts={frame_pts:.6f}, NALs={ncount}, first_SEI={first_sei}")
                        except Exception as e:
                            print(f"[Background TC] Debug print error: {e}")
                    if not frame_nals:
                        continue
                    
                    # Try to extract timecode from SEI
                    sei_timecode = self._extract_timecode_from_sei(frame_nals)
                    if sei_timecode:
                        # Format display showing both raw and normalized when available
                        if isinstance(sei_timecode, dict):
                            raw = sei_timecode.get('raw')
                            norm = sei_timecode.get('normalized')
                            display_tc = f"RAW: {raw}  (NORM: {norm})" if raw and norm else (raw or norm)
                            # Append warnings if present
                            w = sei_timecode.get('warnings') if isinstance(sei_timecode, dict) else None
                            if w:
                                display_tc = f"{display_tc} [WARN: {', '.join(w)}]"
                                if DEBUG: print(f"[Background TC] Warning for frame {frame_idx}: {w}")
                        else:
                            display_tc = str(sei_timecode)

                        # Update frame data with timecode
                        frame_data_list = list(frame_data)
                        if len(frame_data_list) >= 8:
                            frame_data_list[7] = display_tc
                        else:
                            # Extend frame data if needed
                            while len(frame_data_list) < 8:
                                frame_data_list.append(None)
                            frame_data_list[7] = display_tc

                        self.current_frames_data[frame_idx] = tuple(frame_data_list)
                        updated_count += 1
                        if DEBUG:
                            print(f"[Background TC] Frame {frame_idx} updated with timecode: {display_tc}")
                            # Also show a short mapping to the cached label list if available
                            try:
                                lbl = None
                                if hasattr(self, '_thumb_timecode_labels') and frame_idx < len(self._thumb_timecode_labels):
                                    lbl = self._thumb_timecode_labels[frame_idx]
                                print(f"[Background TC] Frame {frame_idx} label_ref={'present' if lbl else 'None'}")
                            except Exception:
                                pass
                            # Schedule creation/update of thumbnail label on the UI thread
                            try:
                                self.root.after(0, lambda i=frame_idx, tc=display_tc: self._create_thumbnail_timecode_label(i, tc))
                            except Exception:
                                pass
                
                except Exception as e:
                    if DEBUG: print(f"[Background TC] Error processing frame {frame_idx}: {e}")
                    continue
            
            if updated_count > 0:
                # Refresh the thumbnail display
                self.root.after(0, self._refresh_thumbnail_timecodes)
                if DEBUG: print(f"[Background TC] Updated {updated_count} frames with SEI timecodes")
            else:
                if DEBUG: print("[Background TC] No new timecodes extracted from SEI")
        
        except Exception as e:
            if DEBUG: print(f"[Background TC] Error in SEI timecode extraction: {e}")
            import traceback
            traceback.print_exc()
    
    
    def show_frame_details(self, frame_data):
        """Show detailed information about a selected frame including PTS/DTS and audio waveform"""
        if len(frame_data) == 8:
            idx, img, pts, dts, frame_type, raw_pts, raw_dts, timecode = frame_data
        elif len(frame_data) == 7:
            idx, img, pts, dts, frame_type, raw_pts, raw_dts = frame_data
            timecode = None
        else:
            messagebox.showinfo("Info", "Frame details not available for this frame")
            return
        
        # Create or update detail window
        if not hasattr(self, 'frame_detail_window') or not self.frame_detail_window.winfo_exists():
            self.frame_detail_window = tk.Toplevel(self.root)
            self.frame_detail_window.title("Frame Details")
            self.frame_detail_window.geometry("1000x800")
            
            # Create notebook for tabs
            detail_notebook = ttk.Notebook(self.frame_detail_window)
            detail_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Tab 1: Frame Info & Audio
            info_tab = ttk.Frame(detail_notebook, padding="10")
            detail_notebook.add(info_tab, text="Info & Audio")
            
            # Frame image panel
            image_frame = ttk.Frame(info_tab, padding="5")
            image_frame.pack(fill=tk.X)
            self.detail_image_label = ttk.Label(image_frame)
            self.detail_image_label.pack()
            
            # Info panel
            info_frame = ttk.Frame(info_tab, padding="10")
            info_frame.pack(fill=tk.X)
            
            self.detail_info_text = tk.Text(info_frame, height=8, width=80)
            self.detail_info_text.pack(fill=tk.BOTH, expand=True)
            
            # Audio waveform panel
            waveform_frame = ttk.Frame(info_tab, padding="10")
            waveform_frame.pack(fill=tk.BOTH, expand=True)
            
            ttk.Label(waveform_frame, text="Audio Waveform (for this frame's time range):", font=('TkDefaultFont', 10)).pack()
            self.detail_waveform_canvas = None
            self.detail_waveform_frame = waveform_frame  # Store reference
            
            # Tab 2: NAL Units & SEI
            nal_tab = ttk.Frame(detail_notebook, padding="10")
            detail_notebook.add(nal_tab, text="NAL Units & SEI")
            
            ttk.Label(nal_tab, text="NAL Units and SEI Messages:", font=('TkDefaultFont', 10)).pack(anchor=tk.W, pady=(0, 5))
            
            # Create NAL tree for this frame
            nal_tree_frame = ttk.Frame(nal_tab)
            nal_tree_frame.pack(fill=tk.BOTH, expand=True)
            
            self.detail_nal_tree = ttk.Treeview(nal_tree_frame, 
                                                 columns=("nal_type", "nal_type_name", "size", "offset", "info"), 
                                                 show='tree headings', height=20)
            
            headings = [("nal_type", "NAL Type", 80), ("nal_type_name", "NAL Type Name", 200), 
                       ("size", "Size", 80), ("offset", "Offset", 100), ("info", "Info/Caption", 300)]
            
            for col, label, w in headings:
                self.detail_nal_tree.heading(col, text=label)
                self.detail_nal_tree.column(col, width=w)
            
            nal_scroll = ttk.Scrollbar(nal_tree_frame, orient=tk.VERTICAL, command=self.detail_nal_tree.yview)
            self.detail_nal_tree.configure(yscrollcommand=nal_scroll.set)
            self.detail_nal_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            nal_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
            
            nal_tree_frame.grid_rowconfigure(0, weight=1)
            nal_tree_frame.grid_columnconfigure(0, weight=1)
            
            # Bind double-click to show SEI details
            self.detail_nal_tree.bind("<Double-1>", self.show_detail_nal_sei_info)
            
            # Tab 3: SPS/PPS Details
            codec_tab = ttk.Frame(detail_notebook, padding="10")
            detail_notebook.add(codec_tab, text="SPS/PPS Details")
            
            ttk.Label(codec_tab, text="Codec Information (SPS/PPS):", font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
            
            # Create scrollable text widget for codec details
            codec_frame = ttk.Frame(codec_tab)
            codec_frame.pack(fill=tk.BOTH, expand=True)
            
            self.detail_codec_text = tk.Text(codec_frame, height=35, width=100, wrap=tk.WORD, font=('Courier', 9))
            codec_scroll = ttk.Scrollbar(codec_frame, orient=tk.VERTICAL, command=self.detail_codec_text.yview)
            self.detail_codec_text.configure(yscrollcommand=codec_scroll.set)
            self.detail_codec_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            codec_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
            
            codec_frame.grid_rowconfigure(0, weight=1)
            codec_frame.grid_columnconfigure(0, weight=1)
            
            # Configure text tags for formatting
            self.detail_codec_text.tag_configure('header', font=('Courier', 10, 'bold'), foreground='#0066CC')
            self.detail_codec_text.tag_configure('subheader', font=('Courier', 9, 'bold'), foreground='#2E7D32')
            self.detail_codec_text.tag_configure('field', font=('Courier', 9))
            self.detail_codec_text.tag_configure('value', font=('Courier', 9, 'bold'))
            self.detail_codec_text.tag_configure('error', foreground='#D32F2F')
            self.detail_codec_text.tag_configure('warning', foreground='#F57C00')
        
        # Update info text
        self.detail_info_text.delete(1.0, tk.END)
        
        # Format DTS and delta values
        dts_str = f"{dts:.6f}" if dts is not None else "N/A"
        dts_raw_str = str(raw_dts) if raw_dts is not None else "N/A"
        delta_str = f"{(pts - dts):.6f}" if dts is not None else "N/A"
        
        # Convert PTS to milliseconds
        pts_ms = pts * 1000 if pts is not None else None
        pts_ms_str = f"{pts_ms:.3f}" if pts_ms is not None else "N/A"
        
        # Calculate expected PCR (same as video PTS in most cases)
        pcr_ms_str = f"{pts_ms:.3f}" if pts_ms is not None else "N/A"
        
        info = f"""Frame Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Frame Number: {idx}
Frame Type: {frame_type}
{f'Timecode: {timecode}' if timecode else ''}

Video PTS (ms): {pts_ms_str}
Video PTS (seconds): {pts:.6f}
Video PTS (raw): {raw_pts if raw_pts is not None else 'N/A'}

DTS (seconds): {dts_str}
DTS (raw): {dts_raw_str}
PTS-DTS delta: {delta_str}

Expected PCR (ms): {pcr_ms_str}

Note: Audio PTS will be extracted from stream...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        self.detail_info_text.insert(1.0, info)
        
        # Render the frame image (if available)
        try:
            if img is not None:
                # Resize to a reasonable preview size while keeping aspect
                max_w, max_h = 640, 360
                w, h = img.size
                scale = min(max_w / w, max_h / h, 1.0)
                new_size = (int(w * scale), int(h * scale))
                preview_img = img if scale == 1.0 else img.resize(new_size, resample=Image.BICUBIC)
                self._detail_photo = ImageTk.PhotoImage(preview_img)
                self.detail_image_label.configure(image=self._detail_photo)
            else:
                self.detail_image_label.configure(text="No frame image available")
        except Exception as e:
            if DEBUG: print(f"Error rendering frame image: {e}")
            self.detail_image_label.configure(text="Error rendering frame image")
        
        # Extract actual Audio PTS from the stream for this frame's time range
        threading.Thread(target=self._extract_stream_timing_info, args=(pts, idx), daemon=True).start()
        
        # Find the relative index of this frame in current_frames_data
        # idx is the absolute frame number, we need the relative index (0-9 typically)
        relative_idx = None
        for i, fd in enumerate(self.current_frames_data):
            if fd[0] == idx:  # Match by absolute frame number
                relative_idx = i
                break
        
        if relative_idx is None:
            print(f"[show_frame_details] Warning: Could not find relative index for absolute frame {idx}")
            relative_idx = 0  # Fallback
        else:
            print(f"[show_frame_details] Frame {idx} (absolute) -> relative index {relative_idx}")
        
        # Populate NAL tree for this frame (use relative index)
        self._populate_frame_nal_tree(relative_idx)
        
        # Populate SPS/PPS details for this frame
        self._populate_codec_details(relative_idx, frame_type)
        
        # Extract and display audio waveform for this frame's time range
        threading.Thread(target=self._extract_frame_audio, args=(pts, idx), daemon=True).start()
        
        self.frame_detail_window.deiconify()
    
    def _extract_frame_audio(self, pts, frame_num):
        """Extract audio samples corresponding to a specific video frame time"""
        container = None
        try:
            container = av.open(self.current_file)
            
            # For MPTS, find the correct audio stream by PID
            audio_stream = None
            if self.last_report:
                audio_pid = None
                for pid, stream_info in self.last_report.get('elementary_streams', {}).items():
                    stream_type = stream_info.get('stream_type')
                    if stream_type in [0x03, 0x04, 0x0F, 0x11, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87]:
                        audio_pid = pid
                        break
                if audio_pid is not None:
                    audio_stream = next((s for s in container.streams.audio 
                                       if s.id == audio_pid), None)
                    if not audio_stream:
                        # Try alternate method by stream index
                        for pmt in self.last_report.get('pmts', {}).values():
                            for idx, stream in enumerate(pmt.get('streams', [])):
                                if stream['pid'] == audio_pid:
                                    audio_streams = list(container.streams.audio)
                                    if idx < len(audio_streams):
                                        audio_stream = audio_streams[idx]
                                    break
            
            if not audio_stream:
                audio_stream = next(iter(container.streams.audio), None)
            
            if not audio_stream:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No audio stream found"))
                return
            
            # Check for problematic audio configurations
            if audio_stream.codec_context and audio_stream.codec_context.name == 'eac3' and audio_stream.channels > 6:
                if DEBUG: print(f"Skipping frame audio extraction for {audio_stream.channels}-channel E-AC-3")
                # Silently skip - don't show error since this is called per-frame
                return
            
            # Calculate time range for this frame
            frame_duration = 1.0 / 30.0 if not self.video_stream_info else 1.0 / float(self.video_stream_info.get('average_rate', 30))
            start_time = pts
            end_time = pts + frame_duration
            
            sample_rate = audio_stream.sample_rate
            
            if DEBUG: print(f"Extracting audio for frame {frame_num}: {start_time:.6f}s to {end_time:.6f}s")
            
            # Seek to a bit before the target time to ensure we don't miss samples
            try:
                seek_time = max(0, start_time - 0.5)  # Seek 0.5s before to be safe
                seek_pts = int(seek_time * av.time_base)
                container.seek(seek_pts)
            except Exception as e:
                if DEBUG: print(f"Seek failed, reading from current position: {e}")
            
            # Extract audio samples in this time range
            samples = []
            found_any = False
            
            for packet in container.demux(audio_stream):
                try:
                    decoded_frames = list(packet.decode())
                except:
                    continue
                
                try:
                    for frame in decoded_frames:
                        try:
                            if frame.pts is not None:
                                frame_time = float(frame.pts * audio_stream.time_base)
                                
                                # Check if we've reached our time range
                                if frame_time >= start_time and frame_time < end_time:
                                    found_any = True
                                    arr = frame.to_ndarray()
                                    # Convert stereo to mono by averaging channels
                                    if len(arr.shape) > 1:
                                        arr = arr.mean(axis=0)  # Average across channels
                                    samples.extend(arr)
                                    if DEBUG: print(f"  Found audio at {frame_time:.6f}s, added {len(arr)} samples")
                                elif frame_time >= end_time:
                                    # We've passed the end time, stop processing
                                    if DEBUG: print(f"  Passed end time at {frame_time:.6f}s")
                                    break
                        except Exception as e:
                            if DEBUG: print(f"Error decoding frame audio: {e}")
                except Exception as e:
                    if DEBUG: print(f"Error processing audio packet: {e}")
                    continue
                
                # Check if we've gone past the end time
                if packet.pts and float(packet.pts * audio_stream.time_base) > end_time:
                    break
            
            if DEBUG: print(f"  Total samples collected: {len(samples)}, found_any={found_any}")
            
            if not samples:
                msg = f"No audio samples found for time range {start_time:.3f}s to {end_time:.3f}s"
                if DEBUG: print(msg)
                self.root.after(0, lambda: self._update_detail_waveform(None, msg))
                return
            
            # Generate waveform
            samples = np.array(samples, dtype=np.float32)
            
            # Normalize
            max_val = np.abs(samples).max()
            if max_val > 0:
                samples = samples / max_val
            
            fig = Figure(figsize=(7, 3), dpi=100)
            ax = fig.add_subplot(111)
            
            time_axis = np.arange(len(samples)) / sample_rate
            ax.plot(time_axis, samples, linewidth=0.8, color='#2196F3')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_title(f'Audio Waveform for Frame {frame_num} (PTS: {pts:.3f}s, {len(samples)} samples)')
            ax.set_ylim(-1.1, 1.1)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            
            self.root.after(0, lambda: self._update_detail_waveform(fig, None))
            
        except Exception as e:
            error_msg = str(e)
            if DEBUG: print(f"Error extracting frame audio: {error_msg}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda msg=error_msg: self._update_detail_waveform(None, f"Error: {msg}"))
        finally:
            if container is not None:
                try:
                    container.close()
                except:
                    pass
    
    def _extract_stream_timing_info(self, video_pts, frame_num):
        """Extract Audio PTS and PCR from the MPEG-TS stream for this frame's time range"""
        try:
            if DEBUG: print(f"Extracting stream timing info for frame {frame_num}, video PTS: {video_pts:.6f}s")
            import struct
            
            # Calculate time range for this frame
            frame_duration = 1.0 / 30.0
            if self.video_stream_info and self.video_stream_info.get('average_rate'):
                frame_duration = 1.0 / float(self.video_stream_info['average_rate'])
            
            start_time = video_pts
            end_time = video_pts + frame_duration
            
            if DEBUG: print(f"  Time range: {start_time:.6f}s to {end_time:.6f}s")
            
            audio_pts_list = []
            pcr_list = []
            
            # Open and read the TS file to extract PTS and PCR from packets
            with open(self.current_file, 'rb') as f:
                packet_num = 0
                while True:
                    packet = f.read(188)
                    if len(packet) < 188:
                        break
                    
                    # Check sync byte
                    if packet[0] != 0x47:
                        continue
                    
                    # Parse TS header
                    adaptation_field = (packet[3] & 0x20) != 0
                    payload_present = (packet[3] & 0x10) != 0
                    
                    offset = 4
                    
                    # Parse adaptation field for PCR
                    if adaptation_field:
                        adaptation_length = packet[4]
                        if adaptation_length > 0 and len(packet) > 5:
                            flags = packet[5]
                            pcr_flag = (flags & 0x10) != 0
                            
                            if pcr_flag and len(packet) >= 11:
                                # Extract PCR (33 bits base + 9 bits extension)
                                pcr_base = (packet[6] << 25) | (packet[7] << 17) | (packet[8] << 9) | (packet[9] << 1) | ((packet[10] >> 7) & 1)
                                pcr_ext = ((packet[10] & 0x01) << 8) | packet[11]
                                pcr_value = (pcr_base * 300 + pcr_ext) / 27000000.0  # Convert to seconds
                                
                                if pcr_value >= start_time and pcr_value < end_time:
                                    pcr_list.append(pcr_value * 1000)  # Convert to ms
                        
                        offset += 1 + adaptation_length
                    
                    # Parse PES header for PTS (simplified - only for audio packets)
                    if payload_present and offset < len(packet):
                        # Check if this looks like a PES packet start
                        if offset + 3 < len(packet) and packet[offset:offset+3] == b'\x00\x00\x01':
                            # This is a PES packet
                            if offset + 8 < len(packet):
                                pts_dts_flags = (packet[offset + 7] & 0xC0) >> 6
                                
                                if pts_dts_flags >= 2 and offset + 13 < len(packet):  # PTS present
                                    # Extract PTS (33 bits)
                                    pts_bytes = packet[offset + 9:offset + 14]
                                    if len(pts_bytes) >= 5:
                                        pts_raw = ((pts_bytes[0] & 0x0E) << 29) | (pts_bytes[1] << 22) | \
                                                 ((pts_bytes[2] & 0xFE) << 14) | (pts_bytes[3] << 7) | \
                                                 ((pts_bytes[4] & 0xFE) >> 1)
                                        pts_sec = pts_raw / 90000.0  # Convert to seconds
                                        
                                        if pts_sec >= start_time and pts_sec < end_time:
                                            audio_pts_list.append(pts_sec * 1000)  # Convert to ms
                    
                    packet_num += 1
                    if packet_num > 100000:  # Limit search to avoid long processing
                        break
            
            # Update the info text with actual values
            audio_pts_str = f"{audio_pts_list[0]:.3f}" if audio_pts_list else "Not found in range"
            pcr_str = f"{pcr_list[0]:.3f}" if pcr_list else "Not found in range"
            
            if DEBUG: print(f"  Found {len(audio_pts_list)} audio PTS values, {len(pcr_list)} PCR values")
            if DEBUG: print(f"  Audio PTS: {audio_pts_str}, PCR: {pcr_str}")
            
            self.root.after(0, lambda: self._update_timing_info(audio_pts_str, pcr_str))
            
        except Exception as e:
            if DEBUG: print(f"Error extracting stream timing info: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_timing_info(self, audio_pts_str, pcr_str):
        """Update the timing information in the detail window"""
        if DEBUG: print(f"Updating timing info: Audio PTS={audio_pts_str}, PCR={pcr_str}")
        
        if not hasattr(self, 'detail_info_text') or not self.detail_info_text.winfo_exists():
            if DEBUG: print("  detail_info_text not found or window closed")
            return
        
        # Get current text and update the Audio PTS line
        current_text = self.detail_info_text.get(1.0, tk.END)
        updated_text = current_text.replace(
            "Note: Audio PTS will be extracted from stream...",
            f"Audio PTS (ms): {audio_pts_str}\nActual PCR (ms): {pcr_str}"
        )
        
        self.detail_info_text.delete(1.0, tk.END)
        self.detail_info_text.insert(1.0, updated_text)
    
    def _update_detail_waveform(self, fig, error_msg):
        """Update the waveform display in the detail window"""
        if not hasattr(self, 'frame_detail_window') or not self.frame_detail_window.winfo_exists():
            return
        
        if not hasattr(self, 'detail_waveform_frame'):
            if DEBUG: print("Warning: detail_waveform_frame not found")
            return
        
        # Clear previous waveform
        if self.detail_waveform_canvas:
            self.detail_waveform_canvas.get_tk_widget().destroy()
            self.detail_waveform_canvas = None
        
        # Clear any old error labels or canvases from the waveform frame
        for child in self.detail_waveform_frame.winfo_children():
            if not isinstance(child, ttk.Label) or child.cget('text') != "Audio Waveform (for this frame's time range):":
                child.destroy()
        
        # Add new waveform or error message
        if fig:
            self.detail_waveform_canvas = FigureCanvasTkAgg(fig, master=self.detail_waveform_frame)
            self.detail_waveform_canvas.draw()
            self.detail_waveform_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        elif error_msg:
            ttk.Label(self.detail_waveform_frame, text=error_msg, foreground="red").pack()
    
    def _extract_timecode_from_sei(self, nal_list):
        """Extract timecode information from SEI messages in NAL units.
        
        Searches for:
        - SEI type 1: pic_timing (contains ctc_counter for picture timing)
        - SEI type 4: user_data_registered_itu_t_t35 (often contains ATSC/SMPTE timecode)
        - SEI type 5: user_data_unregistered (may contain custom timecode formats)
        
        Returns:
            Timecode string (HH:MM:SS:FF or HH:MM:SS;FF) or None
        """
        try:
            for nal in nal_list:
                # Only look at SEI NAL units
                if nal.get("nal_type") != 6:
                    continue
                
                # Check for SEI messages in this NAL
                sei_headers = nal.get("sei_headers", [])
                for sei in sei_headers:
                    payload_type = sei.get("type")
                    payload_hex = sei.get("payload_hex", "")
                    
                    # Type 4: ITU-T T.35 user data (often contains timecode)
                    if payload_type == 4 and len(payload_hex) >= 8:
                        try:
                            payload_bytes = bytes.fromhex(payload_hex)
                            
                            # Check for ATSC GA94 identifier (closed captions / timecode)
                            if len(payload_bytes) >= 8 and payload_bytes[0:4] == b'GA94':
                                # Check for timecode structure (typically after GA94 id)
                                if len(payload_bytes) >= 10:
                                    # ATSC timecode format at offset 4-7 or later
                                    tc_start = 4
                                    if len(payload_bytes) >= tc_start + 4:
                                        tc_data = payload_bytes[tc_start:tc_start+4]
                                        # Validate BCD-like structure
                                        if all((b & 0x0F) < 10 and ((b >> 4) & 0x0F) < 10 for b in tc_data):
                                            try:
                                                drop_frame = (tc_data[0] & 0x80) != 0
                                                hours = ((tc_data[0] & 0x3F) >> 4) * 10 + (tc_data[0] & 0x0F)
                                                minutes = ((tc_data[1] & 0x7F) >> 4) * 10 + (tc_data[1] & 0x0F)
                                                seconds = ((tc_data[2] & 0x7F) >> 4) * 10 + (tc_data[2] & 0x0F)
                                                frames = ((tc_data[3] & 0x3F) >> 4) * 10 + (tc_data[3] & 0x0F)
                                                
                                                # Validate ranges
                                                if hours < 24 and minutes < 60 and seconds < 60 and frames < 60:
                                                    separator = ';' if drop_frame else ':'
                                                    tc_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{frames:02d}"
                                                    if DEBUG: print(f"[TC Extract] Found ATSC timecode in SEI type 4: {tc_str}")
                                                    return tc_str
                                            except:
                                                pass
                            
                            # Check for SMPTE RP 188 or other timecode patterns
                            # (raw BCD timecode without GA94 prefix)
                            if len(payload_bytes) >= 4:
                                tc_data = payload_bytes[0:4]
                                if all((b & 0x0F) < 10 and ((b >> 4) & 0x0F) < 10 for b in tc_data):
                                    try:
                                        drop_frame = (tc_data[0] & 0x80) != 0
                                        hours = ((tc_data[0] & 0x3F) >> 4) * 10 + (tc_data[0] & 0x0F)
                                        minutes = ((tc_data[1] & 0x7F) >> 4) * 10 + (tc_data[1] & 0x0F)
                                        seconds = ((tc_data[2] & 0x7F) >> 4) * 10 + (tc_data[2] & 0x0F)
                                        frames = ((tc_data[3] & 0x3F) >> 4) * 10 + (tc_data[3] & 0x0F)
                                        
                                        if hours < 24 and minutes < 60 and seconds < 60 and frames < 60:
                                            separator = ';' if drop_frame else ':'
                                            tc_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{frames:02d}"
                                            if DEBUG: print(f"[TC Extract] Found BCD timecode in SEI type 4: {tc_str}")
                                            return tc_str
                                    except:
                                        pass
                        except:
                            pass
                    
                    # Type 5: Unregistered user data (may contain timecode)
                    elif payload_type == 5 and len(payload_hex) >= 8:
                        try:
                            payload_bytes = bytes.fromhex(payload_hex)
                            
                            # Check for various timecode UUID identifiers
                            # Common: SMPTE ST 12-1:2014 timecode
                            if len(payload_bytes) >= 20:  # UUID (16) + timecode (4)
                                # Try to parse last 4 bytes as BCD timecode
                                tc_data = payload_bytes[-4:]
                                if all((b & 0x0F) < 10 and ((b >> 4) & 0x0F) < 10 for b in tc_data):
                                    try:
                                        drop_frame = (tc_data[0] & 0x80) != 0
                                        hours = ((tc_data[0] & 0x3F) >> 4) * 10 + (tc_data[0] & 0x0F)
                                        minutes = ((tc_data[1] & 0x7F) >> 4) * 10 + (tc_data[1] & 0x0F)
                                        seconds = ((tc_data[2] & 0x7F) >> 4) * 10 + (tc_data[2] & 0x0F)
                                        frames = ((tc_data[3] & 0x3F) >> 4) * 10 + (tc_data[3] & 0x0F)
                                        
                                        if hours < 24 and minutes < 60 and seconds < 60 and frames < 60:
                                            separator = ';' if drop_frame else ':'
                                            tc_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{frames:02d}"
                                            if DEBUG: print(f"[TC Extract] Found timecode in SEI type 5: {tc_str}")
                                            return tc_str
                                    except:
                                        pass
                        except:
                            pass
                    
                    # Type 1: Picture timing (may contain frame timing info)
                    elif payload_type == 1:
                        # Try to use analyzer-provided detailed fields (preferred)
                        # `fields` is a list of tuples (name, value) returned by analyzer._parse_sei_payload
                        fields = sei.get('fields') or sei.get('detailed_fields') or []
                        if fields:
                            # Convert to dict for easy lookup
                            fd = {k: v for k, v in fields if isinstance(k, str)}
                            try:
                                # Look for full timestamp fields first
                                if 'hours' in fd and 'minutes' in fd and 'seconds' in fd:
                                    hours = int(str(fd.get('hours')).split()[0])
                                    minutes = int(str(fd.get('minutes')).split()[0])
                                    seconds = int(str(fd.get('seconds')).split()[0])
                                    # Frames may be in 'n_frames' or 'time_offset' (90000 units) or absent
                                    frames = None
                                    if 'n_frames' in fd:
                                        try:
                                            frames = int(str(fd.get('n_frames')).split()[0])
                                        except:
                                            frames = None
                                    if frames is None:
                                        # Try to infer frames from time_offset if available and frame rate known
                                        if 'time_offset' in fd and hasattr(self, 'video_stream_info') and self.video_stream_info and self.video_stream_info.get('average_rate'):
                                            try:
                                                # time_offset is in 90kHz units per analyzer parsing note
                                                to = int(str(fd.get('time_offset')).split()[0])
                                                fps = float(self.video_stream_info.get('average_rate'))
                                                frames = int(round((to / 90000.0) * fps))
                                            except Exception:
                                                frames = None
                                    if frames is None:
                                        frames = 0
                                    # Raw timecode string directly from parsed fields
                                    raw_hours = hours
                                    raw_minutes = minutes
                                    raw_seconds = seconds
                                    raw_frames = frames
                                    raw_str = f"{raw_hours:02d}:{raw_minutes:02d}:{raw_seconds:02d}:{raw_frames:02d}"
                                    # Normalized timecode (wrap hours to 0-23 and clamp components)
                                    try:
                                        norm_hours = raw_hours % 24
                                    except Exception:
                                        norm_hours = raw_hours if isinstance(raw_hours, int) else 0
                                    norm_minutes = raw_minutes % 60
                                    norm_seconds = raw_seconds % 60
                                    norm_frames = raw_frames % 100
                                    norm_str = f"{norm_hours:02d}:{norm_minutes:02d}:{norm_seconds:02d}:{norm_frames:02d}"
                                    # Check for field-based, discontinuity or dropped-count flags and surface warnings
                                    warnings = []
                                    try:
                                        if 'nuit_field_based_flag' in fd and int(str(fd.get('nuit_field_based_flag')).split()[0]) == 1:
                                            warnings.append('field_based')
                                    except Exception:
                                        pass
                                    try:
                                        if 'discontinuity_flag' in fd and int(str(fd.get('discontinuity_flag')).split()[0]) == 1:
                                            warnings.append('discontinuity')
                                    except Exception:
                                        pass
                                    try:
                                        if 'cnt_dropped_flag' in fd and int(str(fd.get('cnt_dropped_flag')).split()[0]) == 1:
                                            warnings.append('cnt_dropped')
                                    except Exception:
                                        pass

                                    if DEBUG: print(f"[TC Extract] Found pic_timing raw -> {raw_str}, normalized -> {norm_str}, warnings={warnings}")
                                    result = {"raw": raw_str, "normalized": norm_str}
                                    if warnings:
                                        result['warnings'] = warnings
                                    return result
                            except Exception:
                                pass
                        # Fallback: try parsing raw payload hex for 4-byte BCD anywhere
                        try:
                            payload_bytes = bytes.fromhex(payload_hex)
                            L = len(payload_bytes)
                            for i in range(0, max(1, L - 3)):
                                tc_data = payload_bytes[i:i+4]
                                if len(tc_data) < 4:
                                    continue
                                if all((b & 0x0F) < 10 and ((b >> 4) & 0x0F) < 10 for b in tc_data):
                                    try:
                                        drop_frame = (tc_data[0] & 0x80) != 0
                                        hours = ((tc_data[0] & 0x3F) >> 4) * 10 + (tc_data[0] & 0x0F)
                                        minutes = ((tc_data[1] & 0x7F) >> 4) * 10 + (tc_data[1] & 0x0F)
                                        seconds = ((tc_data[2] & 0x7F) >> 4) * 10 + (tc_data[2] & 0x0F)
                                        frames = ((tc_data[3] & 0x3F) >> 4) * 10 + (tc_data[3] & 0x0F)
                                        if hours < 24 and minutes < 60 and seconds < 60 and frames < 100:
                                            separator = ';' if drop_frame else ':'
                                            tc_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{frames:02d}"
                                            if DEBUG: print(f"[TC Extract] Found pic_timing BCD at offset {i}: {tc_str}")
                                            # Provide both raw and normalized BCD-derived timecodes
                                            raw_str = tc_str
                                            # Normalize hours modulo 24
                                            try:
                                                hh = int(tc_str[0:2]) % 24
                                                norm_str = f"{hh:02d}{tc_str[2:]}"
                                            except Exception:
                                                norm_str = tc_str
                                            return {"raw": raw_str, "normalized": norm_str}
                                    except Exception:
                                        continue
                        except Exception:
                            pass
            
            return None
            
        except Exception as e:
            if DEBUG: print(f"[TC Extract] Error extracting timecode from SEI: {e}")
            return None
    
    def _populate_frame_nal_tree(self, frame_idx):
        """Populate NAL tree for a specific frame - shows ALL NAL units (SPS, PPS, IDR, slices, SEI).
        
        Uses PTS-based lookup from ffprobe to accurately find NAL units for the decoded frame,
        independent of frame indexing logic.
        """
        if not hasattr(self, 'detail_nal_tree'):
            return
        
        # Clear existing tree
        self.detail_nal_tree.delete(*self.detail_nal_tree.get_children())
        
        # Get NAL/SEI data on-demand (parse only when needed)
        if not hasattr(self, 'last_report') or not self.last_report:
            self.detail_nal_tree.insert('', 'end', values=("-", "No analysis data available", "-", "-", "-"))
            return
        
        # Check if this is an MP4/MOV file FIRST (before checking for current_frames_data)
        file_type = self.last_report.get('file_type', 'TS')
        is_mp4_format = file_type in ['MP4/MOV', 'MP4', 'MOV']
        # Get the frame data to access PTS (needed for per-frame extraction)
        if not hasattr(self, 'current_frames_data') or frame_idx >= len(self.current_frames_data):
            self.detail_nal_tree.insert('', 'end', values=("-", f"Frame {frame_idx} data not available", "-", "-", "-"))
            return
        
        frame_data = self.current_frames_data[frame_idx]
        frame_pts = frame_data[2] if len(frame_data) > 2 else None  # PTS is at index 2
        abs_frame_num = frame_data[0] if len(frame_data) > 0 else frame_idx  # Absolute frame number in file
        
        if frame_pts is None:
            self.detail_nal_tree.insert('', 'end', values=("-", f"No PTS for frame {frame_idx}", "-", "-", "-"))
            return
        
        # Use ffprobe to get NAL units near this PTS instead of flawed frame grouping
        # Pass absolute frame number instead of relative index
        if is_mp4_format:
            # Try MP4 per-frame extraction first
            frame_nals = self._extract_mp4_nals_by_pts(frame_pts, abs_frame_num)
            # If nothing found, fall back to codec-config display
            if not frame_nals:
                self._populate_mp4_nal_tree()
                return
        else:
            frame_nals = self._extract_nals_by_pts(frame_pts, abs_frame_num)
        found = len(frame_nals) > 0
        
        if not found:
            self.detail_nal_tree.insert('', 'end', values=("-", f"No NAL units found for frame {frame_idx} (PTS: {frame_pts:.6f}s)", "-", "-", "-"))
            return
        
        # Try to extract timecode from SEI messages and update thumbnail if found
        sei_timecode = self._extract_timecode_from_sei(frame_nals)
                    
        # Debug: print NAL/SEI summary to console for verification
        try:
            if DEBUG:
                print(f"[NAL DISPLAY] Frame {frame_idx} (abs #{abs_frame_num}) - PTS={frame_pts:.6f} - NAL count={len(frame_nals)}")
                for nn in frame_nals:
                    try:
                        nal_t = nn.get('nal_type')
                        nal_name = nn.get('nal_type_name')
                        off = nn.get('offset')
                        sz = nn.get('size')
                        au = nn.get('au_index') if 'au_index' in nn else None
                        est = nn.get('estimated_pts') if 'estimated_pts' in nn else None
                        seih = nn.get('sei_headers') or []
                        print(f"  NAL: type={nal_t}({nal_name}) offset={off} size={sz} au={au} est_pts={est} sei_count={len(seih)}")
                        for si, sh in enumerate(seih[:3]):
                            try:
                                print(f"    SEI[{si}]: type={sh.get('type')} name={sh.get('type_name')} len={sh.get('length')} summary={sh.get('summary')}")
                            except Exception:
                                pass
                    except Exception:
                        pass
        except Exception:
            pass
        if sei_timecode and len(self.current_frames_data) > frame_idx:
            # Normalize the value we store/display: show both raw and normalized
            if isinstance(sei_timecode, dict):
                raw = sei_timecode.get('raw')
                norm = sei_timecode.get('normalized')
                display_tc = f"RAW: {raw}  (NORM: {norm})" if raw and norm else (raw or norm)
            else:
                display_tc = str(sei_timecode)

            # Update the frame data with extracted timecode
            frame_data_list = list(self.current_frames_data[frame_idx])
            if len(frame_data_list) >= 8:
                frame_data_list[7] = display_tc  # Timecode is at index 7
                self.current_frames_data[frame_idx] = tuple(frame_data_list)
                if DEBUG: print(f"[Frame Details] Updated frame {frame_idx} timecode from SEI: {display_tc}")
                # Refresh the thumbnail display to show new timecode
                self._refresh_thumbnail_timecodes()
        
        # Add summary node
        summary_text = f"Frame {frame_idx}: {len(frame_nals)} NAL units"
        summary_node = self.detail_nal_tree.insert('', 'end', text=summary_text,
                                                   values=("", f"{len(frame_nals)} NAL units total", "", "", ""))
        
        # Check if we have SPS/PPS info from analysis (even if not in this frame's NAL list)
        sps_info = None
        pps_count = None
        if hasattr(self, 'last_report'):
            es = self.last_report.get('elementary_streams', {})
            for es_pid, info in es.items():
                # Check for H.264 or HEVC video info
                if info.get('stream_type') in [0x1B, 0x24]:
                    # Check for H.264 SPS
                    if info.get('h264_sps'):
                        sps = info['h264_sps']
                        sps_info = f"{sps.get('profile_name', 'Unknown')} Profile @ Level {sps.get('level', '?')}"
                        if 'width' in sps and 'height' in sps:
                            resolution = f"{sps['width']}x{sps['height']}"
                            # Check for 4K
                            if sps['width'] >= 3840 or sps['height'] >= 2160:
                                resolution += " (4K)"
                            sps_info = resolution + (f" @ {sps['frame_rate']} fps" if 'frame_rate' in sps else "") + f" | {sps_info}"
                        pps = info.get('h264_pps')
                        if pps and isinstance(pps, list):
                            pps_count = len(pps)
                    # Check for HEVC SPS/VPS
                    elif 'type' in info.get('video_header', {}) and 'HEVC' in info['video_header'].get('type', ''):
                        header = info['video_header']
                        codec_info = "HEVC (H.265)"
                        if 'resolution_name' in header:
                            codec_info = f"{header['resolution_name']} | {codec_info}"
                        if header.get('is_10bit'):
                            codec_info += " 10-bit"
                        if 'profile_idc' in header:
                            codec_info += f" Profile {header['profile_idc']}"
                        if 'level_idc' in header:
                            codec_info += f" @ Level {header['level_idc']}"
                        sps_info = codec_info
                        break
                    break
        # Add SPS/PPS reference at the top if available (even if not in frame)
        if sps_info:
            ref_node = self.detail_nal_tree.insert(summary_node, 'end',
                                                   values=(7, "SPS (from stream)", "", "", sps_info))
            self.detail_nal_tree.item(ref_node, tags=('reference',))
            if pps_count is not None:
                pps_node = self.detail_nal_tree.insert(summary_node, 'end',
                                                       values=(8, "PPS (from stream)", "", "", f"{pps_count} PPS entries"))
                self.detail_nal_tree.item(pps_node, tags=('reference',))
        
        # Add each NAL unit
        for nal in frame_nals:
            # Skip unknown/invalid NAL types (type 0 is not valid in H.264)
            nal_type = nal.get("nal_type", 0)
            if nal_type == 0:
                continue
                
            nal_info = ""
            
            # Add slice type info if available
            if "slice_type_name" in nal:
                nal_info = f"Slice: {nal.get('slice_type_name')}"
            
            # Add ref_idc info for key frames
            if nal.get("nal_ref_idc", 0) > 0:
                if nal_info:
                    nal_info += f", ref_idc={nal.get('nal_ref_idc', 0)}"
                else:
                    nal_info = f"ref_idc={nal.get('nal_ref_idc', 0)}"

            node = self.detail_nal_tree.insert(summary_node, 'end', 
                                                values=(nal_type, 
                                                       nal.get("nal_type_name", "Unknown"), 
                                                       f"{nal.get('size', 0)} bytes", 
                                                       f"0x{nal.get('offset', 0):08X}", 
                                                       nal_info))

            # Add NAL header details as children
            header_node = self.detail_nal_tree.insert(node, 'end', 
                                                      values=("", "NAL Header", "", "", ""),
                                                      tags=('nal_header',))
            
            # Validate and display forbidden_zero_bit with spec check
            forbidden_zero_bit = nal.get("forbidden_zero_bit", 0)
            is_valid, violation = H264SpecValidator.validate_field(
                "forbidden_zero_bit", forbidden_zero_bit, "forbidden_zero_bit")
            fzb_tags = ('nal_detail',) if is_valid else ('spec_violation',)
            self.detail_nal_tree.insert(header_node, 'end',
                                        values=("", "forbidden_zero_bit", str(forbidden_zero_bit), 
                                               f"⚠ {violation}" if not is_valid else "", ""),
                                        tags=fzb_tags)
            
            # Validate and display nal_ref_idc with spec check
            nal_ref_idc = nal.get("nal_ref_idc", 0)
            is_valid, violation = H264SpecValidator.validate_field(
                "nal_ref_idc", nal_ref_idc, "nal_ref_idc")
            nri_tags = ('nal_detail',) if is_valid else ('spec_violation',)
            self.detail_nal_tree.insert(header_node, 'end',
                                        values=("", "nal_ref_idc", str(nal_ref_idc),
                                               f"⚠ {violation}" if not is_valid else "", ""),
                                        tags=nri_tags)
            
            # Validate and display nal_unit_type with spec check
            is_valid, violation = H264SpecValidator.validate_field(
                "nal_unit_type", nal_type, "nal_unit_type")
            nut_tags = ('nal_detail',) if is_valid else ('spec_violation',)
            self.detail_nal_tree.insert(header_node, 'end',
                                        values=("", "nal_unit_type", f"{nal_type}, '{nal.get('nal_type_name', '')}'",
                                               f"⚠ {violation}" if not is_valid else "", ""),
                                        tags=nut_tags)
            
            # Validate NAL header combination
            violations = H264SpecValidator.validate_nal_header(forbidden_zero_bit, nal_ref_idc, nal_type)
            if violations:
                for violation_msg in violations:
                    self.detail_nal_tree.insert(header_node, 'end',
                                               values=("", f"⚠ SPEC VIOLATION", violation_msg, "", ""),
                                               tags=('spec_violation_note',))

            # Add slice header details as children for slice NALs
            if nal_type in (1, 5) and nal.get("slice_header_fields"):
                for key, val in nal.get("slice_header_fields", []):
                    # Validate slice-specific fields
                    field_type = None
                    if "slice_type" in key.lower():
                        field_type = "slice_type"
                    elif "frame_num" in key.lower():
                        field_type = "frame_num"
                    elif "pic_order_cnt" in key.lower():
                        field_type = "pic_order_cnt"
                    
                    if field_type:
                        is_valid, violation = H264SpecValidator.validate_field(key, val, field_type)
                        tags = ('slice_field',) if is_valid else ('spec_violation',)
                        node_item = self.detail_nal_tree.insert(node, 'end', 
                                                    values=("", key, "", "", 
                                                           f"⚠ {violation}" if not is_valid else str(val)),
                                                    tags=tags)
                    else:
                        self.detail_nal_tree.insert(node, 'end', 
                                                    values=("", key, "", "", str(val)),
                                                    tags=('slice_field',))

            # Add SEI messages as children
            for sei_msg_idx, sei in enumerate(nal.get("sei_headers", []) if isinstance(nal.get("sei_headers", []), list) else []):
                caption_info = sei.get("summary", "")
                payload_hex = sei.get("payload_hex", "")
                payload_preview = payload_hex[:32] + ("..." if len(payload_hex) > 32 else "")

                sei_type = sei.get('type', 0)
                sei_node = self.detail_nal_tree.insert(node, 'end', 
                                                      values=(f'SEI-{sei_type}', 
                                                              sei.get("type_name", f'Type {sei_type}'), 
                                                              f"{sei.get('length', 0)} bytes",
                                                              payload_preview,
                                                              caption_info),
                                                      tags=('sei_payload', payload_hex))

                # Add SEI message header
                self.detail_nal_tree.insert(sei_node, 'end',
                                            values=("", f"<sei_message {sei_msg_idx}>", "", "", ""),
                                            tags=('sei_msg_label',))
                self.detail_nal_tree.insert(sei_node, 'end',
                                            values=("", "PayloadType", str(sei_type), "", ""),
                                            tags=('sei_detail',))
                self.detail_nal_tree.insert(sei_node, 'end',
                                            values=("", "PayloadSize", f"{sei.get('length',0)} (0x{sei.get('length',0):02X})", "", ""),
                                            tags=('sei_detail',))
                self.detail_nal_tree.insert(sei_node, 'end',
                                            values=("", sei.get("type_name", ""), "", "", ""),
                                            tags=('sei_detail',))

                # Add detailed fields if available (support both 'detailed_fields' and analyzer 'fields')
                sei_fields = sei.get("detailed_fields") or sei.get('fields') or []
                if sei_fields:
                    for field_key, field_val in sei_fields:
                        # Indent conditional/loop labels
                        if field_val == "" and ("if(" in field_key or "for(" in field_key or field_key.startswith("<")):
                            self.detail_nal_tree.insert(sei_node, 'end',
                                                       values=("", field_key, "", "", ""),
                                                       tags=('sei_label',))
                        else:
                            # Validate SEI-specific fields
                            field_type = None
                            if "pic_struct" in field_key.lower():
                                field_type = "pic_struct"
                            elif "cc_valid" in field_key.lower():
                                field_type = "cc_valid"
                            elif "cc_type" in field_key.lower():
                                field_type = "cc_type"
                            elif "one_bit" in field_key.lower():
                                field_type = "one_bit"
                            elif "reserved" in field_key.lower():
                                field_type = "reserved"

                            if field_type:
                                is_valid, violation = H264SpecValidator.validate_field(
                                    field_key, field_val, field_type)
                                tags = ('sei_detail',) if is_valid else ('spec_violation',)
                                self.detail_nal_tree.insert(sei_node, 'end',
                                                           values=("", field_key, str(field_val), 
                                                                  f"⚠ {violation}" if not is_valid else "", ""),
                                                           tags=tags)
                            else:
                                self.detail_nal_tree.insert(sei_node, 'end',
                                                           values=("", field_key, str(field_val), "", ""),
                                                           tags=('sei_detail',))

        print(f"[_populate_frame_nal_tree] Tree populated with {len(frame_nals)} NAL units")
        
        # Configure tag colors
        self.detail_nal_tree.tag_configure('reference', foreground='#2196F3')
        self.detail_nal_tree.tag_configure('nal_header', foreground='#424242', font=('TkDefaultFont', 9, 'bold'))
        self.detail_nal_tree.tag_configure('nal_detail', foreground='#616161')
        self.detail_nal_tree.tag_configure('slice_field', foreground='#455A64')
        self.detail_nal_tree.tag_configure('sei_msg_label', foreground='#0277BD', font=('TkDefaultFont', 9, 'bold'))
        self.detail_nal_tree.tag_configure('sei_label', foreground='#0277BD')
        self.detail_nal_tree.tag_configure('sei_detail', foreground='#616161')
        
        # SPEC VIOLATION TAGS - Red highlight for H.264 non-compliance
        self.detail_nal_tree.tag_configure('spec_violation', foreground='#D32F2F', background='#FFEBEE', font=('TkDefaultFont', 9, 'bold'))
        self.detail_nal_tree.tag_configure('spec_violation_note', foreground='#C62828', background='#FFCDD2', font=('TkDefaultFont', 8, 'italic'))
        
        # Expand the summary node
        self.detail_nal_tree.item(summary_node, open=True)
        print(f"[_populate_frame_nal_tree] Completed successfully")
    
    def _populate_mp4_nal_tree(self):
        """Populate NAL tree for MP4/MOV files - shows SPS/PPS from codec config.
        
        MP4 files don't have frame-by-frame NAL extraction like TS files.
        Instead, we show the codec configuration (SPS/PPS) extracted from the MP4 tracks.
        """
        if not hasattr(self, 'detail_nal_tree'):
            return
        
        if not hasattr(self, 'last_report') or not self.last_report:
            self.detail_nal_tree.insert('', 'end', values=("-", "No analysis data available", "-", "-", "-"))
            return
        
        es = self.last_report.get('elementary_streams', {})
        
        if not es:
            self.detail_nal_tree.insert('', 'end', values=("-", "No elementary streams found", "-", "-", "-"))
            return
        
        # Add header
        header_node = self.detail_nal_tree.insert('', 'end', text="MP4 Codec Configuration NALs",
                                                  values=("", "Extracted from codec config", "", "", ""))
        
        # Display NALs from each track
        for track_key, info in sorted(es.items()):
            track_id = info.get('track_id', track_key.replace('track_', ''))
            codec = info.get('codec', 'Unknown')
            nal_count = info.get('nal_count', 0)
            
            # Create track node
            track_text = f"Track {track_id}: {codec}"
            track_node = self.detail_nal_tree.insert(header_node, 'end', text=track_text,
                                                     values=("", f"{nal_count} NALs from config", "", "", ""))
            
            # Add SPS if available
            if 'h264_sps' in info:
                sps = info['h264_sps']
                sps_text = f"SPS (NAL type 7): {sps.get('width', '?')}x{sps.get('height', '?')} @ {sps.get('frame_rate', '?')} fps"
                sps_info = f"{sps.get('profile_name', 'Unknown')} Profile Level {sps.get('level', '?')}"
                self.detail_nal_tree.insert(track_node, 'end',
                                          values=(7, "SPS", sps.get('width', 0), sps.get('height', 0), sps_info))
            
            # Add PPS if available
            if info.get('h264_pps_found'):
                self.detail_nal_tree.insert(track_node, 'end',
                                          values=(8, "PPS", "-", "-", "Picture Parameter Set"))
            
            # Add HEVC info if available
            if 'hevc_sps' in info:
                sps = info['hevc_sps']
                sps_text = f"HEVC SPS: {sps.get('width', '?')}x{sps.get('height', '?')}"
                self.detail_nal_tree.insert(track_node, 'end',
                                          values=(33, "HEVC SPS", sps.get('width', 0), sps.get('height', 0), sps_text))
            
            if info.get('hevc_vps_found'):
                self.detail_nal_tree.insert(track_node, 'end',
                                          values=(32, "HEVC VPS", "-", "-", "Video Parameter Set"))
            
            if info.get('hevc_pps_found'):
                self.detail_nal_tree.insert(track_node, 'end',
                                          values=(34, "HEVC PPS", "-", "-", "Picture Parameter Set"))
        
        # Expand the header node
        self.detail_nal_tree.item(header_node, open=True)
        print(f"[_populate_mp4_nal_tree] Completed successfully")
    
    def _extract_nals_by_pts(self, target_pts, frame_idx):
        """Extract NAL units for a frame using PTS-based lookup instead of flawed frame grouping.
        
        Uses a cache to store NAL data for the current visible frame range to avoid
        re-extracting when navigating. Cache is cleared when extracting a new range.
        
        This method finds all NAL units (SPS, PPS, slices, SEI) that belong to a specific video frame
        by extracting them from the TS stream using the frame's PTS as reference.
        
        Args:
            target_pts: PTS timestamp of the frame in seconds
            frame_idx: Frame index for reference
            
        Returns:
            List of NAL unit dictionaries containing all NAL/SEI data for this frame
        """
        try:
            if not hasattr(self, 'analyzer') or not self.analyzer:
                return []
            
            # Find H.264 video PID
            h264_pid = None
            for pid, codec_type in self.analyzer.video_pids.items():
                if codec_type == 0x1B:  # H.264
                    h264_pid = pid
                    break
            
            if not h264_pid:
                return []
            
            # Check if we have cached NAL data
            if not hasattr(self, '_nal_cache'):
                self._nal_cache = {}
            
            # Use cache key to identify if this is the same frame window
            cache_key = f"frame_{frame_idx}"
            
            # If we have cached data for this frame, use it
            if cache_key in self._nal_cache:
                return self._nal_cache[cache_key]
            
            # Extract all NAL units from the entire video stream (no limits)
            # This is done once per visible window, then cached
            if not hasattr(self, '_all_nals_unlimited') or not self._all_nals_unlimited:
                # Check buffer size before extraction
                if h264_pid in self.analyzer.video_pes_buffers:
                    buffer_size = len(self.analyzer.video_pes_buffers[h264_pid])
                else:
                    pass
                self._all_nals_unlimited = self.analyzer.extract_nal_sei_unlimited(h264_pid)
                if not self._all_nals_unlimited:
                    return []
                
                # Check for SPS/PPS in extracted NALs
                sps_count = sum(1 for nal in self._all_nals_unlimited if nal.get('nal_type') == 7)
                pps_count = sum(1 for nal in self._all_nals_unlimited if nal.get('nal_type') == 8)
                if sps_count > 0:
                    # Show where the first SPS appears
                    for i, nal in enumerate(self._all_nals_unlimited[:100]):  # Check first 100 NALs
                        if nal.get('nal_type') == 7:
                            break
            
            # Group NALs by frame access unit (AUD or slice boundary)
            if not hasattr(self, '_frame_nals_grouped') or self._frame_nals_grouped is None:
                self._frame_nals_grouped = self._group_nals_by_frame_correct(self._all_nals_unlimited)
                if self._frame_nals_grouped is None or len(self._frame_nals_grouped) == 0:
                    return []
                # Count slices in grouped data for debugging
                slice_count = 0
                for group in self._frame_nals_grouped:
                    if any(nal["nal_type"] in (1, 5) for nal in group):
                        slice_count += 1
            
            frame_nals_list = self._frame_nals_grouped
            
            # Verify frame_nals_list is not None before iterating
            if frame_nals_list is None or len(frame_nals_list) == 0:
                return []
            
            # Direct index lookup - frame_idx is now the absolute frame number in the file
            # which matches the index in frame_nals_list
            print(f"\n[NAL Extract] Looking for frame {frame_idx}, grouped list has {len(frame_nals_list)} frames")
            if frame_idx < len(frame_nals_list):
                nal_group = frame_nals_list[frame_idx]
                if nal_group:
                    # Show detailed NAL info for debugging
                    first_offset = nal_group[0].get('offset', 'N/A') if len(nal_group) > 0 else 'N/A'
                    for i, n in enumerate(nal_group):
                        nal_type = n.get('nal_type', '?')
                        nal_name = n.get('nal_type_name', 'Unknown')
                        nal_size = n.get('size', 0)
                        nal_offset = n.get('offset', 0)
                        print(f"    NAL {i}: Type {nal_type:2} ({nal_name:15s}) Size {nal_size:6} bytes at 0x{nal_offset:08X}")
                    # Cache this result
                    self._nal_cache[cache_key] = nal_group
                    return nal_group
                else:
            
            
            # Fallback: try to find by counting slices only
                    pass
            slice_frame_count = 0
            for idx, nal_group in enumerate(frame_nals_list):
                has_slice = any(nal["nal_type"] in (1, 5) for nal in nal_group)
                if has_slice:
                    if DEBUG: print(f"  Group {idx}: slice frame #{slice_frame_count}")
                    if slice_frame_count == frame_idx:
                        self._nal_cache[cache_key] = nal_group
                        return nal_group
                    slice_frame_count += 1
            
            return []
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return []

    def _split_mp4_sample_to_nals(self, sample_bytes: bytes, length_size: int, codec_type: str = 'H.264'):
        """Split an MP4/H264 or HEVC sample (length-prefixed) into NAL units.

        Returns list of dicts: { 'nal_type': int, 'data': bytes, 'size': int }
        """
        nals = []
        pos = 0
        total = len(sample_bytes)
        try:
            while pos + length_size <= total:
                nlen = int.from_bytes(sample_bytes[pos:pos+length_size], 'big')
                pos += length_size
                if nlen <= 0 or pos + nlen > total:
                    # malformed or no more data
                    break
                nalu = sample_bytes[pos:pos+nlen]
                pos += nlen
                if not nalu:
                    continue
                if codec_type == 'H.264':
                    nal_header = nalu[0]
                    nal_type = nal_header & 0x1F
                else:
                    # HEVC nal unit header: nal_unit_type is bits 1..6 of first byte
                    nal_header = nalu[0]
                    nal_type = (nal_header >> 1) & 0x3F
                nals.append({'nal_type': nal_type, 'data': nalu, 'size': len(nalu)})
        except Exception:
            return nals
        return nals

    def _extract_mp4_nals_by_pts(self, target_pts, frame_idx):
        """On-demand MP4 per-frame NAL extraction using PyAV.

        Finds the packet/frame closest to target_pts and extracts length-prefixed NALs
        using the track's codec configuration (avcC/hvcC).
        """
        try:
            if not AV_AVAILABLE:
                return []

            if not hasattr(self, 'last_report') or not self.last_report:
                return []

            report = self.last_report
            tracks = report.get('tracks', {}) or {}
            video_tracks = report.get('video_tracks', []) or []
            if not video_tracks:
                return []

            # Use first video stream for extraction
            track_id = video_tracks[0]
            track_info = tracks.get(track_id) if isinstance(tracks, dict) else tracks
            if not track_info:
                return []

            codec_type = track_info.get('codec_type', 'H.264')
            codec_config = track_info.get('codec_config')

            # Determine lengthSize for avcC (H.264); default to 4
            length_size = 4
            if codec_config and codec_type == 'H.264' and len(codec_config) >= 5:
                length_size_minus_one = codec_config[4] & 0x3
                length_size = length_size_minus_one + 1

            # Use cache if available
            if not hasattr(self, '_mp4_nal_cache'):
                self._mp4_nal_cache = {}
            cache_key = f"mp4_frame_{frame_idx}"
            if cache_key in self._mp4_nal_cache:
                return self._mp4_nal_cache[cache_key]

            # Open container and search for packet/frame close to target_pts
            container = None
            try:
                container = av.open(self.current_file)
                video_stream = next((s for s in container.streams if s.type == 'video'), None)
                if not video_stream:
                    return []

                # Demux packets and look for packet/frame with matching PTS (within small epsilon)
                closest_packet = None
                closest_diff = None
                for packet in container.demux(video_stream):
                    try:
                        # packet.pts may be None for some containers; compute seconds if available
                        if packet.pts is None:
                            continue
                        pkt_time = float(packet.pts * packet.time_base)
                        diff = abs(pkt_time - float(target_pts))
                        if closest_diff is None or diff < closest_diff:
                            closest_diff = diff
                            closest_packet = packet
                        # early stop if very close
                        if diff < 0.001:
                            break
                    except Exception:
                        continue

                if not closest_packet:
                    return []

                sample_bytes = bytes(closest_packet)
                raw_nals = self._split_mp4_sample_to_nals(sample_bytes, length_size, codec_type=codec_type)
                # Enrich to match TS-style NAL dicts expected by GUI
                enriched = []
                for n in raw_nals:
                    nal_type = n.get('nal_type')
                    data = n.get('data')
                    size = n.get('size', len(data) if data else 0)
                    # Default values
                    nal_type_name = f"NAL {nal_type}"
                    if codec_type == 'H.264':
                        if nal_type == 6:
                            nal_type_name = 'SEI'
                        elif nal_type == 7:
                            nal_type_name = 'SPS'
                        elif nal_type == 8:
                            nal_type_name = 'PPS'
                        elif nal_type == 5:
                            nal_type_name = 'IDR'
                        elif nal_type == 1:
                            nal_type_name = 'Non-IDR'
                    else:
                        # HEVC mapping (partial)
                        if nal_type == 32:
                            nal_type_name = 'VPS'
                        elif nal_type == 33:
                            nal_type_name = 'SPS'
                        elif nal_type == 34:
                            nal_type_name = 'PPS'
                        elif nal_type == 39:
                            nal_type_name = 'SEI'

                    # Parse header bits where possible
                    forbidden_zero_bit = 0
                    nal_ref_idc = 0
                    try:
                        if data and len(data) > 0 and codec_type == 'H.264':
                            hdr = data[0]
                            forbidden_zero_bit = (hdr >> 7) & 0x1
                            nal_ref_idc = (hdr >> 5) & 0x3
                    except Exception:
                        forbidden_zero_bit = 0
                        nal_ref_idc = 0

                    enriched.append({
                        'nal_type': nal_type,
                        'nal_type_name': nal_type_name,
                        'data': data,
                        'size': size,
                        'offset': 0,
                        'forbidden_zero_bit': forbidden_zero_bit,
                        'nal_ref_idc': nal_ref_idc,
                    })

                    # Parse SEI payloads into headers if this is an SEI NAL and analyzer is available
                    if nal_type == 6 and data and hasattr(self, 'analyzer') and self.analyzer:
                        try:
                            rbsp = self.analyzer._remove_emulation_prevention(data[1:])
                            sei_pos = 0
                            sei_headers = []
                            current_sps = None
                            # Attempt to get current SPS from report if available
                            try:
                                es = (self.last_report or {}).get('elementary_streams', {})
                                for k, info in (es.items() if isinstance(es, dict) else []):
                                    if info.get('h264_sps'):
                                        current_sps = info.get('h264_sps')
                                        break
                            except Exception:
                                current_sps = None

                            while sei_pos + 2 <= len(rbsp):
                                payload_type = 0
                                type_iter = 0
                                while sei_pos < len(rbsp) and rbsp[sei_pos] == 0xFF and type_iter < 1000:
                                    payload_type += 255
                                    sei_pos += 1
                                    type_iter += 1
                                if sei_pos >= len(rbsp):
                                    break
                                payload_type += rbsp[sei_pos]
                                sei_pos += 1
                                payload_size = 0
                                size_iter = 0
                                while sei_pos < len(rbsp) and rbsp[sei_pos] == 0xFF and size_iter < 1000:
                                    payload_size += 255
                                    sei_pos += 1
                                    size_iter += 1
                                if sei_pos >= len(rbsp):
                                    break
                                payload_size += rbsp[sei_pos]
                                sei_pos += 1
                                payload_end = sei_pos + payload_size
                                if payload_end > len(rbsp):
                                    break
                                payload = rbsp[sei_pos:payload_end]
                                sei_type_name = self.analyzer.get_sei_type_name(payload_type) if hasattr(self.analyzer, 'get_sei_type_name') else f"SEI type {payload_type}"
                                sei_summary = self.analyzer.summarize_sei(payload_type, payload) if hasattr(self.analyzer, 'summarize_sei') else ''
                                sei_entry = {
                                    'type': payload_type,
                                    'type_name': sei_type_name,
                                    'length': payload_size,
                                    'payload_hex': payload.hex(),
                                    'summary': sei_summary,
                                }
                                # Try to get detailed fields using analyzer helper
                                try:
                                    detailed = self.analyzer._parse_sei_payload(payload_type, payload, current_sps)
                                    if detailed:
                                        sei_entry['detailed_fields'] = detailed
                                except Exception:
                                    pass
                                sei_headers.append(sei_entry)
                                sei_pos = payload_end

                            if sei_headers:
                                enriched[-1]['sei_headers'] = sei_headers
                        except Exception:
                            pass

                self._mp4_nal_cache[cache_key] = enriched
                return enriched
            finally:
                if container is not None:
                    try:
                        container.close()
                    except:
                        pass
        except Exception as e:
            if DEBUG: print(f"[_extract_mp4_nals_by_pts] Error: {e}")
            return []

    def _display_nal_group_in_tree(self, nal_group, frame_idx):
        """Helper to display a list of NAL dicts in the detail tree."""
        if not hasattr(self, 'detail_nal_tree') or not self.detail_nal_tree:
            return

        # Add summary node
        summary_text = f"Frame {frame_idx}: {len(nal_group)} NAL units"
        summary_node = self.detail_nal_tree.insert('', 'end', text=summary_text,
                                                   values=("", f"{len(nal_group)} NAL units total", "", "", ""))

        for i, n in enumerate(nal_group):
            nal_type = n.get('nal_type')
            size = n.get('size', 0)
            nal_name = f"NAL {nal_type}"
            if nal_type == 6:
                nal_name = "SEI"
            elif nal_type == 7:
                nal_name = "SPS"
            elif nal_type == 8:
                nal_name = "PPS"
            self.detail_nal_tree.insert(summary_node, 'end', values=(nal_type, nal_name, size, "-", ""))
        self.detail_nal_tree.item(summary_node, open=True)
    
    def _group_nals_by_frame_correct(self, nal_list):
        """Group NAL units by frame access units using correct logic.
        
        A frame (access unit) in H.264 consists of:
        - Optional: AUD (type 9) marks the start of a new access unit
        - Optional: SPS (type 7), PPS (type 8) before IDR frames
        - Optional: SEI (type 6) before/after slices
        - Required: One or more slices (type 1 or 5)
        - Optional: Filler data (type 12)
        
        Frame boundary detection:
        1. AUD (type 9) always starts a new access unit
        2. If no AUD, use first_mb_in_slice == 0 to detect new frame
        3. SPS/PPS before an IDR also indicate frame boundary
        
        Returns:
            List of frame groups, where each group contains all NALs for that frame.
        """
        if not nal_list:
            return []

        # If NAL entries already include an `au_index`, prefer grouping by that
        have_au = any(('au_index' in n and n.get('au_index') is not None) for n in nal_list)
        if have_au:
            groups = {}
            for n in nal_list:
                a = n.get('au_index')
                if a is None:
                    # place unknowns in -1 bucket preserving order
                    a = -1
                groups.setdefault(a, []).append(n)
            # Sort groups by AU index ascending and within group by offset
            grouped_list = []
            for a in sorted(groups.keys()):
                grp = groups[a]
                try:
                    grp.sort(key=lambda x: (x.get('offset') if x.get('offset') is not None else 0))
                except Exception:
                    pass
                grouped_list.append(grp)
            return grouped_list

        frames = []
        current_frame = []

        for nal in nal_list:
            nal_type = nal["nal_type"]
            
            # AUD (type 9) always marks the start of a new access unit
            if nal_type == 9:
                # Save previous frame if it has content
                if current_frame:
                    frames.append(current_frame)
                # Start new frame with this AUD
                current_frame = [nal]
                continue
            
            # SPS (7) or PPS (8) before an IDR slice indicate a new frame is starting
            # This handles streams without AUD
            if nal_type in (7, 8) and current_frame:
                # Check if current frame already has slices - if so, this is a new frame
                has_slices = any(n["nal_type"] in (1, 5) for n in current_frame)
                if has_slices:
                    frames.append(current_frame)
                    current_frame = [nal]
                    continue

            # If this is a slice (1 non-IDR or 5 IDR), try to detect access unit boundary
            # by checking first_mb_in_slice when available from earlier parsing.
            if nal_type in (1, 5):
                # Try to obtain first_mb_in_slice from parsed slice header fields
                first_mb = None
                sh_fields = nal.get('slice_header_fields') or []
                for fname, fval in sh_fields:
                    if fname == 'first_mb_in_slice':
                        try:
                            first_mb = int(fval)
                        except Exception:
                            # fval may be string; attempt to parse digits
                            try:
                                first_mb = int(''.join(ch for ch in str(fval) if ch.isdigit()))
                            except Exception:
                                first_mb = None
                        break

                # If first_mb is present and equals 0, this starts a new access unit
                if first_mb is not None:
                    if first_mb == 0 and current_frame:
                        frames.append(current_frame)
                        current_frame = [nal]
                        continue
            
            # Add NAL to current frame
            current_frame.append(nal)
        
        # Add final frame
        if current_frame:
            frames.append(current_frame)
        
        return frames
    
    def _populate_codec_details(self, frame_idx, frame_type):
        """Populate detailed SPS and PPS information in the codec tab for a specific frame"""
        if not hasattr(self, 'detail_codec_text'):
            return
        
        # Clear existing content
        self.detail_codec_text.delete(1.0, tk.END)
        self.detail_codec_text.config(state=tk.NORMAL)
        
        if not hasattr(self, 'last_report') or not self.last_report:
            self.detail_codec_text.insert(tk.END, "No analysis data available\n")
            return
        
        # Add frame-specific header
        self.detail_codec_text.insert(tk.END, "═" * 80 + "\n", 'header')
        self.detail_codec_text.insert(tk.END, f"  CODEC INFORMATION FOR FRAME {frame_idx} ({frame_type} Frame)\n", 'header')
        self.detail_codec_text.insert(tk.END, "═" * 80 + "\n\n", 'header')
        
        # Get elementary stream info
        es = self.last_report.get('elementary_streams', {})
        
        found_codec = False
        for pid, info in es.items():
            stream_type = info.get('stream_type')
            
            # Check for H.264 stream
            if stream_type == 0x1B and info.get('h264_sps'):
                found_codec = True
                sps = info['h264_sps']
                
                # SPS Header
                self.detail_codec_text.insert(tk.END, "═" * 80 + "\n", 'header')
                self.detail_codec_text.insert(tk.END, f"  H.264 SEQUENCE PARAMETER SET (SPS) - PID 0x{pid:04X}\n", 'header')
                self.detail_codec_text.insert(tk.END, "═" * 80 + "\n\n", 'header')
                
                # Basic Profile/Level
                self.detail_codec_text.insert(tk.END, "▶ Profile and Level\n", 'subheader')
                self.detail_codec_text.insert(tk.END, f"  Profile IDC:           {sps.get('profile_idc', 'N/A')} ({sps.get('profile_name', 'Unknown')})\n", 'field')
                self.detail_codec_text.insert(tk.END, f"  Constraint Flags:      0x{sps.get('constraint_flags', 0):02X}\n", 'field')
                self.detail_codec_text.insert(tk.END, f"  Level IDC:             {sps.get('level_idc', 'N/A')} (Level {sps.get('level', 'N/A')})\n", 'field')
                self.detail_codec_text.insert(tk.END, f"  NAL Ref IDC:           {sps.get('nal_ref_idc', 'N/A')}\n\n", 'field')
                
                # Resolution
                if 'width' in sps and 'height' in sps:
                    self.detail_codec_text.insert(tk.END, "▶ Video Resolution\n", 'subheader')
                    self.detail_codec_text.insert(tk.END, f"  Width:                 {sps['width']} pixels\n", 'field')
                    self.detail_codec_text.insert(tk.END, f"  Height:                {sps['height']} pixels\n", 'field')
                    self.detail_codec_text.insert(tk.END, f"  Aspect Ratio:          {sps['width']}:{sps['height']}\n\n", 'field')
                
                # Frame Rate
                if 'frame_rate' in sps:
                    self.detail_codec_text.insert(tk.END, "▶ Frame Rate (from VUI timing_info)\n", 'subheader')
                    self.detail_codec_text.insert(tk.END, f"  Frame Rate:            ", 'field')
                    self.detail_codec_text.insert(tk.END, f"{sps['frame_rate']:.3f} fps\n\n", 'value')
                
                # VUI Parameters (if available in SPS)
                if any(k.startswith('vui_') for k in sps.keys()):
                    self.detail_codec_text.insert(tk.END, "▶ VUI Parameters\n", 'subheader')
                    
                    # Aspect Ratio
                    if 'vui_aspect_ratio_idc' in sps:
                        aspect_names = {
                            1: "1:1 (Square)", 2: "12:11", 3: "10:11", 4: "16:11", 
                            5: "40:33", 6: "24:11", 7: "20:11", 8: "32:11",
                            9: "80:33", 10: "18:11", 11: "15:11", 12: "64:33",
                            13: "160:99", 14: "4:3", 15: "3:2", 16: "2:1", 255: "Extended SAR"
                        }
                        ar_idc = sps.get('vui_aspect_ratio_idc')
                        ar_name = aspect_names.get(ar_idc, f"Reserved ({ar_idc})")
                        self.detail_codec_text.insert(tk.END, f"  aspect_ratio_idc:      {ar_idc} ({ar_name})\n", 'field')
                        if ar_idc == 255 and 'vui_sar_width' in sps:
                            self.detail_codec_text.insert(tk.END, f"  SAR Width:             {sps['vui_sar_width']}\n", 'field')
                            self.detail_codec_text.insert(tk.END, f"  SAR Height:            {sps['vui_sar_height']}\n", 'field')
                    
                    # Video Signal Type
                    if 'vui_video_format' in sps:
                        video_formats = {
                            0: "Component", 1: "PAL", 2: "NTSC", 3: "SECAM",
                            4: "MAC", 5: "Unspecified", 6: "Reserved", 7: "Reserved"
                        }
                        vf = sps.get('vui_video_format')
                        self.detail_codec_text.insert(tk.END, f"  video_format:          {vf} ({video_formats.get(vf, 'Unknown')})\n", 'field')
                    
                    if 'vui_video_full_range_flag' in sps:
                        vfr = sps.get('vui_video_full_range_flag')
                        range_str = "Full Range (0-255)" if vfr else "Limited Range (16-235)"
                        self.detail_codec_text.insert(tk.END, f"  video_full_range:      {vfr} ({range_str})\n", 'field')
                    
                    # Color Description
                    if 'vui_colour_primaries' in sps:
                        primaries_names = {
                            1: "BT.709", 2: "Unspecified", 4: "BT.470M", 5: "BT.470BG",
                            6: "SMPTE 170M", 7: "SMPTE 240M", 8: "FILM", 9: "BT.2020",
                            10: "SMPTE ST 428", 11: "DCI-P3", 12: "Display P3"
                        }
                        cp = sps.get('vui_colour_primaries')
                        self.detail_codec_text.insert(tk.END, f"  colour_primaries:      {cp} ({primaries_names.get(cp, 'Unknown')})\n", 'field')
                    
                    if 'vui_transfer_characteristics' in sps:
                        transfer_names = {
                            1: "BT.709", 2: "Unspecified", 4: "Gamma 2.2", 5: "Gamma 2.8",
                            6: "SMPTE 170M", 7: "SMPTE 240M", 8: "Linear", 9: "Logarithmic (100:1)",
                            10: "Logarithmic (316:1)", 11: "xvYCC", 12: "BT.1361", 13: "sRGB",
                            14: "BT.2020 (10-bit)", 15: "BT.2020 (12-bit)", 16: "SMPTE ST 2084 (PQ)",
                            17: "SMPTE ST 428", 18: "ARIB STD-B67 (HLG)"
                        }
                        tc = sps.get('vui_transfer_characteristics')
                        self.detail_codec_text.insert(tk.END, f"  transfer_char:         {tc} ({transfer_names.get(tc, 'Unknown')})\n", 'field')
                    
                    if 'vui_matrix_coefficients' in sps:
                        matrix_names = {
                            0: "RGB/Identity", 1: "BT.709", 2: "Unspecified", 4: "FCC",
                            5: "BT.470BG", 6: "SMPTE 170M", 7: "SMPTE 240M", 8: "YCgCo",
                            9: "BT.2020 Non-constant", 10: "BT.2020 Constant"
                        }
                        mc = sps.get('vui_matrix_coefficients')
                        self.detail_codec_text.insert(tk.END, f"  matrix_coefficients:   {mc} ({matrix_names.get(mc, 'Unknown')})\n", 'field')
                    
                    # Timing Info
                    if 'vui_num_units_in_tick' in sps:
                        self.detail_codec_text.insert(tk.END, f"  num_units_in_tick:     {sps['vui_num_units_in_tick']}\n", 'field')
                    if 'vui_time_scale' in sps:
                        self.detail_codec_text.insert(tk.END, f"  time_scale:            {sps['vui_time_scale']}\n", 'field')
                    if 'vui_fixed_frame_rate_flag' in sps:
                        ffr = sps.get('vui_fixed_frame_rate_flag')
                        self.detail_codec_text.insert(tk.END, f"  fixed_frame_rate:      {ffr} ({'Yes' if ffr else 'No'})\n", 'field')
                    
                    self.detail_codec_text.insert(tk.END, "\n")
                
                # Additional SPS fields
                if 'seq_parameter_set_id' in sps:
                    self.detail_codec_text.insert(tk.END, "▶ Additional SPS Parameters\n", 'subheader')
                    self.detail_codec_text.insert(tk.END, f"  seq_parameter_set_id:  {sps.get('seq_parameter_set_id')}\n", 'field')
                    if 'chroma_format_idc' in sps:
                        chroma_names = {0: "Monochrome", 1: "4:2:0", 2: "4:2:2", 3: "4:4:4"}
                        cf = sps.get('chroma_format_idc')
                        self.detail_codec_text.insert(tk.END, f"  chroma_format_idc:     {cf} ({chroma_names.get(cf, 'Unknown')})\n", 'field')
                    if 'bit_depth_luma' in sps:
                        self.detail_codec_text.insert(tk.END, f"  bit_depth_luma:        {sps['bit_depth_luma']} bits\n", 'field')
                    if 'bit_depth_chroma' in sps:
                        self.detail_codec_text.insert(tk.END, f"  bit_depth_chroma:      {sps['bit_depth_chroma']} bits\n", 'field')
                    if 'log2_max_frame_num' in sps:
                        self.detail_codec_text.insert(tk.END, f"  log2_max_frame_num:    {sps['log2_max_frame_num']}\n", 'field')
                    if 'pic_order_cnt_type' in sps:
                        self.detail_codec_text.insert(tk.END, f"  pic_order_cnt_type:    {sps['pic_order_cnt_type']}\n", 'field')
                    if 'max_num_ref_frames' in sps:
                        self.detail_codec_text.insert(tk.END, f"  max_num_ref_frames:    {sps['max_num_ref_frames']}\n", 'field')
                    self.detail_codec_text.insert(tk.END, "\n")
                
                # Errors and Warnings
                if sps.get('errors'):
                    self.detail_codec_text.insert(tk.END, "▶ Errors\n", 'subheader')
                    for err in sps['errors']:
                        self.detail_codec_text.insert(tk.END, f"  ✗ {err}\n", 'error')
                    self.detail_codec_text.insert(tk.END, "\n")
                
                if sps.get('warnings'):
                    self.detail_codec_text.insert(tk.END, "▶ Warnings\n", 'subheader')
                    for warn in sps['warnings']:
                        self.detail_codec_text.insert(tk.END, f"  ⚠ {warn}\n", 'warning')
                    self.detail_codec_text.insert(tk.END, "\n")
            
            # Check for H.264 PPS
            if stream_type == 0x1B and info.get('h264_pps'):
                pps = info['h264_pps']
                
                # PPS Header
                self.detail_codec_text.insert(tk.END, "═" * 80 + "\n", 'header')
                self.detail_codec_text.insert(tk.END, f"  H.264 PICTURE PARAMETER SET (PPS) - PID 0x{pid:04X}\n", 'header')
                self.detail_codec_text.insert(tk.END, "═" * 80 + "\n\n", 'header')
                
                self.detail_codec_text.insert(tk.END, "▶ Basic Information\n", 'subheader')
                self.detail_codec_text.insert(tk.END, f"  NAL Unit Type:         {pps.get('nal_unit_type', 'N/A')}\n", 'field')
                self.detail_codec_text.insert(tk.END, f"  NAL Ref IDC:           {pps.get('nal_ref_idc', 'N/A')}\n\n", 'field')
                
                # PPS-specific fields (if parsed)
                if 'pic_parameter_set_id' in pps:
                    self.detail_codec_text.insert(tk.END, "▶ PPS Parameters\n", 'subheader')
                    self.detail_codec_text.insert(tk.END, f"  pic_parameter_set_id:  {pps.get('pic_parameter_set_id')}\n", 'field')
                    if 'seq_parameter_set_id' in pps:
                        self.detail_codec_text.insert(tk.END, f"  seq_parameter_set_id:  {pps.get('seq_parameter_set_id')}\n", 'field')
                    if 'entropy_coding_mode_flag' in pps:
                        ecm = pps.get('entropy_coding_mode_flag')
                        mode = "CABAC" if ecm else "CAVLC"
                        self.detail_codec_text.insert(tk.END, f"  entropy_coding_mode:   {ecm} ({mode})\n", 'field')
                    if 'num_ref_idx_l0_default' in pps:
                        self.detail_codec_text.insert(tk.END, f"  num_ref_idx_l0:        {pps['num_ref_idx_l0_default']}\n", 'field')
                    if 'num_ref_idx_l1_default' in pps:
                        self.detail_codec_text.insert(tk.END, f"  num_ref_idx_l1:        {pps['num_ref_idx_l1_default']}\n", 'field')
                    self.detail_codec_text.insert(tk.END, "\n")
                
                # PPS Errors and Warnings
                if pps.get('errors'):
                    self.detail_codec_text.insert(tk.END, "▶ Errors\n", 'subheader')
                    for err in pps['errors']:
                        self.detail_codec_text.insert(tk.END, f"  ✗ {err}\n", 'error')
                    self.detail_codec_text.insert(tk.END, "\n")
                
                if pps.get('warnings'):
                    self.detail_codec_text.insert(tk.END, "▶ Warnings\n", 'subheader')
                    for warn in pps['warnings']:
                        self.detail_codec_text.insert(tk.END, f"  ⚠ {warn}\n", 'warning')
                    self.detail_codec_text.insert(tk.END, "\n")
            
            # Check for HEVC stream (H.265)
            if stream_type == 0x24:
                # Check if we have video_header with HEVC data
                if hasattr(self, 'last_report') and 'video_headers' in self.last_report:
                    video_header = self.last_report['video_headers'].get(pid)
                    if video_header and 'type' in video_header and 'HEVC' in video_header['type']:
                        found_codec = True
                        
                        # HEVC VPS Header (if available)
                        if 'vps_id' in video_header:
                            self.detail_codec_text.insert(tk.END, "═" * 80 + "\n", 'header')
                            self.detail_codec_text.insert(tk.END, f"  HEVC VIDEO PARAMETER SET (VPS) - PID 0x{pid:04X}\n", 'header')
                            self.detail_codec_text.insert(tk.END, "═" * 80 + "\n\n", 'header')
                            
                            self.detail_codec_text.insert(tk.END, "▶ Basic Information\n", 'subheader')
                            self.detail_codec_text.insert(tk.END, f"  VPS ID:                {video_header.get('vps_id', 'N/A')}\n", 'field')
                            if 'vps_max_layers' in video_header:
                                self.detail_codec_text.insert(tk.END, f"  Max Layers:            {video_header['vps_max_layers']}\n", 'field')
                            if 'vps_max_sub_layers' in video_header:
                                self.detail_codec_text.insert(tk.END, f"  Max Sub Layers:        {video_header['vps_max_sub_layers']}\n", 'field')
                            self.detail_codec_text.insert(tk.END, "\n")
                        
                        # HEVC SPS Header
                        self.detail_codec_text.insert(tk.END, "═" * 80 + "\n", 'header')
                        self.detail_codec_text.insert(tk.END, f"  HEVC SEQUENCE PARAMETER SET (SPS) - PID 0x{pid:04X}\n", 'header')
                        self.detail_codec_text.insert(tk.END, "═" * 80 + "\n\n", 'header')
                        
                        # Profile and Level
                        self.detail_codec_text.insert(tk.END, "▶ Profile and Level\n", 'subheader')
                        profile_idc = video_header.get('profile_idc', 'N/A')
                        profile_names = {1: "Main", 2: "Main 10", 3: "Main Still Picture"}
                        profile_name = profile_names.get(profile_idc, f"Profile {profile_idc}")
                        self.detail_codec_text.insert(tk.END, f"  Profile IDC:           {profile_idc} ({profile_name})\n", 'field')
                        level_idc = video_header.get('level_idc', 'N/A')
                        if level_idc != 'N/A':
                            level_str = f"{level_idc / 30:.1f}"
                            self.detail_codec_text.insert(tk.END, f"  Level IDC:             {level_idc} (Level {level_str})\n", 'field')
                        else:
                            self.detail_codec_text.insert(tk.END, f"  Level IDC:             {level_idc}\n", 'field')
                        self.detail_codec_text.insert(tk.END, "\n")
                        
                        # Resolution (with 4K indicator)
                        if 'width' in video_header and 'height' in video_header:
                            self.detail_codec_text.insert(tk.END, "▶ Video Resolution\n", 'subheader')
                            width = video_header['width']
                            height = video_header['height']
                            self.detail_codec_text.insert(tk.END, f"  Width:                 {width} pixels\n", 'field')
                            self.detail_codec_text.insert(tk.END, f"  Height:                {height} pixels\n", 'field')
                            
                            # Display resolution name with 4K indicator
                            resolution_name = video_header.get('resolution_name', f"{width}x{height}")
                            if video_header.get('is_4k'):
                                self.detail_codec_text.insert(tk.END, f"  Resolution:            ", 'field')
                                self.detail_codec_text.insert(tk.END, f"{resolution_name}\n", 'value_highlight')
                            else:
                                self.detail_codec_text.insert(tk.END, f"  Resolution:            {resolution_name}\n", 'field')
                            
                            self.detail_codec_text.insert(tk.END, f"  Aspect Ratio:          {width}:{height}\n", 'field')
                            self.detail_codec_text.insert(tk.END, "\n")
                        
                        # Bit Depth (10-bit indicator)
                        if 'bit_depth_luma' in video_header or 'bit_depth_chroma' in video_header:
                            self.detail_codec_text.insert(tk.END, "▶ Bit Depth\n", 'subheader')
                            if 'bit_depth_luma' in video_header:
                                bd_luma = video_header['bit_depth_luma']
                                if video_header.get('is_10bit'):
                                    self.detail_codec_text.insert(tk.END, f"  Luma Bit Depth:        ", 'field')
                                    self.detail_codec_text.insert(tk.END, f"{bd_luma} bits (High Quality)\n", 'value_highlight')
                                else:
                                    self.detail_codec_text.insert(tk.END, f"  Luma Bit Depth:        {bd_luma} bits\n", 'field')
                            if 'bit_depth_chroma' in video_header:
                                bd_chroma = video_header['bit_depth_chroma']
                                if video_header.get('is_10bit'):
                                    self.detail_codec_text.insert(tk.END, f"  Chroma Bit Depth:      ", 'field')
                                    self.detail_codec_text.insert(tk.END, f"{bd_chroma} bits (High Quality)\n", 'value_highlight')
                                else:
                                    self.detail_codec_text.insert(tk.END, f"  Chroma Bit Depth:      {bd_chroma} bits\n", 'field')
                            self.detail_codec_text.insert(tk.END, "\n")
                        
                        # Chroma Format
                        if 'chroma_format_idc' in video_header:
                            self.detail_codec_text.insert(tk.END, "▶ Color Format\n", 'subheader')
                            chroma_idc = video_header['chroma_format_idc']
                            chroma_names = {0: "Monochrome", 1: "4:2:0", 2: "4:2:2", 3: "4:4:4"}
                            chroma_name = chroma_names.get(chroma_idc, f"Unknown ({chroma_idc})")
                            self.detail_codec_text.insert(tk.END, f"  Chroma Format:         {chroma_idc} ({chroma_name})\n", 'field')
                            self.detail_codec_text.insert(tk.END, "\n")
                        
                        # Additional SPS Parameters
                        if 'sps_id' in video_header:
                            self.detail_codec_text.insert(tk.END, "▶ Additional SPS Parameters\n", 'subheader')
                            self.detail_codec_text.insert(tk.END, f"  SPS ID:                {video_header.get('sps_id')}\n", 'field')
                            if 'sps_max_sub_layers' in video_header:
                                self.detail_codec_text.insert(tk.END, f"  Max Sub Layers:        {video_header['sps_max_sub_layers']}\n", 'field')
                            if 'log2_max_pic_order_cnt' in video_header:
                                self.detail_codec_text.insert(tk.END, f"  Log2 Max POC:          {video_header['log2_max_pic_order_cnt']}\n", 'field')
                            self.detail_codec_text.insert(tk.END, "\n")
                        
                        # HEVC PPS (if available)
                        if 'pps_id' in video_header:
                            self.detail_codec_text.insert(tk.END, "═" * 80 + "\n", 'header')
                            self.detail_codec_text.insert(tk.END, f"  HEVC PICTURE PARAMETER SET (PPS) - PID 0x{pid:04X}\n", 'header')
                            self.detail_codec_text.insert(tk.END, "═" * 80 + "\n\n", 'header')
                            
                            self.detail_codec_text.insert(tk.END, "▶ Basic Information\n", 'subheader')
                            self.detail_codec_text.insert(tk.END, f"  PPS ID:                {video_header.get('pps_id')}\n", 'field')
                            if 'pps_sps_id' in video_header:
                                self.detail_codec_text.insert(tk.END, f"  Referenced SPS ID:     {video_header['pps_sps_id']}\n", 'field')
                            self.detail_codec_text.insert(tk.END, "\n")
        
        if not found_codec:
            self.detail_codec_text.insert(tk.END, "No H.264 or HEVC SPS/PPS information found in stream.\n\n", 'field')
            self.detail_codec_text.insert(tk.END, "Note: SPS and PPS are typically sent at the beginning of video streams.\n", 'field')
            self.detail_codec_text.insert(tk.END, "If this is an MPEG-2 stream, SPS/PPS do not apply.\n", 'field')
        
        # Make text read-only
        self.detail_codec_text.config(state=tk.DISABLED)

    def _create_thumbnail_timecode_label(self, idx, timecode):
        """Create a timecode label for thumbnail at index `idx` on the UI thread."""
        try:
            if not hasattr(self, '_thumb_frames') or idx >= len(self._thumb_frames):
                if DEBUG: print(f"[Create TC] No thumbnail frame for index {idx}")
                return
            frame_widget = self._thumb_frames[idx]
            # If label already exists, update
            if hasattr(self, '_thumb_timecode_labels') and idx < len(self._thumb_timecode_labels) and self._thumb_timecode_labels[idx]:
                lbl = self._thumb_timecode_labels[idx]
                lbl.config(text=f"TC: {timecode}")
                # If warning present in text, color it red
                try:
                    if 'WARN:' in str(timecode):
                        lbl.config(foreground='red')
                    else:
                        lbl.config(foreground='#1976D2')
                except Exception:
                    pass
                if DEBUG: print(f"[Create TC] Updated existing label for {idx}: {timecode}")
                return

            # Color red when a warning marker is included
            fg = 'red' if (isinstance(timecode, str) and 'WARN:' in timecode) else '#1976D2'
            new_lbl = ttk.Label(frame_widget, text=f"TC: {timecode}", font=('TkDefaultFont', 8, 'bold'), foreground=fg, justify=tk.CENTER)
            new_lbl.pack()
            # Ensure list length
            if not hasattr(self, '_thumb_timecode_labels'):
                self._thumb_timecode_labels = []
            while len(self._thumb_timecode_labels) <= idx:
                self._thumb_timecode_labels.append(None)
            self._thumb_timecode_labels[idx] = new_lbl
            if DEBUG: print(f"[Create TC] Created label for thumbnail {idx}: {timecode}")
        except Exception as e:
            if DEBUG: print(f"[Create TC] Error creating label for {idx}: {e}")
    
    def show_detail_nal_sei_info(self, event):
        """Show full SEI payload when double-clicking a SEI message in frame details"""
        selected = self.detail_nal_tree.selection()
        if not selected:
            return
        
        item = self.detail_nal_tree.item(selected[0])
        values = item.get("values", [])
        tags = self.detail_nal_tree.item(selected[0], 'tags')
        
        # Check if this is a SEI message
        if len(values) >= 5 and str(values[0]).startswith("SEI-"):
            # Extract full payload from tags
            full_payload = ""
            if len(tags) >= 2:
                full_payload = tags[1]  # Second tag contains the full payload
            else:
                full_payload = values[3]  # Fallback to preview
            
            caption_info = values[4] if len(values) > 4 else ""
            
            # Create popup window
            popup = tk.Toplevel(self.frame_detail_window)
            popup.title(f"{values[1]} - Full Details")
            popup.geometry("700x500")
            
            # Frame for content
            content_frame = ttk.Frame(popup, padding="10")
            content_frame.pack(fill=tk.BOTH, expand=True)
            
            # Info text
            info_text = tk.Text(content_frame, wrap=tk.WORD, height=25)
            info_scrollbar = ttk.Scrollbar(content_frame, orient=tk.VERTICAL, command=info_text.yview)
            info_text.configure(yscrollcommand=info_scrollbar.set)
            
            info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Format the information
            info = f"""SEI Message Details:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type: {values[0]}
Type Name: {values[1]}
Size: {values[2]}

Caption/Service Info:
{caption_info if caption_info else 'None'}

Full Payload (Hex):
{full_payload}

Payload Length: {len(full_payload)} characters ({len(full_payload)//2} bytes)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            info_text.insert(1.0, info)
            info_text.configure(state='disabled')
            
            # Close button
            ttk.Button(popup, text="Close", command=popup.destroy).pack(pady=5)
    
    def extract_audio_waveform(self, start_sample=0):
        """Extract and display audio waveform"""
        if not self.current_file or not os.path.isfile(self.current_file):
            messagebox.showerror("Error", "Please analyze a TS file first")
            return
        
        # Clear previous thumbnails
        for widget in self.thumbnails_inner_frame.winfo_children():
            widget.destroy()
        self.thumbnail_images.clear()
        
        self.current_media_type = 'audio'
        self.current_frame_start = start_sample
        
        self.extract_audio_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Extracting audio waveform...", foreground="blue")
        
        # Run extraction in separate thread
        threading.Thread(target=self._extract_audio_worker, args=(start_sample,), daemon=True).start()
    
    def _extract_audio_worker(self, start_sample=0):
        """Worker thread to extract audio samples and generate waveform"""
        container = None
        try:
            container = av.open(self.current_file)
            
            # For MPTS, find the correct audio stream by PID
            audio_stream = None
            if self.last_report:
                audio_pid = None
                for pid, stream_info in self.last_report.get('elementary_streams', {}).items():
                    stream_type = stream_info.get('stream_type')
                    if stream_type in [0x03, 0x04, 0x0F, 0x11, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87]:
                        audio_pid = pid
                        break
                if audio_pid is not None:
                    audio_stream = next((s for s in container.streams.audio 
                                       if s.id == audio_pid), None)
                    if not audio_stream:
                        # Try alternate method by stream index
                        for pmt in self.last_report.get('pmts', {}).values():
                            for idx, stream in enumerate(pmt.get('streams', [])):
                                if stream['pid'] == audio_pid:
                                    audio_streams = list(container.streams.audio)
                                    if idx < len(audio_streams):
                                        audio_stream = audio_streams[idx]
                                    break
            
            if not audio_stream:
                audio_stream = next(iter(container.streams.audio), None)
            
            if not audio_stream:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No audio stream found"))
                self.root.after(0, lambda: self.extract_audio_btn.config(state=tk.NORMAL))
                return
            
            # Check for problematic audio configurations
            if audio_stream.codec_context and audio_stream.codec_context.name == 'eac3' and audio_stream.channels > 6:
                self.root.after(0, lambda: messagebox.showwarning(
                    "Audio Format Not Supported",
                    f"This file contains {audio_stream.channels}-channel E-AC-3 audio which may cause stability issues.\\n\\n"
                    f"Standalone audio waveform extraction is not available for this format."
                ))
                self.root.after(0, lambda: self.extract_audio_btn.config(state=tk.NORMAL))
                return
            
            # Calculate total samples
            if audio_stream.duration and audio_stream.sample_rate:
                duration_sec = float(audio_stream.duration * audio_stream.time_base)
                total_samples = int(duration_sec * audio_stream.sample_rate)
            else:
                total_samples = 0
            
            # Store stream info for navigation
            self.audio_stream_info = {
                'total_samples': total_samples,
                'sample_rate': audio_stream.sample_rate,
                'channels': audio_stream.channels,
                'time_base': audio_stream.time_base
            }
            
            # Extract audio samples from start_sample
            sample_arrays = []
            sample_count = 0
            samples_to_extract = 100000  # Extract 100k samples at a time
            total_extracted = 0
            
            for packet in container.demux(audio_stream):
                try:
                    decoded_frames = list(packet.decode())
                except:
                    continue
                
                try:
                    for frame in decoded_frames:
                        try:
                            arr = frame.to_ndarray()
                            # Keep multi-channel structure
                            # arr shape is (channels, samples) for multi-channel or (samples,) for mono
                            # Transpose to (samples, channels) for easier processing
                            if len(arr.shape) > 1:
                                arr = arr.T  # Now shape is (samples, channels)
                            else:
                                arr = arr.reshape(-1, 1)  # Mono: reshape to (samples, 1)
                            
                            num_samples = arr.shape[0]
                            
                            # Skip samples before start_sample
                            if sample_count + num_samples <= start_sample:
                                sample_count += num_samples
                                continue
                            
                            # Extract portion of frame if we're in the middle
                            if sample_count < start_sample:
                                offset = start_sample - sample_count
                                arr = arr[offset:]
                                sample_count = start_sample
                            
                            # Add samples until we reach our limit
                            remaining = samples_to_extract - total_extracted
                            if remaining > 0:
                                chunk = arr[:remaining]
                                sample_arrays.append(chunk)
                                total_extracted += len(chunk)
                                sample_count += len(chunk)
                        except Exception:
                            # Skip corrupted frames
                            pass
                        
                        if total_extracted >= samples_to_extract:
                            break
                except av.error.InvalidDataError:
                    # Skip corrupted packets
                    continue
                except Exception:
                    # Skip other decoding errors
                    continue
                if total_extracted >= samples_to_extract:
                    break
            
            if not sample_arrays:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No audio samples extracted"))
                self.root.after(0, lambda: self.extract_audio_btn.config(state=tk.NORMAL))
                return
            
            # Concatenate all sample arrays - maintains (samples, channels) shape
            samples = np.concatenate(sample_arrays, axis=0)
            
            if DEBUG: print(f"[DEBUG] Samples shape after concatenation: {samples.shape}")
            if DEBUG: print(f"[DEBUG] Samples dtype: {samples.dtype}")
            
            # Generate waveform plot
            num_channels = samples.shape[1] if len(samples.shape) > 1 else 1
            if DEBUG: print(f"[DEBUG] Number of channels detected: {num_channels}")
            
            # Create matplotlib figure with subplots for each channel
            fig = Figure(figsize=(10, 2 + 2*num_channels), dpi=100)
            if DEBUG: print(f"[DEBUG] Creating {num_channels} subplots")
            
            # Downsample for display
            display_samples = 2000
            if len(samples) > display_samples:
                step = len(samples) // display_samples
                samples_display = samples[::step]
            else:
                samples_display = samples
            
            time_axis = np.arange(len(samples_display)) / audio_stream.sample_rate * (len(samples) / len(samples_display))
            
            # Plot each channel separately
            for ch in range(num_channels):
                ax = fig.add_subplot(num_channels, 1, ch + 1)
                if DEBUG: print(f"[DEBUG] Creating subplot {ch+1}/{num_channels}")
                
                if num_channels == 1:
                    channel_data = samples_display.flatten()
                else:
                    channel_data = samples_display[:, ch]
                
                if DEBUG: print(f"[DEBUG] Channel {ch+1} data shape: {channel_data.shape}, min: {channel_data.min()}, max: {channel_data.max()}")
                
                ax.plot(time_axis, channel_data, linewidth=0.5)
                ax.set_ylabel(f'Ch {ch+1}')
                ax.grid(True, alpha=0.3)
                
                # Only show x-label on bottom subplot
                if ch == num_channels - 1:
                    ax.set_xlabel('Time (s)')
                else:
                    ax.set_xticklabels([])
            
            fig.suptitle(f'Audio Waveform ({audio_stream.sample_rate} Hz, {audio_stream.channels} ch)', fontsize=12)
            fig.tight_layout()
            
            # Display in UI thread
            self.root.after(0, self._display_audio_waveform, fig, total_samples)
            
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to extract audio waveform:\\n{msg}"))
        finally:
            if container is not None:
                try:
                    container.close()
                except:
                    pass
            self.root.after(0, lambda: self.extract_audio_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.status_label.config(text="Ready", foreground="green"))
    
    def _display_audio_waveform(self, fig, total_samples=0):
        """Display audio waveform in the GUI"""
        # Create canvas for matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master=self.thumbnails_inner_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.thumbnail_images.append(canvas)  # Keep reference
        self.status_label.config(text="Audio waveform extracted", foreground="green")
        
        # Update navigation state
        if self.audio_stream_info:
            sample_rate = self.audio_stream_info.get('sample_rate', 1)
            time_sec = self.current_frame_start / sample_rate
            self.current_position_var.set(f"Position: {time_sec:.2f}s (sample {self.current_frame_start})")
            
            # Enable/disable navigation buttons
            if self.current_frame_start > 0:
                self.prev_10_btn.config(state=tk.NORMAL)
            else:
                self.prev_10_btn.config(state=tk.DISABLED)
            
            if total_samples > 0 and self.current_frame_start + 100000 < total_samples:
                self.next_10_btn.config(state=tk.NORMAL)
            else:
                self.next_10_btn.config(state=tk.DISABLED)
            
            self.jump_btn.config(state=tk.NORMAL)
    
    def navigate_prev_10(self):
        """Navigate to previous 10 frames/samples"""
        if self.current_media_type == 'video':
            try:
                num_frames = int(self.num_frames_var.get())
            except ValueError:
                num_frames = 10
            new_start = max(0, self.current_frame_start - num_frames)
            self.extract_video_thumbnails(new_start)
        elif self.current_media_type == 'audio':
            # Move back by 100k samples (approximately the window size)
            new_start = max(0, self.current_frame_start - 100000)
            self.extract_audio_waveform(new_start)
    
    def navigate_next_10(self):
        """Navigate to next 10 frames/samples"""
        if self.current_media_type == 'video':
            try:
                num_frames = int(self.num_frames_var.get())
            except ValueError:
                num_frames = 10
            new_start = self.current_frame_start + num_frames
            if self.total_video_frames > 0 and new_start >= self.total_video_frames:
                return
            self.extract_video_thumbnails(new_start)
        elif self.current_media_type == 'audio':
            # Move forward by 100k samples
            new_start = self.current_frame_start + 100000
            if self.audio_stream_info:
                total = self.audio_stream_info.get('total_samples', 0)
                if total > 0 and new_start >= total:
                    return
            self.extract_audio_waveform(new_start)
    
    def jump_to_frame(self):
        """Jump to specific frame/time"""
        try:
            target = self.jump_frame_var.get().strip()
            if not target:
                return
            
            if self.current_media_type == 'video':
                # Can be frame number or time in seconds
                if '.' in target or 's' in target.lower():
                    # Time in seconds
                    time_sec = float(target.rstrip('sS'))
                    if self.video_stream_info and self.video_stream_info.get('average_rate'):
                        frame_num = int(time_sec * float(self.video_stream_info['average_rate']))
                    else:
                        messagebox.showerror("Error", "Cannot convert time to frame: video info not available")
                        return
                else:
                    frame_num = int(target)
                
                if frame_num < 0:
                    frame_num = 0
                if self.total_video_frames > 0 and frame_num >= self.total_video_frames:
                    frame_num = self.total_video_frames - 1
                
                self.extract_video_thumbnails(frame_num)
                
            elif self.current_media_type == 'audio':
                # Can be sample number or time in seconds
                if '.' in target or 's' in target.lower():
                    # Time in seconds
                    time_sec = float(target.rstrip('sS'))
                    if self.audio_stream_info and self.audio_stream_info.get('sample_rate'):
                        sample_num = int(time_sec * self.audio_stream_info['sample_rate'])
                    else:
                        messagebox.showerror("Error", "Cannot convert time to sample: audio info not available")
                        return
                else:
                    sample_num = int(target)
                
                if sample_num < 0:
                    sample_num = 0
                if self.audio_stream_info:
                    total = self.audio_stream_info.get('total_samples', 0)
                    if total > 0 and sample_num >= total:
                        sample_num = total - 1
                
                self.extract_audio_waveform(sample_num)
            
            self.jump_frame_var.set("")  # Clear input
            
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid frame/time value: {target}\\n{e}")
    
    def apply_frame_filter(self):
        """Apply frame type filter (all, I-frames, or IDR-frames)"""
        if self.current_media_type != 'video' or not self.current_file:
            return
        
        # Re-extract thumbnails with the new filter starting from current position
        self.extract_video_thumbnails(start_frame=self.current_frame_start)

    def apply_frame_order(self):
        """Reorder thumbnails display according to selected frame order (PTS or DTS)"""
        if self.current_media_type != 'video' or not self.current_file:
            return
        # If we already have frames loaded, re-display them with new ordering
        if hasattr(self, 'current_frames_data') and self.current_frames_data:
            # Re-display using current frames (will sort inside display)
            self._display_video_thumbnails(self.current_frames_data, self.total_video_frames if hasattr(self, 'total_video_frames') else 0)
    
    def navigate_prev_idr(self):
        """Navigate to previous I or IDR frame"""
        if self.current_media_type != 'video':
            return
        
        # Search backwards from current position - 1
        search_start = max(0, self.current_frame_start - 1)
        
        self.status_label.config(text="Searching for previous I/IDR frame...", foreground="blue")
        threading.Thread(target=self._find_prev_idr_worker, args=(search_start,), daemon=True).start()
    
    def navigate_next_idr(self):
        """Navigate to next I or IDR frame"""
        if self.current_media_type != 'video':
            return
        
        # Search forwards from current position + 1
        try:
            num_frames = int(self.num_frames_var.get())
        except ValueError:
            num_frames = 10
        
        search_start = self.current_frame_start + num_frames
        
        self.status_label.config(text="Searching for next I/IDR frame...", foreground="blue")
        threading.Thread(target=self._find_next_idr_worker, args=(search_start,), daemon=True).start()
    
    def _find_prev_idr_worker(self, search_start):
        """Worker thread to find previous I or IDR frame"""
        try:
            container = av.open(self.current_file)
            
            # For MPTS, find the correct video stream by PID
            video_stream = None
            if self.last_report:
                video_pid = None
                for pid, stream_info in self.last_report.get('elementary_streams', {}).items():
                    stream_type = stream_info.get('stream_type')
                    if stream_type in [0x1B, 0x24, 0x02]:  # H.264, H.265, MPEG-2
                        video_pid = pid
                        break
                if video_pid is not None:
                    video_stream = next((s for s in container.streams.video 
                                       if s.id == video_pid), None)
                    if not video_stream:
                        # Try alternate method by stream index
                        for pmt in self.last_report.get('pmts', {}).values():
                            for idx, stream in enumerate(pmt.get('streams', [])):
                                if stream['pid'] == video_pid:
                                    video_streams = list(container.streams.video)
                                    if idx < len(video_streams):
                                        video_stream = video_streams[idx]
                                    break
            
            if not video_stream:
                video_stream = next(iter(container.streams.video), None)
            
            if not video_stream:
                return
            
            # Scan backwards to find I/IDR frame
            frame_count = 0
            found_frame = None
            
            for packet in container.demux(video_stream):
                try:
                    decoded_frames = list(packet.decode())
                except:
                    continue
                
                try:
                    for frame in decoded_frames:
                        if frame_count <= search_start:
                            frame_type = self._get_frame_type(frame)
                            if frame_type in ['I', 'IDR']:
                                found_frame = frame_count
                        
                        frame_count += 1
                        
                        # Stop if we've gone beyond search range
                        if frame_count > search_start:
                            break
                except:
                    continue
                
                if frame_count > search_start:
                    break
            
            if found_frame is not None:
                # Jump to found frame
                self.root.after(0, lambda: self.extract_video_thumbnails(found_frame))
            else:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No I/IDR frame found before current position"))
                self.root.after(0, lambda: self.status_label.config(text="Ready", foreground="green"))
        
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to search for I/IDR frame:\\n{msg}"))
            self.root.after(0, lambda: self.status_label.config(text="Ready", foreground="green"))
    
    def _find_next_idr_worker(self, search_start):
        """Worker thread to find next I or IDR frame"""
        try:
            container = av.open(self.current_file)
            
            # For MPTS, find the correct video stream by PID
            video_stream = None
            if self.last_report:
                video_pid = None
                for pid, stream_info in self.last_report.get('elementary_streams', {}).items():
                    stream_type = stream_info.get('stream_type')
                    if stream_type in [0x1B, 0x24, 0x02]:  # H.264, H.265, MPEG-2
                        video_pid = pid
                        break
                if video_pid is not None:
                    video_stream = next((s for s in container.streams.video 
                                       if s.id == video_pid), None)
                    if not video_stream:
                        # Try alternate method by stream index
                        for pmt in self.last_report.get('pmts', {}).values():
                            for idx, stream in enumerate(pmt.get('streams', [])):
                                if stream['pid'] == video_pid:
                                    video_streams = list(container.streams.video)
                                    if idx < len(video_streams):
                                        video_stream = video_streams[idx]
                                    break
            
            if not video_stream:
                video_stream = next(iter(container.streams.video), None)
            
            if not video_stream:
                return
            
            # Scan forward to find I/IDR frame
            frame_count = 0
            found_frame = None
            
            for packet in container.demux(video_stream):
                try:
                    decoded_frames = list(packet.decode())
                except:
                    continue
                
                try:
                    for frame in decoded_frames:
                        if frame_count >= search_start:
                            frame_type = self._get_frame_type(frame)
                            if frame_type in ['I', 'IDR']:
                                found_frame = frame_count
                                break
                        
                        frame_count += 1
                except:
                    continue
                
                if found_frame is not None:
                    break
            
            if found_frame is not None:
                # Jump to found frame
                self.root.after(0, lambda: self.extract_video_thumbnails(found_frame))
            else:
                self.root.after(0, lambda: messagebox.showinfo("Info", "No I/IDR frame found after current position"))
                self.root.after(0, lambda: self.status_label.config(text="Ready", foreground="green"))
        
        except Exception as e:
            error_msg = str(e)
            self.root.after(0, lambda msg=error_msg: messagebox.showerror("Error", f"Failed to search for I/IDR frame:\\n{msg}"))
            self.root.after(0, lambda: self.status_label.config(text="Ready", foreground="green"))

def main():
    root = tk.Tk()
    # Launch maximized/fullscreen for better use of screen real estate
    try:
        root.state('zoomed')
    except tk.TclError:
        try:
            root.attributes('-zoomed', True)
        except Exception:
            root.attributes('-fullscreen', True)
    app = TSAnalyserGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
