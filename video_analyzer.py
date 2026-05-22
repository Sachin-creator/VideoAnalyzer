#!/usr/bin/env python3
"""MPEG-TS analyser with TR101-290 style checks

This tool performs common checks from TR101-290 (broadcast transport stream
monitoring) including:
- sync byte validation
- transport_error_indicator counts
- continuity counter errors per PID
- null packet percentage
- PAT/PMT discovery and basic PMT parsing
- PCR extraction and PCR jitter/inter-arrival checks per PCR PID
- approximate bitrate estimate using PCRs or file duration
- HRD/T-STD buffer analysis for decoder compliance

This is not a full standards compliance test-suite but provides practical
checks that catch common broadcast transport issues.

Usage:
  python3 video_analyzer.py input.ts [--json] [--pcr-jitter-ms 50]

"""
from __future__ import annotations
import argparse
import os
import sys
import json
import time


# Import MP4/MOV parser
try:
    from mp4_parser import MP4Parser
    MP4_PARSER_AVAILABLE = True
except ImportError:
    MP4_PARSER_AVAILABLE = False

# Import SCTE-35 validator
try:
    from scte35_validator import SCTE35Validator, SCTE35ValidationError
    SCTE35_VALIDATOR_AVAILABLE = True
except ImportError:
    SCTE35_VALIDATOR_AVAILABLE = False

# Import HEVC parser
try:
    from hevc_parser import parse_hevc_vps, parse_hevc_sps, parse_hevc_pps, find_hevc_nal_units, HEVC_NAL_UNIT_TYPES
    HEVC_PARSER_AVAILABLE = True
except ImportError:
    HEVC_PARSER_AVAILABLE = False

from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Import buffer analyzer if available
try:
    from buffer_analyzer import BufferAnalyzer, T_STD_Analyzer
    BUFFER_ANALYSIS_AVAILABLE = True
except ImportError:
    BUFFER_ANALYSIS_AVAILABLE = False
    BufferAnalyzer = None
    T_STD_Analyzer = None

SYNC_BYTE = 0x47
TS_PACKET_SIZE = 188
DEBUG = False  # Set to True for verbose debug output

# CEA-608 character mapping per ANSI/CEA-608-B
# Note: The actual structure of CEA-608 is complex. For simplicity and robustness,
# we treat byte pairs (c1, c2) where both are in printable ASCII range (0x20-0x7E)
# as literal characters, since many captioning streams include ASCII text directly.

def cea608_decode_chars(c1: int, c2: int) -> str:
    """Decode CEA-608 character pair into displayable text.
    
    CEA-608 bytes include parity bit in MSB (bit 7), which must be stripped
    to get the actual 7-bit data value.
    
    Many modern caption streams embed ASCII text directly in the payload,
    so we accept printable ASCII pairs as-is. Control codes and invalid
    sequences are filtered out.
    """
    # Strip parity bit (MSB) to get 7-bit data
    c1 = c1 & 0x7F
    c2 = c2 & 0x7F
    
    # Null/padding: 0x00 0x00 or repeated nulls
    if (c1 == 0 and c2 == 0) or (c1 == 0x80 and c2 == 0x80):
        return ''
    
    # Both bytes must be in printable ASCII range (0x20-0x7E)
    # to avoid control codes and invalid sequences
    if 0x20 <= c1 <= 0x7E and 0x20 <= c2 <= 0x7E:
        # Found valid printable pair
        return chr(c1) + chr(c2)
    
    # Control codes, low bytes, or unprintable—skip
    return ''

# KLV Universal Label for MISB metadata (STANAG 4609)
KLV_UL_MISB_0601 = bytes([0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01, 0x0E, 0x01, 0x03, 0x01, 0x01, 0x00, 0x00, 0x00])
KLV_UL_MISB_0102 = bytes([0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01, 0x0E, 0x01, 0x03, 0x03, 0x00, 0x00, 0x00, 0x00])

###############################################
# MISB ST 0601 (UAS Datalink) basic tag support
# This is a partial implementation focused on common telemetry:
#   Heading, Pitch, Roll, Sensor Lat/Lon/Alt, Frame Center Lat/Lon/Elev
# Tag parsing assumes Local Set encoding: <Tag><Length><Value>
# Tag: 1 byte (for tags < 128) ; Length: 1 byte ; Value: 'Length' bytes
# For full compliance, extended tag/length BER forms should be added later.
###############################################

# Scaling helpers derived from MISB ST 0601 specification (simplified ranges)
def _scale_unsigned(value: int, bits: int, low: float, high: float) -> float:
    max_val = (1 << bits) - 1
    return (value / max_val) * (high - low) + low

def _scale_signed(value: int, bits: int, low: float, high: float) -> float:
    max_val = (1 << bits) - 1
    # Convert two's complement
    if value & (1 << (bits - 1)):
        value = value - (1 << bits)
    span = high - low
    return ((value - (-(1 << (bits - 1)))) / max_val) * span + low

# Tag decode map: tag -> (name, decoder_fn)
MISB_ST0601_TAGS = {
    # 1: Checksum or Metadata Universal (record only)
    1: ("UAS Datalink Checksum", lambda b: int.from_bytes(b,'big')),
    2: ("Platform Designation", lambda b: b.decode('latin-1', 'ignore').strip()),
    3: ("Platform Mission ID", lambda b: b.decode('latin-1', 'ignore').strip()),
    5: ("Platform Heading Angle (deg)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, 0.0, 360.0), 4) if len(b)==2 else None),
    6: ("Platform Pitch Angle (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 16, -20.0, 20.0), 4) if len(b)==2 else None),
    7: ("Platform Roll Angle (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 16, -50.0, 50.0), 4) if len(b)==2 else None),
    11: ("Platform Ground Speed (m/s)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 8*len(b), 0.0, 255.0), 3) if len(b) in (1,2) else None),
    12: ("Platform Ground Track Angle (deg)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, 0.0, 360.0), 4) if len(b)==2 else None),
    10: ("Platform True Airspeed (m/s)", lambda b: int.from_bytes(b,'big') if len(b)==1 else int.from_bytes(b,'big')),
    13: ("Sensor Latitude (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 32, -90.0, 90.0), 6) if len(b)==4 else None),
    14: ("Sensor Longitude (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 32, -180.0, 180.0), 6) if len(b)==4 else None),
    15: ("Sensor True Altitude (m)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, -900.0, 19000.0), 2) if len(b)==2 else None),
    16: ("Sensor Horizontal FOV (deg)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, 0.0, 180.0), 4) if len(b)==2 else None),
    17: ("Sensor Vertical FOV (deg)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, 0.0, 180.0), 4) if len(b)==2 else None),
    18: ("Sensor Relative Azimuth (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 16, -180.0, 180.0), 4) if len(b)==2 else None),
    19: ("Sensor Relative Elevation (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 16, -90.0, 90.0), 4) if len(b)==2 else None),
    20: ("Sensor Relative Roll (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 16, -180.0, 180.0), 4) if len(b)==2 else None),
    22: ("Slant Range (m)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, 0.0, 500000.0), 2) if len(b)==2 else None),
    23: ("Frame Center Latitude (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 32, -90.0, 90.0), 6) if len(b)==4 else None),
    24: ("Frame Center Longitude (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 32, -180.0, 180.0), 6) if len(b)==4 else None),
    25: ("Frame Center Elevation (m)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, -900.0, 19000.0), 2) if len(b)==2 else None),
    65: ("Target Width (m)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, 0.0, 10000.0), 3) if len(b)==2 else None),
    42: ("UAS Datalink LS Version", lambda b: int.from_bytes(b,'big')),
    48: ("Sensor Azimuth (deg)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, 0.0, 360.0), 4) if len(b)==2 else None),
    66: ("Target Location Latitude (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 32, -90.0, 90.0), 6) if len(b)==4 else None),
    67: ("Target Location Longitude (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 32, -180.0, 180.0), 6) if len(b)==4 else None),
    # Frame corners and geometry
    35: ("Frame Corner Lat/Lon Set A", lambda b: b.hex()),
    36: ("Frame Corner Lat/Lon Set B", lambda b: b.hex()),
    47: ("Sensor Elevation Angle (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 16, -90.0, 90.0), 4) if len(b)==2 else None),
    51: ("Platform Location Latitude (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 32, -90.0, 90.0), 6) if len(b)==4 else None),
    56: ("Platform Location Longitude (deg)", lambda b: round(_scale_signed(int.from_bytes(b,'big'), 32, -180.0, 180.0), 6) if len(b)==4 else None),
    75: ("Sensor Mode", lambda b: int.from_bytes(b,'big')),  
    79: ("Target Elevation (m)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, -900.0, 19000.0), 2) if len(b)==2 else None),
    80: ("Ground Sample Distance (m)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, 0.0, 500.0), 4) if len(b)==2 else None),
    90: ("Frame Center HAE (m)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, -900.0, 19000.0), 2) if len(b)==2 else None),
    91: ("Platform HAE (m)", lambda b: round(_scale_unsigned(int.from_bytes(b,'big'), 16, -900.0, 19000.0), 2) if len(b)==2 else None),
}

def _read_ber_field(data: bytes, offset: int) -> Tuple[int, int]:
    """Read BER-encoded field (short or long form). Returns (value, bytes_consumed)."""
    if offset >= len(data):
        return 0, 0
    first = data[offset]
    if first & 0x80 == 0:
        return first & 0x7F, 1
    count = first & 0x7F
    if offset + 1 + count > len(data):
        return 0, 0
    val = 0
    for i in range(count):
        val = (val << 8) | data[offset + 1 + i]
    return val, 1 + count

def parse_misb_st0601_local_set(value_bytes: bytes) -> Dict[str, object]:
    """Parse a MISB ST 0601 Local Set payload supporting BER Tag/Length.
    Returns only decoded telemetry map."""
    telemetry, _seen_tags, _unknown_tags = parse_misb_st0601_local_set_with_tags(value_bytes)
    return telemetry

def parse_misb_st0601_local_set_with_tags(value_bytes: bytes) -> Tuple[Dict[str, object], List[int], List[int]]:
    """Parse ST0601 Local Set and also return seen tags and unknown tags."""
    telemetry = {}
    seen_tags: List[int] = []
    unknown_tags: List[int] = []
    offset = 0
    length = len(value_bytes)
    while offset < length:
        tag, tag_consumed = _read_ber_field(value_bytes, offset)
        if tag_consumed == 0:
            break
        offset += tag_consumed
        val_len, len_consumed = _read_ber_field(value_bytes, offset)
        if len_consumed == 0:
            break
        offset += len_consumed
        if offset + val_len > length:
            break
        raw = value_bytes[offset:offset + val_len]
        offset += val_len
        seen_tags.append(tag)
        entry = MISB_ST0601_TAGS.get(tag)
        if not entry:
            unknown_tags.append(tag)
            continue
        name, decoder = entry
        try:
            decoded = decoder(raw)
        except Exception:
            decoded = None
        if decoded is not None:
            telemetry[name] = decoded
    return telemetry, seen_tags, unknown_tags


def parse_ber_length(data: bytes, offset: int) -> Tuple[int, int]:
    """
    Parse BER (Basic Encoding Rules) encoded length
    Returns (length_value, bytes_consumed)
    """
    if offset >= len(data):
        return 0, 0
    
    first_byte = data[offset]
    if first_byte & 0x80 == 0:
        # Short form: length is in the lower 7 bits
        return first_byte, 1
    else:
        # Long form: lower 7 bits indicate number of subsequent length bytes
        num_bytes = first_byte & 0x7F
        if num_bytes == 0 or offset + num_bytes >= len(data):
            return 0, 0
        
        length = 0
        for i in range(num_bytes):
            length = (length << 8) | data[offset + 1 + i]
        return length, 1 + num_bytes


def detect_klv_metadata(data: bytes) -> List[Dict]:
    """
    Detect KLV (Key-Length-Value) metadata in data
    Returns list of KLV packets found with their properties
    """
    klv_packets = []
    offset = 0
    
    while offset < len(data) - 16:  # Need at least 16 bytes for UL key
        # Look for KLV Universal Label (16-byte key starting with 0x06 0x0E 0x2B 0x34)
        if data[offset:offset+4] == bytes([0x06, 0x0E, 0x2B, 0x34]):
            key = data[offset:offset+16]
            
            # Parse BER length
            length, length_bytes = parse_ber_length(data, offset + 16)
            if length_bytes == 0:
                offset += 1
                continue
            
            value_offset = offset + 16 + length_bytes
            
            # Check if we have complete value
            if value_offset + length > len(data):
                offset += 1
                continue
            
            # Identify KLV type
            klv_type = "Unknown KLV"
            is_misb = False
            standard = None
            
            if key == KLV_UL_MISB_0601:
                klv_type = "MISB ST 0601 (UAS Datalink)"
                is_misb = True
                standard = "MISB ST 0601"
            elif key == KLV_UL_MISB_0102:
                klv_type = "MISB ST 0102 (Security Metadata)"
                is_misb = True
                standard = "MISB ST 0102"
            elif key[:12] == bytes([0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01, 0x0E, 0x01, 0x03, 0x01]):
                klv_type = "MISB Metadata"
                is_misb = True
                standard = "MISB"
            
            decoded = None
            if standard == "MISB ST 0601":
                value_bytes = data[value_offset:value_offset+length]
                decoded, seen_tags, unknown_tags = parse_misb_st0601_local_set_with_tags(value_bytes)
            else:
                seen_tags, unknown_tags = [], []
            klv_packets.append({
                'offset': offset,
                'key': key.hex(),
                'length': length,
                'type': klv_type,
                'is_misb': is_misb,
                'standard': standard,
                'value_offset': value_offset,
                'decoded': decoded,
                'seen_tags': seen_tags,
                'unknown_tags': unknown_tags
            })
            
            offset = value_offset + length
        else:
            offset += 1
    
    return klv_packets


def parse_ts_header(pkt: bytes) -> Dict[str, int]:
    # Assumes len(pkt) == 188
    if len(pkt) < 4:
        raise ValueError("packet too short")
    if pkt[0] != SYNC_BYTE:
        return {"sync": False}
    hdr1 = pkt[1]
    hdr2 = pkt[2]
    hdr3 = pkt[3]
    transport_error_indicator = (hdr1 & 0x80) != 0
    payload_unit_start_indicator = (hdr1 & 0x40) != 0
    transport_priority = (hdr1 & 0x20) != 0
    pid = ((hdr1 & 0x1F) << 8) | hdr2
    transport_scrambling_control = (hdr3 & 0xC0) >> 6
    adaptation_field_control = (hdr3 & 0x30) >> 4
    continuity_counter = hdr3 & 0x0F
    return {
        "sync": True,
        "tei": int(transport_error_indicator),
        "pusi": int(payload_unit_start_indicator),
        "pid": pid,
        "scrambling": transport_scrambling_control,
        "afc": adaptation_field_control,
        "cc": continuity_counter,
    }


def extract_pcr_from_adaptation(adapt_bytes: bytes) -> Optional[float]:
    # adapt_bytes begins with adaptation_field_length then flags etc.
    if not adapt_bytes:
        return None
    if len(adapt_bytes) < 2:
        return None
    af_len = adapt_bytes[0]
    if af_len == 0:
        return None
    flags = adapt_bytes[1]
    pcr_flag = (flags & 0x10) != 0
    if not pcr_flag:
        return None
    # PCR present at bytes 2..7 (6 bytes) if available
    if len(adapt_bytes) < 8:
        return None
    pcr_bytes = adapt_bytes[2:8]
    if len(pcr_bytes) != 6:
        return None
    p = int.from_bytes(pcr_bytes, byteorder='big')
    # PCR_base is upper 33 bits
    pcr_base = p >> 15
    pcr_ext = p & 0x1FF
    # compute seconds: (pcr_base*300 + pcr_ext) / 27000000
    pcr_seconds = (pcr_base * 300 + pcr_ext) / 27000000.0
    return pcr_seconds


def extract_pts_dts(payload: bytes, pusi: bool) -> Tuple[Optional[float], Optional[float]]:
    """Extract PTS and DTS from PES packet if present"""
    if not pusi or not payload or len(payload) < 14:
        return None, None
    
    # Check for PES start code (0x000001)
    if len(payload) < 3 or payload[0] != 0x00 or payload[1] != 0x00 or payload[2] != 0x01:
        return None, None
    
    # Stream ID at offset 3
    stream_id = payload[3]
    # Skip non-video/audio streams
    if stream_id == 0xBE or stream_id == 0xBF:  # padding/private stream 2
        return None, None
    
    if len(payload) < 9:
        return None, None
    
    # PES header data at offset 6-8
    pts_dts_flags = (payload[7] & 0xC0) >> 6
    header_data_length = payload[8]
    
    pts = None
    dts = None
    
    # PTS present (10 or 11)
    if pts_dts_flags >= 2 and len(payload) >= 14:
        pts_bytes = payload[9:14]
        if len(pts_bytes) == 5:
            pts_val = ((pts_bytes[0] & 0x0E) << 29) | (pts_bytes[1] << 22) | \
                      ((pts_bytes[2] & 0xFE) << 14) | (pts_bytes[3] << 7) | \
                      ((pts_bytes[4] & 0xFE) >> 1)
            pts = pts_val / 90000.0  # Convert to seconds
    
    # DTS present (11)
    if pts_dts_flags == 3 and len(payload) >= 19:
        dts_bytes = payload[14:19]
        if len(dts_bytes) == 5:
            dts_val = ((dts_bytes[0] & 0x0E) << 29) | (dts_bytes[1] << 22) | \
                      ((dts_bytes[2] & 0xFE) << 14) | (dts_bytes[3] << 7) | \
                      ((dts_bytes[4] & 0xFE) >> 1)
            dts = dts_val / 90000.0  # Convert to seconds
    
    return pts, dts


def get_stream_type_name(stream_type: int) -> str:
    """Return human-readable name for stream type"""
    stream_types = {
        0x00: "Reserved",
        0x01: "MPEG-1 Video",
        0x02: "MPEG-2 Video",
        0x03: "MPEG-1 Audio",
        0x04: "MPEG-2 Audio",
        0x05: "Private Sections",
        0x06: "PES Private Data",
        0x0F: "MPEG-2 AAC Audio",
        0x10: "MPEG-4 Video",
        0x11: "MPEG-4 AAC Audio (LATM)",
        0x1B: "H.264/AVC Video",
        0x24: "H.265/HEVC Video",
        0x80: "PCM Audio",
        0x81: "AC-3 Audio",
        0x82: "DTS Audio",
        0x83: "TrueHD Audio",
        0x84: "E-AC-3 Audio",
        0x85: "DTS-HD Audio",
        0x86: "SCTE-35",
        0x87: "E-AC-3 Audio",
        0xA1: "E-AC-3 Audio (Secondary)",
        0xA2: "DTS Audio (Secondary)",
    }
    return stream_types.get(stream_type, f"Unknown (0x{stream_type:02X})")


def detect_ac3_bitrate(payload: bytes) -> Optional[int]:
    """Best-effort AC-3 bitrate sniff from a small payload sample.
    Returns bitrate in bps if detected, else None.
    """
    if not payload or len(payload) < 7:
        return None
    sync = b"\x0b\x77"
    idx = payload.find(sync)
    scan_limit = min(len(payload), 8192)
    pos = 0
    while idx == -1 and pos + 6 < scan_limit:
        pos += 1
        idx = payload[pos:scan_limit].find(sync)
        if idx != -1:
            idx += pos
    if idx == -1 or idx + 7 > len(payload):
        return None
    b4 = payload[idx + 4]
    frmsizecod = b4 & 0x3F
    if frmsizecod > 37:
        return None
    bitrate_table = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640]
    index = frmsizecod // 2
    if index >= len(bitrate_table):
        return None
    return bitrate_table[index] * 1000


def detect_adts_bitrate(payload: bytes) -> Optional[int]:
    """Detect AAC ADTS bitrate from sync/header. Returns bps or None."""
    if not payload or len(payload) < 7:
        return None
    scan_limit = min(len(payload), 8192)
    pos = 0
    while pos + 7 <= scan_limit:
        if payload[pos] == 0xFF and (payload[pos + 1] & 0xF0) == 0xF0:
            br_index = (payload[pos + 2] & 0xF8) >> 3
            if br_index == 0x0F:
                return None
            br_table = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384]
            if 0 <= br_index < len(br_table):
                return br_table[br_index] * 1000
        pos += 1
    return None


def detect_mpeg_audio_bitrate(payload: bytes) -> Optional[int]:
    """Detect MPEG audio (Layer II/III) bitrate. Returns bps or None."""
    if not payload or len(payload) < 4:
        return None
    scan_limit = min(len(payload), 8192)
    pos = 0
    while pos + 4 <= scan_limit:
        b0, b1, b2, b3 = payload[pos:pos+4]
        if b0 == 0xFF and (b1 & 0xE0) == 0xE0:
            version_id = (b1 >> 3) & 0x03
            layer = (b1 >> 1) & 0x03
            br_index = (b2 >> 4) & 0x0F
            if layer in (0x01, 0x02, 0x03) and br_index not in (0, 0x0F):
                layer_idx = {0x01: 'L3', 0x02: 'L2', 0x03: 'L1'}.get(layer)
                if layer_idx is None:
                    return None
                if version_id == 0x03:  # MPEG1
                    l3 = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384]
                    l2 = [32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448]
                    table = l3 if layer_idx == 'L3' else l2 if layer_idx == 'L2' else None
                else:  # MPEG2/2.5
                    l3 = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]
                    l2 = [32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256]
                    table = l3 if layer_idx == 'L3' else l2 if layer_idx == 'L2' else None
                if table and 1 <= br_index <= len(table):
                    return table[br_index - 1] * 1000
        pos += 1
    return None


def detect_audio_bitrate(payload: bytes, stream_type: Optional[int]) -> Optional[int]:
    """Try multiple audio header sniffers to recover nominal bitrate."""
    if not payload:
        return None
    sniff = detect_ac3_bitrate(payload)
    if sniff:
        return sniff
    sniff = detect_adts_bitrate(payload)
    if sniff:
        return sniff
    sniff = detect_mpeg_audio_bitrate(payload)
    if sniff:
        return sniff
    return None


def parse_descriptors(data: bytes, length: int) -> List[Dict[str, object]]:
    """Parse descriptor loop and return list of descriptors"""
    descriptors = []
    pos = 0
    while pos + 1 < length and pos + 1 < len(data):
        tag = data[pos]
        desc_len = data[pos + 1]
        if pos + 2 + desc_len > len(data):
            break
        desc_data = data[pos + 2: pos + 2 + desc_len]
        entry = {
            'tag': tag,
            'tag_name': get_descriptor_name(tag),
            'length': desc_len,
            'data': desc_data.hex()
        }
        # Decode selected descriptors for richer info
        try:
            if tag == 0x59:  # Subtitling descriptor
                entry['decoded'] = parse_subtitling_descriptor(desc_data)
            elif tag == 0x56:  # Teletext descriptor
                entry['decoded'] = parse_teletext_descriptor(desc_data)
        except Exception:
            pass
        descriptors.append(entry)
        pos += 2 + desc_len
    return descriptors


def get_descriptor_name(tag: int) -> str:
    """Return human-readable descriptor name"""
    descriptor_tags = {
        0x02: "Video Stream",
        0x03: "Audio Stream",
        0x05: "Registration",
        0x09: "CA (Conditional Access)",
        0x0A: "ISO 639 Language",
        0x0E: "Maximum Bitrate",
        0x28: "AVC Video",
        0x2A: "AVC Timing and HRD",
        0x38: "HEVC Video",
        0x52: "Stream Identifier",
        0x56: "Teletext",
        0x59: "Subtitling",
        0x6A: "AC-3",
        0x7A: "E-AC-3",
        0x7B: "DTS Audio",
        0x81: "AC-3 (ATSC)",
        0xCC: "AC-3 Audio (DVB)",
    }
    return descriptor_tags.get(tag, f"Unknown (0x{tag:02X})")


def parse_subtitling_descriptor(desc_data: bytes) -> List[Dict[str, object]]:
    """Parse DVB Subtitling descriptor (tag 0x59) per ETSI EN 300 468.
    Returns list of entries: language, type, composition_page_id, ancillary_page_id.
    Each entry is 8 bytes.
    """
    out: List[Dict[str, object]] = []
    pos = 0
    while pos + 8 <= len(desc_data):
        lang = desc_data[pos:pos+3].decode('latin-1', 'ignore')
        subt_type = desc_data[pos+3]
        comp_page = (desc_data[pos+4] << 8) | desc_data[pos+5]
        anc_page = (desc_data[pos+6] << 8) | desc_data[pos+7]
        # Map common subtitling types
        type_names = {
            0x10: 'Normal subtitles',
            0x11: 'Normal subtitles (for the hard of hearing)',
            0x20: 'Teletext subtitles',
            0x21: 'Teletext subtitles (for the hard of hearing)',
        }
        out.append({
            'language': lang,
            'subtitling_type': subt_type,
            'type_name': type_names.get(subt_type, f'type 0x{subt_type:02X}'),
            'composition_page_id': comp_page,
            'ancillary_page_id': anc_page,
        })
        pos += 8
    return out


def parse_teletext_descriptor(desc_data: bytes) -> List[Dict[str, object]]:
    """Parse Teletext descriptor (tag 0x56) per ETSI EN 300 468.
    Returns list of entries: language, teletext_type, magazine_page.
    Each entry is 5 bytes.
    """
    out: List[Dict[str, object]] = []
    pos = 0
    while pos + 5 <= len(desc_data):
        lang = desc_data[pos:pos+3].decode('latin-1', 'ignore')
        txt_type = desc_data[pos+3]
        mp = desc_data[pos+4]
        type_names = {
            0x01: 'Initial teletext page',
            0x02: 'Teletext subtitles',
            0x03: 'Additional info',
            0x04: 'Programme schedule',
            0x05: 'Subtitle for hard of hearing',
        }
        out.append({
            'language': lang,
            'teletext_type': txt_type,
            'type_name': type_names.get(txt_type, f'type 0x{txt_type:02X}'),
            'magazine_and_page': mp,
        })
        pos += 5
    return out


def parse_ac3_descriptor(desc_data: bytes) -> Dict[str, object]:
    """
    Parse ATSC AC-3 Audio Descriptor (tag 0x81) or DVB AC-3 descriptor (tag 0x6A/0xCC)
    Returns sample rate, channels, bitrate info
    """
    result = {}
    if len(desc_data) < 1:
        return result
    
    # ATSC AC-3 descriptor format (A/52)
    try:
        # Byte 0: sample_rate_code (3 bits), bsid (5 bits)
        sample_rate_code = (desc_data[0] & 0xE0) >> 5
        bsid = desc_data[0] & 0x1F
        
        sample_rates = {0: 48000, 1: 44100, 2: 32000}
        if sample_rate_code in sample_rates:
            result['sample_rate'] = sample_rates[sample_rate_code]
        
        # BSID 16 or greater = E-AC-3
        if bsid >= 16:
            result['codec'] = 'E-AC-3'
        else:
            result['codec'] = 'AC-3'
        
        # Byte 1: bit_rate_code (6 bits), surround mode, etc
        if len(desc_data) >= 2:
            bit_rate_code = (desc_data[1] & 0xFC) >> 2
            # AC-3 bitrates in kbps
            bitrates = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640]
            if bit_rate_code < len(bitrates):
                result['bitrate'] = bitrates[bit_rate_code] * 1000
        
        # Byte 2: audio coding mode (channels)
        if len(desc_data) >= 3:
            acmod = (desc_data[2] & 0xE0) >> 5
            # AC-3 audio coding modes
            channels_map = {
                0: 2,  # 1+1 (dual mono)
                1: 1,  # 1/0 (mono)
                2: 2,  # 2/0 (stereo)
                3: 3,  # 3/0
                4: 3,  # 2/1
                5: 4,  # 3/1
                6: 4,  # 2/2
                7: 5   # 3/2
            }
            result['channels'] = channels_map.get(acmod, 2)
            
            # Check for LFE (subwoofer)
            if desc_data[2] & 0x10:
                result['channels'] += 1  # Add .1 for LFE
                result['lfe'] = True
    except:
        pass
    
    return result


def get_enhanced_stream_description(stream_type: int, descriptors: List[Dict]) -> str:
    """
    Generate enhanced stream description with audio format details
    Returns a detailed description string for display
    """
    type_name = get_stream_type_name(stream_type)
    
    # E-AC-3 Audio - stream type 0x87
    if stream_type == 0x87:
        # This is always E-AC-3
        # Try to get details from descriptor 0xCC (DVB AC-3 descriptor)
        for desc in descriptors:
            if desc.get('tag') == 0xCC:
                try:
                    data = bytes.fromhex(desc.get('data', ''))
                    ac3_info = parse_ac3_descriptor(data)
                    
                    sample_rate = ac3_info.get('sample_rate', 0)
                    channels = ac3_info.get('channels', 0)
                    bitrate = ac3_info.get('bitrate', 0)
                    lfe = ac3_info.get('lfe', False)
                    
                    parts = ['E-AC-3']  # Force E-AC-3 for type 0x87
                    if sample_rate:
                        parts.append(f"{sample_rate/1000:.1f}kHz")
                    if channels:
                        ch_str = f"{channels-1}.1" if lfe else f"{channels}.0"
                        parts.append(f"{ch_str}ch")
                    if bitrate:
                        parts.append(f"{bitrate/1000:.0f}kbps")
                    
                    return " ".join(parts)
                except:
                    pass
        return "E-AC-3"  # Default for 0x87
    
    # AC-3 / E-AC-3 with descriptors
    for desc in descriptors:
        tag = desc.get('tag')
        data_hex = desc.get('data', '')
        
        if tag in [0x81, 0x6A, 0xCC]:  # AC-3 descriptors
            try:
                data = bytes.fromhex(data_hex)
                ac3_info = parse_ac3_descriptor(data)
                
                if ac3_info:
                    codec = ac3_info.get('codec', 'AC-3')
                    sample_rate = ac3_info.get('sample_rate', 0)
                    channels = ac3_info.get('channels', 0)
                    bitrate = ac3_info.get('bitrate', 0)
                    lfe = ac3_info.get('lfe', False)
                    
                    parts = [codec]
                    if sample_rate:
                        parts.append(f"{sample_rate/1000:.1f}kHz")
                    if channels:
                        ch_str = f"{channels-1}.1" if lfe else f"{channels}.0"
                        parts.append(f"{ch_str}ch")
                    if bitrate:
                        parts.append(f"{bitrate/1000:.0f}kbps")
                    
                    return " ".join(parts)
            except:
                pass
        
        # Check for HDMV LPCM (BluRay PCM)
        if tag == 0x05:  # Registration descriptor
            try:
                data = bytes.fromhex(data_hex)
                if len(data) >= 4:
                    format_id = data[:4].decode('ascii', errors='ignore')
                    if format_id == 'BSSD':
                        # This is BluRay LPCM - try to get detailed info
                        pcm_info = parse_pcm_audio_info(descriptors)
                        sample_rate = pcm_info.get('sample_rate', 48000)
                        channels = pcm_info.get('channels', 2)
                        bit_depth = pcm_info.get('bit_depth', 16)
                        
                        # Check if we have actual descriptor data or just defaults
                        has_lpcm_desc = any(d.get('tag') in [0x80, 0x81] for d in descriptors)
                        if has_lpcm_desc:
                            return f"HDMV LPCM {sample_rate/1000:.0f}kHz {channels}ch {bit_depth}-bit"
                        else:
                            # No detailed descriptor, just show that it's BluRay PCM
                            return "HDMV LPCM (BluRay PCM Audio)"
            except:
                pass
    
    # Check for standard PCM (stream type 0x80)
    if stream_type == 0x80:
        pcm_info = parse_pcm_audio_info(descriptors)
        sample_rate = pcm_info.get('sample_rate', 48000)
        channels = pcm_info.get('channels', 2)
        bit_depth = pcm_info.get('bit_depth', 16)
        
        return f"PCM {sample_rate/1000:.0f}kHz {channels}ch {bit_depth}-bit"
    
    return type_name


def parse_pcm_audio_info(descriptors: List[Dict]) -> Dict[str, int]:
    """
    Extract PCM audio parameters from descriptors
    Returns dict with 'channels', 'sample_rate', 'bit_depth'
    
    For HDMV LPCM (Blu-ray), descriptor tag 0x80 or 0x81
    """
    info = {
        'channels': 2,  # Default stereo
        'sample_rate': 48000,  # Default 48kHz  
        'bit_depth': 16  # Default 16-bit
    }
    
    for desc in descriptors:
        tag = desc.get('tag')
        data_hex = desc.get('data', '')
        
        # HDMV LPCM audio stream descriptor (Blu-ray)
        if tag == 0x80 or tag == 0x81:
            try:
                data = bytes.fromhex(data_hex)
                if len(data) >= 4:
                    # Byte 0: sampling frequency (upper 4 bits)
                    # 0001 = 48kHz, 0100 = 96kHz, 0101 = 192kHz
                    sample_code = (data[0] & 0xF0) >> 4
                    if sample_code == 0x01:
                        info['sample_rate'] = 48000
                    elif sample_code == 0x04:
                        info['sample_rate'] = 96000
                    elif sample_code == 0x05:
                        info['sample_rate'] = 192000
                    
                    # Byte 1: number of channels (lower 4 bits)
                    # 0001 = mono, 0011 = stereo, 0110 = 6ch, 0111 = 8ch
                    channel_code = data[1] & 0x0F
                    if channel_code == 0x01:
                        info['channels'] = 1
                    elif channel_code == 0x03:
                        info['channels'] = 2
                    elif channel_code == 0x06:
                        info['channels'] = 6
                    elif channel_code == 0x07 or channel_code == 0x08:
                        info['channels'] = 8
                    
                    # Byte 2: bits per sample (upper 2 bits)
                    # 01 = 16-bit, 10 = 20-bit, 11 = 24-bit
                    bit_code = (data[2] & 0xC0) >> 6
                    if bit_code == 0x01:
                        info['bit_depth'] = 16
                    elif bit_code == 0x02:
                        info['bit_depth'] = 20
                    elif bit_code == 0x03:
                        info['bit_depth'] = 24
            except:
                pass
    
    return info


def parse_ac3_audio_info(descriptors: List[Dict], stream_type: int) -> Dict[str, object]:
    """
    Extract AC-3 audio parameters from descriptors
    Returns dict with 'format' (ATSC/DVB), 'channels', 'sample_rate', 'bitrate'
    
    Descriptor tags:
    - 0x81: ATSC AC-3 audio descriptor
    - 0x6A: DVB AC-3 descriptor  
    - 0xCC: AC-3 descriptor (alternate)
    """
    info = {
        'format': 'Unknown',
        'channels': 0,
        'sample_rate': 48000,  # AC-3 is typically 48kHz
        'bitrate': 0,
        'acmod': None,
        'bsmod': None
    }
    
    # Determine format based on stream type
    if stream_type == 0x81:
        info['format'] = 'ATSC AC-3'
    
    for desc in descriptors:
        tag = desc.get('tag')
        data_hex = desc.get('data', '')
        
        try:
            data = bytes.fromhex(data_hex)
            
            # ATSC AC-3 audio descriptor (A/52)
            if tag == 0x81:
                info['format'] = 'ATSC AC-3'
                if len(data) >= 3:
                    # Byte 0: sample_rate_code (3 bits), bsid (5 bits)
                    sample_rate_code = (data[0] & 0xE0) >> 5
                    bsid = data[0] & 0x1F
                    
                    # Byte 1: bit_rate_code (6 bits), surround_mode (2 bits)
                    bit_rate_code = (data[1] & 0xFC) >> 2
                    
                    # Byte 2: bsmod (3 bits), num_channels (4 bits), full_svc (1 bit)
                    bsmod = (data[2] & 0xE0) >> 5
                    num_channels = (data[2] & 0x1E) >> 1
                    
                    info['bsmod'] = bsmod
                    
                    # Decode sample rate (typically 48kHz for AC-3)
                    sample_rates = {0: 48000, 1: 44100, 2: 32000}
                    info['sample_rate'] = sample_rates.get(sample_rate_code, 48000)
                    
                    # Decode bitrate (in kbps)
                    bitrates = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640]
                    if bit_rate_code < len(bitrates):
                        info['bitrate'] = bitrates[bit_rate_code]
                    
                    # Channel count
                    info['channels'] = num_channels
                    
            # DVB AC-3 descriptor (EN 300 468)
            elif tag == 0x6A:
                info['format'] = 'DVB AC-3'
                if len(data) >= 1:
                    # Byte 0: various flags
                    component_type_flag = (data[0] & 0x80) >> 7
                    bsid_flag = (data[0] & 0x40) >> 6
                    mainid_flag = (data[0] & 0x20) >> 5
                    asvc_flag = (data[0] & 0x10) >> 4
                    
                    idx = 1
                    if component_type_flag and idx < len(data):
                        component_type = data[idx]
                        idx += 1
                    if bsid_flag and idx < len(data):
                        bsid = data[idx]
                        idx += 1
                    if mainid_flag and idx < len(data):
                        mainid = data[idx]
                        idx += 1
                    if asvc_flag and idx < len(data):
                        asvc = data[idx]
                        idx += 1
                    
                    # Additional info parsing if available
                    if idx + 2 < len(data):
                        additional_info = data[idx:]
                        # Parse acmod if present
                        if len(additional_info) >= 1:
                            acmod = (additional_info[0] & 0x70) >> 4
                            info['acmod'] = acmod
                            # Map acmod to channel count
                            acmod_channels = {
                                0: 2,  # 1+1 (dual mono)
                                1: 1,  # 1/0 (mono)
                                2: 2,  # 2/0 (stereo)
                                3: 3,  # 3/0 (L, C, R)
                                4: 3,  # 2/1 (L, R, S)
                                5: 4,  # 3/1 (L, C, R, S)
                                6: 4,  # 2/2 (L, R, SL, SR)
                                7: 5   # 3/2 (L, C, R, SL, SR)
                            }
                            info['channels'] = acmod_channels.get(acmod, 2)
                            
        except:
            pass
    
    return info


def parse_eac3_audio_info(descriptors: List[Dict]) -> Dict[str, object]:
    """
    Extract E-AC-3 audio parameters from descriptors
    Returns dict with 'format', 'channels', 'sample_rate', 'bitrate'
    
    Descriptor tags:
    - 0x7A: DVB E-AC-3 descriptor
    - 0xCC: E-AC-3 descriptor (alternate)
    """
    info = {
        'format': 'DVB E-AC-3',
        'channels': 0,
        'sample_rate': 48000,
        'bitrate': 0
    }
    
    for desc in descriptors:
        tag = desc.get('tag')
        data_hex = desc.get('data', '')
        
        try:
            data = bytes.fromhex(data_hex)
            
            # DVB E-AC-3 descriptor (EN 300 468)
            if tag == 0x7A:
                info['format'] = 'DVB E-AC-3'
                if len(data) >= 1:
                    # Byte 0: various flags
                    component_type_flag = (data[0] & 0x80) >> 7
                    bsid_flag = (data[0] & 0x40) >> 6
                    mainid_flag = (data[0] & 0x20) >> 5
                    asvc_flag = (data[0] & 0x10) >> 4
                    mixinfoexists = (data[0] & 0x08) >> 3
                    substream1_flag = (data[0] & 0x04) >> 2
                    substream2_flag = (data[0] & 0x02) >> 1
                    substream3_flag = (data[0] & 0x01)
                    
                    idx = 1
                    if component_type_flag and idx < len(data):
                        component_type = data[idx]
                        idx += 1
                    if bsid_flag and idx < len(data):
                        bsid = data[idx]
                        idx += 1
                    if mainid_flag and idx < len(data):
                        mainid = data[idx]
                        idx += 1
                    if asvc_flag and idx < len(data):
                        asvc = data[idx]
                        idx += 1
                    
                    # Additional info parsing if available
                    if idx < len(data):
                        additional_info = data[idx:]
                        if len(additional_info) >= 1:
                            # Parse acmod if present
                            acmod = (additional_info[0] & 0x70) >> 4
                            # Map acmod to channel count
                            acmod_channels = {
                                0: 2,  # 1+1 (dual mono)
                                1: 1,  # 1/0 (mono)
                                2: 2,  # 2/0 (stereo)
                                3: 3,  # 3/0 (L, C, R)
                                4: 3,  # 2/1 (L, R, S)
                                5: 4,  # 3/1 (L, C, R, S)
                                6: 4,  # 2/2 (L, R, SL, SR)
                                7: 5   # 3/2 (L, C, R, SL, SR)
                            }
                            info['channels'] = acmod_channels.get(acmod, 2)
                            
        except:
            pass
    
    return info


def parse_mpeg2_sequence_header(data: bytes) -> Optional[Dict[str, object]]:
    """Parse MPEG-2 sequence header and detect syntax errors"""
    # Look for sequence header start code: 0x000001B3
    if len(data) < 12:
        return None
    
    for i in range(len(data) - 11):
        if data[i] == 0x00 and data[i+1] == 0x00 and data[i+2] == 0x01 and data[i+3] == 0xB3:
            # Found sequence header
            errors = []
            warnings = []
            
            try:
                # Parse sequence header (minimum 12 bytes after start code)
                if i + 12 > len(data):
                    errors.append("Incomplete sequence header")
                    return {'errors': errors, 'warnings': warnings}
                
                # Horizontal size (12 bits)
                horizontal_size = (data[i+4] << 4) | (data[i+5] >> 4)
                # Vertical size (12 bits)
                vertical_size = ((data[i+5] & 0x0F) << 8) | data[i+6]
                # Aspect ratio (4 bits)
                aspect_ratio = data[i+7] >> 4
                # Frame rate (4 bits)
                frame_rate_code = data[i+7] & 0x0F
                # Bit rate (18 bits)
                bit_rate = (data[i+8] << 10) | (data[i+9] << 2) | (data[i+10] >> 6)
                # VBV buffer size (10 bits)
                vbv_buffer = ((data[i+10] & 0x1F) << 5) | (data[i+11] >> 3)
                
                # Validate values
                if horizontal_size == 0 or horizontal_size > 16383:
                    errors.append(f"Invalid horizontal size: {horizontal_size}")
                if vertical_size == 0 or vertical_size > 16383:
                    errors.append(f"Invalid vertical size: {vertical_size}")
                if aspect_ratio == 0 or aspect_ratio > 4:
                    warnings.append(f"Reserved aspect ratio code: {aspect_ratio}")
                if frame_rate_code == 0 or frame_rate_code > 8:
                    errors.append(f"Invalid frame rate code: {frame_rate_code}")
                if bit_rate == 0x3FFFF:  # VBR marker
                    warnings.append("VBR stream (bitrate = 0x3FFFF)")
                
                aspect_ratio_names = {
                    1: "1:1 (Square)",
                    2: "4:3",
                    3: "16:9",
                    4: "2.21:1"
                }
                
                frame_rate_values = {
                    1: 23.976, 2: 24.0, 3: 25.0, 4: 29.97,
                    5: 30.0, 6: 50.0, 7: 59.94, 8: 60.0
                }
                
                return {
                    'type': 'MPEG-2 Sequence Header',
                    'horizontal_size': horizontal_size,
                    'vertical_size': vertical_size,
                    'aspect_ratio_code': aspect_ratio,
                    'aspect_ratio': aspect_ratio_names.get(aspect_ratio, f"Unknown ({aspect_ratio})"),
                    'frame_rate_code': frame_rate_code,
                    'frame_rate': frame_rate_values.get(frame_rate_code, 0.0),
                    'bit_rate': bit_rate * 400,  # in bps
                    'vbv_buffer_size': vbv_buffer * 16 * 1024,  # in bits
                    'errors': errors,
                    'warnings': warnings,
                    'offset': i
                }
            except Exception as e:
                errors.append(f"Parse error: {str(e)}")
                return {'errors': errors, 'warnings': warnings}
    
    return None


class BitReader:
    """Helper class for reading H.264 bit streams"""
    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0  # bit position
    
    def read_bits(self, n: int) -> int:
        """Read n bits and return as integer"""
        if n == 0:
            return 0
        result = 0
        for _ in range(n):
            byte_pos = self.pos // 8
            bit_pos = 7 - (self.pos % 8)
            if byte_pos >= len(self.data):
                return 0
            bit = (self.data[byte_pos] >> bit_pos) & 1
            result = (result << 1) | bit
            self.pos += 1
        return result
    
    def bits_available(self) -> bool:
        """Check if there are more bits to read"""
        return (self.pos // 8) < len(self.data)
    
    def read_ue(self) -> int:
        """Read unsigned exponential-Golomb code"""
        leading_zeros = 0
        while self.read_bits(1) == 0:
            leading_zeros += 1
            if leading_zeros > 32:  # Prevent infinite loop
                return 0
        if leading_zeros == 0:
            return 0
        value = self.read_bits(leading_zeros)
        return (1 << leading_zeros) - 1 + value
    
    def read_se(self) -> int:
        """Read signed exponential-Golomb code"""
        value = self.read_ue()
        if value == 0:
            return 0
        return (value + 1) // 2 if value % 2 else -(value // 2)
    
    def skip_scaling_list(self, size: int):
        """Skip scaling list (used in SPS parsing)"""
        last_scale = 8
        next_scale = 8
        for _ in range(size):
            if next_scale != 0:
                delta_scale = self.read_se()
                next_scale = (last_scale + delta_scale + 256) % 256
            last_scale = next_scale if next_scale != 0 else last_scale


def parse_h264_sps(data: bytes) -> Optional[Dict[str, object]]:
    """Parse H.264 SPS (Sequence Parameter Set) and detect syntax errors"""
    # Look for SPS NAL unit: 0x00000001 followed by NAL type 7 (0x67 or 0x27)
    errors = []
    warnings = []
    
    for i in range(len(data) - 5):
        # Check for start code (0x000001 or 0x00000001)
        start_code_len = 0
        if data[i] == 0x00 and data[i+1] == 0x00 and data[i+2] == 0x01:
            start_code_len = 3
        elif i < len(data) - 6 and data[i] == 0x00 and data[i+1] == 0x00 and data[i+2] == 0x00 and data[i+3] == 0x01:
            start_code_len = 4
        
        if start_code_len > 0:
            nal_start = i + start_code_len
            if nal_start >= len(data):
                continue
            
            nal_header = data[nal_start]
            nal_ref_idc = (nal_header >> 5) & 0x03
            nal_unit_type = nal_header & 0x1F
            
            # Check for SPS (type 7)
            if nal_unit_type == 7:
                try:
                    if nal_start + 4 > len(data):
                        errors.append("Incomplete SPS NAL unit")
                        return {'errors': errors, 'warnings': warnings}
                    
                    # Check forbidden_zero_bit
                    if nal_header & 0x80:
                        errors.append("SPS forbidden_zero_bit is not zero")
                    
                    # NAL ref_idc should be non-zero for SPS
                    if nal_ref_idc == 0:
                        errors.append("SPS nal_ref_idc is zero (should be non-zero)")
                    
                    # Profile and level
                    profile_idc = data[nal_start + 1]
                    constraint_flags = data[nal_start + 2]
                    level_idc = data[nal_start + 3]
                    
                    # Validate profile
                    valid_profiles = [66, 77, 88, 100, 110, 122, 244, 44, 83, 86, 118, 128]
                    if profile_idc not in valid_profiles:
                        warnings.append(f"Unknown profile_idc: {profile_idc}")
                    
                    # Validate level (should be valid level code)
                    valid_levels = [10, 11, 12, 13, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52]
                    if level_idc not in valid_levels:
                        warnings.append(f"Non-standard level_idc: {level_idc}")
                    
                    profile_names = {
                        66: "Baseline",
                        77: "Main",
                        88: "Extended",
                        100: "High",
                        110: "High 10",
                        122: "High 4:2:2",
                        244: "High 4:4:4",
                        44: "CAVLC 4:4:4"
                    }
                    
                    # Try to parse resolution and framerate from SPS RBSP
                    width = None
                    height = None
                    framerate = None
                    
                    try:
                        # Remove emulation prevention bytes (0x000003 -> 0x0000)
                        # Start after NAL header (4 bytes: profile, constraint, level, then RBSP data)
                        rbsp_data = data[nal_start + 4:]
                        
                        # Remove emulation prevention bytes
                        out = bytearray()
                        zeros = 0
                        for b in rbsp_data:
                            if zeros >= 2 and b == 0x03:
                                # Skip emulation prevention byte
                                zeros = 0
                                continue
                            out.append(b)
                            if b == 0x00:
                                zeros += 1
                            else:
                                zeros = 0
                        rbsp_data = bytes(out)
                        
                        # Start parsing SPS RBSP
                        br = BitReader(rbsp_data)
                        
                        seq_parameter_set_id = br.read_ue()
                        
                        # Parse chroma format (for High profiles)
                        if profile_idc in [100, 110, 122, 244, 44, 83, 86, 118, 128]:
                            chroma_format_idc = br.read_ue()
                            if chroma_format_idc == 3:
                                separate_colour_plane_flag = br.read_bits(1)
                            bit_depth_luma = br.read_ue() + 8
                            bit_depth_chroma = br.read_ue() + 8
                            qpprime_y_zero_transform_bypass_flag = br.read_bits(1)
                            seq_scaling_matrix_present_flag = br.read_bits(1)
                            if seq_scaling_matrix_present_flag:
                                for i in range(8 if chroma_format_idc != 3 else 12):
                                    seq_scaling_list_present_flag = br.read_bits(1)
                                    if seq_scaling_list_present_flag:
                                        br.skip_scaling_list(16 if i < 6 else 64)
                        
                        log2_max_frame_num = br.read_ue() + 4
                        pic_order_cnt_type = br.read_ue()
                        
                        if pic_order_cnt_type == 0:
                            log2_max_pic_order_cnt_lsb = br.read_ue() + 4
                        elif pic_order_cnt_type == 1:
                            delta_pic_order_always_zero_flag = br.read_bits(1)
                            offset_for_non_ref_pic = br.read_se()
                            offset_for_top_to_bottom_field = br.read_se()
                            num_ref_frames_in_pic_order_cnt_cycle = br.read_ue()
                            for _ in range(num_ref_frames_in_pic_order_cnt_cycle):
                                offset_for_ref_frame = br.read_se()
                        
                        max_num_ref_frames = br.read_ue()
                        gaps_in_frame_num_value_allowed_flag = br.read_bits(1)
                        
                        # Resolution
                        pic_width_in_mbs = br.read_ue() + 1
                        pic_height_in_map_units = br.read_ue() + 1
                        
                        frame_mbs_only_flag = br.read_bits(1)
                        if not frame_mbs_only_flag:
                            mb_adaptive_frame_field_flag = br.read_bits(1)
                        
                        # Store for framerate calculation
                        is_progressive = frame_mbs_only_flag
                        
                        direct_8x8_inference_flag = br.read_bits(1)
                        
                        # Cropping
                        frame_cropping_flag = br.read_bits(1)
                        crop_left = crop_right = crop_top = crop_bottom = 0
                        if frame_cropping_flag:
                            crop_left = br.read_ue()
                            crop_right = br.read_ue()
                            crop_top = br.read_ue()
                            crop_bottom = br.read_ue()
                        
                        # Calculate resolution
                        width = pic_width_in_mbs * 16 - (crop_left + crop_right) * 2
                        height = (2 - frame_mbs_only_flag) * pic_height_in_map_units * 16 - (crop_top + crop_bottom) * 2
                        
                        # VUI parameters for framerate, HRD, and display info
                        vui_params = {}
                        vui_parameters_present_flag = br.read_bits(1)
                        if vui_parameters_present_flag:
                            def parse_hrd_parameters(reader: BitReader):
                                hrd = {}
                                hrd['cpb_cnt_minus1'] = reader.read_ue()
                                hrd['bit_rate_scale'] = reader.read_bits(4)
                                hrd['cpb_size_scale'] = reader.read_bits(4)
                                hrd['cpb'] = []
                                for _ in range(hrd['cpb_cnt_minus1'] + 1):
                                    bit_rate_value_minus1 = reader.read_ue()
                                    cpb_size_value_minus1 = reader.read_ue()
                                    cbr_flag = reader.read_bits(1)
                                    hrd['cpb'].append({
                                        'bit_rate_value_minus1': bit_rate_value_minus1,
                                        'cpb_size_value_minus1': cpb_size_value_minus1,
                                        'cbr_flag': cbr_flag
                                    })
                                hrd['initial_cpb_removal_delay_length_minus1'] = reader.read_bits(5)
                                hrd['cpb_removal_delay_length_minus1'] = reader.read_bits(5)
                                hrd['dpb_output_delay_length_minus1'] = reader.read_bits(5)
                                hrd['time_offset_length'] = reader.read_bits(5)
                                return hrd

                            aspect_ratio_info_present_flag = br.read_bits(1)
                            if aspect_ratio_info_present_flag:
                                aspect_ratio_idc = br.read_bits(8)
                                vui_params['vui_aspect_ratio_idc'] = aspect_ratio_idc
                                if aspect_ratio_idc == 255:  # Extended_SAR
                                    sar_width = br.read_bits(16)
                                    sar_height = br.read_bits(16)
                                    vui_params['vui_sar_width'] = sar_width
                                    vui_params['vui_sar_height'] = sar_height
                            
                            overscan_info_present_flag = br.read_bits(1)
                            if overscan_info_present_flag:
                                overscan_appropriate_flag = br.read_bits(1)
                                vui_params['vui_overscan_appropriate_flag'] = overscan_appropriate_flag
                            
                            video_signal_type_present_flag = br.read_bits(1)
                            if video_signal_type_present_flag:
                                video_format = br.read_bits(3)
                                video_full_range_flag = br.read_bits(1)
                                vui_params['vui_video_format'] = video_format
                                vui_params['vui_video_full_range_flag'] = video_full_range_flag
                                
                                colour_description_present_flag = br.read_bits(1)
                                if colour_description_present_flag:
                                    colour_primaries = br.read_bits(8)
                                    transfer_characteristics = br.read_bits(8)
                                    matrix_coefficients = br.read_bits(8)
                                    vui_params['vui_colour_primaries'] = colour_primaries
                                    vui_params['vui_transfer_characteristics'] = transfer_characteristics
                                    vui_params['vui_matrix_coefficients'] = matrix_coefficients
                            
                            chroma_loc_info_present_flag = br.read_bits(1)
                            if chroma_loc_info_present_flag:
                                chroma_sample_loc_type_top_field = br.read_ue()
                                chroma_sample_loc_type_bottom_field = br.read_ue()
                                vui_params['vui_chroma_sample_loc_top'] = chroma_sample_loc_type_top_field
                                vui_params['vui_chroma_sample_loc_bottom'] = chroma_sample_loc_type_bottom_field
                            
                            # Timing info for framerate
                            timing_info_present_flag = br.read_bits(1)
                            if timing_info_present_flag:
                                num_units_in_tick = br.read_bits(32)
                                time_scale = br.read_bits(32)
                                fixed_frame_rate_flag = br.read_bits(1)
                                
                                vui_params['vui_num_units_in_tick'] = num_units_in_tick
                                vui_params['vui_time_scale'] = time_scale
                                vui_params['vui_fixed_frame_rate_flag'] = fixed_frame_rate_flag
                                
                                if num_units_in_tick > 0 and time_scale > 0:
                                    if 'is_progressive' in locals() and is_progressive:
                                        framerate = time_scale / (2 * num_units_in_tick)
                                    else:
                                        framerate = time_scale / num_units_in_tick

                            # HRD parameters
                            nal_hrd_parameters_present_flag = br.read_bits(1)
                            if nal_hrd_parameters_present_flag:
                                vui_params['nal_hrd'] = parse_hrd_parameters(br)
                                vui_params['nal_hrd_present_flag'] = 1
                            else:
                                vui_params['nal_hrd_present_flag'] = 0
                            
                            vcl_hrd_parameters_present_flag = br.read_bits(1)
                            if vcl_hrd_parameters_present_flag:
                                vui_params['vcl_hrd'] = parse_hrd_parameters(br)
                                vui_params['vcl_hrd_present_flag'] = 1
                            else:
                                vui_params['vcl_hrd_present_flag'] = 0

                            if nal_hrd_parameters_present_flag or vcl_hrd_parameters_present_flag:
                                vui_params['low_delay_hrd_flag'] = br.read_bits(1)
                            
                            # pic_struct_present_flag (CRITICAL for SEI pic_timing)
                            vui_params['pic_struct_present_flag'] = br.read_bits(1)
                        else:
                            vui_params['pic_struct_present_flag'] = 0
                    
                    except Exception as e:
                        # VUI parsing is optional, don't fail the whole SPS parse
                        warnings.append(f"Could not parse VUI parameters: {str(e)}")
                    
                    result = {
                        'type': 'H.264 SPS',
                        'nal_unit_type': nal_unit_type,
                        'nal_ref_idc': nal_ref_idc,
                        'profile_idc': profile_idc,
                        'profile_name': profile_names.get(profile_idc, f"Unknown ({profile_idc})"),
                        'constraint_flags': constraint_flags,
                        'level_idc': level_idc,
                        'level': f"{level_idc // 10}.{level_idc % 10}",
                        'errors': errors,
                        'warnings': warnings,
                        'offset': i
                    }
                    
                    # Add parsed SPS parameters if available
                    try:
                        if 'seq_parameter_set_id' in locals():
                            result['seq_parameter_set_id'] = seq_parameter_set_id
                        if 'frame_mbs_only_flag' in locals():
                            result['frame_mbs_only_flag'] = frame_mbs_only_flag
                        if 'chroma_format_idc' in locals():
                            result['chroma_format_idc'] = chroma_format_idc
                        if 'bit_depth_luma' in locals():
                            result['bit_depth_luma'] = bit_depth_luma
                        if 'bit_depth_chroma' in locals():
                            result['bit_depth_chroma'] = bit_depth_chroma
                        if 'log2_max_frame_num' in locals():
                            result['log2_max_frame_num'] = log2_max_frame_num
                        if 'pic_order_cnt_type' in locals():
                            result['pic_order_cnt_type'] = pic_order_cnt_type
                        if 'log2_max_pic_order_cnt_lsb' in locals():
                            result['log2_max_pic_order_cnt_lsb'] = log2_max_pic_order_cnt_lsb
                        if 'max_num_ref_frames' in locals():
                            result['max_num_ref_frames'] = max_num_ref_frames
                    except:
                        pass
                    
                    if width is not None and height is not None:
                        result['width'] = width
                        result['height'] = height
                    
                    if framerate is not None:
                        result['frame_rate'] = round(framerate, 3)
                    
                    # Merge VUI parameters into result
                    if 'vui_params' in locals() and vui_params:
                        result.update(vui_params)
                    
                    return result
                except Exception as e:
                    errors.append(f"Parse error: {str(e)}")
                    return {'errors': errors, 'warnings': warnings}
    
    return None


def parse_h264_pps(data: bytes) -> Optional[Dict[str, object]]:
    """Parse H.264 PPS (Picture Parameter Set) and detect syntax errors"""
    errors = []
    warnings = []
    
    for i in range(len(data) - 5):
        # Check for start code
        start_code_len = 0
        if data[i] == 0x00 and data[i+1] == 0x00 and data[i+2] == 0x01:
            start_code_len = 3
        elif i < len(data) - 6 and data[i] == 0x00 and data[i+1] == 0x00 and data[i+2] == 0x00 and data[i+3] == 0x01:
            start_code_len = 4
        
        if start_code_len > 0:
            nal_start = i + start_code_len
            if nal_start >= len(data):
                continue
            
            nal_header = data[nal_start]
            nal_ref_idc = (nal_header >> 5) & 0x03
            nal_unit_type = nal_header & 0x1F
            
            # Check for PPS (type 8)
            if nal_unit_type == 8:
                try:
                    # Check forbidden_zero_bit
                    if nal_header & 0x80:
                        errors.append("PPS forbidden_zero_bit is not zero")
                    
                    # NAL ref_idc should be non-zero for PPS
                    if nal_ref_idc == 0:
                        errors.append("PPS nal_ref_idc is zero (should be non-zero)")
                    
                    return {
                        'type': 'H.264 PPS',
                        'nal_unit_type': nal_unit_type,
                        'nal_ref_idc': nal_ref_idc,
                        'errors': errors,
                        'warnings': warnings,
                        'offset': i
                    }
                except Exception as e:
                    errors.append(f"Parse error: {str(e)}")
                    return {'errors': errors, 'warnings': warnings}
    
    return None


def parse_pat(payload: bytes) -> Dict[str, object]:
    # returns detailed PAT information
    result = {
        'programs': {},
        'transport_stream_id': None,
        'version': None,
        'current_next': None,
        'warnings': [],
    }
    if not payload:
        return result
    ptr = payload[0]
    if ptr + 1 >= len(payload):
        return result
    start = 1 + ptr
    if start + 8 > len(payload):
        return result
    table_id = payload[start]
    if table_id != 0x00:
        return result
    
    section_length = ((payload[start+1] & 0x0F) << 8) | payload[start+2]
    transport_stream_id = (payload[start+3] << 8) | payload[start+4]
    version = (payload[start+5] & 0x3E) >> 1
    current_next = payload[start+5] & 0x01
    
    result['transport_stream_id'] = transport_stream_id
    result['version'] = version
    result['current_next'] = current_next
    
    # program loop starts at start+8 and ends before CRC (4 bytes)
    pos = start + 8
    end = start + 3 + section_length - 4
    corrupted_count = 0
    while pos + 3 <= end and pos + 3 < len(payload):
        program_number = (payload[pos] << 8) | payload[pos+1]
        pid = ((payload[pos+2] & 0x1F) << 8) | payload[pos+3]
        # Skip invalid program entries (program 0 is network PID, 0xFFFF is often corruption)
        if program_number == 0xFFFF or pid == 0x1FFF:
            corrupted_count += 1
        elif program_number != 0:
            result['programs'][program_number] = pid
        pos += 4
    
    if corrupted_count > 0:
        result['warnings'].append(f"PAT contains {corrupted_count} corrupted/invalid program entries (program=0xFFFF or PMT PID=0x1FFF)")
    
    return result


def parse_pmt(payload: bytes) -> Dict[str, object]:
    # returns detailed PMT information including descriptors
    info = {
        "pcr_pid": None,
        "program_number": None,
        "version": None,
        "current_next": None,
        "program_descriptors": [],
        "streams": [],
        "warnings": []
    }
    if not payload:
        return info
    ptr = payload[0]
    start = 1 + ptr
    if start + 12 > len(payload):
        info["warnings"].append("PMT payload too short (< 12 bytes)")
        return info
    table_id = payload[start]
    if table_id != 0x02:
        return info
    
    section_length = ((payload[start+1] & 0x0F) << 8) | payload[start+2]
    section_end = start + 3 + section_length
    
    # Check section length validity
    if section_end > len(payload):
        info["warnings"].append(f"PMT section length ({section_length}) exceeds payload size")
    
    # program_number at start+3..+4
    program_number = (payload[start+3] << 8) | payload[start+4]
    version = (payload[start+5] & 0x3E) >> 1
    current_next = payload[start+5] & 0x01
    
    # PCR PID at start+8..+9
    pcr_pid = ((payload[start+8] & 0x1F) << 8) | payload[start+9]
    
    # Check for invalid PCR PID (0x1FFF is NULL, shouldn't be PCR)
    if pcr_pid == 0x1FFF:
        info["warnings"].append("PMT has invalid PCR PID (0x1FFF - NULL PID)")
    
    info["pcr_pid"] = pcr_pid
    info["program_number"] = program_number
    info["version"] = version
    info["current_next"] = current_next
    
    # Program descriptors
    program_info_length = ((payload[start+10] & 0x0F) << 8) | payload[start+11]
    if program_info_length > 0 and start + 12 + program_info_length <= len(payload):
        prog_desc_data = payload[start+12:start+12+program_info_length]
        info["program_descriptors"] = parse_descriptors(prog_desc_data, program_info_length)
    elif program_info_length > 0:
        info["warnings"].append(f"PMT program info length ({program_info_length}) exceeds available data")
    
    pos = start + 12 + program_info_length
    
    # Parse elementary streams
    stream_count = 0
    corrupted_streams = 0
    while pos + 4 <= section_end - 4 and pos + 4 < len(payload):
        stream_type = payload[pos]
        es_pid = ((payload[pos+1] & 0x1F) << 8) | payload[pos+2]
        es_info_length = ((payload[pos+3] & 0x0F) << 8) | payload[pos+4]
        
        # Check for suspicious ES info length (very large values indicate corruption)
        if es_info_length > 1000:
            corrupted_streams += 1
            if corrupted_streams <= 3:  # Report first 3 corrupted streams
                info["warnings"].append(f"Stream entry has suspicious ES info length ({es_info_length} bytes) at PID 0x{es_pid:04X}")
            break  # Stop parsing if we hit corruption
        
        # Parse ES descriptors
        es_descriptors = []
        if es_info_length > 0 and pos + 5 + es_info_length <= len(payload):
            es_desc_data = payload[pos+5:pos+5+es_info_length]
            es_descriptors = parse_descriptors(es_desc_data, es_info_length)
        
        # Detect SCTE-35 stream type (0x86)
        type_name = get_stream_type_name(stream_type)
        if stream_type == 0x86:
            type_name = "SCTE-35"
        info["streams"].append({
            "type": stream_type,
            "type_name": type_name,
            "pid": es_pid,
            "info_len": es_info_length,
            "descriptors": es_descriptors
        })
        stream_count += 1
        pos += 5 + es_info_length
    
    if corrupted_streams > 3:
        info["warnings"].append(f"Total {corrupted_streams} corrupted stream entries detected in PMT")
    
    return info


class BufferAnalyzer:
    """Analyzes HRD (Hypothetical Reference Decoder) and T-STD buffer occupancy"""
    
    def __init__(self, bitrate_bps: float, buffer_size_bits: int, is_video: bool = True):
        self.bitrate = bitrate_bps  # R_x in T-STD
        self.buffer_size = buffer_size_bits  # BS_x or EB_n
        self.is_video = is_video
        self.buffer_level = 0.0  # Current buffer occupancy in bits
        self.max_buffer_level = 0.0
        self.min_buffer_level = float('inf')
        self.buffer_samples = []  # (time, level, event_type)
        self.overflows = 0
        self.underflows = 0
        self.last_access_time = 0.0
    
    def add_data(self, data_bits: int, arrival_time: float):
        """Add data to buffer at arrival_time"""
        # Leak data based on time elapsed
        time_elapsed = arrival_time - self.last_access_time
        if time_elapsed > 0:
            leaked_bits = self.bitrate * time_elapsed
            self.buffer_level = max(0, self.buffer_level - leaked_bits)
        
        # Add incoming data
        self.buffer_level += data_bits
        self.last_access_time = arrival_time
        
        # Track statistics
        self.max_buffer_level = max(self.max_buffer_level, self.buffer_level)
        self.min_buffer_level = min(self.min_buffer_level, self.buffer_level)
        
        # Check for overflow
        if self.buffer_level > self.buffer_size:
            self.overflows += 1
            self.buffer_samples.append((arrival_time, self.buffer_level, 'overflow'))
        
        # Check for underflow (buffer empty when data needed)
        if self.buffer_level < 0:
            self.underflows += 1
            self.buffer_samples.append((arrival_time, self.buffer_level, 'underflow'))
        else:
            self.buffer_samples.append((arrival_time, self.buffer_level, 'normal'))
        
        # Limit samples to prevent memory issues
        if len(self.buffer_samples) > 10000:
            # Downsample: keep every 10th sample
            self.buffer_samples = self.buffer_samples[::10]
    
    def get_stats(self):
        """Return buffer statistics"""
        utilization = (self.max_buffer_level / self.buffer_size * 100) if self.buffer_size > 0 else 0
        return {
            'max_level_bits': self.max_buffer_level,
            'min_level_bits': self.min_buffer_level if self.min_buffer_level != float('inf') else 0,
            'buffer_size_bits': self.buffer_size,
            'max_utilization_percent': utilization,
            'overflows': self.overflows,
            'underflows': self.underflows,
            'samples': self.buffer_samples[-1000:]  # Last 1000 samples for graphing
        }


class TSAnalyser:
    # Add per-frame NAL/SEI extraction
    def extract_nal_sei_per_frame(self, pid: int):
        """Extract detailed NAL and SEI info per frame for H.264 video PID."""
        if pid not in self.video_pes_buffers:
            return []
        buffer = bytes(self.video_pes_buffers[pid])
        
        # Limit buffer size for performance - process up to 10MB
        # This should cover most video files (100+ frames typically fit in 5-10MB)
        # For very large files, only the first portion will be analyzed
        max_parse_size = 10 * 1024 * 1024  # Increased from 100KB to 10MB
        if len(buffer) > max_parse_size:
            buffer = buffer[:max_parse_size]
        
        positions = []
        i = 0
        dlen = len(buffer)
        
        # Optimized NAL start code search using bytes.find()
        while i < dlen - 4:
            # Look for 0x000001 or 0x00000001 start codes
            zero_pos = buffer.find(b'\x00\x00', i, dlen - 2)
            if zero_pos == -1:
                break
            
            # Check if it's followed by 0x01 (3-byte start code)
            if zero_pos + 2 < dlen and buffer[zero_pos + 2] == 0x01:
                positions.append(zero_pos + 3)
                i = zero_pos + 3
            # Check if it's followed by 0x0001 (4-byte start code)
            elif zero_pos + 3 < dlen and buffer[zero_pos + 2] == 0x00 and buffer[zero_pos + 3] == 0x01:
                positions.append(zero_pos + 4)
                i = zero_pos + 4
            else:
                i = zero_pos + 2
        
        if not positions:
            return []
        
        positions.append(dlen)
        frames = []

        current_sps = None
        # Access unit grouping index - helps associate SEI with frames
        au_index = 0
        au_has_slice = False

        # Limit number of NAL units to process for performance
        # Increased from 500 to 5000 to cover more frames in longer files
        # At ~3-5 NAL units per frame, this covers ~1000-1600 frames
        max_nal_units = 5000
        for idx in range(min(len(positions)-1, max_nal_units)):
            start = positions[idx]
            end = positions[idx+1]
            nal_unit = buffer[start:end]
            if not nal_unit:
                continue
            nal_header = nal_unit[0]
            nal_unit_type = nal_header & 0x1F
            nal_ref_idc = (nal_header >> 5) & 0x03
            forbidden_zero_bit = (nal_header >> 7) & 0x01
            entry = {
                "offset": start,
                "nal_type": nal_unit_type,
                "nal_type_name": self.get_nal_type_name(nal_unit_type),
                "size": len(nal_unit),
                "nal_ref_idc": nal_ref_idc,
                "forbidden_zero_bit": forbidden_zero_bit,
            }
            # Annotate access unit grouping for downstream use
            entry["au_index"] = au_index
            entry["estimated_pts"] = self.last_pts_by_pid.get(pid)
            # Annotate access unit index for downstream grouping
            entry["au_index"] = au_index
            # Attach an estimated PTS if available (best-effort)
            entry["estimated_pts"] = self.last_pts_by_pid.get(pid)
            # Track SPS for later slice parsing
            if nal_unit_type == 7:
                try:
                    sps_parse = parse_h264_sps(b"\x00\x00\x01" + nal_unit)
                    if sps_parse:
                        current_sps = sps_parse
                except Exception:
                    pass

            # If slice, try to parse slice header using current SPS context
            if nal_unit_type in (1, 5):
                try:
                    clean_rbsp = self._remove_emulation_prevention(nal_unit[1:])
                    slice_info = self._parse_h264_slice_header(clean_rbsp, current_sps)
                    if slice_info:
                        entry["slice_type"] = slice_info.get("slice_type")
                        entry["slice_type_name"] = slice_info.get("slice_type_name")
                        entry["slice_header_fields"] = slice_info.get("fields", [])
                except Exception:
                    pass
            # If SEI, parse SEI headers with type labeling
            if nal_unit_type == 6:
                sei_headers = []
                rbsp = self._remove_emulation_prevention(nal_unit[1:])
                
                # Find rbsp_trailing_bits() - DO NOT remove them prematurely
                # The trailing bits are part of the H.264 spec but removing them incorrectly
                # can corrupt the actual SEI payload data
                rbsp_len = len(rbsp)
                trailing_removed = False
                if rbsp_len > 0:
                    last_byte = rbsp[-1]
                    if last_byte in (0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01):
                        pass  # Potential trailing bits not removed for safety
                
                sei_pos = 0
                sei_count = 0
                while sei_pos + 2 <= rbsp_len:
                    # Parse payload_type (may span multiple bytes if >= 255)
                    payload_type = 0
                    type_iter = 0
                    while sei_pos < rbsp_len and rbsp[sei_pos] == 0xFF and type_iter < 1000:
                        payload_type += 255
                        sei_pos += 1
                        type_iter += 1
                    if sei_pos >= rbsp_len:
                        break
                    payload_type += rbsp[sei_pos]
                    sei_pos += 1
                    
                    # Parse payload_size (may span multiple bytes if >= 255)
                    payload_size = 0
                    size_iter = 0
                    while sei_pos < rbsp_len and rbsp[sei_pos] == 0xFF and size_iter < 1000:
                        payload_size += 255
                        sei_pos += 1
                        size_iter += 1
                    if sei_pos >= rbsp_len:
                        break
                    payload_size += rbsp[sei_pos]
                    sei_pos += 1
                    
                    payload_end = sei_pos + payload_size
                    if payload_end > rbsp_len:
                        # Try to salvage - check if we're only 1 byte off (might be trailing bits issue)
                        if payload_end == rbsp_len + 1 and not trailing_removed:
                            # The trailing byte we didn't remove is probably part of payload
                            payload_size = rbsp_len - sei_pos
                            payload_end = rbsp_len
                        else:
                            break
                    
                    payload = rbsp[sei_pos:payload_end]
                    
                    sei_type_name = self.get_sei_type_name(payload_type)
                    sei_summary = self.summarize_sei(payload_type, payload)
                    
                    # Parse detailed SEI fields
                    sei_fields = self._parse_sei_payload(payload_type, payload, current_sps)
                    
                    sei_headers.append({
                        "type": payload_type,
                        "type_name": sei_type_name,
                        "length": payload_size,
                        "payload_hex": payload.hex(),
                        "summary": sei_summary,
                        "fields": sei_fields if sei_fields else [],
                    })
                    sei_pos = payload_end
                    sei_count += 1
                entry["sei_headers"] = sei_headers

            # Update access unit grouping heuristics
            if nal_unit_type == 9:  # AUD - start a new access unit
                au_index += 1
                au_has_slice = False
            elif nal_unit_type in (1, 5):  # slice NALs: may indicate start of new AU
                if au_has_slice:
                    # start new AU when another primary slice is encountered
                    au_index += 1
                au_has_slice = True
            frames.append(entry)
        return frames

    def extract_nal_sei_unlimited(self, pid: int):
        """Extract ALL NAL and SEI units from entire buffer without limits.
        
        This is used for on-demand extraction when user navigates to different frames.
        Processes the entire PES buffer without size or count limits, allowing
        extraction of NAL/SEI data for any frame in the file.
        
        Returns:
            List of ALL NAL units (potentially large)
        """
        if pid not in self.video_pes_buffers:
            return []
        buffer = bytes(self.video_pes_buffers[pid])
        
        if not buffer:
            return []
        
        positions = []
        i = 0
        dlen = len(buffer)
        
        # Find all NAL start codes in the entire buffer
        while i < dlen - 4:
            zero_pos = buffer.find(b'\x00\x00', i, dlen - 2)
            if zero_pos == -1:
                break
            
            if zero_pos + 2 < dlen and buffer[zero_pos + 2] == 0x01:
                positions.append(zero_pos + 3)
                i = zero_pos + 3
            elif zero_pos + 3 < dlen and buffer[zero_pos + 2] == 0x00 and buffer[zero_pos + 3] == 0x01:
                positions.append(zero_pos + 4)
                i = zero_pos + 4
            else:
                i = zero_pos + 2
        
        if not positions:
            return []
        
        positions.append(dlen)
        frames = []
        
        current_sps = None
        au_index = 0
        au_has_slice = False

        # Process ALL NAL units without count limits
        for idx in range(len(positions) - 1):
            start = positions[idx]
            end = positions[idx + 1]
            nal_unit = buffer[start:end]
            if not nal_unit:
                continue
            
            nal_header = nal_unit[0]
            nal_unit_type = nal_header & 0x1F
            nal_ref_idc = (nal_header >> 5) & 0x03
            forbidden_zero_bit = (nal_header >> 7) & 0x01
            
            entry = {
                "offset": start,
                "nal_type": nal_unit_type,
                "nal_type_name": self.get_nal_type_name(nal_unit_type),
                "size": len(nal_unit),
                "nal_ref_idc": nal_ref_idc,
                "forbidden_zero_bit": forbidden_zero_bit,
            }
            
            # Track SPS for later slice parsing
            if nal_unit_type == 7:
                try:
                    sps_parse = parse_h264_sps(b"\x00\x00\x01" + nal_unit)
                    if sps_parse:
                        current_sps = sps_parse
                except Exception:
                    pass

            # Parse slice header if this is a slice
            if nal_unit_type in (1, 5):
                try:
                    clean_rbsp = self._remove_emulation_prevention(nal_unit[1:])
                    slice_info = self._parse_h264_slice_header(clean_rbsp, current_sps)
                    if slice_info:
                        entry["slice_type"] = slice_info.get("slice_type")
                        entry["slice_type_name"] = slice_info.get("slice_type_name")
                        entry["slice_header_fields"] = slice_info.get("fields", [])
                except Exception:
                    pass
            
            # Parse SEI headers if this is SEI
            if nal_unit_type == 6:
                sei_headers = []
                rbsp = self._remove_emulation_prevention(nal_unit[1:])
                
                # Find rbsp_trailing_bits() - search from end for stop bit (1 followed by 0-7 zeros)
                # The trailing bits are ONLY the last byte if it matches the stop bit pattern
                # We should NOT remove it if the payload needs all bytes
                rbsp_len = len(rbsp)
                trailing_removed = False
                if rbsp_len > 0:
                    last_byte = rbsp[-1]
                    # Check if last byte is ONLY trailing bits (stop bit pattern with no data)
                    # Only remove if it's a pure trailing bits byte: 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01
                    if last_byte in (0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01):
                        # However, we need to be careful - these bytes could be part of actual payload
                        # Only remove if this appears to be after a complete SEI payload structure
                        # For now, let's NOT remove trailing bits and see if that fixes the issue
                        pass  # Potential trailing bits not removed for safety
                
                sei_pos = 0
                sei_count = 0
                while sei_pos + 2 <= rbsp_len:
                    # Parse payload_type (may span multiple bytes if >= 255)
                    payload_type = 0
                    type_iter = 0
                    while sei_pos < rbsp_len and rbsp[sei_pos] == 0xFF and type_iter < 1000:
                        payload_type += 255
                        sei_pos += 1
                        type_iter += 1
                    if sei_pos >= rbsp_len:
                        break
                    payload_type += rbsp[sei_pos]
                    sei_pos += 1
                    
                    # Parse payload_size (may span multiple bytes if >= 255)
                    payload_size = 0
                    size_iter = 0
                    while sei_pos < rbsp_len and rbsp[sei_pos] == 0xFF and size_iter < 1000:
                        payload_size += 255
                        sei_pos += 1
                        size_iter += 1
                    if sei_pos >= rbsp_len:
                        break
                    payload_size += rbsp[sei_pos]
                    sei_pos += 1
                    
                    payload_end = sei_pos + payload_size
                    if payload_end > rbsp_len:
                        # Try to salvage - check if we're only 1 byte off (might be trailing bits issue)
                        if payload_end == rbsp_len + 1 and not trailing_removed:
                            # The trailing byte we didn't remove is probably part of payload
                            payload_size = rbsp_len - sei_pos
                            payload_end = rbsp_len
                        else:
                            break
                    
                    payload = rbsp[sei_pos:payload_end]
                    
                    sei_type_name = self.get_sei_type_name(payload_type)
                    sei_summary = self.summarize_sei(payload_type, payload)
                    
                    # Parse detailed SEI fields
                    sei_fields = self._parse_sei_payload(payload_type, payload, current_sps)
                    
                    sei_entry = {
                        "type": payload_type,
                        "type_name": sei_type_name,
                        "length": payload_size,
                        "payload_hex": payload.hex(),
                        "summary": sei_summary,
                        "fields": sei_fields if sei_fields else [],
                    }
                    sei_headers.append(sei_entry)
                    sei_pos = payload_end
                    sei_count += 1
                entry["sei_headers"] = sei_headers
            
            # Update access unit grouping heuristics
            if nal_unit_type == 9:  # AUD
                au_index += 1
                au_has_slice = False
            elif nal_unit_type in (1, 5):
                if au_has_slice:
                    au_index += 1
                au_has_slice = True

            # Annotate entry with access-unit index and an estimated PTS (best-effort)
            try:
                entry["au_index"] = au_index
            except Exception:
                entry["au_index"] = None

            try:
                entry["estimated_pts"] = self.last_pts_by_pid.get(pid) if hasattr(self, 'last_pts_by_pid') else None
            except Exception:
                entry["estimated_pts"] = None

            frames.append(entry)
        
        return frames

    def _parse_sei_payload(self, payload_type: int, payload: bytes, sps: Optional[Dict[str, object]] = None) -> Optional[List[Tuple[str, str]]]:
        """Parse SEI payload into detailed field tuples per H.264 Spec Annex D.

        Dispatches to type-specific parsers for all standard SEI message types.
        
        Implemented SEI Types (H.264 Spec D):
        - Type 0: buffering_period - CPB removal delay information
        - Type 1: pic_timing - Picture timing and clock timestamp data
        - Type 2: pan_scan_rect - Pan and scan rectangle (UNIMPLEMENTED)
        - Type 3: filler_payload - Filler NAL unit
        - Type 4: user_data_registered_itu_t_t35 - ITU-T T.35 data (ATSC captions)
        - Type 5: user_data_unregistered - User-defined data with UUID
        - Type 6: recovery_point - Decoder recovery point information
        - Types 7+: Not yet implemented (frame_packing, stereoscopic_info, etc.)
        
        Returns: List[(field_name, field_value)] or None if type unhandled
        
        Spec compliance notes:
        - pic_timing (type 1): Depends on seq_parameter_set_data() from active SPS
        - pic_struct_present_flag must come from SPS vui_parameters
        - NumClockTS derived from pic_struct per Table D.1
        - buffering_period requires HRD parameters from SPS VUI
        """
        if not payload:
            return None

        try:
            # Dispatch to type-specific parsers
            if payload_type == 0:
                return self._parse_sei_buffering_period(payload, sps)
            elif payload_type == 1:
                return self._parse_sei_pic_timing(payload, sps)
            elif payload_type == 3:
                return self._parse_sei_filler_payload(payload)
            elif payload_type == 4:
                return self._parse_sei_itu_t_t35(payload)
            elif payload_type == 5:
                return self._parse_sei_user_data_unregistered(payload)
            elif payload_type == 6:
                return self._parse_sei_recovery_point(payload)
            # Default: unimplemented types show raw hex
            else:
                return [("(unimplemented SEI type)", f"{len(payload)} bytes: {payload.hex()}")]
                
        except Exception as e:
            pass  # Silently handle SEI parse errors

        return None

    def _parse_sei_buffering_period(self, payload: bytes, sps: Optional[Dict[str, object]] = None) -> List[Tuple[str, str]]:
        """Parse buffering_period SEI (type 0) per H.264 Spec D.2.1 with spec compliance.

        Syntax: buffering_period() {
            seq_parameter_set_id     ue(v)
            if( NalHrdBpPresentFlag )
              for( SchedSelIdx = 0; SchedSelIdx < CpbCnt; SchedSelIdx++ )
                initial_cpb_removal_delay[SchedSelIdx]     u(CpbDpBSize)
                initial_cpb_removal_delay_offset[SchedSelIdx]  u(CpbDpBSize)
            if( VclHrdBpPresentFlag )
              ...similar fields...
        }
        
        Purpose: Specifies CPB (Coded Picture Buffer) removal delays for decoding/playback timing.
        Requires SPS context to determine bit depths (CpbDpBSize from HRD parameters).
        
        Note: This is a simplified parser. Full parsing requires:
        1. SPS to extract HRD parameters
        2. NalHrdBpPresentFlag, VclHrdBpPresentFlag, and CpbCnt from VUI
        3. CpbDpBSize = 4 * (CpbLog2SizeMaxMinus1 + 1) from HRD
        """
        fields = []
        if len(payload) < 1:
            return fields
        
        try:
            br = BitReader(payload)
            seq_parameter_set_id = br.read_ue()
            fields.append(("seq_parameter_set_id", f"{seq_parameter_set_id}"))

            # Parse HRD-dependent fields if SPS VUI carries HRD parameters
            nal_hrd = sps.get('nal_hrd') if sps else None
            vcl_hrd = sps.get('vcl_hrd') if sps else None

            def parse_initial_delays(hrd: Dict[str, object], prefix: str):
                cpb_cnt = hrd.get('cpb_cnt_minus1', -1) + 1
                delay_len = hrd.get('initial_cpb_removal_delay_length_minus1', 23) + 1
                entries = []
                for idx in range(cpb_cnt):
                    initial_delay = br.read_bits(delay_len)
                    initial_offset = br.read_bits(delay_len)
                    entries.append((initial_delay, initial_offset))
                return cpb_cnt, delay_len, entries

            if nal_hrd:
                cpb_cnt, delay_len, entries = parse_initial_delays(nal_hrd, "nal")
                fields.append(("NalHrdBpPresentFlag", "1"))
                fields.append(("cpb_cnt", str(cpb_cnt)))
                fields.append(("initial_cpb_removal_delay_length", f"{delay_len} bits"))
                for idx, (d, o) in enumerate(entries):
                    fields.append((f"nal_initial_cpb_removal_delay[{idx}]", str(d)))
                    fields.append((f"nal_initial_cpb_removal_delay_offset[{idx}]", str(o)))

            if vcl_hrd:
                cpb_cnt, delay_len, entries = parse_initial_delays(vcl_hrd, "vcl")
                fields.append(("VclHrdBpPresentFlag", "1"))
                fields.append(("cpb_cnt", str(cpb_cnt)))
                fields.append(("initial_cpb_removal_delay_length", f"{delay_len} bits"))
                for idx, (d, o) in enumerate(entries):
                    fields.append((f"vcl_initial_cpb_removal_delay[{idx}]", str(d)))
                    fields.append((f"vcl_initial_cpb_removal_delay_offset[{idx}]", str(o)))

            if not nal_hrd and not vcl_hrd:
                remaining = payload[br.pos // 8:]
                if remaining:
                    fields.append(("(HRD-dependent fields)", f"remaining {len(remaining)} bytes: {remaining.hex()}"))
                else:
                    fields.append(("(parse complete)", "seq_parameter_set_id only"))
            
        except Exception as e:
            fields.append(("(parse error)", str(e)))
        
        return fields

    def _parse_sei_recovery_point(self, payload: bytes) -> List[Tuple[str, str]]:
        """Parse recovery_point SEI (type 6) per H.264 Spec D.2.6 with spec compliance.

        Syntax: recovery_point() {
            recovery_frame_cnt      ue(v)   // Frame number (in display order) where decoding can start
            exact_match_flag        u(1)    // 1: recovery frame exactly matches encoded, 0: may differ
            broken_link_flag        u(1)    // 1: bitstream broken, must skip to recovery frame
            changing_slice_group_idc u(2)   // 0-3: transitions in slice group map
        }
        
        Purpose: Indicates point in bitstream where a decoder can be safely started after interruption.
        Commonly used in streams with error resilience or interrupted playback.
        
        H.264 Spec D.2.6 normative requirements:
        - Recovery frame must be an IDR or recovery point reference (per Picture Order Count)
        - exact_match_flag=1 means no post-processing needed
        - broken_link_flag indicates whether sequence/group structures change
        """
        fields = []
        if len(payload) < 1:
            return fields
        
        try:
            br = BitReader(payload)
            
            # recovery_frame_cnt (UE(v))
            recovery_frame_cnt = br.read_ue()
            fields.append(("recovery_frame_cnt", f"{recovery_frame_cnt}"))
            
            # Single flags (U(1) each)
            exact_match_flag = br.read_bits(1)
            broken_link_flag = br.read_bits(1)
            changing_slice_group_idc = br.read_bits(2)
            
            fields.append(("exact_match_flag", f"{exact_match_flag}"))
            fields.append(("broken_link_flag", f"{broken_link_flag}"))
            fields.append(("changing_slice_group_idc", f"{changing_slice_group_idc}"))
            
            # Validation per spec
            if changing_slice_group_idc > 3:
                fields.append(("⚠ SPEC VIOLATION", f"changing_slice_group_idc {changing_slice_group_idc} out of range [0-3]"))
            
        except Exception as e:
            fields.append(("(parse error)", str(e)))
        
        return fields

    def _parse_sei_filler_payload(self, payload: bytes) -> List[Tuple[str, str]]:
        """Parse filler_payload SEI (type 3) per H.264 Spec D.2.4 with spec compliance.

        Syntax: filler_payload() {
            for( i = 0; i < payloadSize; i++ )
              ff_byte   u(8)   // All bytes must be 0xFF per spec
        }
        
        Purpose: Adds padding to bitstream for alignment or bandwidth control.
        Used to reach specific bitrate targets or align NAL units to byte boundaries.
        
        H.264 Spec D.2.4 Normative:
        - All bytes in filler_payload must be 0xFF (255 decimal)
        - If any byte is not 0xFF, decoder behavior is undefined
        - Decoders may skip filler payload without processing
        """
        fields = []
        if len(payload) == 0:
            fields.append(("(empty payload)", "0 bytes"))
            return fields
        
        try:
            # Check if all bytes are 0xFF (spec-compliant)
            invalid_bytes = []
            for i, byte in enumerate(payload):
                if byte != 0xFF:
                    invalid_bytes.append((i, byte))
            
            if invalid_bytes:
                fields.append(("⚠ SPEC VIOLATION", f"Non-0xFF bytes found: {invalid_bytes[:5]}"))
            
            fields.append(("payload_size", f"{len(payload)} bytes"))
            fields.append(("all_bytes_0xff", "✓ VALID" if not invalid_bytes else "✗ INVALID"))
            
            # Show a sample
            if len(payload) <= 16:
                fields.append(("hex_content", payload.hex()))
            else:
                fields.append(("hex_content (first 16 bytes)", payload[:16].hex()))
            
        except Exception as e:
            fields.append(("(parse error)", str(e)))
        
        return fields

    def _parse_sei_pic_timing(self, payload: bytes, sps: Optional[Dict[str, object]] = None) -> List[Tuple[str, str]]:
        """Parse pic_timing SEI (type 1) per H.264 Spec D.2.2.
        
        SPEC COMPLIANCE NOTES:
        - CpbDpbDelaysPresentFlag determines if cpb_removal_delay/dpb_output_delay are present
        - pic_struct presence depends on pic_struct_present_flag in SPS vui_parameters
        - NumClockTS is derived from pic_struct value (0-12) per Table D.1
        - For each clock_timestamp: if flag=1, parse full timestamp structure
        - Parsing requires knowledge of SPS vui_timing_info for proper field widths
        
        Per H.264 D.2.2:
        pic_timing( payloadSize ) {
            if( CpbDpbDelaysPresentFlag ) {
                cpb_removal_delay  // length = cpb_removal_delay_length_minus1 + 1
                dpb_output_delay   // length = dpb_output_delay_length_minus1 + 1
            }
            if( pic_struct_present_flag ) {
                pic_struct
                for( i = 0; i < NumClockTS; i++ ) {
                    clock_timestamp_flag[i]
                    if( clock_timestamp_flag[i] ) {
                        ... timecode fields ...
                    }
                }
            }
        }
        """
        fields = []
        if len(payload) < 1:
            return fields

        try:
            br = BitReader(payload)
            
            # Check if we have SPS VUI information
            cpb_dpb_delays_present = False
            cpb_removal_delay_length = 24  # default
            dpb_output_delay_length = 24   # default
            pic_struct_present_flag = True  # assume present
            
            if sps and 'vui_parameters' in sps:
                vui = sps['vui_parameters']
                if 'nal_hrd_parameters_present_flag' in vui or 'vcl_hrd_parameters_present_flag' in vui:
                    cpb_dpb_delays_present = vui.get('nal_hrd_parameters_present_flag', False) or vui.get('vcl_hrd_parameters_present_flag', False)
                    
                    # Get delay lengths from HRD parameters
                    if cpb_dpb_delays_present and 'hrd_parameters' in vui:
                        hrd = vui['hrd_parameters']
                        cpb_removal_delay_length = hrd.get('cpb_removal_delay_length_minus1', 23) + 1
                        dpb_output_delay_length = hrd.get('dpb_output_delay_length_minus1', 23) + 1
                
                pic_struct_present_flag = vui.get('pic_struct_present_flag', True)
            
            # Parse cpb_removal_delay and dpb_output_delay if present
            if cpb_dpb_delays_present:
                cpb_removal_delay = br.read_bits(cpb_removal_delay_length)
                fields.append(("cpb_removal_delay", f"{cpb_removal_delay} ({cpb_removal_delay_length} bits)"))
                
                dpb_output_delay = br.read_bits(dpb_output_delay_length)
                fields.append(("dpb_output_delay", f"{dpb_output_delay} ({dpb_output_delay_length} bits)"))
            
            # Parse pic_struct and clock timestamps if present
            if not pic_struct_present_flag:
                fields.append(("(note)", "pic_struct_present_flag=0 in SPS VUI, no timing info"))
                return fields
            
            # Per spec: pic_struct is present when pic_struct_present_flag=1
            pic_struct = br.read_bits(4)
            fields.append(("pic_struct", f"{pic_struct} (range 0-12, see Table D.1)"))
            
            # Validate pic_struct per H.264 spec
            if pic_struct > 12:
                fields.append(("⚠ SPEC VIOLATION", f"pic_struct {pic_struct} exceeds valid range 0-12"))
            
            # Determine NumClockTS based on pic_struct value (H.264 Spec Table D.1)
            # This mapping is normative in the spec
            num_clock_ts_map = {
                0: 1,   # Frame picture
                1: 1,   # Top field
                2: 1,   # Bottom field
                3: 2,   # Top field, bottom field, in that order
                4: 2,   # Bottom field, top field, in that order
                5: 3,   # Top field, bottom field, top field repeated
                6: 3,   # Bottom field, top field, bottom field repeated
                7: 2,   # Frame picture, top field, bottom field
                8: 2,   # Frame picture, bottom field, top field
                9: 3,   # Top field, bottom field, top field repeated, bottom field repeated
                10: 3,  # Bottom field, top field, bottom field repeated, top field repeated
                11: 2,  # Top field, bottom field, top field repeated, bottom field repeated, ...
                12: 2,  # Bottom field, top field, bottom field repeated, top field repeated, ...
            }
            num_clock_ts = num_clock_ts_map.get(pic_struct, 1)
            fields.append(("NumClockTS", f"{num_clock_ts} (normatively derived from pic_struct per Table D.1)"))
            
            fields.append(("for( i = 0; i < NumClockTS; i++ )", ""))
            
            for i in range(num_clock_ts):
                if not br.bits_available():
                    break
                
                clock_ts_flag = br.read_bits(1)
                fields.append((f"<clock_timestamp {i}>", ""))
                fields.append(("clock_timestamp_flag", f"{clock_ts_flag} (1=timestamp present, 0=omitted)"))
                
                # If clock_timestamp_flag = 1, parse timing information per spec D.2.2
                if clock_ts_flag:
                    # ct_type: 0=progressive, 1=interlaced, 2=unknown, 3=reserved
                    ct_type = br.read_bits(2)
                    ct_type_names = {0: "Progressive", 1: "Interlaced", 2: "Unknown", 3: "Reserved"}
                    fields.append(("ct_type", f"{ct_type} ({ct_type_names.get(ct_type, 'Invalid')})"))
                    
                    # nuit_field_based_flag: 0=frame-based, 1=field-based timing
                    nuit_field_based_flag = br.read_bits(1)
                    fields.append(("nuit_field_based_flag", f"{nuit_field_based_flag}"))
                    
                    # counting_type: determines which time components are present (0-7)
                    counting_type = br.read_bits(5)
                    fields.append(("counting_type", f"{counting_type} (0-4 standard, 5-7 auxiliary)"))
                    
                    # full_timestamp_flag: indicates whether full timestamp is encoded
                    full_timestamp_flag = br.read_bits(1)
                    fields.append(("full_timestamp_flag", f"{full_timestamp_flag}"))
                    
                    # discontinuity_flag: indicates potential discontinuity in timestamp
                    discontinuity_flag = br.read_bits(1)
                    fields.append(("discontinuity_flag", f"{discontinuity_flag}"))
                    
                    # cnt_dropped_flag: indicates dropped frames (drop frame timecode)
                    cnt_dropped_flag = br.read_bits(1)
                    fields.append(("cnt_dropped_flag", f"{cnt_dropped_flag}"))
                    
                    # n_frames: number of frames in current second
                    n_frames = br.read_bits(8)
                    fields.append(("n_frames", f"{n_frames}"))
                    
                    # Timestamp values - presence and bit widths per H.264 spec
                    if full_timestamp_flag:
                        # Full timestamp: all time components present
                        hours = br.read_bits(5)
                        fields.append(("hours", f"{hours} (0-23)"))
                        
                        minutes = br.read_bits(6)
                        fields.append(("minutes", f"{minutes} (0-59)"))
                        
                        seconds = br.read_bits(6)
                        fields.append(("seconds", f"{seconds} (0-59)"))
                        
                        time_offset = br.read_bits(24)
                        fields.append(("time_offset", f"{time_offset} (0x{time_offset:06X}) (90000 Hz units)"))
                    else:
                        # Partial timestamp per counting_type (spec Table D.4)
                        # Note: Presence of subsequent fields depends on counting_type
                        if counting_type == 0:
                            seconds_value = br.read_bits(6)
                            fields.append(("seconds_value", f"{seconds_value}"))
                        elif counting_type == 1:
                            minutes_value = br.read_bits(6)
                            fields.append(("minutes_value", f"{minutes_value}"))
                            seconds_value = br.read_bits(6)
                            fields.append(("seconds_value", f"{seconds_value}"))
                        elif counting_type >= 2:
                            hours_value = br.read_bits(5)
                            fields.append(("hours_value", f"{hours_value}"))
                            minutes_value = br.read_bits(6)
                            fields.append(("minutes_value", f"{minutes_value}"))
                            seconds_value = br.read_bits(6)
                            fields.append(("seconds_value", f"{seconds_value}"))
                    
                    # time_offset_length per counting_type (not always present)
                    # This field presence is VUI-dependent; simplified implementation
                # Continue loop regardless of flag value
                
        except Exception as e:
            fields.append(("(parse error)", str(e)))

        return fields

    def _parse_sei_user_data_unregistered(self, payload: bytes) -> List[Tuple[str, str]]:
        """Parse user_data_unregistered SEI (type 5) per H.264 Spec D.2.5.

        Syntax: user_data_unregistered() {
            uuid_iso_iec_11578  u(128)     // 16 bytes
            for( i = 0; i < payloadSize - 16; i++ )
              user_data_payload_byte   u(8)
        }
        
        The 128-bit UUID is used to identify user data format. Common UUIDs:
        - RFC 4122 format or proprietary identifiers
        - Used for DVB, DTMF, and custom metadata
        """
        fields = []
        if len(payload) < 16:
            fields.append(("(incomplete)", "payload < 16 bytes, cannot extract UUID"))
            return fields
        
        try:
            # Extract 128-bit UUID (16 bytes)
            uuid_bytes = payload[0:16]
            
            # Format as RFC 4122 string: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
            uuid_str = '-'.join([
                uuid_bytes[0:4].hex(),
                uuid_bytes[4:6].hex(),
                uuid_bytes[6:8].hex(),
                uuid_bytes[8:10].hex(),
                uuid_bytes[10:16].hex()
            ])
            fields.append(("uuid_iso_iec_11578", uuid_str))
            
            # If there's user_data_payload after UUID
            if len(payload) > 16:
                user_data = payload[16:]
                fields.append(("user_data_payload", f"{len(user_data)} bytes: {user_data.hex()}"))
            
            # Known UUID patterns (informational)
            if uuid_bytes == bytes(16):
                fields.append(("(note)", "UUID is all zeros (unusual)"))
            
        except Exception as e:
            fields.append(("(parse error)", str(e)))
        
        return fields

    def _parse_sei_itu_t_t35(self, payload: bytes) -> List[Tuple[str, str]]:
        """Parse user_data_registered_itu_t_t35 SEI (type 4) - ATSC closed captions."""
        fields = []
        if len(payload) < 2:
            return fields

        try:
            pos = 0
            country_code = payload[pos]
            pos += 1
            fields.append(("itu_t_t35_country_code", f"{country_code} (0x{country_code:02X})"))

            if pos + 1 >= len(payload):
                return fields

            provider_code = (payload[pos] << 8) | payload[pos + 1]
            pos += 2
            fields.append(("itu_t_t35_provider_code", f"{provider_code} (0x{provider_code:04X})"))

            # ATSC data (country=0xB5, provider=0x0031)
            if country_code == 0xB5 and provider_code == 0x0031:
                fields.append(("ATSC data", ""))
                if pos + 4 > len(payload):
                    return fields

                # User identifier (4 ASCII bytes)
                user_id = payload[pos:pos + 4]
                user_id_str = user_id.decode('ascii', errors='replace')
                user_id_hex = ' '.join(f'{b:02X}' for b in user_id)
                fields.append(("user_identifier", f"'{user_id_str}' (hex {user_id_hex})"))
                pos += 4

                # Check for GA94 (ATSC1_data with closed captions)
                if user_id == b'GA94':
                    fields.append(("ATSC1_data", ""))
                    if pos >= len(payload):
                        return fields

                    user_data_type_code = payload[pos]
                    fields.append(("user_data_type_code", str(user_data_type_code)))
                    pos += 1

                    # cc_data (type code 0x03)
                    if user_data_type_code == 0x03 and pos < len(payload):
                        fields.append(("cc_data", ""))
                        reserved = (payload[pos] >> 5) & 0x07
                        process_cc_data = (payload[pos] >> 6) & 0x01
                        zero_bit = payload[pos] & 0x01
                        fields.append(("reserved", str(reserved)))
                        fields.append(("process_cc_data_flag", str(process_cc_data)))
                        fields.append(("zero_bit", str(zero_bit)))
                        pos += 1

                        if pos >= len(payload):
                            return fields

                        cc_count = payload[pos] & 0x1F
                        fields.append(("cc_count", f"{cc_count} (0x{cc_count:02X})"))
                        pos += 1

                        if pos >= len(payload):
                            return fields

                        reserved_byte = payload[pos]
                        fields.append(("reserved", f"{reserved_byte} (0x{reserved_byte:02X})"))
                        pos += 1

                        # Parse cc_data triplets
                        fields.append(("for ( i=0 ; i < cc_count ; i++ )", ""))
                        for cc_idx in range(cc_count):
                            if pos + 3 > len(payload):
                                break

                            cc_byte = payload[pos]
                            cc_data_1 = payload[pos + 1]
                            cc_data_2 = payload[pos + 2]
                            pos += 3

                            one_bit = (cc_byte >> 7) & 0x01
                            reserved = (cc_byte >> 3) & 0x0F
                            cc_valid = (cc_byte >> 2) & 0x01
                            cc_type = cc_byte & 0x03

                            fields.append((f"<CC {cc_idx}>", ""))
                            fields.append(("one_bit", str(one_bit)))
                            fields.append(("reserved", f"{reserved} (0x{reserved:01X})"))
                            fields.append(("cc_valid", str(cc_valid)))

                            # cc_type names
                            cc_type_names = {
                                0: "CEA-608, field 1",
                                1: "CEA-608, field 2",
                                2: "CEA-708, CCP Data",
                                3: "CEA-708, CCP Data"
                            }
                            cc_type_name = cc_type_names.get(cc_type, f"Unknown ({cc_type})")
                            fields.append(("cc_type", f"{cc_type}, '{cc_type_name}'"))
                            fields.append(("cc_data_1", f"{cc_data_1} (0x{cc_data_1:02X})"))
                            fields.append(("cc_data_2", f"{cc_data_2} (0x{cc_data_2:02X})"))

                        # Marker bits at end
                        if pos < len(payload):
                            marker = payload[pos]
                            fields.append(("marker_bits", f"{marker} (0x{marker:02X})"))

                # DTG1 AFD data
                elif user_id == b'DTG1':
                    fields.append(("afd_data", ""))
                    if pos >= len(payload):
                        return fields

                    afd_byte = payload[pos]
                    zero_bit = (afd_byte >> 7) & 0x01
                    active_format_flag = (afd_byte >> 6) & 0x01
                    alignment_bits = (afd_byte >> 4) & 0x03
                    fields.append(("zero_bit", str(zero_bit)))
                    fields.append(("active_format_flag", str(active_format_flag)))
                    fields.append(("alignment_bits", f"{alignment_bits} (0x{alignment_bits:01X})"))

                    if active_format_flag and pos + 1 < len(payload):
                        fields.append(("if (active_format_flag == '1')", ""))
                        pos += 1
                        active_byte = payload[pos]
                        reserved = (active_byte >> 4) & 0x0F
                        active_format = active_byte & 0x0F

                        afd_meanings = {
                            0: "Reserved",
                            1: "Box (4:3)",
                            2: "Box (16:9)",
                            3: "Box (14:9)",
                            4: "Full frame (4:3)",
                            8: "Full frame (16:9)",
                            9: "Full frame (14:9)",
                            10: "16:9 letterbox for 4:3 coded frames; 16:9 full frame for 16:9 coded frames",
                            11: "14:9 letterbox for 4:3 coded frames; 14:9 full frame for 14:9 coded frames",
                            13: "4:3 full frame (14:9 pillar)",
                            14: "16:9 full frame (4:3 pillar)",
                            15: "16:9 full frame (14:9 pillar)"
                        }
                        afd_desc = afd_meanings.get(active_format, f"Unknown ({active_format})")
                        fields.append(("reserved", f"{reserved} (0x{reserved:01X})"))
                        fields.append(("active_format", f"{active_format} (0x{active_format:01X}), '{afd_desc}'"))

        except Exception as e:
            fields.append(("(parse error)", str(e)))

        return fields

    @staticmethod
    def get_sei_type_name(payload_type: int) -> str:
        """Return human-readable name for SEI payload type."""
        # H.264 SEI type mapping (subset)
        sei_types = {
            1: "pic_timing",
            4: "user_data_registered_itu_t_t35", # often CC
            5: "user_data_unregistered",
            6: "recovery_point",
            9: "scene_info",
            45: "mastering_display_colour_volume",
        }
        return sei_types.get(payload_type, f"SEI type {payload_type}")

    def summarize_sei(self, payload_type: int, payload: bytes) -> str:
        # Provide a short summary for known SEI types
        if payload_type == 1:
            return "Picture timing info"
        elif payload_type == 4:
            # ATSC/CEA-708/608 closed captions (user_data_registered_itu_t_t35)
            # payload layout: country_code (1), provider_code (2), user_identifier (4, optional), user_data...
            if len(payload) >= 3 and payload[0] == 0xB5:
                provider_code = (payload[1] << 8) | payload[2]
                summary = "Closed captions (ATSC/CEA-708): "
                try:
                    summary += f"country_code={payload[0]}, provider_code={provider_code}"
                    # Include user identifier if present
                    if len(payload) >= 7:
                        user_id = payload[3:7]
                        try:
                            user_id_str = user_id.decode('ascii', errors='replace')
                            summary += f", user_id='{user_id_str}'"
                        except Exception:
                            summary += f", user_id=0x{user_id.hex()}"
                    # Report remaining user data length heuristically
                    if len(payload) > 7:
                        summary += f", cc_payload_len={len(payload) - 7}"
                except Exception as e:
                    summary += f" (parse error: {e})"
                return summary
            return "User data registered (possible CC)"
        elif payload_type == 5:
            return "User data unregistered"
        elif payload_type == 6:
            return "Recovery point"
        elif payload_type == 9:
            return "Scene info"
        elif payload_type == 45:
            return "Mastering display colour volume"
        return "Unknown/other SEI"

    def _parse_h264_slice_header(self, rbsp: bytes, sps: Optional[Dict[str, object]] = None):
        """Parse slice_header() per H.264 Spec 7.3.3 with SPS-aware field presence.

        Extracts commonly useful fields per H.264 specification:
        - Always present: first_mb_in_slice, slice_type, pic_parameter_set_id
        - Frame num (bits from SPS.log2_max_frame_num, range 0-15, default 16)
        - POC fields (type and values depend on SPS.pic_order_cnt_type)
        - Reference list presence & modifiers (slice_type dependent)
        - Weighted prediction flags (profile/SPS dependent)
        - Deblocking filter flags (always present)
        
        SPS parameters required for correct parsing:
        - log2_max_frame_num (4-15 bits, encodes as value+4)
        - pic_order_cnt_type (0, 1, or 2)
        - log2_max_pic_order_cnt_lsb (4-15 bits, encodes as value+4)
        - frame_mbs_only_flag (affects field_pic_flag presence)
        - num_ref_frames (presence of ref_pic_list_reordering fields)
        - weighted_pred_flag / weighted_bipred_idc (if frame_mbs_only_flag)
        
        Intended for UI display showing structure per spec, not full bit-exact decode.
        """
        if not rbsp:
            return None

        try:
            br = BitReader(rbsp)
            fields = []

            # first_mb_in_slice - UE(v), always present
            first_mb = br.read_ue()
            fields.append(("first_mb_in_slice", f"{first_mb}"))

            # slice_type - UE(v), always present, value 0-9
            # 0-4: P/B/I/SP/SI (primary), 5-9: P/B/I/SP/SI (all refs)
            slice_type_val = br.read_ue()
            if slice_type_val > 9:
                fields.append(("⚠ SPEC VIOLATION", f"slice_type {slice_type_val} out of range [0-9]"))
                return {"slice_type": slice_type_val, "slice_type_name": "Invalid", "fields": fields}
            
            slice_types = {
                0: "P", 1: "B", 2: "I", 3: "SP", 4: "SI",
                5: "P (all)", 6: "B (all)", 7: "I (all)", 8: "SP (all)", 9: "SI (all)"
            }
            slice_type_name = slice_types.get(slice_type_val, str(slice_type_val))
            fields.append(("slice_type", f"{slice_type_val} ({slice_type_name})"))

            # pic_parameter_set_id - UE(v), always present
            pps_id = br.read_ue()
            fields.append(("pic_parameter_set_id", f"{pps_id}"))

            # Colour plane information (high profile)
            # Requires SPS context - simplified to omit here

            # frame_num - UE from SPS.log2_max_frame_num
            log2_max_frame_num = sps.get('log2_max_frame_num', 16) if sps else 16
            if log2_max_frame_num < 4 or log2_max_frame_num > 16:
                fields.append(("⚠ SPEC VIOLATION", f"SPS.log2_max_frame_num {log2_max_frame_num} out of range [4-16]"))
            frame_num = br.read_bits(log2_max_frame_num) if log2_max_frame_num <= 16 else 0
            fields.append(("frame_num", f"{frame_num} (bits={log2_max_frame_num})"))

            # Field flags (for interlaced coding)
            frame_mbs_only_flag = sps.get('frame_mbs_only_flag', 1) if sps else 1
            field_pic_flag = 0
            if not frame_mbs_only_flag:
                field_pic_flag = br.read_bits(1)
                fields.append(("field_pic_flag", f"{field_pic_flag}"))
                if field_pic_flag:
                    bottom_field_flag = br.read_bits(1)
                    fields.append(("bottom_field_flag", f"{bottom_field_flag}"))

            # Picture Order Count (POC) - depends on SPS.pic_order_cnt_type
            pic_order_cnt_type = sps.get('pic_order_cnt_type', 0) if sps else 0
            if pic_order_cnt_type not in (0, 1, 2):
                fields.append(("⚠ SPEC VIOLATION", f"pic_order_cnt_type {pic_order_cnt_type} invalid (0-2)"))
            
            if pic_order_cnt_type == 0:
                log2_max_poc_lsb = sps.get('log2_max_pic_order_cnt_lsb', 8) if sps else 8
                if log2_max_poc_lsb < 4 or log2_max_poc_lsb > 16:
                    fields.append(("⚠ SPEC VIOLATION", f"log2_max_pic_order_cnt_lsb {log2_max_poc_lsb} out of range [4-16]"))
                poc_lsb = br.read_bits(log2_max_poc_lsb) if log2_max_poc_lsb <= 16 else 0
                fields.append(("pic_order_cnt_lsb", f"{poc_lsb} (bits={log2_max_poc_lsb})"))
                
                # delta_pic_order_cnt_bottom (if not field picture)
                if not field_pic_flag and sps and sps.get('delta_pic_order_always_zero_flag') == 0:
                    try:
                        delta = br.read_se()
                        fields.append(("delta_pic_order_cnt_bottom", f"{delta}"))
                    except:
                        pass
            elif pic_order_cnt_type == 1:
                if not field_pic_flag:
                    delta = br.read_se()
                    fields.append(("delta_pic_order_cnt[0]", f"{delta}"))
                if not field_pic_flag and sps and sps.get('delta_pic_order_always_zero_flag') == 0:
                    delta = br.read_se()
                    fields.append(("delta_pic_order_cnt[1]", f"{delta}"))
            # For type 2: POC is derived, no fields

            # IDR picture ID (when nal_unit_type == 5)
            # Note: caller should inject nal_unit_type via sps or separate parameter

            # Reference index overrides (for non-I slices)
            slice_type_modulo = slice_type_val % 5
            if slice_type_modulo not in (2, 4):  # Not I/SI
                num_ref_idx_active_override_flag = br.read_bits(1)
                fields.append(("num_ref_idx_active_override_flag", f"{num_ref_idx_active_override_flag}"))
                if num_ref_idx_active_override_flag:
                    num_ref_idx_l0_active_minus1 = br.read_ue()
                    fields.append(("num_ref_idx_l0_active_minus1", f"{num_ref_idx_l0_active_minus1}"))
                    if slice_type_modulo == 1:  # B slice
                        num_ref_idx_l1_active_minus1 = br.read_ue()
                        fields.append(("num_ref_idx_l1_active_minus1", f"{num_ref_idx_l1_active_minus1}"))

            # Reference picture list reordering (for non-I slices)
            if slice_type_modulo not in (2, 4):  # Not I/SI
                ref_pic_list_reordering_flag_l0 = br.read_bits(1)
                fields.append(("ref_pic_list_reordering_flag_l0", f"{ref_pic_list_reordering_flag_l0}"))
                if ref_pic_list_reordering_flag_l0:
                    fields.append(("(ref_pic_list_reordering_l0 loop)", "present but not fully parsed"))
                
                if slice_type_modulo == 1:  # B slice
                    ref_pic_list_reordering_flag_l1 = br.read_bits(1)
                    fields.append(("ref_pic_list_reordering_flag_l1", f"{ref_pic_list_reordering_flag_l1}"))
                    if ref_pic_list_reordering_flag_l1:
                        fields.append(("(ref_pic_list_reordering_l1 loop)", "present but not fully parsed"))

            # Weighted prediction
            if (sps and sps.get('weighted_pred_flag') and slice_type_modulo in (0, 3)) or \
               (sps and sps.get('weighted_bipred_idc') == 1 and slice_type_modulo == 1):
                fields.append(("(pred_weight_table)", "present but not fully parsed"))

            # Decoded reference picture marking (if nal_ref_idc != 0)
            fields.append(("dec_ref_pic_marking", "(shown if nal_ref_idc > 0)"))

            # Return structured result
            return {
                "slice_type": slice_type_val,
                "slice_type_name": slice_type_name,
                "fields": fields
            }
        
        except Exception as e:
            return {"slice_type": -1, "slice_type_name": "Parse Error", "fields": [("error", str(e))]}

    def _read_ue(self, data: bytes):
        # Read unsigned Exp-Golomb code from bytes
        bits = 0
        val = 0
        total_bits = len(data) * 8
        bit_idx = 0
        zeros = 0
        while bit_idx < total_bits:
            byte = data[bit_idx // 8]
            bit = (byte >> (7 - (bit_idx % 8))) & 1
            if bit == 0:
                zeros += 1
            else:
                break
            bit_idx += 1
        bit_idx += 1
        if zeros > 0:
            val = 1 << zeros
            for i in range(zeros):
                if bit_idx < total_bits:
                    byte = data[bit_idx // 8]
                    bit = (byte >> (7 - (bit_idx % 8))) & 1
                    val |= bit << (zeros - 1 - i)
                    bit_idx += 1
        return val - 1, bit_idx

    @staticmethod
    def get_nal_type_name(nal_type: int) -> str:
        names = {
            1: "Non-IDR Slice", 5: "IDR Slice", 6: "SEI", 7: "SPS", 8: "PPS", 9: "AUD",
            10: "End Seq", 11: "End Stream", 12: "Filler", 13: "SPS Ext", 19: "IDR (Aux)"
        }
        return names.get(nal_type, f"Unknown ({nal_type})")
    
    def __init__(self, path: str, pcr_jitter_ms: float = 50.0, tei_threshold_pct: float = 0.1, cont_threshold_pct: float = 0.1, progress_callback=None):
        self.path = path
        
        # Detect file format
        self.file_format = self._detect_format()
        self.is_mp4 = (self.file_format in ['mp4', 'mov'])
        
        # Store MP4 analysis result
        self.mp4_analysis = None
        
        self.pcr_jitter_sec = max(0.0, pcr_jitter_ms / 1000.0)
        # thresholds expressed as percent (e.g. 0.1 means 0.1%)
        self.tei_threshold_pct = max(0.0, float(tei_threshold_pct))
        self.cont_threshold_pct = max(0.0, float(cont_threshold_pct))
        self.progress_callback = progress_callback
        
        # Initialize T-STD buffer analyzer
        self.tstd_analyzer = T_STD_Analyzer() if BUFFER_ANALYSIS_AVAILABLE else None
        # State attributes
        self.total_packets = 0
        self.sync_errors = 0
        self.tei_errors = 0
        self.pid_counts: Dict[int, int] = defaultdict(int)
        self.pid_types: Dict[int, str] = {}
        self.null_packets = 0
        self.continuity_errors: Dict[int, int] = defaultdict(int)
        self.last_cc: Dict[int, Optional[int]] = {}
        self.pcr_records: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.pts_records: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.dts_records: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        # Track most-recent PTS seen per PID for stamping caption blocks
        self.last_pts_by_pid: Dict[int, Optional[float]] = {}
        self.pat_info: Dict = {}
        self.pmts: Dict[int, Dict] = {}
        self.video_pids: Dict[int, int] = {}
        self.video_headers: Dict[int, Dict] = {}
        self.video_pes_buffers: Dict[int, bytearray] = {}
        self.video_syntax_errors: Dict[int, List[str]] = defaultdict(list)
        self.video_nal_stats: Dict[int, Dict[str, object]] = {}
        # Subtitles/Teletext tracking
        self.teletext_pids: Dict[int, int] = {}  # PID -> stream_type
        self.dvb_subtitle_pids: Dict[int, int] = {}  # PID -> stream_type
        self.teletext_messages: Dict[int, List[Dict[str, object]]] = defaultdict(list)
        self.dvb_subtitle_summary: Dict[int, Dict[str, object]] = {}
        self.pes_counts: Dict[int, int] = defaultdict(int)
        self.pid_payload_bytes: Dict[int, int] = defaultdict(int)
        self.pid_payload_sample: Dict[int, bytearray] = defaultdict(bytearray)
        self.pid_pes_errors: Dict[int, List[str]] = defaultdict(list)
        self.file_size = os.path.getsize(self.path)
        self.pat_warnings: List[str] = []
        self.pmt_warnings: List[str] = []
        
        # SCTE-35 timing: track PTS when each message was received
        self.scte35_message_pts: Dict[int, List[float]] = defaultdict(list)  # PID -> list of PTS values
        
        # Buffer analysis (HRD/T-STD)
        self.buffer_analyzers: Dict[int, BufferAnalyzer] = {}
        self.audio_pids: Dict[int, int] = {}  # PID -> stream_type
        
        # KLV metadata tracking
        self.klv_pids: Dict[int, Dict] = {}  # PID -> {type, count, sync_type, packets}
        self.klv_in_video: Dict[int, List[Dict]] = {}  # PID -> list of KLV packets found in video
        self.stanag_4609_compliance: Dict = {}  # Compliance check results
        
        # M2TS detection attributes
        self.packet_size = TS_PACKET_SIZE
        self.is_m2ts = False
        self.ts_offset = 0
        
        # Detect M2TS format (192-byte packets with 4-byte timestamp)
        with open(self.path, 'rb') as f:
            first_packets = f.read(192 * 5)
            if len(first_packets) >= 192 * 3:
                # Check if sync bytes are at offset 4 (M2TS)
                m2ts_syncs = sum(1 for i in range(3) if first_packets[i*192 + 4] == 0x47)
                # Check if sync bytes are at offset 0 (TS)
                ts_syncs = sum(1 for i in range(3) if first_packets[i*188] == 0x47)
                
                if m2ts_syncs >= 2:
                    self.is_m2ts = True
                    self.packet_size = 192
                    self.ts_offset = 4
        
        self.reset()

    def reset(self):
        self.total_packets = 0
        self.sync_errors = 0
        self.tei_errors = 0
        self.pid_counts: Dict[int, int] = defaultdict(int)
        self.pid_types: Dict[int, str] = {}  # Track PID type (PAT, PMT, Video, Audio, SCTE-35, etc.)
        self.null_packets = 0
        self.continuity_errors: Dict[int, int] = defaultdict(int)
        self.last_cc: Dict[int, Optional[int]] = {}
        self.pcr_records: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.pts_records: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.dts_records: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.pat_info: Dict = {}
        self.pmts: Dict[int, Dict] = {}
        self.video_pids: Dict[int, int] = {}
        self.video_headers: Dict[int, Dict] = {}
        self.video_pes_buffers: Dict[int, bytearray] = {}
        self.video_syntax_errors: Dict[int, List[str]] = defaultdict(list)
        self.video_nal_stats: Dict[int, Dict[str, object]] = {}
        self.pes_counts: Dict[int, int] = defaultdict(int)
        self.pid_payload_bytes: Dict[int, int] = defaultdict(int)
        self.pid_payload_sample: Dict[int, bytearray] = defaultdict(bytearray)
        self.pid_pes_errors: Dict[int, List[str]] = defaultdict(list)
        
        # Reset KLV tracking
        self.klv_pids: Dict[int, Dict] = {}
        self.klv_in_video: Dict[int, List[Dict]] = {}
        self.stanag_4609_compliance: Dict = {}

    def extract_scte35_messages(self):
        """Extract SCTE-35 splice_info_section messages from SCTE-35 PIDs with timing."""
        scte35_pids = [pid for pid, t in self.pid_types.items() if t == 'SCTE-35']
        messages = {}
        if not scte35_pids:
            return messages
        
        # Parse accumulated SCTE-35 data from buffers
        for pid in scte35_pids:
            if pid not in self.video_pes_buffers:
                continue
            
            buffer = bytes(self.video_pes_buffers[pid])
            pid_messages = []
            buffer_len = len(buffer)
            
            # Get PTS values for this PID to stamp messages
            pts_list = self.scte35_message_pts.get(pid, [])
            message_idx = 0
            
            # Look for SCTE-35 sections (table_id = 0xFC)
            # Optimized: Use bytes.find() instead of byte-by-byte scanning
            i = 0
            while i < buffer_len - 3:
                # Fast search for potential SCTE-35 table_id (0xFC)
                fc_pos = buffer.find(b'\xfc', i, buffer_len)
                if fc_pos == -1:
                    break  # No more 0xFC bytes found
                
                # Check if this is preceded by PES start code
                if fc_pos >= 9 and buffer[fc_pos-9:fc_pos-6] == b'\x00\x00\x01':
                    # PES packet format - verify it's the payload start
                    pes_start = fc_pos - 9
                    if pes_start + 8 < buffer_len:
                        pes_header_len = buffer[pes_start + 8]
                        expected_payload = pes_start + 9 + pes_header_len
                        if fc_pos == expected_payload:
                            # Valid PES-wrapped SCTE-35 section
                            if fc_pos + 2 < buffer_len:
                                section_length = ((buffer[fc_pos + 1] & 0x0F) << 8) | buffer[fc_pos + 2]
                                section_end = fc_pos + 3 + section_length
                                if section_end <= buffer_len:
                                    section_data = buffer[fc_pos:section_end]
                                    parsed = self.parse_scte35(section_data)
                                    # Add packet PTS if available
                                    if message_idx < len(pts_list):
                                        parsed['packet_pts_seconds'] = pts_list[message_idx]
                                    message_idx += 1
                                    pid_messages.append(parsed)
                                    i = section_end
                                    continue
                
                # Try as direct section (no PES wrapper)
                if fc_pos + 2 < buffer_len:
                    section_length = ((buffer[fc_pos + 1] & 0x0F) << 8) | buffer[fc_pos + 2]
                    section_end = fc_pos + 3 + section_length
                    if section_end <= buffer_len and section_length > 0 and section_length < 4096:
                        section_data = buffer[fc_pos:section_end]
                        parsed = self.parse_scte35(section_data)
                        # Only add if it's a valid parse (not an error)
                        if 'error' not in parsed or 'table_id' in parsed:
                            # Add packet PTS if available
                            if message_idx < len(pts_list):
                                parsed['packet_pts_seconds'] = pts_list[message_idx]
                            message_idx += 1
                            pid_messages.append(parsed)
                        i = section_end
                        continue
                
                # Move past this byte and continue searching
                i = fc_pos + 1
            
            if pid_messages:
                messages[f"0x{pid:04X}"] = pid_messages
        
        return messages

    def parse_scte35(self, payload: bytes) -> dict:
        """Parse SCTE-35 splice_info_section and build a tree for GUI display.

        Focus: splice_insert() command decoded into nested fields as requested.
        """
        if not payload or payload[0] != 0xFC:
            return {"error": "Not SCTE-35 splice_info_section"}

        def bits_to_int(bitbuf: bytes, start_bit: int, num_bits: int) -> int:
            # Extract num_bits starting at start_bit (0-based) big-endian
            val = 0
            for i in range(num_bits):
                bit_idx = start_bit + i
                byte_idx = bit_idx // 8
                off = 7 - (bit_idx % 8)
                if byte_idx >= len(bitbuf):
                    break
                bit = (bitbuf[byte_idx] >> off) & 1
                val = (val << 1) | bit
            return val

        try:
            # SCTE-35 Specification Validation
            validation_result = {}
            if SCTE35_VALIDATOR_AVAILABLE:
                validator = SCTE35Validator()
                validation_result = validator.validate_splice_info_section(payload)
            
            table_id = payload[0]
            section_length = ((payload[1] & 0x0F) << 8) | payload[2]
            protocol_version = payload[3]
            encrypted_packet = (payload[4] & 0x80) >> 7
            encryption_algorithm = (payload[4] >> 1) & 0x3F
            # Field bit layout per SCTE-35, section 6.3.1
            pts_adjustment = ((payload[4] & 0x01) << 32) | (payload[5] << 24) | (payload[6] << 16) | (payload[7] << 8) | payload[8]
            cw_index = payload[9]
            tier = ((payload[10] << 8) | payload[11]) >> 4  # upper 12 bits
            splice_command_length = ((payload[11] & 0x0F) << 8) | payload[12]
            splice_command_type = payload[13] if len(payload) > 13 else 0

            splice_command_types = {
                0x00: "null",
                0x05: "splice_insert",
                0x06: "time_signal",
                0x07: "bandwidth_reservation",
                0xFF: "private"
            }
            command_name = splice_command_types.get(splice_command_type, f"type_{splice_command_type}")

            tree = []
            header_node = {
                "label": "splice_info_section",
                "value": "",
                "info": "",
                "hex": "",
                "children": [
                    {"label": "table_id", "value": f"0x{table_id:02X}", "info": "", "hex": "", "children": []},
                    {"label": "section_length", "value": str(section_length), "info": "bytes", "hex": "", "children": []},
                    {"label": "protocol_version", "value": str(protocol_version), "info": "", "hex": "", "children": []},
                    {"label": "encrypted_packet", "value": str(encrypted_packet), "info": "1=encrypted", "hex": "", "children": []},
                    {"label": "encryption_algorithm", "value": str(encryption_algorithm), "info": "", "hex": "", "children": []},
                    {"label": "pts_adjustment", "value": str(pts_adjustment), "info": "90kHz", "hex": "", "children": []},
                    {"label": "cw_index", "value": str(cw_index), "info": "", "hex": "", "children": []},
                    {"label": "tier", "value": f"0x{tier:03X}", "info": "12 bits", "hex": "", "children": []},
                    {"label": "splice_command_length", "value": str(splice_command_length), "info": "", "hex": "", "children": []},
                    {"label": "splice_command_type", "value": f"0x{splice_command_type:02X}", "info": command_name, "hex": "", "children": []},
                ]
            }

            # Parse splice_insert if present
            # Track timing data for graphing
            splice_pts_time = None
            splice_duration_ticks = None
            out_of_network_indicator = None
            splice_immediate_flag = None
            
            if splice_command_type == 0x05:
                cmd_start = 14
                cmd_end = cmd_start + splice_command_length
                cmd_bytes = payload[cmd_start:cmd_end]
                b = cmd_bytes
                pos = 0
                def get_bits(n):
                    nonlocal pos
                    val = bits_to_int(b, pos, n)
                    pos += n
                    return val

                splice_children = []
                splice_event_id = get_bits(32)
                splice_children.append({"label": "splice_event_id", "value": str(splice_event_id), "info": "uimsbf", "hex": "", "children": []})
                cancel = get_bits(1)
                splice_children.append({"label": "splice_event_cancel_indicator", "value": str(cancel), "info": "bslbf", "hex": "", "children": []})
                reserved = get_bits(7)
                splice_children.append({"label": "reserved", "value": str(reserved), "info": "", "hex": "", "children": []})

                if cancel == 0:
                    out_of_network_indicator = get_bits(1)
                    program_splice_flag = get_bits(1)
                    duration_flag = get_bits(1)
                    splice_immediate_flag = get_bits(1)
                    event_id_compliance_flag = get_bits(1)
                    reserved3 = get_bits(3)
                    splice_children.extend([
                        {"label": "out_of_network_indicator", "value": str(out_of_network_indicator), "info": "bslbf", "hex": "", "children": []},
                        {"label": "program_splice_flag", "value": str(program_splice_flag), "info": "bslbf", "hex": "", "children": []},
                        {"label": "duration_flag", "value": str(duration_flag), "info": "bslbf", "hex": "", "children": []},
                        {"label": "splice_immediate_flag", "value": str(splice_immediate_flag), "info": "bslbf", "hex": "", "children": []},
                        {"label": "event_id_compliance_flag", "value": str(event_id_compliance_flag), "info": "bslbf", "hex": "", "children": []},
                        {"label": "reserved", "value": str(reserved3), "info": "3 bits", "hex": "", "children": []},
                    ])

                    def parse_splice_time() -> tuple:
                        """Returns (tree_node, pts_time_value or None)"""
                        time_children = []
                        tsf = get_bits(1)
                        time_children.append({"label": "time_specified_flag", "value": str(tsf), "info": "bslbf", "hex": "", "children": []})
                        pts_val = None
                        if tsf:
                            time_children.append({"label": "reserved", "value": str(get_bits(6)), "info": "6 bslbf", "hex": "", "children": []})
                            pts_val = get_bits(33)
                            time_children.append({"label": "pts_time", "value": str(pts_val), "info": "33-bit, 90kHz", "hex": "", "children": []})
                        else:
                            time_children.append({"label": "reserved", "value": str(get_bits(7)), "info": "7 bslbf", "hex": "", "children": []})
                        return ({"label": "splice_time()", "value": "", "info": "", "hex": "", "children": time_children}, pts_val)

                    def parse_break_duration() -> tuple:
                        """Returns (tree_node, duration_value)"""
                        bd_children = []
                        auto = get_bits(1)
                        bd_children.append({"label": "auto_return", "value": str(auto), "info": "bslbf", "hex": "", "children": []})
                        bd_children.append({"label": "reserved", "value": str(get_bits(6)), "info": "6 bslbf", "hex": "", "children": []})
                        dur = get_bits(33)
                        bd_children.append({"label": "duration", "value": str(dur), "info": "33-bit, 90kHz", "hex": "", "children": []})
                        return ({"label": "break_duration()", "value": "", "info": "", "hex": "", "children": bd_children}, dur)

                    if program_splice_flag == 1:
                        if splice_immediate_flag == 0:
                            time_node, pts_val = parse_splice_time()
                            splice_children.append(time_node)
                            splice_pts_time = pts_val
                    else:
                        component_count = get_bits(8)
                        comp_children = [
                            {"label": "component_count", "value": str(component_count), "info": "uimsbf", "hex": "", "children": []}
                        ]
                        for _ in range(component_count):
                            tag = get_bits(8)
                            tag_children = [{"label": "component_tag", "value": str(tag), "info": "uimsbf", "hex": "", "children": []}]
                            if splice_immediate_flag == 0:
                                time_node, pts_val = parse_splice_time()
                                tag_children.append(time_node)
                                # Use first component's PTS
                                if splice_pts_time is None:
                                    splice_pts_time = pts_val
                            comp_children.append({"label": "component", "value": "", "info": "", "hex": "", "children": tag_children})
                        splice_children.append({"label": "components", "value": "", "info": "", "hex": "", "children": comp_children})

                    if duration_flag == 1:
                        dur_node, dur_val = parse_break_duration()
                        splice_children.append(dur_node)
                        splice_duration_ticks = dur_val

                    splice_children.extend([
                        {"label": "unique_program_id", "value": str(get_bits(16)), "info": "uimsbf", "hex": "", "children": []},
                        {"label": "avail_num", "value": str(get_bits(8)), "info": "uimsbf", "hex": "", "children": []},
                        {"label": "avails_expected", "value": str(get_bits(8)), "info": "uimsbf", "hex": "", "children": []},
                    ])

                splice_node = {"label": "splice_insert()", "value": "", "info": "", "hex": "", "children": splice_children}
                header_node["children"].append(splice_node)

            tree.append(header_node)

            # Calculate timing for graphing
            # PTS values are in 90kHz ticks, convert to seconds
            # Note: splice_time PTS should be adjusted by pts_adjustment before use
            splice_time_seconds = None
            duration_seconds = None
            
            if splice_pts_time is not None:
                # Apply pts_adjustment as per SCTE-35 spec section 9.4.1
                # adjusted_pts_time = (pts_time + pts_adjustment) mod 2^33
                adjusted_pts = (splice_pts_time + pts_adjustment) & 0x1FFFFFFFF  # 33-bit mask
                splice_time_seconds = adjusted_pts / 90000.0
            
            if splice_duration_ticks is not None:
                duration_seconds = splice_duration_ticks / 90000.0

            return {
                "table_id": table_id,
                "section_length": section_length,
                "protocol_version": protocol_version,
                "encrypted_packet": encrypted_packet,
                "encryption_algorithm": encryption_algorithm,
                "pts_adjustment": pts_adjustment,
                "cw_index": cw_index,
                "tier": tier,
                "splice_command_length": splice_command_length,
                "splice_command_type": splice_command_type,
                "command_name": command_name,
                "tree": tree,
                "raw_hex": payload.hex()[:64] + ("..." if len(payload) > 32 else ""),
                # Timing data for graphing
                "splice_time_seconds": splice_time_seconds,
                "duration_seconds": duration_seconds,
                "splice_pts_time": splice_pts_time,
                "splice_duration_ticks": splice_duration_ticks,
                "out_of_network_indicator": out_of_network_indicator,
                "splice_immediate_flag": splice_immediate_flag,
            }
        except Exception as e:
            return {"error": f"SCTE-35 parse error: {e}"}

    def _detect_format(self) -> str:
        """Detect file format (TS, M2TS, MP4, MOV)"""
        with open(self.path, 'rb') as f:
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
            
            # Check for MPEG-TS sync bytes
            if len(header) >= 188:
                if header[0] == 0x47:  # Sync byte at position 0
                    return 'ts'
                elif len(header) >= 192 and header[4] == 0x47:  # M2TS
                    return 'm2ts'
        
        return 'ts'  # Default to TS
    
    def _parse_h264_sps(self, nal_data: bytes) -> Dict:
        """Parse H.264 SPS NAL unit (wrapper for module-level function)"""
        if len(nal_data) < 4:
            raise ValueError("NAL data too short")
        
        # The NAL data from MP4 avcC already includes the NAL header
        # Format: nal_header (1 byte: 0x67 for SPS) + profile + constraint + level + RBSP
        # Just prepend the start code, the NAL header is already there
        
        data_with_start_code = b'\x00\x00\x00\x01' + nal_data
        
        result = parse_h264_sps(data_with_start_code)
        if result:
            return result
        return {}
    
    def _parse_hevc_sps(self, nal_data: bytes) -> Dict:
        """Parse H.265/HEVC SPS NAL unit (wrapper for module-level function)"""
        # For now, return empty dict - HEVC SPS parsing to be implemented
        # TODO: Implement parse_hevc_sps similar to parse_h264_sps
        return {}
    
    def analyze_mp4(self) -> Dict:
        """Analyze MP4/MOV file and extract NAL units"""
        if not MP4_PARSER_AVAILABLE:
            return {
                'error': 'MP4 parser not available',
                'file_type': 'MP4/MOV',
                'elementary_streams': {}
            }
        
        
        parser = MP4Parser(self.path)
        mp4_info = parser.parse()
        
        # Extract NAL units from video tracks
        elementary_streams = {}
        video_nal_stats = {}
        
        for track_id in parser.video_tracks:
            track_info = parser.tracks[track_id]
            codec_type = track_info.get('codec_type', 'Unknown')
            codec = track_info.get('codec', 'Unknown')
            
            
            # Extract NALs from configuration
            nals = parser.extract_nals_from_track(track_id)
            
            # Build elementary stream info
            stream_info = {
                'stream_type': codec,
                'codec': codec_type,
                'track_id': track_id,
                'nal_count': len(nals),
            }
            
            # Parse NALs based on codec
            if codec_type == 'H.264':
                # Parse SPS/PPS
                for nal_type, nal_data in nals:
                    if nal_type == 7:  # SPS
                        try:
                            sps_info = self._parse_h264_sps(nal_data)
                            stream_info['h264_sps'] = sps_info
                        except Exception as e:
                            pass
                    elif nal_type == 8:  # PPS
                        stream_info['h264_pps_found'] = True
            
            elif codec_type == 'H.265':
                # Parse VPS/SPS/PPS
                for nal_type, nal_data in nals:
                    if nal_type == 32:  # VPS
                        stream_info['hevc_vps_found'] = True
                    elif nal_type == 33:  # SPS
                        try:
                            sps_info = self._parse_hevc_sps(nal_data)
                            stream_info['hevc_sps'] = sps_info
                        except Exception as e:
                            pass
                    elif nal_type == 34:  # PPS
                        stream_info['hevc_pps_found'] = True
            
            elementary_streams[f"track_{track_id}"] = stream_info
            
            # Add to video_nal_stats for GUI compatibility
            video_nal_stats[track_id] = {
                'codec': codec_type,
                'caption_lines': [],
                'caption_lines_field0': [],
                'caption_lines_field1': [],
                'caption_708_lines': [],
                'closed_captions': [],
            }

        # If PyAV is available, do a pass over video packets to extract SEI user_data (closed captions)
        try:
            import av
            # Use demux to find SEI user_data blocks (payload_type 4)
            try:
                container = av.open(self.path)
                video_stream = next((s for s in container.streams if s.type == 'video'), None)
                if video_stream:
                    # Determine length_size (default 4) from codec config for first video track
                    for track_id in parser.video_tracks:
                        track_info = parser.tracks.get(track_id, {})
                        codec_config = track_info.get('codec_config')
                        length_size = 4
                        if codec_config and len(codec_config) >= 5:
                            try:
                                length_size = (codec_config[4] & 0x3) + 1
                            except Exception:
                                length_size = 4

                        # Demux packets and parse SEI NALs
                        for packet in container.demux(video_stream):
                            try:
                                if packet.pts is None:
                                    continue
                                pkt_time = float(packet.pts * packet.time_base)
                                sample_bytes = bytes(packet)
                                pos = 0
                                total = len(sample_bytes)
                                while pos + length_size <= total:
                                    nlen = int.from_bytes(sample_bytes[pos:pos+length_size], 'big')
                                    pos += length_size
                                    if nlen <= 0 or pos + nlen > total:
                                        break
                                    nalu = sample_bytes[pos:pos+nlen]
                                    pos += nlen
                                    if not nalu:
                                        continue
                                    # H.264 nal header
                                    nal_header = nalu[0]
                                    nal_type = nal_header & 0x1F
                                    if nal_type == 6:
                                        rbsp = self._remove_emulation_prevention(nalu[1:])
                                        sei_pos = 0
                                        while sei_pos < len(rbsp) - 1:
                                            # Accumulate payload_type
                                            payload_type = 0
                                            while sei_pos < len(rbsp) and rbsp[sei_pos] == 0xFF:
                                                payload_type += 255
                                                sei_pos += 1
                                            if sei_pos >= len(rbsp):
                                                break
                                            payload_type += rbsp[sei_pos]
                                            sei_pos += 1
                                            # Accumulate payload_size
                                            payload_size = 0
                                            while sei_pos < len(rbsp) and rbsp[sei_pos] == 0xFF:
                                                payload_size += 255
                                                sei_pos += 1
                                            if sei_pos >= len(rbsp):
                                                break
                                            payload_size += rbsp[sei_pos]
                                            sei_pos += 1
                                            payload_end = sei_pos + payload_size
                                            if payload_end > len(rbsp):
                                                break
                                            payload = rbsp[sei_pos:payload_end]

                                            # User data registered ITU-T T.35 (payload_type 4)
                                            if payload_type == 4 and len(payload) >= 3:
                                                country_code = payload[0]
                                                provider_code = (payload[1] << 8) | payload[2]
                                                user_id_ascii = None
                                                if len(payload) >= 7:
                                                    user_id_ascii = payload[3:7].decode('latin-1', errors='ignore').strip()
                                                cc_data_bytes = None
                                                # ATSC (GA94) detection
                                                user_data_type_code = payload[7] if len(payload) >= 8 else None
                                                if country_code == 0xB5 and provider_code == 0x0031 and user_id_ascii == 'GA94' and user_data_type_code == 0x03:
                                                    cc_data_bytes = payload[8:]

                                                blocks = []
                                                if cc_data_bytes:
                                                    flags = cc_data_bytes[0] if len(cc_data_bytes) >= 1 else 0
                                                    cc_count = flags & 0x1F
                                                    em_data_present = (flags & 0x40) != 0
                                                    offset = 1
                                                    if em_data_present and len(cc_data_bytes) > offset:
                                                        offset += 1
                                                    for block_idx in range(cc_count):
                                                        if offset + 3 > len(cc_data_bytes):
                                                            break
                                                        blk = cc_data_bytes[offset:offset+3]
                                                        offset += 3
                                                        cc_valid = (blk[0] & 0x04) != 0
                                                        cc_type = blk[0] & 0x03
                                                        c1_raw = blk[1]
                                                        c2_raw = blk[2]
                                                        c1 = c1_raw & 0x7F
                                                        c2 = c2_raw & 0x7F
                                                        ch1 = chr(c1) if 0x20 <= c1 <= 0x7E else ' '
                                                        ch2 = chr(c2) if 0x20 <= c2 <= 0x7E else ' '
                                                        txt = (ch1 + ch2).strip()
                                                        blocks.append({"valid": cc_valid, "type": cc_type, "hex": blk.hex(), "text": txt})

                                                if blocks:
                                                    block_data = {
                                                        "country_code": country_code,
                                                        "provider_code": provider_code,
                                                        "user_id": user_id_ascii,
                                                        "blocks": blocks,
                                                        "pts": pkt_time
                                                    }
                                                    # Append to video_nal_stats for this track
                                                    if track_id in video_nal_stats:
                                                        video_nal_stats[track_id]['closed_captions'].append(block_data)

                                            sei_pos = payload_end
                            except Exception:
                                continue
                try:
                    container.close()
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            # av not available or error opening - skip per-packet MP4 SEI extraction
            pass
        
        return {
            'file_type': 'MP4/MOV',
            'format': self.file_format,
            'elementary_streams': elementary_streams,
            'video_nal_stats': video_nal_stats,
            'tracks': mp4_info['tracks'],
            'video_tracks': mp4_info['video_tracks'],
        }
    
    def analyze(self):
        
        # Route to MP4 analyzer if file is MP4/MOV
        if self.is_mp4:
            self.mp4_analysis = self.analyze_mp4()
            return self.mp4_analysis
        
        import time
        t_start = time.time()
        bytes_processed = 0
        last_pct = -1
        
        t_read = 0
        t_parse = 0
        t_progress = 0
        packet_count = 0
        
        with open(self.path, 'rb') as f:
            pkt_index = 0
            while True:
                # Measure file read time
                t0 = time.time()
                pkt = f.read(self.packet_size)
                t_read += time.time() - t0
                
                if not pkt:
                    break
                if len(pkt) != self.packet_size:
                    # truncated tail
                    break
                
                packet_count += 1
                
                # Update progress
                t1 = time.time()
                bytes_processed += len(pkt)
                if self.progress_callback:
                    current_pct = int((bytes_processed / self.file_size) * 100)
                    if current_pct != last_pct:
                        last_pct = current_pct
                        self.progress_callback(current_pct)
                t_progress += time.time() - t1
                
                # Measure packet parsing time
                t2 = time.time()
                
                # Extract TS packet (skip M2TS timestamp if present)
                ts_pkt = pkt[self.ts_offset:self.ts_offset + self.packet_size] if self.is_m2ts else pkt
                self.total_packets += 1
                hdr = parse_ts_header(ts_pkt)
                if not hdr.get('sync', False):
                    self.sync_errors += 1
                    pkt_index += 1
                    continue
                pid = hdr['pid']
                self.pid_counts[pid] += 1
                if hdr.get('tei', 0):
                    self.tei_errors += 1
                if pid == 0x1FFF:
                    self.null_packets += 1
                # continuity counter checking (skip NULL packets - PID 0x1FFF)
                if pid != 0x1FFF:
                    cc = hdr['cc']
                    last = self.last_cc.get(pid)
                    if last is None:
                        self.last_cc[pid] = cc
                    else:
                        # expected next cc (mod 16) when payload/adaptation present
                        exp = (last + 1) & 0x0F
                        # if adaptation_field_control indicates no payload and no adaptation, continuity might still increment
                        if cc != exp:
                            self.continuity_errors[pid] += 1
                        self.last_cc[pid] = cc

                # adaptation field parsing for PCR
                afc = hdr['afc']
                adapt_len = 0
                if afc in (2, 3):
                    # adaptation field exists
                    # adaptation starts after 4 bytes header
                    adapt_len = ts_pkt[4]
                    adapt_slice = pkt[4: 4 + 1 + adapt_len] if adapt_len >= 0 else b''
                    pcr = extract_pcr_from_adaptation(adapt_slice)
                    if pcr is not None:
                        self.pcr_records[pid].append((pkt_index, pcr))

                # Accumulate payload bytes for bitrate estimation (all payload packets, not only PUSI)
                payload_offset = 4
                if afc == 3:  # adaptation + payload
                    payload_offset += 1 + adapt_len
                if afc in (1, 3) and payload_offset < self.packet_size:
                    self.pid_payload_bytes[pid] += (self.packet_size - payload_offset)
                    payload = ts_pkt[payload_offset:]
                    # Accumulate small payload sample for codec sniffing (up to 8192 bytes per PID)
                    if len(self.pid_payload_sample[pid]) < 8192:
                        need = 8192 - len(self.pid_payload_sample[pid])
                        self.pid_payload_sample[pid].extend(payload[:need])
                else:
                    payload = b''

                # Extract PTS/DTS from PES packets
                if hdr.get('pusi', 0) and afc in (1, 3) and payload:
                        # PES header parsing & validation
                        if len(payload) >= 6 and payload[0] == 0x00 and payload[1] == 0x00 and payload[2] == 0x01:
                            self.pes_counts[pid] += 1
                            stream_id = payload[3]
                            pes_packet_length = (payload[4] << 8) | payload[5]
                            # Validate pes_packet_length if non-zero (0 allowed for video streams meaning unspecified)
                            if pes_packet_length != 0 and pes_packet_length + 6 > len(payload):
                                self.pid_pes_errors[pid].append(
                                    f"PES length {pes_packet_length} exceeds available payload bytes {len(payload)}"
                                )
                            # Basic stream_id classification sanity check
                            if stream_id in (0xBE, 0xBF, 0xF0, 0xF1, 0xF2, 0xF8):
                                # Padding or private / reserved
                                pass
                            # PTS/DTS flags consistency check (reuse earlier extraction if needed)
                            if len(payload) >= 9:
                                pts_dts_flags = (payload[7] & 0xC0) >> 6
                                if pts_dts_flags == 1:
                                    self.pid_pes_errors[pid].append("PES has DTS only flag (invalid per ISO/IEC 13818-1)")
                        pts, dts = extract_pts_dts(payload, True)
                        if pts is not None:
                            self.pts_records[pid].append((pkt_index, pts))
                            # remember latest PTS for this PID to stamp captions
                            self.last_pts_by_pid[pid] = pts
                            # For SCTE-35 PIDs, track PTS of each message
                            if self.pid_types.get(pid) == 'SCTE-35':
                                self.scte35_message_pts[pid].append(pts)
                        if dts is not None:
                            self.dts_records[pid].append((pkt_index, dts))
                        
                        # Feed packet to T-STD buffer analyzer if available
                        # Check if this PID has a buffer initialized (includes video, audio, and private data with buffers)
                        if self.tstd_analyzer and self.tstd_analyzer.analyzers and pid in self.tstd_analyzer.analyzers:
                            pcr_time = self.pcr_records[pid][-1][1] if self.pcr_records.get(pid) else None
                            pts_time = pts
                            dts_time = dts
                            packet_bits = self.packet_size * 8
                            self.tstd_analyzer.process_packet(pid, packet_bits, pcr_time, pts_time, dts_time, hdr.get('pusi', 0))
                        
                        # For video, SCTE-35, Teletext, and DVB Subtitle PIDs, accumulate PES data for extraction
                        if pid in self.video_pids or self.pid_types.get(pid) == 'SCTE-35' or pid in self.teletext_pids or pid in self.dvb_subtitle_pids:
                            if pid not in self.video_pes_buffers:
                                self.video_pes_buffers[pid] = bytearray()
                            # Limit buffer size to prevent memory issues (100MB for video, 100KB for SCTE-35)
                            # Increased to 100MB to allow NAL extraction for entire video when navigating
                            max_buffer_size = 100 * 1024 if self.pid_types.get(pid) == 'SCTE-35' else 100 * 1024 * 1024
                            if len(self.video_pes_buffers[pid]) < max_buffer_size:
                                self.video_pes_buffers[pid].extend(payload)
                                # Per-frame mode: parse SEI/NALs on every video PES start to catch captions
                                if hasattr(self, 'per_frame_mode') and self.per_frame_mode and pid in self.video_pids:
                                    try:
                                        # Parse video headers once, then parse NALs for captions per PES
                                        if pid not in self.video_headers:
                                            self._parse_video_header(pid)
                                        # Parse NALs immediately for this accumulated buffer
                                        self._parse_h264_nalus(pid, bytes(self.video_pes_buffers[pid]))
                                    except Exception as e:
                                        self.video_syntax_errors[pid].append(f"Per-frame parse error: {e}")
                elif afc in (1, 3) and (pid in self.video_pids or self.pid_types.get(pid) == 'SCTE-35' or pid in self.teletext_pids or pid in self.dvb_subtitle_pids):
                    payload_offset = 4
                    if afc == 3:
                        payload_offset += 1 + adapt_len
                    if payload_offset < TS_PACKET_SIZE:
                        payload = ts_pkt[payload_offset:]
                        self.pid_payload_bytes[pid] += len(payload)
                        # Append to existing buffer
                        if pid in self.video_pes_buffers:
                            # Limit buffer size to prevent memory issues (100MB for video, 100KB for SCTE-35)
                            max_buffer_size = 100 * 1024 if self.pid_types.get(pid) == 'SCTE-35' else 100 * 1024 * 1024
                            if len(self.video_pes_buffers[pid]) < max_buffer_size:
                                self.video_pes_buffers[pid].extend(payload)
                            # Header parse: in per-frame mode parse ASAP; otherwise keep thresholds
                            if pid in self.video_pids and pid not in self.video_headers:
                                try:
                                    if hasattr(self, 'per_frame_mode') and self.per_frame_mode:
                                        self._parse_video_header(pid)
                                    else:
                                        buffer_size = len(self.video_pes_buffers[pid])
                                        if not hasattr(self, '_header_parse_attempts'):
                                            self._header_parse_attempts = {}
                                        if pid not in self._header_parse_attempts:
                                            self._header_parse_attempts[pid] = 0
                                        if buffer_size >= 5000 and self._header_parse_attempts[pid] == 0:
                                            self._header_parse_attempts[pid] = 1
                                            self._parse_video_header(pid)
                                        elif buffer_size >= 50000 and buffer_size < 500000:
                                            threshold = 50000 * self._header_parse_attempts[pid]
                                            if buffer_size >= threshold and threshold <= 500000:
                                                self._header_parse_attempts[pid] += 1
                                                self._parse_video_header(pid)
                                        elif buffer_size >= 500000:
                                            threshold = 500000 + (5000000 * (self._header_parse_attempts[pid] - 10))
                                            if buffer_size >= threshold:
                                                self._header_parse_attempts[pid] += 1
                                                self._parse_video_header(pid)
                                except Exception as e:
                                    self.video_syntax_errors[pid].append(f"Header parse error: {e}")
                
                # KLV metadata detection in PES packets
                # Check for KLV in private data streams (stream type 0x06) - Asynchronous KLV
                # Check for KLV embedded in video streams - Synchronous KLV
                if hdr.get('pusi', 0) and afc in (1, 3):
                    payload_offset = 4
                    if afc == 3:
                        payload_offset += 1 + adapt_len
                    if payload_offset < TS_PACKET_SIZE:
                        payload = ts_pkt[payload_offset:]
                        
                        # Detect KLV in this payload
                        klv_packets = detect_klv_metadata(payload)
                        if klv_packets:
                            # Determine if this is synchronous (in video) or asynchronous (separate PID)
                            is_video_pid = pid in self.video_pids
                            sync_type = "Synchronous (Embedded in Video)" if is_video_pid else "Asynchronous (Separate PID)"
                            
                            if is_video_pid:
                                # Synchronous KLV embedded in video
                                if pid not in self.klv_in_video:
                                    self.klv_in_video[pid] = []
                                self.klv_in_video[pid].extend(klv_packets)
                            else:
                                # Asynchronous KLV on separate PID
                                if pid not in self.klv_pids:
                                    self.klv_pids[pid] = {
                                        'type': 'KLV Metadata',
                                        'count': 0,
                                        'sync_type': sync_type,
                                        'packets': [],
                                        'stream_type': self.audio_pids.get(pid, 0x06)  # Usually type 0x06
                                    }
                                    self.pid_types[pid] = "KLV Metadata"
                                
                                self.klv_pids[pid]['count'] += len(klv_packets)
                                self.klv_pids[pid]['packets'].extend(klv_packets)

                # Mark PID 0 as PAT
                if pid == 0:
                    self.pid_types[0] = "PAT"
                
                # Mark NULL packets
                if pid == 0x1FFF:
                    self.pid_types[pid] = "NULL"
                
                # parse PAT/PMT payloads when pusi set and pid corresponds
                if hdr.get('pusi', 0) and hdr.get('pid') == 0:
                    # payload begins at offset 4 or 4+adapt_len when adaptation present
                    payload_offset = 4
                    if afc in (2, 3):
                        adapt_len = ts_pkt[4]
                        payload_offset += 1 + adapt_len
                    if payload_offset < TS_PACKET_SIZE:
                        payload = ts_pkt[payload_offset:]
                        pat = parse_pat(payload)
                        if pat and pat.get('programs'):
                            self.pat_info = pat
                            # Capture PAT warnings
                            if pat.get('warnings'):
                                self.pat_warnings.extend(pat['warnings'])
                            # Mark PMT PIDs
                            for prog_num, pmt_pid in pat['programs'].items():
                                self.pid_types[pmt_pid] = f"PMT (Program {prog_num})"
                
                # if pusi and pid is a PMT PID, parse PMT
                pmt_pids = [p for p in self.pat_info.get('programs', {}).values()] if self.pat_info else []
                if hdr.get('pusi', 0) and pid in pmt_pids:
                    payload_offset = 4
                    if afc in (2, 3):
                        adapt_len = ts_pkt[4]
                        payload_offset += 1 + adapt_len
                    if payload_offset < TS_PACKET_SIZE:
                        payload = ts_pkt[payload_offset:]
                        pmt = parse_pmt(payload)
                        if pmt and pmt.get('pcr_pid') is not None:
                            self.pmts[pid] = pmt
                            # Capture PMT warnings
                            if pmt.get('warnings'):
                                for warning in pmt['warnings']:
                                    self.pmt_warnings.append(f"PMT PID 0x{pid:04X}: {warning}")
                            # Mark PCR PID
                            if pmt['pcr_pid'] not in self.pid_types:
                                self.pid_types[pmt['pcr_pid']] = "PCR"
                            # Mark elementary stream PIDs
                            for stream in pmt.get('streams', []):
                                stream_pid = stream['pid']
                                stream_type = stream['type']
                                if stream_pid not in self.pid_types:
                                    self.pid_types[stream_pid] = stream['type_name']
                                # Track video PIDs for header parsing
                                if stream_type in [0x01, 0x02, 0x10, 0x1B, 0x24]:  # MPEG-1/2, MPEG-4, H.264, H.265
                                    self.video_pids[stream_pid] = stream_type
                                # Track audio PIDs (including private data that might be HDMV LPCM)
                                if stream_type in [0x03, 0x04, 0x06, 0x0F, 0x11, 0x81, 0x84, 0x87]:  # Audio types + private data
                                    self.audio_pids[stream_pid] = stream_type
                                # Detect Teletext and DVB Subtitles via descriptors
                                for desc in stream.get('descriptors', []):
                                    if desc.get('tag') == 0x56:  # Teletext
                                        self.pid_types[stream_pid] = 'Teletext'
                                        self.teletext_pids[stream_pid] = stream_type
                                    elif desc.get('tag') == 0x59:  # Subtitling
                                        self.pid_types[stream_pid] = 'DVB Subtitles'
                                        self.dvb_subtitle_pids[stream_pid] = stream_type
                                
                                # Initialize T-STD buffer for this stream
                                if self.tstd_analyzer and stream_pid not in [0, 0x1FFF]:
                                    # For PCM audio, try to extract channel/sample rate from descriptors
                                    if stream_type == 0x80:  # PCM
                                        pcm_info = parse_pcm_audio_info(stream.get('descriptors', []))
                                        # Calculate PCM bitrate: sample_rate × bit_depth × channels
                                        pcm_bitrate = pcm_info['sample_rate'] * pcm_info['bit_depth'] * pcm_info['channels']
                                        
                                        # Add buffer and set decode rate
                                        buf = self.tstd_analyzer.add_pid_buffer(stream_pid, stream['type_name'])
                                        self.tstd_analyzer.update_bitrate(stream_pid, pcm_bitrate)
                                    else:
                                        self.tstd_analyzer.add_pid_buffer(stream_pid, stream['type_name'])

                t_parse += time.time() - t2
                pkt_index += 1
        
        t_total = time.time() - t_start
    
    def _parse_video_header(self, pid: int):
        """Parse video header from accumulated PES buffer"""
        if pid not in self.video_pes_buffers or pid not in self.video_pids:
            return
        
        buffer = bytes(self.video_pes_buffers[pid])
        stream_type = self.video_pids[pid]
        
        # MPEG-2 video (types 0x01, 0x02)
        if stream_type in [0x01, 0x02]:
            header = parse_mpeg2_sequence_header(buffer)
            if header:
                self.video_headers[pid] = header
                if header.get('errors'):
                    for error in header['errors']:
                        self.video_syntax_errors[pid].append(f"MPEG-2: {error}")
                if header.get('warnings'):
                    for warning in header['warnings']:
                        self.video_syntax_errors[pid].append(f"MPEG-2 Warning: {warning}")
        
        # H.264 video (type 0x1B)
        elif stream_type == 0x1B:
            sps = parse_h264_sps(buffer)
            pps = parse_h264_pps(buffer)
            
            if sps:
                self.video_headers[pid] = sps
                if sps.get('errors'):
                    for error in sps['errors']:
                        self.video_syntax_errors[pid].append(f"H.264 SPS: {error}")
                if sps.get('warnings'):
                    for warning in sps['warnings']:
                        self.video_syntax_errors[pid].append(f"H.264 SPS Warning: {warning}")
            if pps:
                if pid not in self.video_headers:
                    self.video_headers[pid] = pps
                else:
                    # annotate existing header entry that PPS was also found
                    self.video_headers[pid]['pps_found'] = True
                if pps.get('errors'):
                    for error in pps['errors']:
                        self.video_syntax_errors[pid].append(f"H.264 PPS: {error}")
                if pps.get('warnings'):
                    for warning in pps['warnings']:
                        self.video_syntax_errors[pid].append(f"H.264 PPS Warning: {warning}")

        # Parse general H.264 NAL unit stats + SEI + CC if H.264
        if stream_type == 0x1B:
            if pid not in self.video_nal_stats:
                self.video_nal_stats[pid] = {"nal_counts": defaultdict(int), "sei_messages": [], "closed_captions": []}
            self._parse_h264_nalus(pid, buffer)

        # H.265/HEVC video (type 0x24)
        elif stream_type == 0x24:
            if HEVC_PARSER_AVAILABLE:
                vps = parse_hevc_vps(buffer)
                sps = parse_hevc_sps(buffer)
                pps = parse_hevc_pps(buffer)
                
                if vps:
                    if pid not in self.video_headers:
                        self.video_headers[pid] = vps
                    if vps.get('error'):
                        self.video_syntax_errors[pid].append(f"HEVC VPS: {vps['error']}")
                
                if sps:
                    if pid not in self.video_headers:
                        self.video_headers[pid] = sps
                    else:
                        # Merge SPS info into existing header
                        self.video_headers[pid].update(sps)
                    if sps.get('errors'):
                        for error in sps['errors']:
                            self.video_syntax_errors[pid].append(f"HEVC SPS: {error}")
                    if sps.get('warnings'):
                        for warning in sps['warnings']:
                            self.video_syntax_errors[pid].append(f"HEVC SPS Warning: {warning}")
                
                if pps:
                    if 'pps_found' not in self.video_headers.get(pid, {}):
                        if pid in self.video_headers:
                            self.video_headers[pid]['pps_found'] = True
                    if pps.get('errors'):
                        for error in pps['errors']:
                            self.video_syntax_errors[pid].append(f"HEVC PPS: {error}")
                    if pps.get('warnings'):
                        for warning in pps['warnings']:
                            self.video_syntax_errors[pid].append(f"HEVC PPS Warning: {warning}")
                
                # Parse HEVC NAL units
                if pid not in self.video_nal_stats:
                    self.video_nal_stats[pid] = {"nal_counts": defaultdict(int), "sei_messages": [], "closed_captions": []}
                self._parse_hevc_nalus(pid, buffer)


    def _parse_hevc_nalus(self, pid: int, data: bytes, max_nalus: int = 2000):
        """
        Parse HEVC (H.265) NAL units from PES buffer data.
        """
        if not HEVC_PARSER_AVAILABLE:
            return
        
        
        stats = self.video_nal_stats[pid]
        nal_counts = stats["nal_counts"]
        
        # Find all HEVC NAL units
        nal_units = find_hevc_nal_units(data, max_nalus)
        
        
        for nal_type, start_pos, nal_data in nal_units:
            nal_counts[nal_type] += 1
            
            # Parse specific NAL unit types
            if nal_type == 32:  # VPS
                vps = parse_hevc_vps(nal_data)
                if vps and pid not in self.video_headers:
                    self.video_headers[pid] = vps
            elif nal_type == 33:  # SPS
                sps = parse_hevc_sps(nal_data)
                if sps:
                    if pid not in self.video_headers:
                        self.video_headers[pid] = sps
                    else:
                        self.video_headers[pid].update(sps)
            elif nal_type == 34:  # PPS
                pps = parse_hevc_pps(nal_data)
                if pps and pid in self.video_headers:
                    self.video_headers[pid]['pps_found'] = True
            elif nal_type == 39:  # PREFIX_SEI
                # TODO: Parse HEVC SEI messages (similar to H.264)
                pass
            elif nal_type == 40:  # SUFFIX_SEI
                # TODO: Parse HEVC SEI messages
                pass


    def _remove_emulation_prevention(self, rbsp: bytes) -> bytes:
        out = bytearray()
        zeros = 0
        for b in rbsp:
            if zeros >= 2 and b == 0x03:
                zeros = 0
                continue
            out.append(b)
            if b == 0x00:
                zeros += 1
            else:
                zeros = 0
        return bytes(out)

    def _parse_h264_nalus(self, pid: int, data: bytes, max_nalus: int = 2000):
        """
        Parse H.264 NAL units from PES buffer data.
        The buffer may contain multiple PES packets, so we need to extract payloads properly.
        """
        stats = self.video_nal_stats[pid]
        nal_counts = stats["nal_counts"]
        sei_messages = stats["sei_messages"]
        closed_captions = stats["closed_captions"]
        
        # Track all unique CEA-608 byte pairs seen across entire file
        if "all_cea608_bytes_seen" not in stats:
            stats["all_cea608_bytes_seen"] = []
        all_cea608_bytes_seen = stats["all_cea608_bytes_seen"]
        
        # Track unique captions to avoid duplicating same caption across all frames
        if "caption_sei_blocks" not in stats:
            stats["caption_sei_blocks"] = []  # List of SEI caption blocks with their content
        caption_sei_blocks = stats["caption_sei_blocks"]
        
        # Track unique ATSC payloads seen
        if "unique_atsc_payloads" not in stats:
            stats["unique_atsc_payloads"] = set()
        unique_atsc_payloads = stats["unique_atsc_payloads"]
        
        if "caption_lines" not in stats:
            stats["caption_lines"] = []  # aggregated human-readable lines (for backward compat)
        caption_lines = stats["caption_lines"]
        if "caption_lines_field0" not in stats:
            stats["caption_lines_field0"] = []  # CEA-608 field 0 captions (interlaced video)
        caption_lines_field0 = stats["caption_lines_field0"]
        if "caption_lines_field1" not in stats:
            stats["caption_lines_field1"] = []  # CEA-608 field 1 captions (interlaced video)
        caption_lines_field1 = stats["caption_lines_field1"]
        if "caption_708_lines" not in stats:
            stats["caption_708_lines"] = []  # heuristic decoded lines for CEA-708
        caption_708_lines = stats["caption_708_lines"]
        if "_cc708_buffer" not in stats:
            stats["_cc708_buffer"] = bytearray()  # accumulate raw cc_type 2/3 bytes
        cc708_buffer: bytearray = stats["_cc708_buffer"]
        if "caption_708_services" not in stats:
            stats["caption_708_services"] = []  # structured service output
        if "caption_standards_found" not in stats:
            stats["caption_standards_found"] = set()  # Track which caption standards we detected
        caption_standards = stats["caption_standards_found"]
        
        # Extract H.264 payload from PES packets
        # PES packets have structure: 0x000001 + stream_id + length + header... + payload
        h264_payload = bytearray()
        pos = 0
        data_len = len(data)
        
        while pos < data_len - 6:
            # Look for PES start code 0x000001
            pes_pos = data.find(b'\x00\x00\x01', pos)
            if pes_pos == -1:
                # No more PES packets; append remaining data as payload
                h264_payload.extend(data[pos:])
                break
            
            # Skip to PES header
            pos = pes_pos + 3
            if pos + 5 > data_len:
                break
                
            stream_id = data[pos]
            pos += 1
            
            # Skip stream_id that's not video (stream_id should be 0xE0-0xEF for video)
            if not (0xE0 <= stream_id <= 0xEF):
                # Not video PES; skip this entire section and continue
                pos = pes_pos + 3
                continue
            
            # Get PES packet length (0 = unbounded)
            pes_packet_length = (data[pos] << 8) | data[pos+1]
            pos += 2
            
            if pes_packet_length == 0:
                # Unbounded packet; append everything until next PES or end
                if data_len > pos:
                    h264_payload.extend(data[pos:])
                break
            
            # PES header structure: 2 bytes flags + header_len + optional fields
            if pos + 2 > data_len:
                break
                
            pes_flags = data[pos]
            pes_header_len = data[pos + 1]
            pos += 2
            
            # Skip optional PES header fields (PTS/DTS, extension, etc.)
            if pos + pes_header_len > data_len:
                break
            pos += pes_header_len
            
            # Now 'pos' points to actual H.264 payload
            payload_len = pes_packet_length - 3 - pes_header_len  # subtract flags + header_len bytes
            if payload_len > 0 and pos + payload_len <= data_len:
                h264_payload.extend(data[pos:pos+payload_len])
                pos += payload_len
            else:
                # Invalid length; try to continue from next position
                pos += 1
        
        # Now parse NALs from extracted payload
        if not h264_payload:
            return
        
        
        # Limit analysis size to avoid huge memory/time
        # TEMPORARILY DISABLED to find captions
        max_parse_size = 50 * 1024 * 1024  # 50MB - parse more data
        if len(h264_payload) > max_parse_size:
            h264_payload = h264_payload[:max_parse_size]
        
        # Optimized: Use bytes.find() for start code search on extracted payload
        positions = []
        i = 0
        dlen = len(h264_payload)
        
        while i < dlen - 4 and len(positions) < max_nalus:
            # Fast search for 0x0000 pattern
            zero_pos = h264_payload.find(b'\x00\x00', i, dlen - 2)
            if zero_pos == -1:
                break
            
            # Check for 3-byte start code (0x000001)
            if zero_pos + 2 < dlen and h264_payload[zero_pos + 2] == 0x01:
                positions.append(zero_pos + 3)
                i = zero_pos + 3
            # Check for 4-byte start code (0x00000001)
            elif zero_pos + 3 < dlen and h264_payload[zero_pos + 2] == 0x00 and h264_payload[zero_pos + 3] == 0x01:
                positions.append(zero_pos + 4)
                i = zero_pos + 4
            else:
                i = zero_pos + 2
        
        if not positions:
            return
        
        # Add end sentinel
        positions.append(dlen)
        for idx in range(len(positions)-1):
            start = positions[idx]
            end = positions[idx+1]
            nal_unit = h264_payload[start:end]
            if not nal_unit:
                continue
            nal_header = nal_unit[0]
            nal_unit_type = nal_header & 0x1F
            nal_counts[nal_unit_type] += 1
            # SEI (type 6)
            if nal_unit_type == 6:
                rbsp = self._remove_emulation_prevention(nal_unit[1:])
                sei_pos = 0
                # Parse multiple SEI messages in one NAL
                sei_message_count = 0

                # Parse all SEI messages in this NAL unit
                # Multiple SEI messages can be in one NAL unit, or just one
                # End is marked by rbsp_trailing_bits (0x80 followed by zeros)
                while sei_pos < len(rbsp) - 1:  # Need at least 2 bytes for type+size
                    # Check for rbsp_trailing_bits pattern (0x80 followed by zeros)
                    # which marks end of SEI messages
                    if rbsp[sei_pos] == 0x80:
                        # Check if rest is zeros OR starts with 0x00 0x00 (potential start code)
                        remaining = rbsp[sei_pos+1:]
                        if len(remaining) == 0 or all(b == 0x00 for b in remaining) or (len(remaining) >= 3 and remaining[0] == 0x00 and remaining[1] == 0x00):
                            break  # End of SEI messages


                    # payloadType accumulation
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
                    # payloadSize accumulation (each 0xFF byte adds 255)
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
                        sei_messages.append({
                            "type": payload_type,
                            "length": payload_size,
                            "error": "SEI payload truncated",
                            "nal_index": idx
                        })
                        break
                    payload = rbsp[sei_pos:payload_end]
                    sei_messages.append({
                        "type": payload_type,
                        "length": payload_size,
                        "nal_index": idx
                    })
                    sei_message_count += 1
                    # Optional: log non-CC SEI types for visibility
                    # if payload_type != 4:
                    #     print(f"[SEI Type] Found SEI payload_type={payload_type}, size={payload_size} bytes")

                    # User data registered ITU-T T.35 (payload_type 4) - Closed Captions
                    if payload_type == 4 and payload_size >= 3:
                        country_code = payload[0]
                        provider_code = payload[1] << 8 | payload[2]
                        user_id_ascii = None
                        if payload_size >= 7:
                            user_id_ascii = payload[3:7].decode('latin-1', errors='ignore').strip()
                        user_data_type_code = payload[7] if payload_size >= 8 else None
                        
                        # Detect caption standard
                        standard_name = "Unknown ITU-T T.35"
                        cc_data_bytes = None
                        
                        # Print FULL hex dump of this SEI payload for inspection
                        
                        pass  # SEI payload processed
                        
                        # ATSC Standard (most common in North America)
                        if country_code == 0xB5 and provider_code == 0x0031 and user_id_ascii == 'GA94' and user_data_type_code == 0x03:
                            standard_name = "ATSC (CEA-608/708)"
                            cc_data_bytes = payload[8:]  # Skip 8-byte header
                            caption_standards.add("ATSC")
                            
                            # Track unique payloads
                            payload_hash = payload.hex()
                            if payload_hash not in unique_atsc_payloads:
                                unique_atsc_payloads.add(payload_hash)
                                # Show first 20 cc_data bytes
                                if len(cc_data_bytes) >= 20:
                        # UK Freeview/DTG user data (DTG1) is not captions; often AFD/bar data
                                    pass
                        elif country_code == 0xB5 and provider_code == 0x0031 and (user_id_ascii and user_id_ascii.strip().upper() in ('DTG1','UKDT')):
                            standard_name = "DTG1 user data (non-caption)"
                            cc_data_bytes = None  # do not treat as CC
                            caption_standards.add("DTG1")
                        # DVB/EBU Standard (Europe)
                        elif country_code == 0x00 and provider_code == 0x0000:
                            standard_name = "DVB/EBU Subtitles"
                            cc_data_bytes = payload[3:] if payload_size > 3 else None
                            caption_standards.add("DVB/EBU")
                        # Japan ISDB-T Standard
                        elif country_code in (0x81, 0xA4):
                            standard_name = "ISDB-T (Japan)"
                            cc_data_bytes = payload[4:] if payload_size > 4 else None
                            caption_standards.add("ISDB-T")
                        # Generic ITU-T T.35 fallback
                        elif payload_size > 4:
                            standard_name = f"Generic ITU-T (country=0x{country_code:02X}, provider=0x{provider_code:04X})"
                            cc_data_bytes = payload[4:]
                            caption_standards.add("Generic-ITU-T")
                            # Log this for debugging
                        
                        if cc_data_bytes:
                            blocks = []
                            if standard_name.startswith("ATSC"):
                                # ATSC: First byte contains flags and cc_count
                                flags = cc_data_bytes[0] if len(cc_data_bytes) >= 1 else 0
                                cc_count = flags & 0x1F
                                em_data_present = (flags & 0x40) != 0  # Bit 6 is process_em_data_flag
                                offset = 1
                                if em_data_present and len(cc_data_bytes) > offset:
                                    # Skip em_data byte
                                    offset += 1
                                all_cea608_bytes = []
                                non_zero_found = False
                                for block_idx in range(cc_count):
                                    if offset + 3 > len(cc_data_bytes):
                                        break
                                    blk = cc_data_bytes[offset:offset+3]
                                    offset += 3
                                    cc_valid = (blk[0] & 0x04) != 0
                                    cc_type = blk[0] & 0x03
                                    c1_raw = blk[1]
                                    c2_raw = blk[2]
                                    c1 = c1_raw & 0x7F
                                    c2 = c2_raw & 0x7F
                                    
                                    # LOG ANY NON-PADDING BYTES (not 0x00 or 0x80 padding patterns)
                                    is_padding = (c1_raw == 0x80 and c2_raw == 0x80) or (c1_raw == 0x00 and c2_raw == 0x00)
                                    if not is_padding:
                                        non_zero_found = True
                                    
                                    # Collect all bytes for analysis
                                    if cc_valid and cc_type in (0, 1):
                                        all_cea608_bytes.append((c1_raw, c2_raw, c1, c2))
                                        byte_pair = f"0x{c1_raw:02X}:0x{c2_raw:02X}"
                                        if byte_pair not in all_cea608_bytes_seen:
                                            all_cea608_bytes_seen.append(byte_pair)
                                    # Convert to printable text using safe range (0x20-0x7E)
                                    ch1 = chr(c1) if 0x20 <= c1 <= 0x7E else ' '
                                    ch2 = chr(c2) if 0x20 <= c2 <= 0x7E else ' '
                                    txt = ch1 + ch2
                                    blocks.append({"valid": cc_valid, "type": cc_type, "hex": blk.hex(), "text": txt.strip()})
                                    if cc_valid and cc_type in (0,1):
                                        decoded = cea608_decode_chars(blk[1], blk[2])
                                        if decoded:
                                            # Store caption with field info: (text, field_number)
                                            caption_tuple = (decoded, cc_type)  # cc_type 0=field0, 1=field1
                                            caption_lines.append(caption_tuple)
                                            if cc_type == 0:
                                                caption_lines_field0.append(decoded)
                                            else:  # cc_type == 1
                                                caption_lines_field1.append(decoded)
                                        if c2 == 0x0D:
                                            caption_tuple = ('\n', cc_type)
                                            caption_lines.append(caption_tuple)
                                            if cc_type == 0:
                                                caption_lines_field0.append('\n')
                                            else:
                                                caption_lines_field1.append('\n')
                                
                                # Log sample of CEA-608 bytes for debugging
                                if all_cea608_bytes:
                                    # Check if we have any non-80 bytes
                                    has_different = False
                                    for c1r, c2r, c1, c2 in all_cea608_bytes:
                                        if c1r != 0x80 or c2r != 0x80:
                                            has_different = True
                                            break
                                    if not has_different:
                                        sample = all_cea608_bytes[:3]
                                    if cc_valid and cc_type in (2,3):
                                        # CEA-708 buffer: use RAW bytes (with parity), not stripped
                                        cc708_buffer.extend([blk[1], blk[2]])
                            else:
                                # Fallback: treat as consecutive 3-byte blocks
                                for b in range(0, len(cc_data_bytes)//3 * 3, 3):
                                    blk = cc_data_bytes[b:b+3]
                                    if len(blk) < 3:
                                        continue
                                    cc_valid = (blk[0] & 0x04) != 0
                                    cc_type = blk[0] & 0x03
                                    c1 = blk[1] & 0x7F
                                    c2 = blk[2] & 0x7F
                                    ch1 = chr(c1) if 0x20 <= c1 <= 0x7E else ' '
                                    ch2 = chr(c2) if 0x20 <= c2 <= 0x7E else ' '
                                    txt = ch1 + ch2
                                    blocks.append({"valid": cc_valid, "type": cc_type, "hex": blk.hex(), "text": txt.strip()})
                                    if cc_valid and cc_type in (0,1):
                                        decoded = cea608_decode_chars(blk[1], blk[2])
                                        if decoded:
                                            # Store caption with field info: (text, field_number)
                                            caption_tuple = (decoded, cc_type)  # cc_type 0=field0, 1=field1
                                            caption_lines.append(caption_tuple)
                                            if cc_type == 0:
                                                caption_lines_field0.append(decoded)
                                            else:  # cc_type == 1
                                                caption_lines_field1.append(decoded)
                                        if c2 == 0x0D:
                                            caption_tuple = ('\n', cc_type)
                                            caption_lines.append(caption_tuple)
                                            if cc_type == 0:
                                                caption_lines_field0.append('\n')
                                            else:
                                                caption_lines_field1.append('\n')
                                    if cc_valid and cc_type in (2,3):
                                        # CEA-708 buffer: use RAW bytes (with parity), not stripped
                                        cc708_buffer.extend([c1_raw, c2_raw])
                            if blocks:
                                # Track this caption block - avoid duplicating identical consecutive blocks
                                block_data = {
                                    "country_code": country_code, 
                                    "provider_code": provider_code, 
                                    "user_id": user_id_ascii, 
                                    "blocks": blocks,
                                    # Stamp this caption SEI block with the most-recent PTS we saw for this PID (may be None)
                                    "pts": self.last_pts_by_pid.get(pid)
                                }
                                
                                # Check if this is a duplicate of the last caption block
                                is_duplicate = False
                                if caption_sei_blocks:
                                    last_block = caption_sei_blocks[-1]
                                    # Compare blocks content (not object reference)
                                    if (last_block.get('blocks') == blocks and
                                        last_block.get('country_code') == country_code and
                                        last_block.get('provider_code') == provider_code):
                                        is_duplicate = True
                                
                                # Only append if not a duplicate
                                if not is_duplicate:
                                    caption_sei_blocks.append(block_data)
                                    closed_captions.append(block_data)
                    sei_pos = payload_end
                # End of SEI parsing

        # CEA-608 decoding: convert cc_data block pairs using CEA-608 character set
        # This runs ONCE after all NAL units are processed
        # Caption lines now contain tuples of (text, field_num) for interlaced video tracking
        # Filter and convert tuples to strings with field annotation
        processed_captions = []
        current_line = []
        current_field = None
        for item in caption_lines:
            if isinstance(item, tuple):
                text, field_num = item
                if text == '\n':
                    # Flush current line with field annotation
                    if current_line:
                        line_text = "".join(current_line).strip()
                        if line_text and field_num is not None:
                            processed_captions.append(f"[Field {field_num}] {line_text}")
                        elif line_text:
                            processed_captions.append(line_text)
                        current_line = []
                        current_field = None
                else:
                    # Accumulate character pairs into a line
                    if current_field is None:
                        current_field = field_num
                    current_line.append(text)
            elif isinstance(item, str):
                if item == '\n':
                    # Flush current line
                    if current_line:
                        line_text = "".join(current_line).strip()
                        if line_text and current_field is not None:
                            processed_captions.append(f"[Field {current_field}] {line_text}")
                        elif line_text:
                            processed_captions.append(line_text)
                        current_line = []
                        current_field = None
                else:
                    current_line.append(item)
            elif isinstance(item, int):
                current_line.append(str(item))
        # Flush any remaining line
        if current_line:
            line_text = "".join(current_line).strip()
            if line_text and current_field is not None:
                processed_captions.append(f"[Field {current_field}] {line_text}")
            elif line_text:
                processed_captions.append(line_text)
        if processed_captions:
            caption_lines.clear()
            caption_lines.extend(processed_captions)

        # CEA-708 (DTVCC) text extraction from accumulated cc708_buffer
        # This runs ONCE after all NAL units are processed and cc708_buffer is fully populated
        # CEA-708 DTVCC structure: cc_type 2/3 bytes form DTVCC packets with service blocks
        # Format: [service_number (bits 5-0) | block_size_code (bits 7-6)][size_byte][block_data]
        if len(cc708_buffer) > 0:
            try:
                services = {}
                buf = bytes(cc708_buffer)
                pos = 0
                iterations = 0
                svc_count = 0
                
                # Try structured DTVCC service block parsing
                while pos < len(buf) and iterations < 5000:
                    iterations += 1
                    if pos + 1 > len(buf):
                        break
                        
                    first = buf[pos]
                    service_number = first & 0x3F  # bits 5:0
                    block_size_code = (first >> 6) & 0x03  # bits 7:6
                    pos += 1
                    
                    # Check for null/padding
                    if service_number == 0 and block_size_code == 0:
                        continue
                    
                    # Decode block size BEFORE logging
                    if block_size_code == 3:
                        # Extended size: next byte contains actual size
                        if pos >= len(buf):
                            break
                        block_size = buf[pos]
                        pos += 1
                    else:
                        # Small sizes: 1, 2, 3 bytes
                        block_size = block_size_code if block_size_code > 0 else 0
                    
                    svc_count += 1
                    
                    # Extract service data block
                    if pos + block_size > len(buf):
                        break
                        
                    service_data = buf[pos:pos + block_size]
                    pos += block_size
                    
                    # Parse service block: contains CEA-708 commands and data
                    svc = services.setdefault(service_number, {
                        "service": service_number,
                        "lines": [],
                        "raw_bytes": bytearray()
                    })
                    svc["raw_bytes"].extend(service_data)
                    
                    # Simple text extraction: collect printable ASCII, break on control codes
                    current_line = []
                    i = 0
                    while i < len(service_data):
                        b = service_data[i]
                        b_stripped = b & 0x7F  # Strip parity bit
                        
                        # Newline/carriage return
                        if b_stripped in (0x0D, 0x0A):
                            if current_line:
                                line_text = ''.join(current_line).strip()
                                if line_text:
                                    svc["lines"].append(line_text)
                                current_line = []
                            i += 1
                        # Printable ASCII text (after stripping parity)
                        elif 0x20 <= b_stripped <= 0x7E:
                            current_line.append(chr(b_stripped))
                            i += 1
                        # Control codes: check both parity-encoded and raw ranges
                        elif (0x80 <= b <= 0x9F) or (0x00 <= b_stripped <= 0x1F):
                            # Control code sequence; skip to end of current line or next control
                            if current_line:
                                line_text = ''.join(current_line).strip()
                                if line_text:
                                    svc["lines"].append(line_text)
                                current_line = []
                            # Skip this control and next byte (if present)
                            i += 2 if i + 1 < len(service_data) else 1
                        else:
                            # Other control / unused byte
                            if current_line:
                                line_text = ''.join(current_line).strip()
                                if line_text:
                                    svc["lines"].append(line_text)
                                current_line = []
                            i += 1
                    
                    # Flush remaining line
                    if current_line:
                        line_text = ''.join(current_line).strip()
                        if line_text:
                            svc["lines"].append(line_text)
                
                # If no services found, try raw DTVCC byte parsing (direct CEA-708 stream)
                if not services and len(buf) > 0:
                    # Raw DTVCC: treat all bytes as potential CEA-708 data
                    services[0] = {
                        "service": 0,
                        "lines": [],
                        "raw_bytes": buf
                    }
                    
                    current_line = []
                    for b in buf:
                        b_stripped = b & 0x7F  # Strip parity bit
                        if b_stripped in (0x0D, 0x0A):
                            if current_line:
                                line_text = ''.join(current_line).strip()
                                if line_text:
                                    services[0]["lines"].append(line_text)
                                current_line = []
                        elif 0x20 <= b_stripped <= 0x7E:
                            current_line.append(chr(b_stripped))
                        elif (0x80 <= b <= 0x9F) or (0x00 <= b_stripped <= 0x1F):
                            if current_line:
                                line_text = ''.join(current_line).strip()
                                if line_text:
                                    services[0]["lines"].append(line_text)
                                current_line = []
                    
                    if current_line:
                        line_text = ''.join(current_line).strip()
                        if line_text:
                            services[0]["lines"].append(line_text)
                
                # Populate caption_708_lines from all services
                for svc_num in sorted(services.keys()):
                    svc = services[svc_num]
                    for line in svc["lines"]:
                        # Only add if not a duplicate of last line (captions often repeat)
                        if not caption_708_lines or caption_708_lines[-1] != line:
                            caption_708_lines.append(line)
                
                # Limit growth to prevent memory issues
                if len(caption_708_lines) > 1000:
                    # Keep only last 1000 unique lines
                    caption_708_lines = caption_708_lines[-1000:]
                
                # Limit buffer growth
                if len(cc708_buffer) > 50000:
                    stats["_cc708_buffer"] = cc708_buffer[-10000:]
            except Exception as e:
                import traceback
                traceback.print_exc()
        
        # Log NAL type distribution
        nal_type_summary = {k: v for k, v in nal_counts.items() if v > 0}
        if nal_type_summary:

            pass
    def check_stanag_4609_compliance(self) -> Dict:
        """
        Check STANAG 4609 compliance for KLV metadata streams
        
        STANAG 4609 defines requirements for Motion Imagery and Metadata:
        - Asynchronous KLV: Separate PID (stream type 0x06 - Private Data)
        - Synchronous KLV: Embedded in video stream (within PES packets)
        - Must use MISB standards (ST 0601, ST 0102, etc.)
        - Registration descriptor should be present for metadata PIDs
        """
        compliance_results = {
            'compliant': False,
            'klv_detected': False,
            'asynchronous_klv': [],
            'synchronous_klv': [],
            'issues': [],
            'recommendations': []
        }
        
        # Check for asynchronous KLV (separate PIDs)
        for pid, klv_info in self.klv_pids.items():
            klv_detected = True
            compliance_results['klv_detected'] = True
            
            async_klv = {
                'pid': f"0x{pid:04X}",
                'pid_decimal': pid,
                'packet_count': klv_info['count'],
                'sync_type': klv_info['sync_type'],
                'stream_type': f"0x{klv_info['stream_type']:02X}",
                'misb_compliant': False,
                'standards': []
            }
            
            # Check if KLV packets use MISB standards
            for klv_pkt in klv_info['packets'][:10]:  # Check first 10 packets
                if klv_pkt.get('is_misb'):
                    async_klv['misb_compliant'] = True
                    if klv_pkt.get('standard') not in async_klv['standards']:
                        async_klv['standards'].append(klv_pkt.get('standard'))
            
            # Check stream type (should be 0x06 for private data)
            if klv_info['stream_type'] != 0x06:
                compliance_results['issues'].append(
                    f"PID 0x{pid:04X}: KLV metadata uses stream type 0x{klv_info['stream_type']:02X} "
                    f"instead of recommended 0x06 (Private Data)"
                )
            
            # Check for registration descriptor
            pmt_has_registration = False
            for pmt_pid, pmt in self.pmts.items():
                for stream in pmt.get('streams', []):
                    if stream['pid'] == pid:
                        for desc in stream.get('descriptors', []):
                            if desc.get('tag') == 0x05:  # Registration descriptor
                                pmt_has_registration = True
                                break
            
            if not pmt_has_registration:
                compliance_results['recommendations'].append(
                    f"PID 0x{pid:04X}: Consider adding registration descriptor for KLV metadata stream"
                )
            
            compliance_results['asynchronous_klv'].append(async_klv)
        
        # Check for synchronous KLV (embedded in video)
        for pid, klv_packets in self.klv_in_video.items():
            compliance_results['klv_detected'] = True
            
            sync_klv = {
                'video_pid': f"0x{pid:04X}",
                'video_pid_decimal': pid,
                'klv_packet_count': len(klv_packets),
                'sync_type': 'Synchronous (Embedded in Video)',
                'misb_compliant': False,
                'standards': []
            }
            
            # Check if KLV packets use MISB standards
            for klv_pkt in klv_packets[:10]:
                if klv_pkt.get('is_misb'):
                    sync_klv['misb_compliant'] = True
                    if klv_pkt.get('standard') not in sync_klv['standards']:
                        sync_klv['standards'].append(klv_pkt.get('standard'))
            
            compliance_results['synchronous_klv'].append(sync_klv)
            
            # STANAG 4609 recommends asynchronous KLV for easier processing
            compliance_results['recommendations'].append(
                f"Video PID 0x{pid:04X}: Contains synchronous KLV. "
                f"STANAG 4609 recommends using asynchronous KLV (separate PID) for easier processing"
            )
        
        # Overall compliance determination
        has_async_klv = len(compliance_results['asynchronous_klv']) > 0
        has_misb_compliant = any(
            klv.get('misb_compliant') for klv in 
            compliance_results['asynchronous_klv'] + compliance_results['synchronous_klv']
        )
        
        compliance_results['compliant'] = (
            compliance_results['klv_detected'] and 
            has_misb_compliant and
            len(compliance_results['issues']) == 0
        )
        
        if not compliance_results['klv_detected']:
            compliance_results['issues'].append("No KLV metadata detected in transport stream")
        elif not has_misb_compliant:
            compliance_results['issues'].append(
                "KLV metadata detected but does not use MISB standards (ST 0601, ST 0102, etc.)"
            )
        
        if not has_async_klv and compliance_results['klv_detected']:
            compliance_results['recommendations'].append(
                "No asynchronous KLV detected. STANAG 4609 recommends using separate PID for metadata"
            )
        
        return compliance_results

    def report(self) -> Dict[str, object]:
        # If this is an MP4/MOV file, return MP4-specific report
        if self.is_mp4 and self.mp4_analysis:
            return {
                "input": self.path,
                "file_type": self.mp4_analysis.get('file_type', 'MP4/MOV'),
                "format": self.mp4_analysis.get('format', self.file_format),
                "elementary_streams": self.mp4_analysis.get('elementary_streams', {}),
                "video_nal_stats": self.mp4_analysis.get('video_nal_stats', {}),
                "tracks": self.mp4_analysis.get('tracks', []),
                "video_tracks": self.mp4_analysis.get('video_tracks', []),
                # Add TS-compatible empty fields for GUI compatibility
                "total_packets": 0,
                "sync_errors": 0,
                "transport_error_indicators": 0,
                "tei_percent": 0,
                "tei_exceeds_threshold_pct": self.tei_threshold_pct,
                "tei_exceeds": False,
                "null_packets": 0,
                "null_percent": 0,
                "pid_count": 0,
                "continuity_errors_total": 0,
                "continuity_errors_per_pid": {},
                "continuity_by_pid": {},
                "pcr_pids": {},
                "pcr_jitter_issues": {},
                "file_size_bytes": self.file_size,
                "packet_size": 0,
                "approx_duration_s": None,
                "approx_bitrate_bps": None,
                "pat": {},
                "pmts": {},
                "pid_info": {},
                "programs": {},
                "pcr_records": {},
                "pts_records": {},
                "dts_records": {},
                "video_headers": {},
                "video_syntax_errors": {},
                "pat_warnings": [],
                "scte35_messages": [],
                "klv_metadata": {"asynchronous_pids": {}, "synchronous_video_pids": {}, "total_klv_pids": 0, "total_video_with_klv": 0},
                "stanag_4609_compliance": {},
                "misb_telemetry": {},
                "pmt_warnings": [],
                "buffer_analysis": {}
            }
        
        # Extract SCTE-35 messages
        scte35_messages = self.extract_scte35_messages()
        
        # Debug: log caption data before report generation
        if DEBUG:
            for pid, stats in self.video_nal_stats.items():
                caption_count = len(stats.get("caption_lines", []))
                cea708_count = len(stats.get("caption_708_lines", []))
                cc_blocks = len(stats.get("closed_captions", []))
        duration = None
        bitrate_bps = None
        # if we have PCRs across any PID we can approximate duration
        all_pcrs = []
        for pid, recs in self.pcr_records.items():
            if recs:
                all_pcrs.extend([r[1] for r in recs])
        if all_pcrs:
            start = min(all_pcrs)
            end = max(all_pcrs)
            duration = max(0.0, end - start)
            if duration > 0 and self.total_packets > 0:
                bitrate_bps = int(self.total_packets * self.packet_size * 8 / duration)

        # PCR jitter checks: compute per-PID deltas
        pcr_jitter_issues = {}
        for pid, recs in self.pcr_records.items():
            if len(recs) < 2:
                continue
            deltas = []
            last = recs[0][1]
            for _, p in recs[1:]:
                deltas.append(p - last)
                last = p
            # find large jumps or negative deltas
            negatives = sum(1 for d in deltas if d < -0.001)
            large = [d for d in deltas if abs(d) > self.pcr_jitter_sec]
            if negatives or large:
                pcr_jitter_issues[pid] = {"count": len(deltas), "negatives": negatives, "large_jumps": len(large), "max_jump_s": max(abs(d) for d in deltas)}

        null_pct = 100.0 * self.null_packets / max(1, self.total_packets)

        cont_errors = sum(self.continuity_errors.values())

        # TEI (transport error indicator) percent
        tei_pct = 100.0 * self.tei_errors / max(1, self.total_packets)
        tei_exceeds = tei_pct > self.tei_threshold_pct

        # Build comprehensive PID info
        pid_info = {}
        for pid, count in self.pid_counts.items():
            pid_info[pid] = {
                'count': count,
                'type': self.pid_types.get(pid, 'Unknown'),
                'continuity_errors': self.continuity_errors.get(pid, 0)
            }

        # continuity-per-pid: compute PID-specific continuity error rate and flag if above threshold
        continuity_by_pid = {}
        for pid, err in self.continuity_errors.items():
            pid_packets = max(1, self.pid_counts.get(pid, 0))
            pct = 100.0 * err / pid_packets
            continuity_by_pid[pid] = {"errors": int(err), "packets": int(pid_packets), "percent": round(pct, 6), "exceeds": pct > self.cont_threshold_pct}

        report = {
            "input": self.path,
            "total_packets": self.total_packets,
            "sync_errors": self.sync_errors,
            "transport_error_indicators": self.tei_errors,
            "tei_percent": round(tei_pct, 6),
            "tei_exceeds_threshold_pct": self.tei_threshold_pct,
            "tei_exceeds": tei_exceeds,
            "null_packets": self.null_packets,
            "null_percent": round(null_pct, 3),
            "pid_count": len(self.pid_counts),
            "continuity_errors_total": cont_errors,
            "continuity_errors_per_pid": {pid: int(c) for pid, c in self.continuity_errors.items()},
            "continuity_by_pid": continuity_by_pid,
            "pcr_pids": {pid: len(recs) for pid, recs in self.pcr_records.items()},
            "pcr_jitter_issues": pcr_jitter_issues,
            "file_size_bytes": self.file_size,
            "packet_size": self.packet_size,
            "approx_duration_s": duration,
            "approx_bitrate_bps": bitrate_bps,
            # Detailed PAT/PMT/PID information
            "pat": self.pat_info,
            "pmts": self.pmts,
            "pid_info": pid_info,
            # Legacy compatibility
            "programs": self.pat_info.get('programs', {}),
            # Raw data for graphing
            "pcr_records": {pid: recs for pid, recs in self.pcr_records.items()},
            "pts_records": {pid: recs for pid, recs in self.pts_records.items()},
            "dts_records": {pid: recs for pid, recs in self.dts_records.items()},
            # Video header and syntax error information
            "video_headers": {pid: header for pid, header in self.video_headers.items()},
            "video_syntax_errors": {pid: errors for pid, errors in self.video_syntax_errors.items()},
            "pat_warnings": list(set(self.pat_warnings)),  # Deduplicate warnings
            "scte35_messages": scte35_messages,  # SCTE-35 splice info sections
            # H.264 NAL/SEI/CC statistics
            "video_nal_stats": {
                pid: {
                    "nal_counts": dict(stats.get("nal_counts", {})),
                    "sei_messages": stats.get("sei_messages", []),
                    "closed_captions": stats.get("closed_captions", []),
                    "caption_lines": [str(l) for l in stats.get("caption_lines", [])],
                    "caption_708_lines": [l for l in stats.get("caption_708_lines", []) if l],
                    "caption_708_services": stats.get("caption_708_services", []),
                    "cea608_bytes_seen": stats.get("all_cea608_bytes_seen", [])[:100]
                } for pid, stats in self.video_nal_stats.items()
            },
            # Elementary stream statistics
            "elementary_streams": {},
            # KLV metadata information
            "klv_metadata": {
                "asynchronous_pids": {
                    f"0x{pid:04X}": {
                        'pid_decimal': pid,
                        'packet_count': info['count'],
                        'sync_type': info['sync_type'],
                        'stream_type': f"0x{info['stream_type']:02X}",
                        'sample_packets': info['packets'][:5]  # First 5 packets
                    } for pid, info in self.klv_pids.items()
                },
                "synchronous_video_pids": {
                    f"0x{pid:04X}": {
                        'video_pid_decimal': pid,
                        'klv_packet_count': len(packets),
                        'sample_packets': packets[:5]  # First 5 packets
                    } for pid, packets in self.klv_in_video.items()
                },
                "total_klv_pids": len(self.klv_pids),
                "total_video_with_klv": len(self.klv_in_video)
            },
            # STANAG 4609 compliance
            "stanag_4609_compliance": self.check_stanag_4609_compliance()
        }

        # Aggregate MISB ST 0601 telemetry from decoded KLV packets
        telemetry_samples = []
        field_latest: Dict[str, object] = {}
        unknown_tag_counts: Dict[int, int] = {}

        # Collect from asynchronous KLV packets
        for pid, info in self.klv_pids.items():
            for pkt in info.get('packets', []):
                if pkt.get('standard') == 'MISB ST 0601' and pkt.get('decoded'):
                    telemetry_samples.append(pkt['decoded'])
                    for k, v in pkt['decoded'].items():
                        field_latest[k] = v
                    for ut in pkt.get('unknown_tags', []):
                        unknown_tag_counts[ut] = unknown_tag_counts.get(ut, 0) + 1
        # Collect from synchronous (embedded) KLV packets
        for pid, packets in self.klv_in_video.items():
            for pkt in packets:
                if pkt.get('standard') == 'MISB ST 0601' and pkt.get('decoded'):
                    telemetry_samples.append(pkt['decoded'])
                    for k, v in pkt['decoded'].items():
                        field_latest[k] = v
                    for ut in pkt.get('unknown_tags', []):
                        unknown_tag_counts[ut] = unknown_tag_counts.get(ut, 0) + 1

        if telemetry_samples:
            # Build summary with statistics and field history
            field_counts: Dict[str, int] = {}
            field_history: Dict[str, List] = {}  # Track all values per field
            field_stats: Dict[str, Dict] = {}  # min, max, avg per field
            
            for sample in telemetry_samples:
                for k, v in sample.items():
                    field_counts[k] = field_counts.get(k, 0) + 1
                    if k not in field_history:
                        field_history[k] = []
                    field_history[k].append(v)
            
            # Calculate statistics for numeric fields
            for field, values in field_history.items():
                numeric_values = [v for v in values if isinstance(v, (int, float))]
                if numeric_values:
                    field_stats[field] = {
                        'min': min(numeric_values),
                        'max': max(numeric_values),
                        'avg': sum(numeric_values) / len(numeric_values)
                    }
                else:
                    field_stats[field] = {'min': '-', 'max': '-', 'avg': '-'}
            
            # Compute presence map across supported tags
            supported_fields = [name for _, (name, _) in sorted(MISB_ST0601_TAGS.items())]
            presence_map = {name: ('Not Present' if name not in field_counts else 'Present') for name in supported_fields}

            report['misb_telemetry'] = {
                'total_samples': len(telemetry_samples),
                'fields_present': len(field_latest),
                'latest_values': field_latest,
                'field_counts': field_counts,
                'field_stats': field_stats,
                'field_history': field_history,  # All values for packet viewer
                'sample_preview': telemetry_samples[:5],
                'supported_fields': supported_fields,
                'presence_map': presence_map,
                'unknown_tags_seen': unknown_tag_counts
            }
        else:
            report['misb_telemetry'] = {
                'total_samples': 0,
                'fields_present': 0,
                'latest_values': {},
                'field_counts': {},
                'supported_fields': [name for _, (name, _) in sorted(MISB_ST0601_TAGS.items())],
                'presence_map': {name: 'Not Present' for _, (name, _) in sorted(MISB_ST0601_TAGS.items())},
                'unknown_tags_seen': {}
            }

        # Pre-compute total payload bits for scaling ES bitrates against TS bitrate
        total_payload_bits = sum(bytes_ for pid, bytes_ in self.pid_payload_bytes.items() if pid != 0x1FFF) * 8

        # Build elementary stream summaries
        for pid in self.pid_counts.keys():
            # Skip pure control/NULL unless they carry PES counts
            if pid == 0x1FFF:
                continue
            pes_cnt = self.pes_counts.get(pid, 0)
            payload_bytes = self.pid_payload_bytes.get(pid, 0)
            pts_list = [t for _, t in self.pts_records.get(pid, [])]
            dts_list = [t for _, t in self.dts_records.get(pid, [])]
            syntax_errors = []
            if pid in self.video_syntax_errors:
                syntax_errors.extend(self.video_syntax_errors[pid])
            if pid in self.pid_pes_errors:
                syntax_errors.extend(self.pid_pes_errors[pid])
            if pes_cnt or payload_bytes or pts_list or dts_list or syntax_errors:
                # Prefer the longest valid duration (PTS/DTS span or TS duration) to avoid inflated rates when PCR coverage is short
                pid_packets = self.pid_counts.get(pid, 0)
                pid_bits = pid_packets * self.packet_size * 8
                payload_bits = payload_bytes * 8 if payload_bytes else 0

                candidate_durations = []
                if duration and duration > 0:
                    candidate_durations.append(duration)
                if len(pts_list) >= 2:
                    pts_span = pts_list[-1] - pts_list[0]
                    if pts_span > 0:
                        candidate_durations.append(pts_span)
                if len(dts_list) >= 2:
                    dts_span = dts_list[-1] - dts_list[0]
                    if dts_span > 0:
                        candidate_durations.append(dts_span)

                es_duration = max(candidate_durations) if candidate_durations else None

                approx_bitrate = None
                # Keep ES bitrate aligned with the TS observation window; fall back to
                # the best available span when PCR-derived duration is missing.
                duration_for_bitrate = duration if duration and duration > 0 else es_duration
                if duration_for_bitrate:
                    payload_rate = int(payload_bits / duration_for_bitrate) if payload_bits > 0 else None
                    packet_rate = int(pid_bits / duration_for_bitrate) if pid_packets > 0 else None

                    # Scale payload-based rate so that the sum of ES payload rates cannot exceed TS bitrate.
                    scaled_payload_rate = None
                    if bitrate_bps and total_payload_bits > 0 and payload_bits > 0:
                        share = payload_bits / total_payload_bits
                        scaled_payload_rate = int(bitrate_bps * share)

                    if payload_rate is not None:
                        approx_bitrate = payload_rate
                        if scaled_payload_rate is not None and scaled_payload_rate < approx_bitrate:
                            approx_bitrate = scaled_payload_rate
                    elif packet_rate is not None:
                        approx_bitrate = packet_rate
                        # A per-PID bitrate should never exceed the measured TS bitrate.
                        if bitrate_bps and approx_bitrate > bitrate_bps:
                            approx_bitrate = bitrate_bps

                es_entry = {
                    "pid": pid,
                    "type": self.pid_types.get(pid, 'Unknown'),
                    "pes_packets": pes_cnt,
                    "payload_bytes": payload_bytes,
                    "approx_bitrate_bps": approx_bitrate,
                    "pts_first": pts_list[0] if pts_list else None,
                    "pts_last": pts_list[-1] if pts_list else None,
                    "pts_count": len(pts_list),
                    "dts_first": dts_list[0] if dts_list else None,
                    "dts_last": dts_list[-1] if dts_list else None,
                    "dts_count": len(dts_list),
                    "syntax_errors": syntax_errors,
                }
                
                # Add stream type information from PMT
                stream_type = None
                if pid in self.video_pids:
                    stream_type = self.video_pids[pid]
                elif pid in self.audio_pids:
                    stream_type = self.audio_pids[pid]
                
                if stream_type is not None:
                    es_entry["stream_type"] = stream_type
                    es_entry["stream_type_name"] = get_stream_type_name(stream_type)
                    st_name = es_entry.get("stream_type_name", "").lower()
                    # Generic audio bitrate sniff override (AC-3, AAC, MPEG audio)
                    if "audio" in st_name or stream_type in (0x03, 0x04, 0x0F, 0x11, 0x81, 0x84, 0x87, 0xA1):
                        sample = bytes(self.pid_payload_sample.get(pid, b""))
                        sniff = detect_audio_bitrate(sample, stream_type)
                        if sniff:
                            es_entry["approx_bitrate_bps"] = sniff
                
                # Add video header info (H.264 SPS or MPEG-2 sequence header)
                if pid in self.video_headers:
                    header = self.video_headers[pid]
                    if header.get('type') == 'H.264 SPS':
                        es_entry['h264_sps'] = header
                    elif header.get('type') == 'MPEG-2 Sequence Header':
                        es_entry['mpeg2_sequence_header'] = header
                
                report["elementary_streams"][pid] = es_entry
        
        # Note: NAL/SEI per-frame extraction is now done on-demand in GUI
        # to improve performance. Use extract_nal_sei_per_frame(pid) method.
        
        # Add PAT warnings
        report["pat_warnings"] = list(set(self.pat_warnings))  # Deduplicate
        
        # Add PMT warnings  
        report["pmt_warnings"] = list(set(self.pmt_warnings))  # Deduplicate
        
        # Add buffer analysis results
        if self.tstd_analyzer and BUFFER_ANALYSIS_AVAILABLE:
            buffer_stats = self.tstd_analyzer.get_all_stats()
            total_overflows = sum(s['overflows'] for s in buffer_stats.values())
            total_underflows = sum(s['underflows'] for s in buffer_stats.values())
            compliant_pids = sum(1 for s in buffer_stats.values() if s['compliant'])
            report["buffer_analysis"] = {
                "enabled": True,
                "per_pid": buffer_stats,
                "summary": {
                    "total_pids": len(buffer_stats),
                    "compliant_pids": compliant_pids,
                    "total_overflows": total_overflows,
                    "total_underflows": total_underflows,
                    "all_compliant": all(s['compliant'] for s in buffer_stats.values())
                }
            }
        else:
            report["buffer_analysis"] = {"enabled": False, "reason": "buffer_analyzer module not available"}
        
        return report


def main(argv=None):
    p = argparse.ArgumentParser(prog="video_analyzer.py")
    p.add_argument("path", nargs='?', help="Input TS file (omit when using --ndi)")
    p.add_argument("--ndi", action="store_true", help="Use live NDI source instead of a file")
    p.add_argument("--ndi-source", help="NDI source name (optional)")
    p.add_argument("--json", action="store_true", help="Print JSON report")
    p.add_argument("--pcr-jitter-ms", type=float, default=50.0, help="PCR jitter threshold in ms to flag (default 50ms)")
    p.add_argument("--tei-threshold-pct", type=float, default=0.1, help="TEI rate threshold (percent) to flag (default 0.1)")
    p.add_argument("--cont-threshold-pct", type=float, default=0.1, help="Continuity error rate threshold per PID (percent) to flag (default 0.1)")
    p.add_argument("--per-frame", action="store_true", help="Parse H.264 SEI per access unit (frame) for caption extraction")
    args = p.parse_args(argv)

    if args.ndi:
        # Live NDI mode: path can be omitted
        pass
    else:
        if not args.path or not os.path.isfile(args.path):
            return 2

    if args.ndi:
        # Live NDI receive -> hand frames to a simple callback. Full
        # integration with the TS analyzer isn't trivial (analyzer
        # consumes TS/PES data). This provides a useful live hook so
        # callers can extend processing of received frames.
        try:
            from ndi_streamer import NDIReceiver
        except Exception as e:
            print("NDI support is not available:", e, file=sys.stderr)
            return 3

        receiver = NDIReceiver()
        try:
            sources = receiver.list_sources()
        except Exception as e:
            print("Failed to list NDI sources:", e, file=sys.stderr)
            return 4

        src = args.ndi_source if args.ndi_source else (sources[0] if sources else None)
        if not src:
            print("No NDI source available", file=sys.stderr)
            return 5

        def _on_frame(frame):
            try:
                # Minimal default action: print shape/info. Replace or
                # extend this callback to run more detailed analysis.
                if frame is None:
                    return
                try:
                    h = len(frame)
                    w = len(frame[0]) if h else 0
                    print(f"NDI frame received: {h}x{w}")
                except Exception:
                    print("NDI frame received (unknown shape)")
            except Exception:
                pass

        print(f"Starting NDI receive from: {src}")
        receiver.start(source_name=src, frame_callback=_on_frame)
        try:
            while True:
                # Keep running until user interrupts
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("Stopping NDI receiver...")
        finally:
            receiver.stop()
        return 0

    # File analysis mode
    a = TSAnalyser(args.path, pcr_jitter_ms=args.pcr_jitter_ms, tei_threshold_pct=args.tei_threshold_pct, cont_threshold_pct=args.cont_threshold_pct)
    # Enable per-frame parsing mode via instance flag
    setattr(a, "per_frame_mode", bool(args.per_frame))
    a.analyze()
    rpt = a.report()
    if args.json:
        # Produce a JSON-safe, size-limited view for external tools / GUIs.
        def _sanitize(o, _depth=0):
            if _depth > 6:
                return str(o)
            if o is None or isinstance(o, (bool, int, float, str)):
                return o
            if isinstance(o, (bytes, bytearray)):
                try:
                    return o.hex()
                except Exception:
                    return str(o)
            if isinstance(o, dict):
                out = {}
                for k, v in o.items():
                    try:
                        out[str(k)] = _sanitize(v, _depth + 1)
                    except Exception:
                        out[str(k)] = str(v)
                return out
            if isinstance(o, (list, tuple, set)):
                res = []
                for v in list(o)[:200]:
                    res.append(_sanitize(v, _depth + 1))
                return res
            # Fallback
            return str(o)

        minimal = {
            'input': rpt.get('input'),
            'file_type': rpt.get('file_type'),
            'elementary_streams': _sanitize(rpt.get('elementary_streams', {})),
            'video_nal_stats': _sanitize(rpt.get('video_nal_stats', {})),
            'tracks': _sanitize(rpt.get('tracks', {})),
            'video_tracks': _sanitize(rpt.get('video_tracks', [])),
            'approx_duration_s': rpt.get('approx_duration_s'),
            'approx_bitrate_bps': rpt.get('approx_bitrate_bps')
        }
        json.dump(minimal, sys.stdout, indent=2)
        sys.stdout.flush()
        return 0
    else:
        if rpt['continuity_errors_total']:
            for pid, c in rpt['continuity_errors_per_pid'].items():
                pass
        if rpt['pcr_pids']:
            for pid, cnt in rpt['pcr_pids'].items():
                pass
        if rpt['pcr_jitter_issues']:
            for pid, info in rpt['pcr_jitter_issues'].items():
                pass
        if rpt['approx_duration_s']:
            pass
        if rpt['programs']:
            for prog, pid in rpt['programs'].items():
                pass
        if rpt['pmts']:
            for pid, pmt in rpt['pmts'].items():
                for stream in pmt.get('streams', []):
                    stream_type = stream.get('type', 0)
                    stream_pid = stream.get('pid', 0)
                    stream_type_name = stream.get('type_name', 'Unknown')

        
        # Buffer analysis results
        if rpt.get('buffer_analysis', {}).get('enabled') and rpt.get('buffer_analysis', {}).get('summary'):
            buf_summary = rpt['buffer_analysis']['summary']
            if buf_summary.get('pids_with_overflows', 0) > 0:
                pass
            if buf_summary.get('pids_with_underflows', 0) > 0:
            
            # Show details for non-compliant streams
                pass
            per_pid = rpt['buffer_analysis']['per_pid']
            for pid, stats in per_pid.items():
                if not stats['compliant']:
                    if stats['overflows'] > 0:
                        pass
                    if stats['underflows'] > 0:
        

                        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
