#!/usr/bin/env python3
"""
HRD/T-STD Buffer Analysis Module

Analyzes MPEG Transport Stream buffer compliance according to:
- ISO/IEC 13818-1 (MPEG-2 Systems) T-STD model (3-stage buffer)
- ISO/IEC 14496-10 (H.264/AVC) HRD model
- ITU-T H.222.0

Implements the complete T-STD three-stage buffer model:
1. Transport Buffer (TB) - 512 bytes, receives TS packets
2. Multiplex Buffer (MBn) - Demux/PES buffering, variable size
3. Elementary Buffer (EBn) - Decoder buffer, codec-specific size

This module simulates decoder buffer behavior to detect:
- Buffer overflows (encoder sending data too fast)
- Buffer underflows (encoder sending data too slow)
- Compliance with standard buffer models
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque


class ThreeStageBufferAnalyzer:
    """
    ISO/IEC 13818-1 T-STD Three-Stage Buffer Model
    
    Implements:
    - Transport Buffer (TB): 512 bytes
    - Multiplex Buffer (MBn): Variable, based on stream type
    - Elementary Buffer (EBn): Variable, based on codec
    """
    
    def __init__(self, pid: int, stream_type: int, eb_size: int = None):
        """
        Initialize 3-stage T-STD buffer analyzer
        
        Args:
            pid: PID being analyzed
            stream_type: MPEG stream type (0x1B for H.264, 0x02 for MPEG-2, etc.)
            eb_size: Elementary buffer size override (auto-calculated if None)
        """
        self.pid = pid
        self.stream_type = stream_type
        
        # === Stage 1: Transport Buffer (TB) ===
        # Fixed 512 bytes per ISO/IEC 13818-1
        self.tb_size = 512
        self.tb_level = 0
        self.tb_max = 0
        self.tb_overflows = 0
        
        # === Stage 2: Multiplex Buffer (MBn) ===
        # Size depends on stream type (RBn × BS)
        self.mb_size = self._calculate_mb_size(stream_type)
        self.mb_level = 0
        self.mb_max = 0
        self.mb_overflows = 0
        self.mb_underflows = 0
        
        # PES packet queue (accumulated before transfer to EB)
        self.pes_queue: deque = deque()  # [(arrival_time, pes_size)]
        
        # === Stage 3: Elementary Buffer (EBn) ===
        # Size depends on codec and profile
        if eb_size is None:
            eb_size = self._calculate_eb_size(stream_type)
        self.eb_size = eb_size
        self.eb_level = 0
        self.eb_max = 0
        self.eb_min = eb_size
        self.eb_overflows = 0
        self.eb_underflows = 0
        
        # Decode/removal tracking
        self.last_decode_time = None
        self.decode_rate = 0  # bytes per second (based on bitrate)
        
        # Set default decode rate for PCM (uncompressed audio)
        # PCM bitrate = sample_rate × bit_depth × channels / 8
        # Default: 48kHz × 16-bit × 2-channel = 192 KB/s
        # For 8-channel: 48kHz × 24-bit × 8-channel = 1152 KB/s
        if stream_type == 0x80:  # PCM
            # Assume 48kHz, 16-bit, stereo as default (can be updated later)
            default_pcm_bitrate = 48000 * 16 * 2  # bits per second
            self.decode_rate = default_pcm_bitrate / 8  # bytes per second
        
        # Overall statistics
        self.total_overflows = 0
        self.total_underflows = 0
        self.history: List[Tuple[float, Dict]] = []  # (time, {tb, mb, eb levels})
        self.last_time = 0.0
        self.last_pts = None
        self.last_dts = None
    
    def _calculate_mb_size(self, stream_type: int) -> int:
        """
        Calculate Multiplex Buffer size (MBn) per ISO/IEC 13818-1
        
        MBn = RBn × BS where:
        - BS = 128 bytes for video, 32 bytes for audio
        - RBn depends on stream type
        """
        if stream_type in [0x01, 0x02]:  # MPEG-1/2 Video
            # Video: RBn typically 400-1250
            return 400 * 128  # 51,200 bytes (400 × BS)
        elif stream_type == 0x1B:  # H.264/AVC
            # H.264: Larger MB for HD streams
            return 1250 * 128  # 160,000 bytes
        elif stream_type == 0x24:  # H.265/HEVC
            # H.265: Similar to H.264
            return 1250 * 128  # 160,000 bytes
        elif stream_type in [0x03, 0x04]:  # MPEG Audio
            # Audio: RBn typically 35
            return 35 * 32  # 1,120 bytes (35 × 32)
        elif stream_type == 0x0F:  # AAC
            # AAC: Similar to MPEG audio
            return 35 * 32  # 1,120 bytes
        elif stream_type in [0x81, 0x87]:  # AC-3, E-AC-3
            # AC-3: Larger for surround
            return 50 * 32  # 1,600 bytes
        elif stream_type == 0x80:  # PCM
            # PCM: Uncompressed audio, needs larger MB for high sample rates
            # 48kHz 16-bit stereo = 192 KB/s, needs buffering for PES assembly
            return 60 * 32  # 1,920 bytes (60 × 32)
        else:
            return 200 * 128  # Default: 25,600 bytes
    
    def _calculate_eb_size(self, stream_type: int) -> int:
        """
        Calculate Elementary Buffer size (EBn) per ISO/IEC 13818-1
        
        For video: BSn (decoder buffer size bound from sequence header)
        For audio: Codec-specific
        """
        if stream_type == 0x1B:  # H.264/AVC
            # H.264: CPB size, typically 10 MB for HD
            return 10 * 1024 * 1024  # 10 MB
        elif stream_type == 0x24:  # H.265/HEVC
            # H.265: Similar to H.264
            return 10 * 1024 * 1024  # 10 MB
        elif stream_type in [0x01, 0x02]:  # MPEG-1/2 Video
            # MPEG-2: vbv_buffer_size, typically 1.75-9.78 Mbit
            return int(1.75 * 1024 * 1024 / 8)  # 1.75 Mbit = 224 KB (minimum)
        elif stream_type in [0x03, 0x04]:  # MPEG Audio
            return 4 * 1024  # 4 KB
        elif stream_type == 0x0F:  # AAC
            return 6 * 1024  # 6 KB (for AAC-LC)
        elif stream_type == 0x81:  # AC-3
            return 6 * 1024  # 6 KB
        elif stream_type == 0x87:  # E-AC-3
            return 8 * 1024  # 8 KB
        elif stream_type == 0x80:  # PCM
            # PCM: Uncompressed audio
            # 48kHz 16-bit stereo = 192 KB/s × ~100ms buffering = ~20 KB
            # 48kHz 24-bit 8-channel = 1152 KB/s × ~100ms = ~115 KB
            return 32 * 1024  # 32 KB (supports up to 8-channel 24-bit)
        else:
            return 2 * 1024 * 1024  # Default: 2 MB
    
    def process_ts_packet(self, time: float, packet_size: int, is_pusi: bool = False, 
                          pts: float = None, dts: float = None):
        """
        Process incoming TS packet through 3-stage buffer model
        
        Args:
            time: Arrival time in seconds
            packet_size: TS packet size in bytes (usually 188)
            is_pusi: Payload Unit Start Indicator (PES packet start)
            pts: Presentation timestamp (seconds)
            dts: Decode timestamp (seconds)
        """
        # Update decode time references
        if pts is not None:
            self.last_pts = pts
        if dts is not None:
            self.last_dts = dts
        
        # === Stage 1: Transport Buffer (TB) ===
        # TS packets arrive instantaneously at transport rate
        # TB is always 512 bytes, empties instantaneously to MB
        self.tb_level = packet_size  # Instantaneous
        if self.tb_level > self.tb_size:
            self.tb_overflows += 1
            self.total_overflows += 1
        self.tb_max = max(self.tb_max, self.tb_level)
        
        # Instantaneous transfer TB → MB
        payload_size = packet_size - 4  # Remove TS header (4 bytes minimum)
        
        # === Stage 2: Multiplex Buffer (MBn) ===
        # Accumulates PES packets, transfers complete PES to EB
        self.mb_level += payload_size
        
        if self.mb_level > self.mb_size:
            self.mb_overflows += 1
            self.total_overflows += 1
            self.mb_level = self.mb_size  # Clamp
        
        self.mb_max = max(self.mb_max, self.mb_level)
        
        # If PUSI, previous PES packet is complete, transfer to EB
        if is_pusi and self.mb_level > 0:
            self._transfer_mb_to_eb(time)
        
        # === Stage 3: Elementary Buffer (EBn) ===
        # Continuous removal at decode rate based on DTS/PTS
        self._decode_from_eb(time)
        
        # Record history snapshot with TB FILLED (before clearing)
        # Sample every 1ms to capture dynamics while limiting memory
        # Compare against actual packet time, not the TB=0 synthetic time
        last_packet_time = self.history[-2][0] if len(self.history) >= 2 else (self.history[0][0] if self.history else 0)
        should_record = (not self.history or 
                        time - last_packet_time >= 0.001)
        
        if should_record:
            # Record TB filled state (188 bytes)
            self.history.append((time, {
                'tb': self.tb_level,  # Captured while still filled (188)
                'mb': self.mb_level,
                'eb': self.eb_level,
                'total': self.tb_level + self.mb_level + self.eb_level
            }))
            
            # Record TB EMPTY state immediately after (shows oscillation)
            # This creates the sawtooth: 188 → 0 → 188 → 0
            self.history.append((time + 0.000001, {  # 1 microsecond later
                'tb': 0,  # TB emptied
                'mb': self.mb_level,
                'eb': self.eb_level,
                'total': self.mb_level + self.eb_level
            }))
        
        # TB empties after transfer (instantaneous)
        self.tb_level = 0
        self.last_time = time
    
    def _transfer_mb_to_eb(self, time: float):
        """Transfer complete PES packet from MB to EB"""
        transfer_size = self.mb_level
        
        # Add to Elementary Buffer
        self.eb_level += transfer_size
        
        if self.eb_level > self.eb_size:
            self.eb_overflows += 1
            self.total_overflows += 1
            self.eb_level = self.eb_size  # Clamp
        
        self.eb_max = max(self.eb_max, self.eb_level)
        
        # Empty MB (creates the drop in sawtooth)
        mb_before_clear = self.mb_level
        self.mb_level = 0
        
        # Record MB clearing to show sawtooth drop
        # Add snapshot showing MB = 0 after PUSI transfer
        if self.history:
            self.history.append((time + 0.000002, {  # 2 microseconds after TB clear
                'tb': 0,  # TB already cleared
                'mb': 0,  # MB now empty (sawtooth drop)
                'eb': self.eb_level,
                'total': self.eb_level
            }))
    
    def _decode_from_eb(self, current_time: float):
        """
        Remove decoded data from Elementary Buffer
        
        Decoding happens continuously at decode_rate (based on stream bitrate)
        or at presentation times (PTS/DTS)
        """
        if self.last_decode_time is None:
            self.last_decode_time = current_time
            return
        
        elapsed = current_time - self.last_decode_time
        
        if self.decode_rate > 0 and elapsed > 0:
            # Continuous decode at constant rate
            decoded_bytes = int(self.decode_rate * elapsed)
            
            self.eb_level -= decoded_bytes
            
            if self.eb_level < 0:
                self.eb_underflows += 1
                self.total_underflows += 1
                self.eb_level = 0  # Clamp
            
            self.eb_min = min(self.eb_min, self.eb_level)
        
        self.last_decode_time = current_time
    
    def set_decode_rate(self, bitrate: int):
        """
        Set Elementary Buffer decode rate based on stream bitrate
        
        Args:
            bitrate: Stream bitrate in bits per second
        """
        self.decode_rate = bitrate / 8  # Convert to bytes per second
        
        # For PCM, also adjust buffer sizes based on bitrate
        # Higher bitrates (8-channel) need larger buffers
        if self.stream_type == 0x80:  # PCM
            # Assume bitrate = sample_rate × bit_depth × channels
            # Estimate channels from bitrate (assumes 48kHz 16-bit baseline)
            baseline_stereo = 48000 * 16 * 2  # 1536000 bps
            ratio = bitrate / baseline_stereo
            
            if ratio > 3:  # 6-channel or more
                # Increase MB and EB for multi-channel
                self.mb_size = int(120 * 32)  # 3840 bytes (doubled for 6-8 channel)
                self.eb_size = int(64 * 1024)  # 64 KB for multi-channel
    
    def get_statistics(self) -> Dict:
        """Get comprehensive 3-stage buffer statistics"""
        # Calculate utilizations for each stage
        # Note: TB utilization is not meaningful (instantaneous, always ~188 bytes)
        # So we report it as percentage of packet arrivals that filled it
        tb_util = (self.tb_max / self.tb_size * 100) if self.tb_size > 0 else 0
        mb_util = (self.mb_max / self.mb_size * 100) if self.mb_size > 0 else 0
        eb_util = (self.eb_max / self.eb_size * 100) if self.eb_size > 0 else 0
        
        # Overall utilization (based on EB as it's the main buffer)
        overall_util = eb_util
        
        # Map stream type to string
        stream_type_names = {
            0x1B: 'H.264',
            0x24: 'H.265',
            0x01: 'MPEG-1 Video',
            0x02: 'MPEG-2 Video',
            0x03: 'MPEG Audio',
            0x04: 'MPEG Audio',
            0x0F: 'AAC',
            0x81: 'AC-3',
            0x87: 'E-AC-3',
            0x80: 'PCM'
        }
        stream_type_str = stream_type_names.get(self.stream_type, f'Type 0x{self.stream_type:02X}')
        
        # Convert history to GUI-compatible format
        # For GUI graph, show EB level (main buffer of interest)
        # But also preserve full history for 3-stage plots
        history_dicts = [{'time': t, 'level': levels['eb'], 
                         'tb': levels['tb'], 'mb': levels['mb'], 'eb': levels['eb']} 
                        for t, levels in self.history]
        
        return {
            'pid': self.pid,
            'stream_type': stream_type_str,
            
            # Overall/combined stats (for backward compatibility)
            'buffer_size': self.eb_size,  # Report EB size as primary
            'buffer_size_bytes': self.eb_size,
            'max_level': self.eb_max,
            'min_level': self.eb_min if self.eb_min != self.eb_size else 0,
            'current_level': self.eb_level,
            'max_utilization_percent': round(overall_util, 2),
            'overflows': self.total_overflows,
            'underflows': self.total_underflows,
            'compliant': self.total_overflows == 0 and self.total_underflows == 0,
            'history': history_dicts,
            
            # Detailed 3-stage breakdown
            'transport_buffer': {
                'size': self.tb_size,
                'max_level': self.tb_max,
                'utilization_percent': round(tb_util, 2),
                'overflows': self.tb_overflows,
                'note': 'TB is instantaneous (fills/empties per packet)'
            },
            'multiplex_buffer': {
                'size': self.mb_size,
                'max_level': self.mb_max,
                'current_level': self.mb_level,
                'utilization_percent': round(mb_util, 2),
                'overflows': self.mb_overflows,
                'underflows': self.mb_underflows,
                'note': 'MB accumulates until PES boundary (PUSI)'
            },
            'elementary_buffer': {
                'size': self.eb_size,
                'max_level': self.eb_max,
                'min_level': self.eb_min if self.eb_min != self.eb_size else 0,
                'current_level': self.eb_level,
                'utilization_percent': round(eb_util, 2),
                'overflows': self.eb_overflows,
                'underflows': self.eb_underflows,
                'note': 'EB continuously decodes at bitrate'
            }
        }


class T_STD_Analyzer:
    """
    Transport Stream System Target Decoder (T-STD) Analyzer
    Implements ISO/IEC 13818-1 three-stage buffer model
    """
    
    def __init__(self):
        """Initialize T-STD analyzer"""
        self.analyzers: Dict[int, ThreeStageBufferAnalyzer] = {}
        self.pcr_times: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    
    def add_pid(self, pid: int, stream_type: int, eb_size: int = None) -> ThreeStageBufferAnalyzer:
        """
        Add a PID for 3-stage T-STD buffer analysis
        
        Args:
            pid: PID to analyze
            stream_type: MPEG stream type
            eb_size: Optional Elementary Buffer size override
            
        Returns:
            ThreeStageBufferAnalyzer instance for this PID
        """
        if pid not in self.analyzers:
            self.analyzers[pid] = ThreeStageBufferAnalyzer(pid, stream_type, eb_size)
        return self.analyzers[pid]
    
    def add_pid_buffer(self, pid: int, stream_type_name: str, eb_size: int = None) -> ThreeStageBufferAnalyzer:
        """
        Add a PID for buffer analysis (alternate interface with stream type name)
        
        Args:
            pid: PID to analyze
            stream_type_name: Stream type name string (e.g., "H.264", "AAC")
            eb_size: Optional Elementary Buffer size override
            
        Returns:
            ThreeStageBufferAnalyzer instance for this PID
        """
        # Map stream type names to numeric codes
        stream_type_map = {
            'H.264': 0x1B,
            'AVC': 0x1B,
            'H.265': 0x24,
            'HEVC': 0x24,
            'MPEG-2 Video': 0x02,
            'MPEG-1 Video': 0x01,
            'AAC': 0x0F,
            'AC-3': 0x81,
            'E-AC-3': 0x87,
            'MPEG Audio': 0x03,
            'PCM': 0x80,
        }
        
        stream_type = stream_type_map.get(stream_type_name, 0x00)
        return self.add_pid(pid, stream_type, eb_size)
    
    def process_packet(self, pid: int, packet_bits: int, pcr_time: float = None, 
                      pts_time: float = None, dts_time: float = None, pusi: int = 0):
        """
        Process a TS packet through 3-stage T-STD model
        
        Args:
            pid: PID of packet
            packet_bits: Size of packet in bits (typically 188*8 = 1504)
            pcr_time: PCR time in seconds (optional)
            pts_time: PTS time in seconds (optional)
            dts_time: DTS time in seconds (optional)
            pusi: Payload unit start indicator (1 = PES start)
        """
        if pid in self.analyzers:
            # Use PCR time if available, otherwise PTS, DTS, or last known time
            time = pcr_time if pcr_time is not None else (pts_time if pts_time is not None else 
                  (dts_time if dts_time is not None else self.analyzers[pid].last_time))
            
            packet_bytes = packet_bits // 8  # Usually 188 bytes
            is_pusi = (pusi == 1)
            
            self.analyzers[pid].process_ts_packet(time, packet_bytes, is_pusi, pts_time, dts_time)
    
    def update_bitrate(self, pid: int, bitrate: int):
        """
        Update Elementary Buffer decode rate for a PID
        
        Args:
            pid: PID
            bitrate: Stream bitrate in bits per second
        """
        if pid in self.analyzers:
            self.analyzers[pid].set_decode_rate(bitrate)
    
    def get_report(self) -> Dict:
        """
        Generate comprehensive buffer analysis report
        
        Returns:
            Dictionary with summary and per-PID statistics
        """
        per_pid = {}
        total_overflows = 0
        total_underflows = 0
        compliant_count = 0
        
        pids_with_overflows = 0
        pids_with_underflows = 0
        
        for pid, analyzer in self.analyzers.items():
            stats = analyzer.get_statistics()
            per_pid[pid] = stats
            
            total_overflows += stats['overflows']
            total_underflows += stats['underflows']
            
            if stats['overflows'] > 0:
                pids_with_overflows += 1
            if stats['underflows'] > 0:
                pids_with_underflows += 1
            
            if stats['compliant']:
                compliant_count += 1
        
        total_pids = len(self.analyzers)
        
        return {
            'summary': {
                'total_pids_analyzed': total_pids,
                'compliant_pids': compliant_count,
                'total_overflows': total_overflows,
                'total_underflows': total_underflows,
                'pids_with_overflows': pids_with_overflows,
                'pids_with_underflows': pids_with_underflows,
                'all_compliant': total_overflows == 0 and total_underflows == 0
            },
            'per_pid': per_pid
        }
    
    def get_all_stats(self) -> Dict:
        """Get per-PID statistics dictionary (backward compatibility)"""
        return self.get_report()['per_pid']
    
    def get_pid_analyzer(self, pid: int) -> Optional[ThreeStageBufferAnalyzer]:
        """Get 3-stage buffer analyzer for specific PID"""
        return self.analyzers.get(pid)


# Backward compatibility alias
BufferAnalyzer = ThreeStageBufferAnalyzer
