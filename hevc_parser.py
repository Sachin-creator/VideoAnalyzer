#!/usr/bin/env python3
"""
HEVC (H.265) Parser for Video Analysis
Supports VPS, SPS, PPS parsing and 4K resolution detection
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class HEVCBitReader:
    """Helper class for reading HEVC bit streams"""
    def __init__(self, data: bytes):
        self.data = data
        self.byte_pos = 0
        self.bit_pos = 0
    
    def read_bits(self, n: int) -> int:
        """Read n bits"""
        result = 0
        for _ in range(n):
            if self.byte_pos >= len(self.data):
                return result
            bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1
            result = (result << 1) | bit
            self.bit_pos += 1
            if self.bit_pos == 8:
                self.bit_pos = 0
                self.byte_pos += 1
        return result
    
    def read_ue(self) -> int:
        """Read unsigned Exp-Golomb coded integer"""
        leading_zeros = 0
        while self.byte_pos < len(self.data) and self.read_bits(1) == 0:
            leading_zeros += 1
        if leading_zeros == 0:
            return 0
        value = self.read_bits(leading_zeros)
        return (1 << leading_zeros) - 1 + value
    
    def read_se(self) -> int:
        """Read signed Exp-Golomb coded integer"""
        ue = self.read_ue()
        if ue % 2 == 0:
            return -(ue // 2)
        return (ue + 1) // 2


# HEVC NAL Unit Types (Table 7-1 in ITU-T H.265)
HEVC_NAL_UNIT_TYPES = {
    0: "TRAIL_N",      # Coded slice segment of a non-TSA, non-STSA trailing picture
    1: "TRAIL_R",      # Coded slice segment of a non-TSA, non-STSA trailing picture
    2: "TSA_N",        # Coded slice segment of a TSA picture
    3: "TSA_R",        # Coded slice segment of a TSA picture
    4: "STSA_N",       # Coded slice segment of an STSA picture
    5: "STSA_R",       # Coded slice segment of an STSA picture
    6: "RADL_N",       # Coded slice segment of a RADL picture
    7: "RADL_R",       # Coded slice segment of a RADL picture
    8: "RASL_N",       # Coded slice segment of a RASL picture
    9: "RASL_R",       # Coded slice segment of a RASL picture
    16: "BLA_W_LP",    # Coded slice segment of a BLA picture
    17: "BLA_W_RADL",  # Coded slice segment of a BLA picture
    18: "BLA_N_LP",    # Coded slice segment of a BLA picture
    19: "IDR_W_RADL",  # Coded slice segment of an IDR picture
    20: "IDR_N_LP",    # Coded slice segment of an IDR picture
    21: "CRA_NUT",     # Coded slice segment of a CRA picture
    32: "VPS_NUT",     # Video parameter set
    33: "SPS_NUT",     # Sequence parameter set
    34: "PPS_NUT",     # Picture parameter set
    35: "AUD_NUT",     # Access unit delimiter
    36: "EOS_NUT",     # End of sequence
    37: "EOB_NUT",     # End of bitstream
    38: "FD_NUT",      # Filler data
    39: "PREFIX_SEI_NUT",  # Supplemental enhancement information
    40: "SUFFIX_SEI_NUT",  # Supplemental enhancement information
}


def remove_hevc_emulation_prevention(data: bytes) -> bytes:
    """Remove emulation prevention bytes (0x000003 -> 0x0000)"""
    out = bytearray()
    i = 0
    while i < len(data):
        if i + 2 < len(data) and data[i] == 0x00 and data[i+1] == 0x00 and data[i+2] == 0x03:
            out.append(0x00)
            out.append(0x00)
            i += 3  # Skip the 0x03
        else:
            out.append(data[i])
            i += 1
    return bytes(out)


def parse_hevc_vps(data: bytes) -> Optional[Dict]:
    """Parse HEVC Video Parameter Set (VPS)"""
    try:
        # Remove start codes if present
        if data.startswith(b'\x00\x00\x01'):
            data = data[3:]
        elif data.startswith(b'\x00\x00\x00\x01'):
            data = data[4:]
        
        # Check NAL unit type (should be 32 for VPS)
        nal_unit_header = (data[0] << 8) | data[1]
        nal_unit_type = (nal_unit_header >> 9) & 0x3F
        
        if nal_unit_type != 32:
            return None
        
        # Remove emulation prevention
        rbsp = remove_hevc_emulation_prevention(data[2:])
        br = HEVCBitReader(rbsp)
        
        result = {'type': 'HEVC VPS'}
        
        # vps_video_parameter_set_id (4 bits)
        vps_id = br.read_bits(4)
        result['vps_id'] = vps_id
        
        # vps_base_layer_internal_flag (1 bit)
        result['vps_base_layer_internal_flag'] = br.read_bits(1)
        
        # vps_base_layer_available_flag (1 bit)
        result['vps_base_layer_available_flag'] = br.read_bits(1)
        
        # vps_max_layers_minus1 (6 bits)
        result['vps_max_layers_minus1'] = br.read_bits(6)
        
        # vps_max_sub_layers_minus1 (3 bits)
        vps_max_sub_layers_minus1 = br.read_bits(3)
        result['vps_max_sub_layers_minus1'] = vps_max_sub_layers_minus1
        
        # vps_temporal_id_nesting_flag (1 bit)
        result['vps_temporal_id_nesting_flag'] = br.read_bits(1)
        
        # vps_reserved_0xffff_16bits (16 bits)
        br.read_bits(16)
        
        return result
    except Exception as e:
        return {'type': 'HEVC VPS', 'error': str(e)}


def parse_hevc_sps(data: bytes) -> Optional[Dict]:
    """Parse HEVC Sequence Parameter Set (SPS) with 4K detection"""
    try:
        # Remove start codes if present
        if data.startswith(b'\x00\x00\x01'):
            data = data[3:]
        elif data.startswith(b'\x00\x00\x00\x01'):
            data = data[4:]
        
        # Check NAL unit type (should be 33 for SPS)
        nal_unit_header = (data[0] << 8) | data[1]
        nal_unit_type = (nal_unit_header >> 9) & 0x3F
        
        if nal_unit_type != 33:
            return None
        
        # Remove emulation prevention
        rbsp = remove_hevc_emulation_prevention(data[2:])
        br = HEVCBitReader(rbsp)
        
        result = {'type': 'HEVC SPS', 'errors': [], 'warnings': []}
        
        # sps_video_parameter_set_id (4 bits)
        sps_vps_id = br.read_bits(4)
        result['sps_vps_id'] = sps_vps_id
        
        # sps_max_sub_layers_minus1 (3 bits)
        sps_max_sub_layers_minus1 = br.read_bits(3)
        result['sps_max_sub_layers_minus1'] = sps_max_sub_layers_minus1
        
        # sps_temporal_id_nesting_flag (1 bit)
        result['sps_temporal_id_nesting_flag'] = br.read_bits(1)
        
        # profile_tier_level()
        # Skip detailed parsing for now, just read essential bits
        # general_profile_space (2 bits)
        br.read_bits(2)
        # general_tier_flag (1 bit)
        br.read_bits(1)
        # general_profile_idc (5 bits)
        profile_idc = br.read_bits(5)
        result['profile_idc'] = profile_idc
        
        # general_profile_compatibility_flag[32]
        for _ in range(32):
            br.read_bits(1)
        
        # general_progressive_source_flag (1 bit)
        result['progressive_source'] = br.read_bits(1)
        
        # general_interlaced_source_flag (1 bit)
        result['interlaced_source'] = br.read_bits(1)
        
        # Skip rest of profile_tier_level for simplicity
        # general_non_packed_constraint_flag, general_frame_only_constraint_flag
        br.read_bits(2)
        # 44 reserved bits
        br.read_bits(44)
        # general_level_idc (8 bits)
        level_idc = br.read_bits(8)
        result['level_idc'] = level_idc
        
        # Skip sub_layer parsing
        for _ in range(sps_max_sub_layers_minus1):
            br.read_bits(2)  # sub_layer_profile_present_flag, sub_layer_level_present_flag
        
        # sps_seq_parameter_set_id
        sps_id = br.read_ue()
        result['sps_id'] = sps_id
        
        # chroma_format_idc
        chroma_format_idc = br.read_ue()
        result['chroma_format_idc'] = chroma_format_idc
        
        if chroma_format_idc == 3:
            # separate_colour_plane_flag
            result['separate_colour_plane_flag'] = br.read_bits(1)
        
        # pic_width_in_luma_samples
        pic_width = br.read_ue()
        result['pic_width'] = pic_width
        
        # pic_height_in_luma_samples
        pic_height = br.read_ue()
        result['pic_height'] = pic_height
        
        # Detect 4K resolutions
        result['is_4k'] = False
        result['resolution_name'] = f"{pic_width}x{pic_height}"
        
        if pic_width == 3840 and pic_height == 2160:
            result['is_4k'] = True
            result['resolution_name'] = "4K UHD (3840x2160)"
        elif pic_width == 4096 and pic_height == 2160:
            result['is_4k'] = True
            result['resolution_name'] = "DCI 4K (4096x2160)"
        elif pic_width >= 3840 or pic_height >= 2160:
            result['is_4k'] = True
            result['resolution_name'] = f"4K+ ({pic_width}x{pic_height})"
        elif pic_width == 1920 and pic_height == 1080:
            result['resolution_name'] = "Full HD (1920x1080)"
        elif pic_width == 1280 and pic_height == 720:
            result['resolution_name'] = "HD (1280x720)"
        
        # conformance_window_flag
        conformance_window_flag = br.read_bits(1)
        if conformance_window_flag:
            result['conf_win_left_offset'] = br.read_ue()
            result['conf_win_right_offset'] = br.read_ue()
            result['conf_win_top_offset'] = br.read_ue()
            result['conf_win_bottom_offset'] = br.read_ue()
        
        # bit_depth_luma_minus8
        bit_depth_luma = br.read_ue() + 8
        result['bit_depth_luma'] = bit_depth_luma
        
        # bit_depth_chroma_minus8
        bit_depth_chroma = br.read_ue() + 8
        result['bit_depth_chroma'] = bit_depth_chroma
        
        # Check for 10-bit encoding
        if bit_depth_luma > 8 or bit_depth_chroma > 8:
            result['is_10bit'] = True
        
        # log2_max_pic_order_cnt_lsb_minus4
        result['log2_max_pic_order_cnt_lsb'] = br.read_ue() + 4
        
        return result
        
    except Exception as e:
        return {'type': 'HEVC SPS', 'error': str(e), 'errors': [str(e)]}


def parse_hevc_pps(data: bytes) -> Optional[Dict]:
    """Parse HEVC Picture Parameter Set (PPS)"""
    try:
        # Remove start codes if present
        if data.startswith(b'\x00\x00\x01'):
            data = data[3:]
        elif data.startswith(b'\x00\x00\x00\x01'):
            data = data[4:]
        
        # Check NAL unit type (should be 34 for PPS)
        nal_unit_header = (data[0] << 8) | data[1]
        nal_unit_type = (nal_unit_header >> 9) & 0x3F
        
        if nal_unit_type != 34:
            return None
        
        # Remove emulation prevention
        rbsp = remove_hevc_emulation_prevention(data[2:])
        br = HEVCBitReader(rbsp)
        
        result = {'type': 'HEVC PPS', 'errors': [], 'warnings': []}
        
        # pps_pic_parameter_set_id
        pps_id = br.read_ue()
        result['pps_id'] = pps_id
        
        # pps_seq_parameter_set_id
        pps_sps_id = br.read_ue()
        result['pps_sps_id'] = pps_sps_id
        
        # dependent_slice_segments_enabled_flag
        result['dependent_slice_segments_enabled'] = br.read_bits(1)
        
        # output_flag_present_flag
        result['output_flag_present'] = br.read_bits(1)
        
        # num_extra_slice_header_bits
        result['num_extra_slice_header_bits'] = br.read_bits(3)
        
        return result
        
    except Exception as e:
        return {'type': 'HEVC PPS', 'error': str(e), 'errors': [str(e)]}


def find_hevc_nal_units(data: bytes, max_units: int = 1000) -> List[Tuple[int, int, bytes]]:
    """
    Find HEVC NAL units in byte stream.
    Returns list of (nal_unit_type, start_pos, nal_data)
    """
    nal_units = []
    i = 0
    
    while i < len(data) - 4 and len(nal_units) < max_units:
        # Look for start code (0x000001 or 0x00000001)
        if data[i] == 0x00 and data[i+1] == 0x00:
            start_code_len = 0
            if data[i+2] == 0x01:
                start_code_len = 3
            elif data[i+2] == 0x00 and i+3 < len(data) and data[i+3] == 0x01:
                start_code_len = 4
            
            if start_code_len > 0:
                nal_start = i + start_code_len
                if nal_start + 2 <= len(data):
                    # Parse NAL unit header (2 bytes)
                    nal_unit_header = (data[nal_start] << 8) | data[nal_start + 1]
                    nal_unit_type = (nal_unit_header >> 9) & 0x3F
                    
                    # Find next start code
                    j = nal_start + 2
                    while j < len(data) - 3:
                        if data[j] == 0x00 and data[j+1] == 0x00 and (data[j+2] == 0x01 or (data[j+2] == 0x00 and j+3 < len(data) and data[j+3] == 0x01)):
                            break
                        j += 1
                    
                    nal_data = data[nal_start:j]
                    nal_units.append((nal_unit_type, nal_start, nal_data))
                    i = j
                    continue
        
        i += 1
    
    return nal_units
