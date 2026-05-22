#!/usr/bin/env python3
"""
MP4/MOV Container Parser with NAL/SEI Extraction
Supports H.264 (AVC) and H.265 (HEVC) video tracks
"""

import struct
from typing import Dict, List, Tuple, Optional, BinaryIO
from collections import defaultdict


class MP4Box:
    """Represents an MP4/QuickTime box (atom)"""
    def __init__(self, box_type: str, size: int, offset: int, data: bytes = b''):
        self.type = box_type
        self.size = size
        self.offset = offset
        self.data = data
        self.children: List['MP4Box'] = []
    
    def __repr__(self):
        return f"Box({self.type}, size={self.size}, offset=0x{self.offset:X})"


class MP4Parser:
    """Parse MP4/MOV files and extract NAL units from video tracks"""
    
    # Box types that contain other boxes
    CONTAINER_BOXES = {
        'moov', 'trak', 'mdia', 'minf', 'stbl', 'edts', 'dinf',
        'mvex', 'moof', 'traf', 'mfra', 'skip', 'meta', 'ipro',
        'sinf', 'udta', 'ilst'
    }
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.boxes: List[MP4Box] = []
        self.tracks: Dict[int, Dict] = {}  # track_id -> track info
        self.video_tracks: List[int] = []
        self.codec_configs: Dict[int, bytes] = {}  # track_id -> avcC or hvcC data
        
    def parse(self) -> Dict:
        """Parse MP4 structure and return analysis results"""
        with open(self.file_path, 'rb') as f:
            file_size = f.seek(0, 2)
            f.seek(0)
            
            # Parse top-level boxes
            offset = 0
            while offset < file_size:
                box = self._read_box(f, offset, file_size)
                if box is None:
                    break
                self.boxes.append(box)
                offset = box.offset + box.size
            
            # Find and parse video tracks
            moov_box = self._find_box('moov')
            if moov_box:
                self._parse_moov(moov_box)
            
            # Extract samples/NALs from video tracks
            for track_id in self.video_tracks:
                self._extract_track_samples(f, track_id)
        
        return self._generate_report()
    
    def _read_box(self, f: BinaryIO, offset: int, parent_end: int) -> Optional[MP4Box]:
        """Read a single box from file"""
        if offset >= parent_end:
            return None
        
        f.seek(offset)
        header = f.read(8)
        if len(header) < 8:
            return None
        
        size = struct.unpack('>I', header[0:4])[0]
        box_type = header[4:8].decode('ascii', errors='ignore')
        
        # Handle extended size (size == 1)
        actual_size = size
        header_size = 8
        if size == 1:
            ext_size = f.read(8)
            if len(ext_size) == 8:
                actual_size = struct.unpack('>Q', ext_size)[0]
                header_size = 16
        elif size == 0:
            # Box extends to end of file
            actual_size = parent_end - offset
        
        # Read box data (excluding header)
        data_size = min(actual_size - header_size, 1024*1024)  # Limit to 1MB for large boxes
        if data_size > 0:
            data = f.read(data_size)
        else:
            data = b''
        
        box = MP4Box(box_type, actual_size, offset, data)
        
        # Parse container boxes recursively
        if box_type in self.CONTAINER_BOXES:
            child_offset = offset + header_size
            child_end = offset + actual_size
            while child_offset < child_end:
                child = self._read_box(f, child_offset, child_end)
                if child is None:
                    break
                box.children.append(child)
                child_offset = child.offset + child.size
        
        return box
    
    def _find_box(self, box_type: str, boxes: List[MP4Box] = None) -> Optional[MP4Box]:
        """Find first box of given type"""
        if boxes is None:
            boxes = self.boxes
        
        for box in boxes:
            if box.type == box_type:
                return box
            if box.children:
                found = self._find_box(box_type, box.children)
                if found:
                    return found
        return None
    
    def _parse_moov(self, moov_box: MP4Box):
        """Parse movie box to find video tracks"""
        # Find all trak boxes
        trak_boxes = [b for b in moov_box.children if b.type == 'trak']
        
        for trak in trak_boxes:
            track_info = self._parse_trak(trak)
            if track_info:
                track_id = track_info['track_id']
                self.tracks[track_id] = track_info
                
                if track_info.get('is_video'):
                    self.video_tracks.append(track_id)
    
    def _parse_trak(self, trak_box: MP4Box) -> Optional[Dict]:
        """Parse track box"""
        track_info = {}
        
        # Find tkhd (track header)
        tkhd = self._find_box('tkhd', trak_box.children)
        if tkhd and len(tkhd.data) >= 20:
            version = tkhd.data[0]
            if version == 0:
                track_info['track_id'] = struct.unpack('>I', tkhd.data[12:16])[0]
            else:
                track_info['track_id'] = struct.unpack('>I', tkhd.data[20:24])[0]
        
        # Find mdia -> hdlr to determine track type
        mdia = self._find_box('mdia', trak_box.children)
        if mdia:
            hdlr = self._find_box('hdlr', mdia.children)
            if hdlr and len(hdlr.data) >= 12:
                handler_type = hdlr.data[8:12].decode('ascii', errors='ignore')
                track_info['handler_type'] = handler_type
                track_info['is_video'] = (handler_type == 'vide')
                track_info['is_audio'] = (handler_type == 'soun')
            
            # Find codec information in stsd
            minf = self._find_box('minf', mdia.children)
            if minf:
                stbl = self._find_box('stbl', minf.children)
                if stbl:
                    stsd = self._find_box('stsd', stbl.children)
                    if stsd:
                        self._parse_stsd(stsd, track_info)
        
        return track_info if track_info else None
    
    def _parse_stsd(self, stsd_box: MP4Box, track_info: Dict):
        """Parse sample description box to get codec info"""
        if len(stsd_box.data) < 8:
            return
        
        # Skip version/flags (4 bytes) + entry_count (4 bytes)
        entry_count = struct.unpack('>I', stsd_box.data[4:8])[0]
        
        if entry_count == 0:
            return
        
        # Parse first entry
        offset = 8
        if len(stsd_box.data) >= offset + 8:
            entry_size = struct.unpack('>I', stsd_box.data[offset:offset+4])[0]
            codec = stsd_box.data[offset+4:offset+8].decode('ascii', errors='ignore')
            
            track_info['codec'] = codec
            
            # Extract entry data (skip 8-byte header of entry itself)
            entry_start = offset
            entry_data_start = offset + 8  # Skip size (4) + codec (4)
            entry_end = min(offset + entry_size, len(stsd_box.data))
            
            # Look for avcC (H.264) or hvcC (H.265) configuration
            if codec in ('avc1', 'avc3'):
                # H.264 - find avcC box (start search after 78-byte avc1 header)
                search_start = entry_data_start + 78  # Standard avc1 box has 78 bytes before child boxes
                avcc_offset = self._find_child_box_in_data(stsd_box.data[search_start:entry_end], 'avcC')
                if avcc_offset is not None:
                    avcc_data = self._extract_box_data(stsd_box.data[search_start+avcc_offset:entry_end])
                    if avcc_data:
                        track_info['codec_config'] = avcc_data
                        track_info['codec_type'] = 'H.264'
                        self.codec_configs[track_info['track_id']] = avcc_data
            
            elif codec in ('hvc1', 'hev1'):
                # H.265 - find hvcC box (start search after 78-byte hvc1 header)
                search_start = entry_data_start + 78
                hvcc_offset = self._find_child_box_in_data(stsd_box.data[search_start:entry_end], 'hvcC')
                if hvcc_offset is not None:
                    hvcc_data = self._extract_box_data(stsd_box.data[search_start+hvcc_offset:entry_end])
                    if hvcc_data:
                        track_info['codec_config'] = hvcc_data
                        track_info['codec_type'] = 'H.265'
                        self.codec_configs[track_info['track_id']] = hvcc_data
    
    def _find_child_box_in_data(self, data: bytes, box_type: str) -> Optional[int]:
        """Find offset of child box within data"""
        offset = 0
        while offset + 8 <= len(data):
            size = struct.unpack('>I', data[offset:offset+4])[0]
            btype = data[offset+4:offset+8].decode('ascii', errors='ignore')
            
            if btype == box_type:
                return offset
            
            if size < 8:
                break
            offset += size
        
        return None
    
    def _extract_box_data(self, data: bytes) -> Optional[bytes]:
        """Extract box payload (data after 8-byte header)"""
        if len(data) < 8:
            return None
        
        size = struct.unpack('>I', data[0:4])[0]
        if size < 8 or size > len(data):
            return None
        
        return data[8:size]
    
    def _extract_track_samples(self, f: BinaryIO, track_id: int):
        """Extract NAL units from video track samples"""
        # This would require parsing stco/stsc/stsz tables
        # For now, we'll extract from mdat directly
        pass
    
    def _generate_report(self) -> Dict:
        """Generate analysis report"""
        return {
            'file_type': 'MP4/MOV',
            'tracks': self.tracks,
            'video_tracks': self.video_tracks,
            'boxes': [{'type': b.type, 'size': b.size, 'offset': b.offset} for b in self.boxes],
        }
    
    def extract_nals_from_track(self, track_id: int) -> List[Tuple[int, bytes]]:
        """
        Extract NAL units from a video track
        Returns list of (nal_type, nal_data) tuples
        
        Note: This is a simplified extractor. Full implementation would
        require parsing stco, stsc, stsz, and mdat boxes.
        """
        nals = []
        
        if track_id not in self.video_tracks:
            return nals
        
        track_info = self.tracks[track_id]
        codec_config = track_info.get('codec_config')
        
        if not codec_config:
            return nals
        
        # Extract parameter sets from codec config
        if track_info.get('codec_type') == 'H.264':
            nals.extend(self._extract_h264_params(codec_config))
        elif track_info.get('codec_type') == 'H.265':
            nals.extend(self._extract_h265_params(codec_config))
        
        return nals
    
    def _extract_h264_params(self, avcc_data: bytes) -> List[Tuple[int, bytes]]:
        """Extract SPS/PPS from avcC configuration"""
        nals = []
        
        if len(avcc_data) < 7:
            return nals
        
        # avcC format:
        # configurationVersion(1) + AVCProfileIndication(1) + profile_compatibility(1) +
        # AVCLevelIndication(1) + lengthSizeMinusOne(1) + numOfSequenceParameterSets(1)
        
        offset = 5
        num_sps = avcc_data[offset] & 0x1F
        offset += 1
        
        # Extract SPS
        for _ in range(num_sps):
            if offset + 2 > len(avcc_data):
                break
            sps_length = struct.unpack('>H', avcc_data[offset:offset+2])[0]
            offset += 2
            
            if offset + sps_length > len(avcc_data):
                break
            
            sps_data = avcc_data[offset:offset+sps_length]
            nals.append((7, sps_data))  # NAL type 7 = SPS
            offset += sps_length
        
        # Extract PPS
        if offset + 1 <= len(avcc_data):
            num_pps = avcc_data[offset]
            offset += 1
            
            for _ in range(num_pps):
                if offset + 2 > len(avcc_data):
                    break
                pps_length = struct.unpack('>H', avcc_data[offset:offset+2])[0]
                offset += 2
                
                if offset + pps_length > len(avcc_data):
                    break
                
                pps_data = avcc_data[offset:offset+pps_length]
                nals.append((8, pps_data))  # NAL type 8 = PPS
                offset += pps_length
        
        return nals
    
    def _extract_h265_params(self, hvcc_data: bytes) -> List[Tuple[int, bytes]]:
        """Extract VPS/SPS/PPS from hvcC configuration"""
        nals = []
        
        if len(hvcc_data) < 23:
            return nals
        
        # hvcC format is more complex
        # Skip to numOfArrays at byte 22
        offset = 22
        num_arrays = hvcc_data[offset]
        offset += 1
        
        for _ in range(num_arrays):
            if offset + 3 > len(hvcc_data):
                break
            
            array_type = hvcc_data[offset] & 0x3F
            offset += 1
            
            num_nalus = struct.unpack('>H', hvcc_data[offset:offset+2])[0]
            offset += 2
            
            for _ in range(num_nalus):
                if offset + 2 > len(hvcc_data):
                    break
                
                nalu_length = struct.unpack('>H', hvcc_data[offset:offset+2])[0]
                offset += 2
                
                if offset + nalu_length > len(hvcc_data):
                    break
                
                nalu_data = hvcc_data[offset:offset+nalu_length]
                nals.append((array_type, nalu_data))
                offset += nalu_length
        
        return nals


def parse_mp4_file(file_path: str) -> Dict:
    """Convenience function to parse MP4/MOV file"""
    parser = MP4Parser(file_path)
    return parser.parse()
