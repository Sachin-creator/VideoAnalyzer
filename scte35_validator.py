"""
SCTE-35 2023r1 Specification Validator
Validates SCTE-35 splice_info_section messages against the specification
and reports errors with severity levels.
"""

import struct
from typing import Dict, List, Tuple, Any


class SCTE35ValidationError:
    """Represents a validation error with severity level"""
    CRITICAL = "CRITICAL"  # Red - violates must/shall requirements
    WARNING = "WARNING"    # Yellow - violates should/recommended
    INFO = "INFO"          # Blue - informational
    
    def __init__(self, severity: str, field: str, message: str, spec_ref: str = ""):
        self.severity = severity
        self.field = field
        self.message = message
        self.spec_ref = spec_ref
    
    def __repr__(self):
        return f"[{self.severity}] {self.field}: {self.message} ({self.spec_ref})"


class SCTE35Validator:
    """Validates SCTE-35 splice_info_section per ANSI/SCTE 35 2023r1"""
    
    # Splice command types per Table 5 (Section 9.2)
    SPLICE_COMMAND_TYPES = {
        0x00: "splice_null",
        0x04: "splice_schedule",
        0x05: "splice_insert",
        0x06: "time_signal",
        0x07: "bandwidth_reservation",
        0xFF: "private_command"
    }
    
    def __init__(self):
        self.errors: List[SCTE35ValidationError] = []
    
    def add_error(self, severity: str, field: str, message: str, spec_ref: str = ""):
        """Add a validation error"""
        self.errors.append(SCTE35ValidationError(severity, field, message, spec_ref))
    
    def validate_crc32(self, payload: bytes) -> bool:
        """Validate CRC_32 field per Section 6.3.1 and 14.6"""
        if len(payload) < 4:
            self.add_error(SCTE35ValidationError.CRITICAL, "CRC_32",
                          "Payload too short for CRC_32", "§6.3.1")
            return False
        
        # CRC_32 is last 4 bytes
        reported_crc = struct.unpack('>I', payload[-4:])[0]
        
        # Calculate CRC on all bytes except CRC itself
        calculated_crc = self._calculate_crc32(payload[:-4])
        
        if reported_crc != calculated_crc:
            self.add_error(SCTE35ValidationError.CRITICAL, "CRC_32",
                          f"CRC mismatch: reported=0x{reported_crc:08X}, calculated=0x{calculated_crc:08X}",
                          "§14.6")
            return False
        return True
    
    def _calculate_crc32(self, data: bytes) -> int:
        """Calculate MPEG-2 CRC-32 per ISO/IEC 13818-1 Annex A"""
        # MPEG-2 CRC polynomial: 0x104C11DB7
        crc = 0xFFFFFFFF
        for byte in data:
            crc ^= (byte << 24)
            for _ in range(8):
                if crc & 0x80000000:
                    crc = (crc << 1) ^ 0x04C11DB7
                else:
                    crc = crc << 1
                crc &= 0xFFFFFFFF
        return crc
    
    def validate_splice_info_section(self, payload: bytes) -> Dict[str, Any]:
        """
        Validate complete splice_info_section per Section 6.3.1
        Returns parsed structure with validation errors
        """
        self.errors = []  # Reset errors
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Minimum length check
        if len(payload) < 14:
            self.add_error(SCTE35ValidationError.CRITICAL, "splice_info_section",
                          f"Payload too short: {len(payload)} bytes (minimum 14)", "§6.3.1")
            result["valid"] = False
            result["errors"] = self.errors
            return result
        
        # table_id SHALL be 0xFC (Section 6.3.1)
        table_id = payload[0]
        if table_id != 0xFC:
            self.add_error(SCTE35ValidationError.CRITICAL, "table_id",
                          f"Invalid table_id: 0x{table_id:02X} (SHALL be 0xFC)", "§6.3.1")
        
        # section_syntax_indicator SHALL be 0 (Section 6.3.1)
        section_syntax_indicator = (payload[1] >> 7) & 0x01
        if section_syntax_indicator != 0:
            self.add_error(SCTE35ValidationError.CRITICAL, "section_syntax_indicator",
                          "SHALL be '0' for SCTE-35", "§6.3.1")
        
        # private_indicator SHALL be 0 (Section 6.3.1)
        private_indicator = (payload[1] >> 6) & 0x01
        if private_indicator != 0:
            self.add_error(SCTE35ValidationError.CRITICAL, "private_indicator",
                          "SHALL be '0'", "§6.3.1")
        
        # sap_type (2 bits) SHALL be '11' (reserved)
        sap_type = (payload[1] >> 4) & 0x03
        if sap_type != 0x03:
            self.add_error(SCTE35ValidationError.WARNING, "sap_type",
                          f"SHOULD be '11' (reserved), found '{sap_type:02b}'", "§6.3.1")
        
        # section_length
        section_length = ((payload[1] & 0x0F) << 8) | payload[2]
        if section_length < 11:
            self.add_error(SCTE35ValidationError.CRITICAL, "section_length",
                          f"Too small: {section_length} (minimum 11)", "§6.3.1")
        if section_length > 4093:
            self.add_error(SCTE35ValidationError.CRITICAL, "section_length",
                          f"Exceeds maximum: {section_length} (max 4093)", "§6.3.1")
        
        expected_total_length = 3 + section_length  # header + section_length value
        if len(payload) != expected_total_length:
            self.add_error(SCTE35ValidationError.CRITICAL, "section_length",
                          f"Mismatch: declared={section_length}, actual payload={len(payload)-3}", "§6.3.1")
        
        # protocol_version SHALL be 0 (Section 6.3.1)
        protocol_version = payload[3]
        if protocol_version != 0:
            self.add_error(SCTE35ValidationError.CRITICAL, "protocol_version",
                          f"SHALL be 0, found {protocol_version}", "§6.3.1")
        
        # encrypted_packet flag
        encrypted_packet = (payload[4] >> 7) & 0x01
        
        # encryption_algorithm (6 bits)
        encryption_algorithm = (payload[4] >> 1) & 0x3F
        if encrypted_packet == 0 and encryption_algorithm != 0:
            self.add_error(SCTE35ValidationError.WARNING, "encryption_algorithm",
                          "SHOULD be 0 when encrypted_packet=0", "§6.3.1")
        elif encrypted_packet == 1:
            # Validate encryption_algorithm values per Table 4 (Section 8)
            valid_algorithms = [0, 1, 2, 3]  # DES, 3DES, AES-128, User Private
            if encryption_algorithm not in valid_algorithms and encryption_algorithm < 32:
                self.add_error(SCTE35ValidationError.WARNING, "encryption_algorithm",
                              f"Reserved value: {encryption_algorithm}", "§8, Table 4")
        
        # pts_adjustment (33 bits)
        pts_adjustment = ((payload[4] & 0x01) << 32) | (payload[5] << 24) | \
                        (payload[6] << 16) | (payload[7] << 8) | payload[8]
        
        # cw_index
        cw_index = payload[9]
        
        # tier (12 bits)
        tier = ((payload[10] << 8) | payload[11]) >> 4
        if tier == 0xFFF:
            self.add_error(SCTE35ValidationError.INFO, "tier",
                          "0xFFF indicates not used", "§6.3.1")
        
        # splice_command_length (12 bits)
        splice_command_length = ((payload[11] & 0x0F) << 8) | payload[12]
        if splice_command_length == 0xFFF:
            self.add_error(SCTE35ValidationError.WARNING, "splice_command_length",
                          "0xFFF indicates unknown length (not recommended)", "§6.3.1")
        
        # splice_command_type
        splice_command_type = payload[13]
        command_name = self.SPLICE_COMMAND_TYPES.get(splice_command_type,
                                                     f"reserved/unknown (0x{splice_command_type:02X})")
        
        # Validate command type
        if splice_command_type not in self.SPLICE_COMMAND_TYPES:
            if splice_command_type >= 0x08 and splice_command_type <= 0xFE:
                self.add_error(SCTE35ValidationError.WARNING, "splice_command_type",
                              f"Reserved value: 0x{splice_command_type:02X}", "§9.2, Table 5")
        
        # Validate splice_command() based on type
        if splice_command_length != 0xFFF:
            cmd_start = 14
            cmd_end = cmd_start + splice_command_length
            if cmd_end > len(payload) - 6:  # Need room for descriptor_loop_length + alignment + CRC
                self.add_error(SCTE35ValidationError.CRITICAL, "splice_command_length",
                              "Extends beyond valid payload boundary", "§6.3.1")
            else:
                cmd_payload = payload[cmd_start:cmd_end]
                
                # Validate specific command types
                if splice_command_type == 0x05:  # splice_insert
                    self._validate_splice_insert(cmd_payload)
                elif splice_command_type == 0x06:  # time_signal
                    self._validate_time_signal(cmd_payload)
                elif splice_command_type == 0x00:  # splice_null
                    if len(cmd_payload) != 0:
                        self.add_error(SCTE35ValidationError.WARNING, "splice_null",
                                      f"splice_null SHOULD have zero length, found {len(cmd_payload)}", "§9.3.1")
        
        # Validate CRC_32
        self.validate_crc32(payload)
        
        # Categorize errors
        for err in self.errors:
            if err.severity == SCTE35ValidationError.CRITICAL:
                result["errors"].append(err)
                result["valid"] = False
            elif err.severity == SCTE35ValidationError.WARNING:
                result["warnings"].append(err)
            else:
                result["info"].append(err)
        
        result["parsed"] = {
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
            "command_name": command_name
        }
        
        return result
    
    def _validate_splice_insert(self, payload: bytes):
        """Validate splice_insert() command per Section 9.3.2"""
        if len(payload) < 5:
            self.add_error(SCTE35ValidationError.CRITICAL, "splice_insert",
                          f"Payload too short: {len(payload)} bytes (minimum 5)", "§9.3.2")
            return
        
        splice_event_id = struct.unpack('>I', payload[0:4])[0]
        splice_event_cancel_indicator = (payload[4] >> 7) & 0x01
        
        if splice_event_cancel_indicator == 0:
            # Full splice_insert structure required
            if len(payload) < 6:
                self.add_error(SCTE35ValidationError.CRITICAL, "splice_insert",
                              "Insufficient data for non-cancelled splice_insert", "§9.3.2")
                return
            
            out_of_network_indicator = (payload[5] >> 7) & 0x01
            program_splice_flag = (payload[5] >> 6) & 0x01
            duration_flag = (payload[5] >> 5) & 0x01
            splice_immediate_flag = (payload[5] >> 4) & 0x01
            
            # event_id_compliance_flag validation (Section 9.3.2)
            event_id_compliance_flag = (payload[5] >> 3) & 0x01
            if event_id_compliance_flag == 0:
                self.add_error(SCTE35ValidationError.WARNING, "event_id_compliance_flag",
                              "SHOULD be '1' for compliance", "§9.3.2")
            
            # Reserved bits (3 bits) SHALL be '111'
            reserved = payload[5] & 0x07
            if reserved != 0x07:
                self.add_error(SCTE35ValidationError.WARNING, "reserved",
                              f"Reserved 3 bits SHOULD be '111', found '{reserved:03b}'", "§9.3.2")
    
    def _validate_time_signal(self, payload: bytes):
        """Validate time_signal() command per Section 9.3.4"""
        if len(payload) < 1:
            self.add_error(SCTE35ValidationError.CRITICAL, "time_signal",
                          "Payload too short for splice_time()", "§9.3.4")
            return
        
        time_specified_flag = (payload[0] >> 7) & 0x01
        
        if time_specified_flag == 1:
            # pts_time is present (33 bits = 6 bits reserved + 33 bits pts_time = 5 bytes total)
            if len(payload) < 5:
                self.add_error(SCTE35ValidationError.CRITICAL, "time_signal",
                              "Insufficient data for pts_time", "§9.3.4")
                return
            
            # Reserved 6 bits SHALL be '111111'
            reserved = payload[0] & 0x3F
            if reserved != 0x3F:
                self.add_error(SCTE35ValidationError.WARNING, "splice_time",
                              f"Reserved 6 bits SHOULD be '111111', found '{reserved:06b}'", "§9.4.1")
        else:
            # Only 1 byte: time_specified_flag(1) + reserved(7)
            if len(payload) != 1:
                self.add_error(SCTE35ValidationError.WARNING, "time_signal",
                              f"Expected 1 byte when time_specified_flag=0, found {len(payload)}", "§9.3.4")
            
            # Reserved 7 bits SHALL be '1111111'
            reserved = payload[0] & 0x7F
            if reserved != 0x7F:
                self.add_error(SCTE35ValidationError.WARNING, "splice_time",
                              f"Reserved 7 bits SHOULD be '1111111', found '{reserved:07b}'", "§9.4.1")
