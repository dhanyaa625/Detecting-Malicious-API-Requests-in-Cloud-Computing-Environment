"""
Demo Input Payloads for the Agentic-PACX Live Dashboard

INSTRUCTIONS:
You can literally copy the string contents between the triple-quotes below 
and paste them directly into the "Submit Your Data" text area on the 
Live Analysis webpage (http://localhost:8080/live_analysis.html).
"""

# =====================================================================
# PAYLOAD 1: Unstructured Raw API Sequence (Select "Simple Text Check")
# =====================================================================
API_ONLY_PAYLOAD = """
kernel32.dll!CreateFileA
ntdll!NtAllocateVirtualMemory
kernel32.dll!VirtualAllocEx
user32.dll!GetMessageW
ws2_32.dll!InternetConnect
kernel32!TerminateProcess
"""

# =====================================================================
# PAYLOAD 2: Fragmented EDR Log (Select "Fragmented Missing Data")
# =====================================================================
JSON_FRAGMENT_PAYLOAD = """
{
    "timestamp": "2026-04-19T10:00:00Z",
    "threat_actor": "unknown",
    "data_available": {
        "headers_found": false,
        "entropy_calculated": false
    },
    "behavioral_apis": [
        "RegSetValueExA",
        "CreateProcessInternalW",
        "LoadLibraryA",
        "GetProcAddress"
    ],
    "network_telemetry": [
        "192.168.1.50:443",
        "8.8.8.8:53"
    ]
}
"""

# =====================================================================
# PAYLOAD 3: Clean Structured Data (Select "Perfect Complete File")
# =====================================================================
CSV_STRUCTURED_PAYLOAD = """
Header_Size,Entropy_Section1,Entropy_Section2,Total_APIs,Network_Outbound
1024,6.883,7.123,45,2
"""

if __name__ == "__main__":
    print("[*] Open live_analysis.html in your browser.")
    print("[*] Copy the string chunks directly from this script to test the UI.")
