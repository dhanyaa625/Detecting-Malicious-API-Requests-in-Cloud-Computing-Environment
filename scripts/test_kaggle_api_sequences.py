"""
Tests the PAC-X pathway ALONE against real dynamic API call sequences
(kaggle_dataset/dynamic_api_call_sequence_per_malware_100_0_306.csv,
43,876 real samples: 100 non-repeated API calls per sample, extracted from
real Cuckoo Sandbox reports -- the "Malware Analysis Datasets: API Call
Sequences" dataset by Angelo Oliveira).

This dataset has NO PE header/section/import-export data, so a graph for
the GNN pathway can't be built from it -- this can only test PAC-X in
isolation, not the fused/arbitration pipeline. What it DOES give: the
first real measurement of PAC-X on genuine dynamic API-log data, which is
exactly the modality the paper describes as PAC-X's intended input but
never had labeled ground truth to actually test.

Code->API-name legend (0-306) sourced from the dataset's own public
documentation (IEEE DataPort), not guessed -- see _API_LEGEND below.

Usage:
    python scripts/test_kaggle_api_sequences.py --csv kaggle_dataset/dynamic_api_call_sequence_per_malware_100_0_306.csv --limit 1000
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.pacx_analyzer import PACXHeuristicAnalyzer  # noqa: E402

# Verbatim from the dataset's public documentation (IEEE DataPort page for
# "Malware Analysis Datasets: API Call Sequences"), positional: index i is
# the API name for code i. 307 entries, codes 0-306.
_API_LEGEND = [
    "NtOpenThread", "ExitWindowsEx", "FindResourceW", "CryptExportKey", "CreateRemoteThreadEx",
    "MessageBoxTimeoutW", "InternetCrackUrlW", "StartServiceW", "GetFileSize", "GetVolumeNameForVolumeMountPointW",
    "GetFileInformationByHandle", "CryptAcquireContextW", "RtlDecompressBuffer", "SetWindowsHookExA", "RegSetValueExW",
    "LookupAccountSidW", "SetUnhandledExceptionFilter", "InternetConnectA", "GetComputerNameW", "RegEnumValueA",
    "NtOpenFile", "NtSaveKeyEx", "HttpOpenRequestA", "recv", "GetFileSizeEx",
    "LoadStringW", "SetInformationJobObject", "WSAConnect", "CryptDecrypt", "GetTimeZoneInformation",
    "InternetOpenW", "CoInitializeEx", "CryptGenKey", "GetAsyncKeyState", "NtQueryInformationFile",
    "GetSystemMetrics", "NtDeleteValueKey", "NtOpenKeyEx", "sendto", "IsDebuggerPresent",
    "RegQueryInfoKeyW", "NetShareEnum", "InternetOpenUrlW", "WSASocketA", "CopyFileExW",
    "connect", "ShellExecuteExW", "SearchPathW", "GetUserNameA", "InternetOpenUrlA",
    "LdrUnloadDll", "EnumServicesStatusW", "EnumServicesStatusA", "WSASend", "CopyFileW",
    "NtDeleteFile", "CreateActCtxW", "timeGetTime", "MessageBoxTimeoutA", "CreateServiceA",
    "FindResourceExW", "WSAAccept", "InternetConnectW", "HttpSendRequestA", "GetVolumePathNameW",
    "RegCloseKey", "InternetGetConnectedStateExW", "GetAdaptersInfo", "shutdown", "NtQueryMultipleValueKey",
    "NtQueryKey", "GetSystemWindowsDirectoryW", "GlobalMemoryStatusEx", "GetFileAttributesExW", "OpenServiceW",
    "getsockname", "LoadStringA", "UnhookWindowsHookEx", "NtCreateUserProcess", "Process32NextW",
    "CreateThread", "LoadResource", "GetSystemTimeAsFileTime", "SetStdHandle", "CoCreateInstanceEx",
    "GetSystemDirectoryA", "NtCreateMutant", "RegCreateKeyExW", "IWbemServices_ExecQuery", "NtDuplicateObject",
    "Thread32First", "OpenSCManagerW", "CreateServiceW", "GetFileType", "MoveFileWithProgressW",
    "NtDeviceIoControlFile", "GetFileInformationByHandleEx", "CopyFileA", "NtLoadKey", "GetNativeSystemInfo",
    "NtOpenProcess", "CryptUnprotectMemory", "InternetWriteFile", "ReadProcessMemory", "gethostbyname",
    "WSASendTo", "NtOpenSection", "listen", "WSAStartup", "socket",
    "OleInitialize", "FindResourceA", "RegOpenKeyExA", "RegEnumKeyExA", "NtQueryDirectoryFile",
    "CertOpenSystemStoreW", "ControlService", "LdrGetProcedureAddress", "GlobalMemoryStatus", "NtSetInformationFile",
    "OutputDebugStringA", "GetAdaptersAddresses", "CoInitializeSecurity", "RegQueryValueExA", "NtQueryFullAttributesFile",
    "DeviceIoControl", "__anomaly__", "DeleteFileW", "GetShortPathNameW", "NtGetContextThread",
    "GetKeyboardState", "RemoveDirectoryA", "InternetSetStatusCallback", "NtResumeThread", "SetFileInformationByHandle",
    "NtCreateSection", "NtQueueApcThread", "accept", "DecryptMessage", "GetUserNameExW",
    "SizeofResource", "RegQueryValueExW", "SetWindowsHookExW", "HttpOpenRequestW", "CreateDirectoryW",
    "InternetOpenA", "GetFileVersionInfoExW", "FindWindowA", "closesocket", "RtlAddVectoredExceptionHandler",
    "IWbemServices_ExecMethod", "GetDiskFreeSpaceExW", "TaskDialog", "WriteConsoleW", "CryptEncrypt",
    "WSARecvFrom", "NtOpenMutant", "CoGetClassObject", "NtQueryValueKey", "NtDelayExecution",
    "select", "HttpQueryInfoA", "GetVolumePathNamesForVolumeNameW", "RegDeleteValueW", "InternetCrackUrlA",
    "OpenServiceA", "InternetSetOptionA", "CreateDirectoryExW", "bind", "NtShutdownSystem",
    "DeleteUrlCacheEntryA", "NtMapViewOfSection", "LdrGetDllHandle", "NtCreateKey", "GetKeyState",
    "CreateRemoteThread", "NtEnumerateValueKey", "SetFileAttributesW", "NtUnmapViewOfSection", "RegDeleteValueA",
    "CreateJobObjectW", "send", "NtDeleteKey", "SetEndOfFile", "GetUserNameExA",
    "GetComputerNameA", "URLDownloadToFileW", "NtFreeVirtualMemory", "recvfrom", "NtUnloadDriver",
    "NtTerminateThread", "CryptUnprotectData", "NtCreateThreadEx", "DeleteService", "GetFileAttributesW",
    "GetFileVersionInfoSizeExW", "OpenSCManagerA", "WriteProcessMemory", "GetSystemInfo", "SetFilePointer",
    "Module32FirstW", "ioctlsocket", "RegEnumKeyW", "RtlCompressBuffer", "SendNotifyMessageW",
    "GetAddrInfoW", "CryptProtectData", "Thread32Next", "NtAllocateVirtualMemory", "RegEnumKeyExW",
    "RegSetValueExA", "DrawTextExA", "CreateToolhelp32Snapshot", "FindWindowW", "CoUninitialize",
    "NtClose", "WSARecv", "CertOpenStore", "InternetGetConnectedState", "RtlAddVectoredContinueHandler",
    "RegDeleteKeyW", "SHGetSpecialFolderLocation", "CreateProcessInternalW", "NtCreateDirectoryObject", "EnumWindows",
    "DrawTextExW", "RegEnumValueW", "SendNotifyMessageA", "NtProtectVirtualMemory", "NetUserGetLocalGroups",
    "GetUserNameW", "WSASocketW", "getaddrinfo", "AssignProcessToJobObject", "SetFileTime",
    "WriteConsoleA", "CryptDecodeObjectEx", "EncryptMessage", "system", "NtSetContextThread",
    "LdrLoadDll", "InternetGetConnectedStateExA", "RtlCreateUserThread", "GetCursorPos", "Module32NextW",
    "RegCreateKeyExA", "NtLoadDriver", "NetUserGetInfo", "SHGetFolderPathW", "GetBestInterfaceEx",
    "CertControlStore", "StartServiceA", "NtWriteFile", "Process32FirstW", "NtReadVirtualMemory",
    "GetDiskFreeSpaceW", "GetFileVersionInfoW", "FindFirstFileExW", "FindWindowExW", "GetSystemWindowsDirectoryA",
    "RegOpenKeyExW", "CoCreateInstance", "NtQuerySystemInformation", "LookupPrivilegeValueW", "NtReadFile",
    "ReadCabinetState", "GetForegroundWindow", "InternetCloseHandle", "FindWindowExA", "ObtainUserAgentString",
    "CryptCreateHash", "GetTempPathW", "CryptProtectMemory", "NetGetJoinInformation", "NtOpenKey",
    "GetSystemDirectoryW", "DnsQuery_A", "RegQueryInfoKeyA", "NtEnumerateKey", "RegisterHotKey",
    "RemoveDirectoryW", "FindFirstFileExA", "CertOpenSystemStoreA", "NtTerminateProcess", "NtSetValueKey",
    "CryptAcquireContextA", "SetErrorMode", "UuidCreate", "RtlRemoveVectoredExceptionHandler", "RegDeleteKeyA",
    "setsockopt", "FindResourceExA", "NtSuspendThread", "GetFileVersionInfoSizeW", "NtOpenDirectoryObject",
    "InternetQueryOptionA", "InternetReadFile", "NtCreateFile", "NtQueryAttributesFile", "HttpSendRequestW",
    "CryptHashMessage", "CryptHashData", "NtWriteVirtualMemory", "SetFilePointerEx", "CertCreateCertificateContext",
    "DeleteUrlCacheEntryW", "__exception__",
]
assert len(_API_LEGEND) == 307, f"Expected 307 entries, got {len(_API_LEGEND)}"


def decode_sequence(codes: list) -> str:
    names = []
    for c in codes:
        try:
            names.append(_API_LEGEND[int(c)])
        except (ValueError, IndexError):
            continue
    return "\n".join(names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="kaggle_dataset/dynamic_api_call_sequence_per_malware_100_0_306.csv")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.csv, low_memory=False)
    t_cols = [f"t_{i}" for i in range(100)]
    assert all(c in df.columns for c in t_cols), "Expected t_0..t_99 columns"
    assert "malware" in df.columns

    per_class = args.limit // 2
    benign = df[df["malware"] == 0]
    malicious = df[df["malware"] == 1]
    benign_sample = benign.sample(n=min(per_class, len(benign)), random_state=args.seed)
    malicious_sample = malicious.sample(n=min(per_class, len(malicious)), random_state=args.seed)
    sample = pd.concat([benign_sample, malicious_sample]).sample(frac=1.0, random_state=args.seed)
    print(f"[+] Testing {len(sample)} real dynamic API-call sequences "
          f"({len(benign_sample)} benign, {len(malicious_sample)} malicious)")
    print(f"[!] Note: source dataset is heavily imbalanced (1,079 benign / 42,797 malicious total); "
          f"benign sample size is capped by real availability.")

    pacx = PACXHeuristicAnalyzer()
    counts = {"correct": 0, "total": 0}
    fp = fn = n_benign = n_malicious = 0
    results = []

    for _, row in sample.iterrows():
        true_label = "Benign" if row["malware"] == 0 else "Malicious"
        codes = [row[c] for c in t_cols]
        api_text = decode_sequence(codes)

        result = pacx.analyze(api_text, input_type="api")
        pred = result["prediction"]

        counts["total"] += 1
        counts["correct"] += int(pred == true_label)
        if true_label == "Benign":
            n_benign += 1
            fp += int(pred == "Malicious")
        else:
            n_malicious += 1
            fn += int(pred == "Benign")

        results.append({"true_label": true_label, "pacx_label": pred, "confidence": result["confidence"]})

    total = max(1, counts["total"])
    summary = {
        "n_tested": counts["total"],
        "pacx_accuracy": round(100 * counts["correct"] / total, 2),
        "pacx_false_positive_rate": round(100 * fp / max(1, n_benign), 2),
        "pacx_false_negative_rate": round(100 * fn / max(1, n_malicious), 2),
    }
    print("\n=== SUMMARY (PAC-X alone, real dynamic API-call sequences) ===")
    print(json.dumps(summary, indent=2))

    os.makedirs("data/kaggle_api_sequence_test", exist_ok=True)
    with open("data/kaggle_api_sequence_test/report.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print("[+] Saved data/kaggle_api_sequence_test/report.json")


if __name__ == "__main__":
    main()
