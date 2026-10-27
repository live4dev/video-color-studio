/* SPDX-License-Identifier: MIT
 * Native Unicode launcher for the private, isolated CPython application runtime.
 * No shell, PATH lookup, registry changes, downloads or system Python are used.
 */
#ifndef UNICODE
#define UNICODE 1
#endif
#ifndef _UNICODE
#define _UNICODE 1
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <wchar.h>

#define BUFFER_CHARS 32768
#ifndef FREEDIVE_GUI
#define FREEDIVE_GUI 1
#endif
#ifndef FREEDIVE_MCP
#define FREEDIVE_MCP 0
#endif

static wchar_t root_path[BUFFER_CHARS];
static wchar_t python_path[BUFFER_CHARS];
static wchar_t bootstrap_path[BUFFER_CHARS];
static wchar_t command_line[BUFFER_CHARS];

static int append(wchar_t *destination, const wchar_t *text) {
    size_t used = wcslen(destination), count = wcslen(text);
    if (used + count >= BUFFER_CHARS) return 0;
    wmemcpy(destination + used, text, count + 1);
    return 1;
}

static const wchar_t *argument_tail(void) {
    const wchar_t *cursor = GetCommandLineW();
    int quoted = 0;
    while (*cursor) {
        if (*cursor == L'"') quoted = !quoted;
        else if (!quoted && (*cursor == L' ' || *cursor == L'\t')) break;
        ++cursor;
    }
    while (*cursor == L' ' || *cursor == L'\t') ++cursor;
    return cursor;
}

static int fail(const wchar_t *message, DWORD error, const wchar_t *arguments) {
#if FREEDIVE_GUI
    /* Automated smoke checks must fail with an exit code rather than wait for a dialog. */
    if (!wcsstr(arguments, L"--smoke-test") && !wcsstr(arguments, L"--smoke-video"))
        MessageBoxW(NULL, message, L"Freedive Color Studio", MB_OK | MB_ICONERROR);
#else
    DWORD count;
    char encoded[2048];
    HANDLE output = GetStdHandle(STD_ERROR_HANDLE);
    int bytes = WideCharToMultiByte(CP_UTF8, 0, message, -1, encoded, sizeof(encoded), NULL, NULL);
    if (output && output != INVALID_HANDLE_VALUE && bytes > 0) {
        /* stderr may be a Codex pipe rather than a console. stdout is protocol-only. */
        WriteFile(output, encoded, (DWORD)(bytes - 1), &count, NULL);
        WriteFile(output, "\r\n", 2, &count, NULL);
    }
#endif
    return error ? (int)error : 1;
}

static int launch(void) {
    const wchar_t *arguments = argument_tail();
    DWORD length = GetModuleFileNameW(NULL, root_path, BUFFER_CHARS);
    if (!length || length >= BUFFER_CHARS)
        return fail(L"The application path is too long.", ERROR_BAD_PATHNAME, arguments);
    wchar_t *slash = wcsrchr(root_path, L'\\');
    if (!slash) return fail(L"Cannot locate the application folder.", 1, arguments);
    *slash = L'\0';
    if (!append(python_path, root_path) ||
#if FREEDIVE_GUI
        !append(python_path, L"\\runtime\\pythonw.exe") ||
#else
        !append(python_path, L"\\runtime\\python.exe") ||
#endif
        !append(bootstrap_path, root_path) ||
        !append(bootstrap_path, L"\\portable_bootstrap.py") ||
        !append(command_line, L"\"") || !append(command_line, python_path) ||
        !append(command_line, L"\" -I -B -X utf8 \"") ||
        !append(command_line, bootstrap_path) ||
#if FREEDIVE_GUI
        !append(command_line, L"\" --launcher-gui ") ||
#elif FREEDIVE_MCP
        !append(command_line, L"\" --launcher-mcp ") ||
#else
        !append(command_line, L"\" --launcher-cli ") ||
#endif
        !append(command_line, arguments))
        return fail(L"The path or command-line arguments are too long.", 1, arguments);
    if (GetFileAttributesW(python_path) == INVALID_FILE_ATTRIBUTES ||
        GetFileAttributesW(bootstrap_path) == INVALID_FILE_ATTRIBUTES)
        return fail(L"The private runtime is missing. Extract the complete ZIP before opening the app.",
                    ERROR_FILE_NOT_FOUND, arguments);
    STARTUPINFOEXW startup = {0};
    PROCESS_INFORMATION process = {0};
    DWORD creation_flags = 0;
    startup.StartupInfo.cb = sizeof(STARTUPINFOW);
#if FREEDIVE_GUI
    creation_flags = CREATE_NO_WINDOW;
#else
    /* Explicitly preserve the MCP/CLI pipes even when the parent supplied
       non-inheritable handles. Restrict inheritance to these three duplicates. */
    HANDLE inherited[3] = {NULL, NULL, NULL};
    const DWORD ids[3] = {STD_INPUT_HANDLE, STD_OUTPUT_HANDLE, STD_ERROR_HANDLE};
    SIZE_T attribute_bytes = 0;
    for (int index = 0; index < 3; ++index) {
        HANDLE original = GetStdHandle(ids[index]);
        if (!original || original == INVALID_HANDLE_VALUE ||
            !DuplicateHandle(GetCurrentProcess(), original, GetCurrentProcess(),
                             &inherited[index], 0, TRUE, DUPLICATE_SAME_ACCESS)) {
            DWORD error = GetLastError();
            for (int prior = 0; prior < index; ++prior) CloseHandle(inherited[prior]);
            return fail(L"Cannot inherit the standard protocol streams.", error, arguments);
        }
    }
    InitializeProcThreadAttributeList(NULL, 1, 0, &attribute_bytes);
    startup.lpAttributeList = HeapAlloc(GetProcessHeap(), 0, attribute_bytes);
    if (!startup.lpAttributeList ||
        !InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &attribute_bytes) ||
        !UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                  inherited, sizeof(inherited), NULL, NULL)) {
        DWORD error = GetLastError();
        for (int index = 0; index < 3; ++index) CloseHandle(inherited[index]);
        if (startup.lpAttributeList) HeapFree(GetProcessHeap(), 0, startup.lpAttributeList);
        return fail(L"Cannot prepare the standard protocol streams.", error, arguments);
    }
    startup.StartupInfo.cb = sizeof(startup);
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = inherited[0];
    startup.StartupInfo.hStdOutput = inherited[1];
    startup.StartupInfo.hStdError = inherited[2];
    creation_flags = EXTENDED_STARTUPINFO_PRESENT;
#endif
    BOOL started = CreateProcessW(python_path, command_line, NULL, NULL, TRUE,
                                  creation_flags, NULL, NULL, &startup.StartupInfo, &process);
    DWORD start_error = started ? 0 : GetLastError();
#if !FREEDIVE_GUI
    DeleteProcThreadAttributeList(startup.lpAttributeList);
    HeapFree(GetProcessHeap(), 0, startup.lpAttributeList);
    for (int index = 0; index < 3; ++index) CloseHandle(inherited[index]);
#endif
    if (!started)
        return fail(L"Cannot start the private application runtime. Check the extracted files.",
                    start_error, arguments);
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess);
    return (int)code;
}

#if FREEDIVE_GUI
int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR command, int show) {
    (void)instance; (void)previous; (void)command; (void)show;
    return launch();
}
#else
int wmain(int argc, wchar_t **argv) {
    (void)argc; (void)argv;
    return launch();
}
#endif
