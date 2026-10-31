// sysutil.cpp — 与平台相关的小工具（单独编译单元，避免 windows.h 的 SIZE/min/max 污染引擎代码）
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "sysutil.h"

#include <cstdio>

namespace gk {

bool set_below_normal_priority() {
#ifdef _WIN32
    return SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS) != 0;
#else
    return false;
#endif
}

std::string exe_path() {
#ifdef _WIN32
    char buf[MAX_PATH * 4];
    const DWORD n = GetModuleFileNameA(nullptr, buf, (DWORD)sizeof(buf));
    if (n == 0 || n >= sizeof(buf)) return std::string();
    return std::string(buf, n);
#else
    return std::string();
#endif
}

bool run_parallel(const std::vector<std::string>& cmds, std::vector<int>* codes) {
#ifdef _WIN32
    if (cmds.empty()) return true;
    const std::string exe = exe_path();
    if (exe.empty()) return false;
    std::vector<PROCESS_INFORMATION> pis(cmds.size());
    std::vector<HANDLE> hs;
    hs.reserve(cmds.size());
    bool ok = true;
    for (size_t i = 0; i < cmds.size(); ++i) {
        std::vector<char> line(exe.size() + 1 + cmds[i].size() + 1);
        std::snprintf(line.data(), line.size(), "\"%s\" %s", exe.c_str(), cmds[i].c_str());
        STARTUPINFOA si;
        ZeroMemory(&si, sizeof(si));
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi;
        ZeroMemory(&pi, sizeof(pi));
        // CREATE_NO_WINDOW：子进程不弹控制台窗口（父进程已有控制台则继承）
        const BOOL born = CreateProcessA(nullptr, line.data(), nullptr, nullptr, FALSE,
                                         CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi);
        if (!born) {
            std::fprintf(stderr, "[错误] 启动子进程失败: %s\n", line.data());
            ok = false;
            pis[i] = PROCESS_INFORMATION{};
            continue;
        }
        pis[i] = pi;
        hs.push_back(pi.hProcess);
        CloseHandle(pi.hThread);
    }
    if (!hs.empty()) WaitForMultipleObjects((DWORD)hs.size(), hs.data(), TRUE, INFINITE);
    for (size_t i = 0; i < pis.size(); ++i) {
        int code = -1;
        if (pis[i].hProcess) {
            DWORD c = 1;
            GetExitCodeProcess(pis[i].hProcess, &c);
            code = (int)c;
            CloseHandle(pis[i].hProcess);
        }
        if (codes) codes->push_back(code);
        if (code != 0) ok = false;
    }
    return ok;
#else
    (void)cmds;
    (void)codes;
    return false;
#endif
}

}  // namespace gk
