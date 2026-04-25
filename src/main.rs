#[cfg(target_os = "windows")]
fn main() -> iced::Result {
    ensure_admin_or_relaunch();
    cannot_max_rs::app::run()
}

#[cfg(not(target_os = "windows"))]
fn main() -> iced::Result {
    cannot_max_rs::app::run()
}

#[cfg(target_os = "windows")]
fn ensure_admin_or_relaunch() {
    match try_ensure_admin_or_relaunch() {
        Ok(AdminEnsureResult::AlreadyElevated) => {}
        Ok(AdminEnsureResult::Relaunched) => std::process::exit(0),
        Err(error) => {
            eprintln!("无法获取管理员权限: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(target_os = "windows")]
#[derive(Debug, Clone, Copy)]
enum AdminEnsureResult {
    AlreadyElevated,
    Relaunched,
}

#[cfg(target_os = "windows")]
fn try_ensure_admin_or_relaunch() -> Result<AdminEnsureResult, String> {
    if is_process_elevated()? {
        return Ok(AdminEnsureResult::AlreadyElevated);
    }

    relaunch_self_as_admin()?;
    Ok(AdminEnsureResult::Relaunched)
}

#[cfg(target_os = "windows")]
fn relaunch_self_as_admin() -> Result<(), String> {
    use std::ffi::OsStr;
    use std::ptr;
    use windows_sys::Win32::UI::Shell::ShellExecuteW;
    use windows_sys::Win32::UI::WindowsAndMessaging::SW_SHOWNORMAL;

    let exe_path = std::env::current_exe().map_err(|error| error.to_string())?;
    let args = build_windows_args();

    let verb = wide_null(OsStr::new("runas"));
    let file = wide_null(exe_path.as_os_str());
    let params = if args.is_empty() {
        None
    } else {
        Some(wide_null(OsStr::new(&args)))
    };

    let result = unsafe {
        ShellExecuteW(
            std::ptr::null_mut(),
            verb.as_ptr(),
            file.as_ptr(),
            params
                .as_ref()
                .map_or(ptr::null(), |value| value.as_ptr()),
            ptr::null(),
            SW_SHOWNORMAL,
        )
    };

    if (result as isize) <= 32 {
        return Err(format!("UAC 提权启动失败，ShellExecuteW 返回: {result:?}"));
    }

    Ok(())
}

#[cfg(target_os = "windows")]
fn is_process_elevated() -> Result<bool, String> {
    use std::mem;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::Security::{
        GetTokenInformation, TOKEN_ELEVATION, TOKEN_QUERY, TokenElevation,
    };
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, OpenProcessToken};

    let mut token: HANDLE = std::ptr::null_mut();
    let open_ok = unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) };
    if open_ok == 0 {
        return Err("OpenProcessToken 失败".to_string());
    }

    let mut elevation = TOKEN_ELEVATION { TokenIsElevated: 0 };
    let mut return_len: u32 = 0;
    let info_ok = unsafe {
        GetTokenInformation(
            token,
            TokenElevation,
            (&mut elevation as *mut TOKEN_ELEVATION).cast(),
            mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut return_len,
        )
    };

    unsafe {
        CloseHandle(token);
    }

    if info_ok == 0 {
        return Err("GetTokenInformation(TokenElevation) 失败".to_string());
    }

    Ok(elevation.TokenIsElevated != 0)
}

#[cfg(target_os = "windows")]
fn build_windows_args() -> String {
    std::env::args_os()
        .skip(1)
        .map(quote_windows_arg)
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(target_os = "windows")]
fn quote_windows_arg(arg: std::ffi::OsString) -> String {
    let value = arg.to_string_lossy();
    if value.is_empty() {
        return "\"\"".to_string();
    }

    let needs_quotes = value.chars().any(|c| c.is_whitespace() || c == '"');
    if !needs_quotes {
        return value.to_string();
    }

    let mut quoted = String::from("\"");
    let mut backslashes = 0;
    for ch in value.chars() {
        match ch {
            '\\' => backslashes += 1,
            '"' => {
                quoted.push_str(&"\\".repeat(backslashes * 2 + 1));
                quoted.push('"');
                backslashes = 0;
            }
            _ => {
                if backslashes > 0 {
                    quoted.push_str(&"\\".repeat(backslashes));
                    backslashes = 0;
                }
                quoted.push(ch);
            }
        }
    }

    if backslashes > 0 {
        quoted.push_str(&"\\".repeat(backslashes * 2));
    }
    quoted.push('"');
    quoted
}

#[cfg(target_os = "windows")]
fn wide_null(value: &std::ffi::OsStr) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    value.encode_wide().chain(std::iter::once(0)).collect()
}
