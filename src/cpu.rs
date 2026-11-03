#[derive(Clone, Copy, Default)]
pub(crate) struct Cpu {
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pub(crate) bmi2: bool,
    #[cfg_attr(not(all(target_arch = "x86_64", target_feature = "sse2")), allow(dead_code))]
    pub(crate) avx2: bool,
}

impl Cpu {
    pub(crate) fn detect() -> Self {
        Cpu { bmi2: bmi2(), avx2: avx2() }
    }

    #[cfg(test)]
    pub(crate) const PLAIN: Cpu = Cpu { bmi2: false, avx2: false };
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn bmi2() -> bool {
    std::arch::is_x86_feature_detected!("bmi1") && std::arch::is_x86_feature_detected!("bmi2")
}

#[cfg(all(target_arch = "x86_64", feature = "std"))]
fn avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
fn avx2() -> bool {
    use core::arch::x86_64::{__cpuid, __cpuid_count, _xgetbv};
    #[target_feature(enable = "xsave")]
    unsafe fn xcr0() -> u64 {
        _xgetbv(0)
    }
    #[allow(unused_unsafe)]
    let max = unsafe { __cpuid(0) }.eax;
    if max < 7 {
        return false;
    }
    #[allow(unused_unsafe)]
    let ecx = unsafe { __cpuid(1) }.ecx;
    if ecx & (1 << 27) == 0 || ecx & (1 << 28) == 0 || unsafe { xcr0() } & 6 != 6 {
        return false;
    }
    #[allow(unused_unsafe)]
    let ebx = unsafe { __cpuid_count(7, 0) }.ebx;
    ebx & (1 << 5) != 0
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2() -> bool {
    false
}

#[cfg(all(target_arch = "x86_64", not(feature = "std")))]
fn bmi2() -> bool {
    use core::arch::x86_64::{__cpuid, __cpuid_count};
    #[allow(unused_unsafe)]
    let max = unsafe { __cpuid(0) }.eax;
    if max < 7 {
        return false;
    }
    #[allow(unused_unsafe)]
    let ebx = unsafe { __cpuid_count(7, 0) }.ebx;
    ebx & (1 << 3) != 0 && ebx & (1 << 8) != 0
}

#[cfg(not(target_arch = "x86_64"))]
fn bmi2() -> bool {
    false
}

macro_rules! multiversion {
    ($vis:vis unsafe fn $name:ident($cpu:ident: Cpu $(, $arg:ident: $ty:ty)*) -> $ret:ty = $imp:path;) => {
        $vis unsafe fn $name($cpu: crate::cpu::Cpu $(, $arg: $ty)*) -> $ret {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "bmi1,bmi2")]
                #[inline(never)]
                unsafe fn fast($($arg: $ty),*) -> $ret {
                    unsafe { $imp($($arg),*) }
                }
                if $cpu.bmi2 {
                    return unsafe { fast($($arg),*) };
                }
            }
            let _ = $cpu;
            unsafe { $imp($($arg),*) }
        }
    };
    ($vis:vis fn $name:ident($cpu:ident: Cpu $(, $arg:ident: $ty:ty)*) -> $ret:ty = $imp:path;) => {
        $vis fn $name($cpu: crate::cpu::Cpu $(, $arg: $ty)*) -> $ret {
            #[cfg(target_arch = "x86_64")]
            {
                #[target_feature(enable = "bmi1,bmi2")]
                #[inline(never)]
                fn fast($($arg: $ty),*) -> $ret {
                    $imp($($arg),*)
                }
                if $cpu.bmi2 {
                    return unsafe { fast($($arg),*) };
                }
            }
            let _ = $cpu;
            $imp($($arg),*)
        }
    };
}

pub(crate) use multiversion;
