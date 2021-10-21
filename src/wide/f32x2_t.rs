// Copyright 2020 Yevhenii Reizner
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#[cfg(all(not(feature = "std"), feature = "libm"))]
use crate::scalar::FloatExt;

use bytemuck::cast;

// Right now, there are no visible benefits of using SIMD for f32x2. So we
// don't.
cfg_if::cfg_if! {
    if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
        use core::arch::aarch64::*;

        #[derive(Clone, Copy, Debug)]
        #[repr(C, align(8))]
        pub struct f32x2(float32x2_t);

        impl Default for f32x2 {
            fn default() -> Self {
                Self::splat(0.0)
            }
        }

        impl PartialEq for f32x2 {
            fn eq(&self, other: &Self) -> bool {
                unsafe { vminv_u32(vceq_f32(self.0, other.0)) != 0 }
            }
        }
    } else {
        #[allow(non_camel_case_types)]
        #[derive(Copy, Clone, Default, PartialEq, Debug)]
        #[repr(C, align(8))]
        pub struct f32x2([f32; 2]);
    }
}

unsafe impl bytemuck::Zeroable for f32x2 {}
unsafe impl bytemuck::Pod for f32x2 {}

impl f32x2 {
    pub fn new(a: f32, b: f32) -> f32x2 {
        Self::from([a, b])
    }

    pub fn splat(x: f32) -> f32x2 {
        Self::from([x, x])
    }

    pub fn abs(self) -> f32x2 {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                unsafe { Self(vabs_f32(self.0)) }
            } else {
                Self::from([
                    self.x().abs(),
                    self.y().abs(),
                ])
            }
        }
    }

    pub fn min(self, other: f32x2) -> f32x2 {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                unsafe { Self(vmin_f32(self.0, other.0)) }
            } else {
                f32x2([
                    super::pmin(self.x(), other.x()),
                    super::pmin(self.y(), other.y()),
                ])
            }
        }
    }

    pub fn max(self, other: f32x2) -> f32x2 {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                unsafe { Self(vmax_f32(self.0, other.0)) }
            } else {
                f32x2([
                    super::pmax(self.x(), other.x()),
                    super::pmax(self.y(), other.y()),
                ])
            }
        }
    }

    pub fn max_component(self) -> f32 {
        // TODO: vpmaxs_f32 isn't available in Rust yet.
        // cfg_if::cfg_if! {
        //     if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
        //         unsafe { vpmaxs_f32(self.0) }
        //     } else {
                super::pmax(self.x(), self.y())
            // }
        // }
    }

    pub fn x(&self) -> f32 { <[f32; 2]>::from(*self)[0] }
    pub fn y(&self) -> f32 { <[f32; 2]>::from(*self)[1] }
}

impl From<[f32; 2]> for f32x2 {
    fn from(v: [f32; 2]) -> Self {
        cast(v)
    }
}

impl From<f32x2> for [f32; 2] {
    fn from(v: f32x2) -> Self {
        cast(v)
    }
}

impl core::ops::Add<f32x2> for f32x2 {
    type Output = f32x2;

    fn add(self, other: f32x2) -> f32x2 {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                unsafe { Self(vadd_f32(self.0, other.0)) }
            } else {
                f32x2([
                    self.x() + other.x(),
                    self.y() + other.y(),
                ])
            }
        }
    }
}

impl core::ops::Sub<f32x2> for f32x2 {
    type Output = f32x2;

    fn sub(self, other: f32x2) -> f32x2 {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                unsafe { Self(vsub_f32(self.0, other.0)) }
            } else {
                f32x2([
                    self.x() - other.x(),
                    self.y() - other.y(),
                ])
            }
        }
    }
}

impl core::ops::Mul<f32x2> for f32x2 {
    type Output = f32x2;

    fn mul(self, other: f32x2) -> f32x2 {
        cfg_if::cfg_if! {
            if #[cfg(all(feature = "simd", target_arch = "aarch64", target_feature = "neon"))] {
                unsafe { Self(vmul_f32(self.0, other.0)) }
            } else {
                f32x2([
                    self.x() * other.x(),
                    self.y() * other.y(),
                ])
            }
        }
    }
}
