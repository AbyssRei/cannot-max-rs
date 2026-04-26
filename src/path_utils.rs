use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// 路径存储策略
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathStorageStrategy {
    Relative,
    Absolute,
}

/// 规范化路径结果
#[derive(Debug, Clone)]
pub struct NormalizedPath {
    pub display: PathBuf,
    pub stored: PathBuf,
    pub strategy: PathStorageStrategy,
}

/// 判断路径是否为 base 的子路径
pub fn is_subpath(path: &Path, base: &Path) -> bool {
    let Ok(canonical_path) = path.canonicalize() else {
        return false;
    };
    let Ok(canonical_base) = base.canonicalize() else {
        return false;
    };
    canonical_path.starts_with(&canonical_base)
}

/// 将路径转换为相对路径（如果是子路径），否则返回原绝对路径
pub fn to_relative_if_subpath(path: &Path, base: &Path) -> PathBuf {
    if let Ok(canonical_path) = path.canonicalize() {
        if let Ok(canonical_base) = base.canonicalize() {
            if let Ok(relative) = canonical_path.strip_prefix(&canonical_base) {
                return relative.to_path_buf();
            }
        }
    }
    path.to_path_buf()
}

/// 规范化路径：判断子路径关系并选择存储策略
pub fn normalize_path(path: &Path, base: &Path) -> NormalizedPath {
    let display = path.to_path_buf();

    if is_subpath(path, base) {
        let stored = to_relative_if_subpath(path, base);
        NormalizedPath {
            display,
            stored,
            strategy: PathStorageStrategy::Relative,
        }
    } else {
        NormalizedPath {
            display,
            stored: path.to_path_buf(),
            strategy: PathStorageStrategy::Absolute,
        }
    }
}

/// 解析配置中的路径（支持相对和绝对）
pub fn resolve_path(path: &Path, base: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn is_subpath_child_directory() {
        let base = std::env::current_dir().unwrap();
        let models_dir = base.join("models");
        let _ = fs::create_dir_all(&models_dir);
        let child = models_dir.join("model.safetensors");
        // Create the file so canonicalize works
        let _ = fs::write(&child, b"test");
        assert!(is_subpath(&child, &base));
        let _ = fs::remove_file(&child);
        let _ = fs::remove_dir(&models_dir);
    }

    #[test]
    fn is_subpath_unrelated_paths() {
        let base = std::env::current_dir().unwrap();
        // Use temp dir which is definitely not a subpath
        let other = std::env::temp_dir();
        if base != other && !other.starts_with(&base) {
            assert!(!is_subpath(&other, &base));
        }
    }

    #[test]
    fn normalize_path_subpath_returns_relative() {
        let base = std::env::current_dir().unwrap();
        let models_dir = base.join("models");
        let _ = fs::create_dir_all(&models_dir);
        let child = models_dir.join("test.safetensors");
        let _ = fs::write(&child, b"test");
        let result = normalize_path(&child, &base);
        assert_eq!(result.strategy, PathStorageStrategy::Relative);
        assert_eq!(result.stored, PathBuf::from("models").join("test.safetensors"));
        let _ = fs::remove_file(&child);
        let _ = fs::remove_dir(&models_dir);
    }

    #[test]
    fn resolve_path_relative() {
        let base = Path::new("C:\\Projects\\cannot-max-rs");
        let relative = Path::new("models/model.safetensors");
        let resolved = resolve_path(relative, base);
        assert_eq!(resolved, base.join("models/model.safetensors"));
    }

    #[test]
    fn resolve_path_absolute() {
        let base = Path::new("C:\\Projects\\cannot-max-rs");
        let absolute = Path::new("D:\\Models\\model.safetensors");
        let resolved = resolve_path(absolute, base);
        assert_eq!(resolved, absolute);
    }
}
