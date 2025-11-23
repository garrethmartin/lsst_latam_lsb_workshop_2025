import yaml
import re
import subprocess

EXCLUDE_NAMES = {
    '_libgcc_mutex', '_openmp_mutex', 'blas', 'boost-cpp', 'brotli-bin', 'bzip2', 'c-ares',
    'cairo', 'cfitsio', 'charls', 'cyrus-sasl', 'dav1d', 'expat', 'fontconfig',
    'freetype', 'freexl', 'fribidi', 'geos', 'geotiff', 'giflib', 'graphite2',
    'hdf4', 'hdf5', 'icu', 'intel-openmp', 'jpeg', 'json-c', 'jxrlib',
    'kealib', 'krb5', 'ld_impl_linux-64', 'lerc', 'libaec', 'libavif', 'libboost',
    'libbrotlicommon', 'libbrotlidec', 'libbrotlienc', 'libcups', 'libcurl',
    'libdeflate', 'libdrm', 'libedit', 'libegl', 'libev', 'libevent', 'libffi',
    'libgcc', 'libgcc-ng', 'libgdal', 'libgfortran-ng', 'libgfortran5', 'libgl',
    'libglib', 'libglvnd', 'libglx', 'libgomp', 'libiconv', 'libkml', 'libkrb5',
    'libllvm15', 'libnetcdf', 'libnghttp2', 'libopengl', 'libpciaccess', 'libpng',
    'libpq', 'libsodium', 'libspatialite', 'libssh2', 'libstdcxx-ng', 'libtiff',
    'libuuid', 'libwebp-base', 'libxcb', 'libxkbcommon', 'libxml2', 'libzip',
    'libzlib', 'libzopfli', 'lmdb', 'lz4-c', 'mesalib', 'minizip', 'mkl',
    'mpi', 'mpich', 'mysql-common', 'mysql-libs', 'ncurses', 'nspr', 'nss',
    'openjpeg', 'openldap', 'openssl', 'pandoc', 'pcre2', 'pixman', 'poppler',
    'poppler-data', 'proj', 'pthread-stubs', 'qhull', 'qtbase', 'qtdeclarative',
    'qtsvg', 'qttools', 'qtwayland', 'qtwebchannel', 'qtwebsockets', 'readline',
    'snappy', 'spirv-tools', 'sqlite', 'tbb', 'tbb-devel', 'tk', 'tzdata',
    'unicodedata2', 'uriparser', 'wayland', 'wcwidth', 'xcb-util', 'xcb-util-cursor',
    'xcb-util-image', 'xcb-util-keysyms', 'xcb-util-renderutil', 'xcb-util-wm',
    'xerces-c', 'xkeyboard-config', 'xorg-libice', 'xorg-libsm', 'xorg-libx11',
    'xorg-libxau', 'xorg-libxdmcp', 'xorg-libxext', 'xorg-libxfixes', 'xorg-libxrandr',
    'xorg-libxrender', 'xorg-libxshmfence', 'xorg-libxxf86vm', 'xorg-xorgproto',
    'xz', 'zfp', 'zlib', 'zlib-ng', 'zstd'
}

def setup_environment(env_file='environment_colab.yml'):
    """Parse environment.yml, filter pip-compatible packages, and install them."""
    with open(env_file) as f:
        env = yaml.safe_load(f)

    dependencies = env.get('dependencies', [])
    pip_packages_raw = []

    for dep in dependencies:
        if isinstance(dep, str):
            if dep.startswith('python'):
                continue
            clean_dep = re.split(r'=[^=]*$', dep)[0]
            pip_packages_raw.append(clean_dep)
        elif isinstance(dep, dict) and 'pip' in dep:
            pip_packages_raw.extend(dep['pip'])

    # Filter out system packages
    filtered_pip_packages = []
    for pkg_spec in pip_packages_raw:
        pkg_name = re.split(r'[=<>!~]', pkg_spec)[0]
        if pkg_name.lower() not in EXCLUDE_NAMES and not pkg_name.startswith('_'):
            filtered_pip_packages.append(pkg_spec)

    # Deduplicate
    filtered_pip_packages = list(set(filtered_pip_packages))

    if filtered_pip_packages:
        print("Installing filtered packages:", filtered_pip_packages)
        try:
            subprocess.check_call(['pip', 'install'] + filtered_pip_packages)
        except subprocess.CalledProcessError:
            print("Batch install failed. Attempting individual installs...")
            for pkg in filtered_pip_packages:
                try:
                    print(f"Installing {pkg}...")
                    subprocess.check_call(['pip', 'install', pkg])
                    print(f"Successfully installed {pkg}")
                except subprocess.CalledProcessError:
                    print(f"Failed to install {pkg}, you may need to handle it manually.")
    else:
        print("No pip packages to install after filtering.")
