# WARGNING: this script assumes it's ran from repo's root
# stops the execution of a script if a command or pipeline has an error
set -e
#  echo $BASE_DIR -> /Users/.../pytorch-0.1.1
# pwd为执行命令所在的目录,即执行命令行所在的目录
BASE_DIR=$(pwd)
cd torch/lib
# /Users/.../torch/lib/tmp_install
# 安装目录为tmp_install
INSTALL_DIR=$(pwd)/tmp_install

BASIC_FLAGS=" -DTH_INDEX_BASE=0 -I$INSTALL_DIR/include -I$INSTALL_DIR/include/TH -I$INSTALL_DIR/include/THC "
LDFLAGS="-L$INSTALL_DIR/lib "

# 不懂
# -Wl:rpath, 后面也是路径，运行的时候用。这条编译指令会在编译时记录到target文件中，所以编译之后的target文件在执行时会按这里给出的路径去找库文件。
#    如：-Wl:rpath=/home/hello/lib
#    表示将/home/hello/lib目录作为程序运行时第一个寻找库文件的目录，程序寻找顺序是：/home/hello/lib-->/usr/lib-->/usr/local/lib
#@loader_path 这个变量表示每一个被加载的 binary (包括可执行程序, dylib, framework 等) 所在的目录 ,在一个进程中, 对于每一个模块, @loader_path 会解析成不用的路径

if [[ $(uname) == 'Darwin' ]]; then
    LDFLAGS="$LDFLAGS -Wl,-rpath,@loader_path"
else
    LDFLAGS="$LDFLAGS -Wl,-rpath,\$ORIGIN"
fi
FLAGS="$BASIC_FLAGS $LDFLAGS"
# 函数  $1 为第一个参数

#find_package采用两种模式搜索库
#Module模式：搜索CMAKE_MODULE_PATH指定路径下的FindXXX.cmake文件，执行该文件从而找到XXX库。
# 其中，具体查找库并给XXX_INCLUDE_DIRS和XXX_LIBRARIES两个变量赋值的操作由FindXXX.cmake模块完成


# make -j带一个参数，可以把项目在进行并行编译，比如在一台双核的机器上，完全可以用make -j4，让make最多允许4个编译命令同时执行，这样可以更有效的利用CPU资源
# 获取cpu核数getconf _NPROCESSORS_ONLN

# 不懂
# otool查看依赖的动态库 otool -L /usr/local/bin/execute_bin
# install_name_tool来修改应用程序对动态库的查找路径  install_name_tool -change old_path new_path execute_bin
#-id:设置链接动态库时将使用的"安装名称".它将在目标动态库文件上运行.
#-change:这会在链接后更改"安装名称"，并将在链接到目标动态库的可执行文件或动态库上运行.
# install_name_tool 的id参数来修改动态库的安装名称
# install_name_tool -id new_path execute_bin
#@rpath是一个类似Shell中的PATH的变量，程序在执行时会从@rpath指定的路径中寻找动态链接库文件
#在gcc中，设置RPATH的办法很简单，就是设置linker的rpath选项：$ gcc -Wl,-rpath,/your/rpath/ test.cpp

function build() {
  mkdir -p build/$1
  cd build/$1
  cmake ../../$1 -DCMAKE_MODULE_PATH="$BASE_DIR/cmake/FindCUDA" \
              -DTorch_FOUND="1" \
              -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
              -DCMAKE_C_FLAGS="$FLAGS" \
              -DCMAKE_CXX_FLAGS="$FLAGS" \
              -DCUDA_NVCC_FLAGS="$BASIC_FLAGS" \
              -DTH_INCLUDE_PATH="$INSTALL_DIR/include"
#  make install -j$(getconf _NPROCESSORS_ONLN)
  cd ../..

#  if [[ $(uname) == 'Darwin' ]]; then
#    cd tmp_install/lib
#    for lib in *.dylib; do
#      echo "Updating install_name for $lib"
#      install_name_tool -id @rpath/$lib $lib
#    done
#    cd ../..
#  fi
}

mkdir -p tmp_install
# 构建TH Library
build TH
# 构建THNN Library
build THNN

if [[ "$1" == "--with-cuda" ]]; then
    build THC
    build THCUNN
fi

#cp $INSTALL_DIR/lib/* .
# 复制到torch/lib目录下，以便 generate_nn_wrappers
cp THNN/generic/THNN.h .
cp THCUNN/THCUNN.h .
