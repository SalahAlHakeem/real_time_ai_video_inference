################################################################################
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

APP:= main

CXX=g++

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream/lib/

ifeq ($(TARGET_DEVICE),aarch64)
  CXXFLAGS:= -DPLATFORM_TEGRA
endif

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0

OBJS:= main.o ds_yml_parse.o

CXXFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/includes \
	 -I/opt/nvidia/deepstream/deepstream/sources/includes/cvcore_headers \
         -I /usr/local/cuda/include -I ./common -I/usr/local/include


CXXFLAGS+= `pkg-config --cflags $(PKGS)`
CXXFLAGS+= `pkg-config --cflags --libs opencv4`

CXXFLAGS+= -D_GLIBCXX_USE_CXX11_ABI=1 -Wno-sign-compare -Wno-deprecated-declarations


LIBS:= `pkg-config --libs $(PKGS)`
LIBS+= `pkg-config --libs opencv4`


LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_inferutils -lnvbufsurface -lnvbufsurftransform -lnvdsgst_helper \
       -lnvds_utils -lm -lstdc++ -lnvds_yml_parser -lgstrtspserver-1.0 -lnvds_batch_jpegenc\
       -L/usr/local/cuda/lib64/ -lcudart -lcuda -lyaml-cpp \
       -L/opt/nvidia/deepstream/deepstream/lib/cvcore_libs \
       -lnvcv_core -lnvcv_tensorops -lnvcv_trtbackend \
       -Wl,-rpath,$(LIB_INSTALL_DIR) \
       -L/usr/local/lib -lrabbitmq \
       -L/usr/lib/x86_64-linux-gnu -lcryptopp


all: $(APP)

%.o: %.cpp $(INCS) Makefile
	$(CXX) -c -o $@ $(CXXFLAGS) $<

main.o: main.cpp $(INCS) Makefile
	$(CXX) -c -o $@ -fpermissive -Wall $(CXXFLAGS) $<


ds_yml_parse.o: common/ds_yml_parse.cpp $(INCS) Makefile
	$(CXX) -c -o $@ -Wall  $(CXXFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

clean:
	rm -rf $(OBJS) $(APP)
